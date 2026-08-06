
import os
# Pin thread counts BEFORE importing numpy/torch/faiss to avoid the OpenMP
# conflict between FAISS and PyTorch that segfaults on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(HERE, "rag_store")
MEDQA = os.path.join(HERE, "data_clean", "questions", "US", "4_options",
                     "phrases_no_exclude_dev.jsonl")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"   # 8 GB RAM: small model avoids swap (llama3.1 8B swaps)
RETRIEVE_K = 8                # store this many; generation slices to --topk
CHUNK_WORDS_IN_PROMPT = 130   # truncate each chunk to bound prompt size
ALL_STRATEGIES = ("neutral", "strict_citations", "chain_of_thought")


# ----------------------------------------------------------------------------
# Load the FAISS store (built by build_index.py)
# ----------------------------------------------------------------------------

def load_store():
    import faiss
    faiss.omp_set_num_threads(1)
    index = faiss.read_index(os.path.join(STORE_DIR, "faiss.index"))
    with open(os.path.join(STORE_DIR, "chunks.pkl"), "rb") as f:
        d = pickle.load(f)
    meta = json.load(open(os.path.join(STORE_DIR, "meta.json")))
    return index, d["chunks"], d["sources"], meta


# ----------------------------------------------------------------------------
# Stage 2 — retrieval (always fetch RETRIEVE_K; slice later)
# ----------------------------------------------------------------------------
def retrieve(question, embed_model, index, chunks, sources, top_k=RETRIEVE_K):
    q = embed_model.encode([question], normalize_embeddings=True,
                           convert_to_numpy=True).astype("float32")
    scores, idx = index.search(q, top_k)
    out = []
    for rank, (i, s) in enumerate(zip(idx[0], scores[0])):
        out.append({"rank": rank + 1, "score": float(s),
                    "source": sources[i], "text": chunks[i]})
    return out


# ----------------------------------------------------------------------------
# Stage 3 — the three prompting strategies
# ----------------------------------------------------------------------------
def build_prompt(strategy, question, options, retrieved):
    def short(t):
        return " ".join(t.split()[:CHUNK_WORDS_IN_PROMPT])
    context = "\n\n".join(
        f"[Source {r['rank']} | {r['source']}]\n{short(r['text'])}" for r in retrieved
    )
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())

    if strategy == "neutral":
        instr = ("Use the context below to help answer the multiple-choice "
                 "medical question. Give a brief explanation, then state the "
                 "final answer as 'Answer: <letter>'.")
    elif strategy == "strict_citations":
        instr = ("Answer the multiple-choice medical question using ONLY the "
                 "context below. Every claim you make must be supported by the "
                 "context and cited as [Source N]. If the context does not "
                 "contain the needed information, say so. End with "
                 "'Answer: <letter>'.")
    elif strategy == "chain_of_thought":
        # Stronger baseline: explicit step-by-step reasoning over the context,
        # no citation requirement and no licence to abstain.
        instr = ("Use the context below to answer the multiple-choice medical "
                 "question. Think step by step: reason through the relevant "
                 "findings one at a time before deciding. Then state the final "
                 "answer as 'Answer: <letter>'.")
    else:
        raise ValueError(strategy)

    prompt = (f"{instr}\n\n=== CONTEXT ===\n{context}\n\n"
              f"=== QUESTION ===\n{question}\n\nOptions:\n{opts}\n\nResponse:")
    return prompt


# ----------------------------------------------------------------------------
# Stage 4 — real local LLM via Ollama
# ----------------------------------------------------------------------------
def ask_ollama(prompt, model=OLLAMA_MODEL, temperature=0.0, num_predict=400):
    r = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }, timeout=600)
    r.raise_for_status()
    return r.json()["response"].strip()


# ----------------------------------------------------------------------------
# Stage 5 — lightweight evaluation (accuracy + parse); grounding via rescore.py
# ----------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text):
    # strip the "Answer: X" line so the final answer isn't scored as a claim
    text = re.sub(r"Answer:\s*[A-D].*$", "", text, flags=re.I | re.S)
    sents = [s.strip() for s in _SENT.split(text) if len(s.strip()) > 15]
    return sents

def parse_answer_letter(text):
    m = re.search(r"Answer:\s*([A-D])", text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])[.):]", text)        # fallback
    return m.group(1).upper() if m else None


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def load_questions(n, seed=42):
    df = pd.read_json(MEDQA, lines=True)
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def phase_retrieve(n, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    index, chunks, sources, meta = load_store()
    print(f"Store: {meta['n_chunks']:,} chunks from {meta['n_textbooks']} "
          f"textbooks ({meta['embed_model']})")
    embed_model = SentenceTransformer(meta["embed_model"], device="cpu")
    qs = load_questions(n, seed=seed)
    out = []
    for qi, row in qs.iterrows():
        retrieved = retrieve(row["question"], embed_model, index, chunks, sources)
        out.append({"qid": int(qi), "question": row["question"],
                    "options": row["options"], "answer_idx": row["answer_idx"],
                    "answer_text_gold": row.get("answer", ""),
                    "retrieved": retrieved})
        print(f"retrieved Q{qi:02d} top={retrieved[0]['source']} "
              f"({retrieved[0]['score']:.3f})")
    path = os.path.join(outdir, f"retrievals_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Saved {len(out)} retrievals (top-{RETRIEVE_K}) -> {path}")


def phase_generate(strategies, topk, seed, outdir, model=OLLAMA_MODEL, tag=""):
    retr_path = os.path.join(outdir, f"retrievals_seed{seed}.json")
    data = json.load(open(retr_path))
    print(f"Loaded {len(data)} retrievals from {retr_path}. Model={model}, "
          f"topk={topk}. {len(data)*len(strategies)} LLM calls.\n")
    rows = []
    details = []
    for item in data:
        qi = item["qid"]
        retrieved = item["retrieved"][:topk]        # slice to requested depth
        for strat in strategies:
            prompt = build_prompt(strat, item["question"], item["options"], retrieved)
            t0 = time.time()
            ans = ask_ollama(prompt, model=model)
            dt = time.time() - t0
            pred = parse_answer_letter(ans)
            correct = (pred == item["answer_idx"])
            rows.append({
                "qid": qi, "strategy": strat, "topk": topk, "seed": seed,
                "correct": bool(correct),
                "predicted": pred, "gold": item["answer_idx"],
                "n_sentences": len(split_sentences(ans)),
                "retrieval_top_score": round(retrieved[0]["score"], 3),
                "retrieval_mean_score": round(
                    float(np.mean([r["score"] for r in retrieved])), 3),
                "latency_s": round(dt, 1),
                "answer_text": ans,
            })
            details.append({"qid": qi, "strategy": strat, "topk": topk,
                            "question": item["question"],
                            "answer_idx": item["answer_idx"],
                            "answer_text_gold": item.get("answer_text_gold", ""),
                            "retrieved": retrieved,
                            "answer_text": ans})
            print(f"Q{qi:02d} {strat:17s} k={topk} correct={correct!s:5} "
                  f"n_sent={len(split_sentences(ans)):2d} ({dt:.1f}s)")

    suffix = tag or f"seed{seed}_k{topk}"
    df = pd.DataFrame(rows)
    res_path = os.path.join(outdir, f"results_{suffix}.csv")
    det_path = os.path.join(outdir, f"results_detail_{suffix}.json")
    df.to_csv(res_path, index=False)
    json.dump(details, open(det_path, "w"), indent=2)
    print(f"\nWrote {res_path}  ({len(df)} rows)")
    print(f"Wrote {det_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["retrieve", "generate"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--strategies", default=",".join(ALL_STRATEGIES),
                    help="comma-separated subset of "
                         "neutral,strict_citations,chain_of_thought")
    ap.add_argument("--outdir", default="pilot_100Q")
    ap.add_argument("--tag", default="", help="override output filename suffix")
    args = ap.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)
    if args.phase == "retrieve":
        phase_retrieve(args.n, args.seed, outdir)
    else:
        strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
        phase_generate(strategies, args.topk, args.seed, outdir, tag=args.tag)
