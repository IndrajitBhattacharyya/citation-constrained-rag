
import os, re, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
from functools import lru_cache

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
REL_THRESHOLD = 0.55          # semantic relevance cut-off for a retrieval "hit"
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "with", "for"}


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def _salient_tokens(text):
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", str(text).lower())
    return [t for t in toks if t not in _STOP]


def _lexical_hit(gold_text, chunks):
    g = str(gold_text).lower().strip()
    if not g:
        return False
    joined = " ".join(c.lower() for c in chunks)
    if g in joined:                                   # whole answer phrase present
        return True
    toks = _salient_tokens(gold_text)
    if not toks:
        return False
    # a hit if any salient content token of the gold answer appears
    return any(t in joined for t in toks)


def audit_at_k(retrievals, k, rel_threshold=REL_THRESHOLD):
    """Return a DataFrame (one row per question) of retrieval-relevance signals
    when only the top-k passages are used."""
    model = _model()
    rows = []
    for item in retrievals:
        gold_text = item.get("answer_text_gold", "")
        chunks = [c["text"] for c in item["retrieved"][:k]]
        # Probe with the gold ANSWER alone (continuous diagnostic only).
        qe = model.encode([str(gold_text)], normalize_embeddings=True, convert_to_numpy=True)
        ce = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        sims = (qe @ ce.T)[0]
        sem_max = float(sims.max())
        lex = _lexical_hit(gold_text, chunks)
        rows.append({
            "qid": int(item["qid"]), "k": k,
            "gold_text": gold_text,
            "lexical_hit": bool(lex),
            "answer_semantic_max": round(sem_max, 3),
            "relevant_hit": bool(lex),          # lexical is the validated signal
        })
    return pd.DataFrame(rows)


def main(retrievals_path, k, out):
    retr = json.load(open(retrievals_path))
    aud = audit_at_k(retr, k)
    if out:
        aud.to_csv(out, index=False)
    print(f"Retrieval audit at k={k} over {len(aud)} questions:")
    print(f"  lexical hit-rate (relevant_hit) = {aud.lexical_hit.mean():.3f}")
    print(f"  mean answer-semantic-max        = {aud.answer_semantic_max.mean():.3f}")
    if out:
        print(f"  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievals", default="pilot_100Q/retrievals_seed42.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args.retrievals, args.k, args.out)
