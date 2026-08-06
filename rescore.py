
import os, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
from functools import lru_cache

from run_pipeline import split_sentences

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THRESHOLD = 0.65

# Keyword rule for abstention. Kept deliberately simple and reported as a
# limitation; an entailment/LLM detector is future work.
ABSTAIN_CUES = (
    "does not contain", "not contain specific", "cannot determine",
    "not enough information", "no information", "cannot be determined",
    "does not provide", "not provided in the context", "context does not",
    "unable to determine", "not mention", "insufficient", "does not include",
    "no relevant information", "not specify", "cannot be answered",
    "not available in the context", "lacks", "does not specify",
)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def _is_abstention(text):
    t = str(text).lower()
    return any(cue in t for cue in ABSTAIN_CUES)


def score_df(df, retr_by_qid, threshold=DEFAULT_THRESHOLD, return_detail=False):
    """Add usr / faithfulness / abstained / knowledge_override to df in place.

    df must have columns: qid, topk, answer_text, correct.
    retr_by_qid maps qid -> list of retrieved chunks (top-8; sliced to topk).
    """
    model = _model()
    usr_list, faith_list, details = [], [], []
    for _, r in df.iterrows():
        k = int(r["topk"]) if "topk" in r and not pd.isna(r["topk"]) else 4
        chunks = [c["text"] for c in retr_by_qid[int(r["qid"])][:k]]
        sents = split_sentences(str(r["answer_text"]))
        if not sents or not chunks:
            usr_list.append(1.0); faith_list.append(0.0); details.append([]); continue
        se = model.encode(sents, normalize_embeddings=True, convert_to_numpy=True)
        ce = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        sims = se @ ce.T
        mx = sims.max(axis=1)
        grounded = mx >= threshold
        usr_list.append(float((~grounded).mean()))
        faith_list.append(float(mx.mean()))
        if return_detail:
            bi = sims.argmax(axis=1)
            details.append([{"sentence": s, "max_sim": round(float(m), 3),
                             "grounded": bool(g)}
                            for s, m, g in zip(sents, mx, grounded)])
    df = df.copy()
    df["usr"] = np.round(usr_list, 3)
    df["faithfulness"] = np.round(faith_list, 3)
    df["abstained"] = df["answer_text"].apply(_is_abstention)
    df["knowledge_override"] = df["correct"].astype(bool) & (df["usr"] > 0.5)
    if return_detail:
        return df, details
    return df


def load_retr(path):
    return {int(item["qid"]): item["retrieved"] for item in json.load(open(path))}


def main(results, retrievals, threshold, out):
    df = pd.read_csv(results)
    retr = load_retr(retrievals)
    df = score_df(df, retr, threshold=threshold)
    out = out or results
    df.to_csv(out, index=False)
    summ = df.groupby("strategy").agg(
        accuracy=("correct", "mean"),
        mean_USR=("usr", "mean"),
        mean_faithfulness=("faithfulness", "mean"),
        abstention_rate=("abstained", "mean"),
        knowledge_override_rate=("knowledge_override", "mean"),
        mean_latency_s=("latency_s", "mean"),
    ).round(3)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"Threshold = {threshold}   ({len(df)} rows) -> {out}")
    print(summ.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--retrievals", required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    main(args.results, args.retrievals, args.threshold, args.out)
