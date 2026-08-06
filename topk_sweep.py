"""
Re-generates answers at several retrieval depths (Top-k) on a fixed question
subset, so we can see how retrieval depth affects grounding (USR), abstention
and accuracy. Uses the top-8 passages already stored by run_pipeline.py, so no
re-embedding is needed — only the LLM is re-run at each k.

"""
import os, json, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import pandas as pd

import run_pipeline as R
from rescore import score_df, load_retr, DEFAULT_THRESHOLD


def main(outdir, subset, ks, strategies):
    retr_path = os.path.join(outdir, "retrievals_seed42.json")
    data = json.load(open(retr_path))[:subset]
    retr_by_qid = load_retr(retr_path)
    total = len(data) * len(ks) * len(strategies)
    print(f"Top-k sweep: {len(data)} questions x {ks} x {strategies} = {total} calls")
    rows = []
    n = 0
    for k in ks:
        for item in data:
            retrieved = item["retrieved"][:k]
            for strat in strategies:
                prompt = R.build_prompt(strat, item["question"], item["options"], retrieved)
                t0 = time.time()
                ans = R.ask_ollama(prompt)
                dt = time.time() - t0
                pred = R.parse_answer_letter(ans)
                n += 1
                rows.append({
                    "qid": item["qid"], "strategy": strat, "topk": k, "seed": 42,
                    "correct": bool(pred == item["answer_idx"]),
                    "predicted": pred, "gold": item["answer_idx"],
                    "n_sentences": len(R.split_sentences(ans)),
                    "retrieval_top_score": round(retrieved[0]["score"], 3),
                    "latency_s": round(dt, 1), "answer_text": ans,
                })
                print(f"[{n}/{total}] Q{item['qid']:02d} {strat:16s} k={k} "
                      f"correct={pred==item['answer_idx']!s:5} ({dt:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "results_topk_sweep_raw.csv"), index=False)
    df = score_df(df, retr_by_qid, threshold=DEFAULT_THRESHOLD)
    df.to_csv(os.path.join(outdir, "results_topk_sweep_scored.csv"), index=False)
    print("\nTop-k sweep summary (USR / abstention / accuracy):")
    print(df.groupby(["strategy", "topk"]).agg(
        usr=("usr", "mean"), abstention=("abstained", "mean"),
        accuracy=("correct", "mean")).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="pilot_100Q")
    ap.add_argument("--subset", type=int, default=30)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--strategies", default="strict_citations")
    args = ap.parse_args()
    outdir = args.outdir if os.path.isabs(args.outdir) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    ks = [int(x) for x in args.ks.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    main(outdir, args.subset, ks, strategies)
