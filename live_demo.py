"""
live_demo.py  —  a stage-safe live demo of the pipeline (no Claude, no internet).
=================================================================================
Runs ONE MedQA question end-to-end in front of an audience:

    question  ->  (cached) top-4 retrieved passages  ->  Neutral & Strict prompts
              ->  qwen2.5:3b answers live (Ollama)  ->  accuracy + abstention

Design choices that make it safe on stage:
  * Uses the CACHED retrievals (pilot_100Q/retrievals_seed42.json), so it does NOT
    load the 209 MB FAISS index or the embedding model — fast, low memory.
  * Only needs Ollama running with qwen2.5:3b.
  * If Ollama is unreachable, it FALLS BACK to the pre-recorded answers in
    scored_main.csv, so the demo never dies in front of the room.

Usage:
    python3 live_demo.py --warm        # do this BEFORE the talk: loads the model
    python3 live_demo.py               # runs the default question (QID 1 = a miss)
    python3 live_demo.py --qid 12      # a retrieval hit (answers correctly)
    python3 live_demo.py --menu        # pick from a few curated questions
"""
import os, sys, json, time, argparse, textwrap
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
RETR = os.path.join(HERE, "pilot_100Q", "retrievals_seed42.json")
SCORED = os.path.join(HERE, "pilot_100Q", "scored_main.csv")

# curated questions for the stage
CURATED = {
    1:  "retrieval MISS  ->  Neutral guesses (wrong), STRICT abstains honestly",
    12: "retrieval HIT   ->  STRICT answers correctly (Anticipation)",
    14: "retrieval HIT   ->  all three answer correctly (Dysthymia)",
}

C_GREEN, C_RED, C_CY, C_BOLD, C_DIM, C_END = "\033[92m", "\033[91m", "\033[96m", "\033[1m", "\033[2m", "\033[0m"


def hr(ch="="): print(ch * 78)
def load_retr():
    return {int(x["qid"]): x for x in json.load(open(RETR))}


def cached_answer(qid, strat):
    import csv
    with open(SCORED) as f:
        for row in csv.DictReader(f):
            if int(row["qid"]) == qid and row["strategy"] == strat:
                return row["answer_text"], row["predicted"], row["abstained"] == "True"
    return "(no cached answer)", "", False


def run(qid):
    import run_pipeline as R
    try:
        import requests
    except Exception:
        requests = None
    data = load_retr()
    if qid not in data:
        print(f"QID {qid} not in cached set. Try one of: {sorted(CURATED)}"); return
    item = data[qid]
    opts = item["options"]; gold = item["answer_idx"]

    hr()
    print(f"{C_BOLD}QUESTION (MedQA QID {qid}){C_END}")
    hr()
    print(textwrap.fill(item["question"], 76))
    print()
    for k, v in opts.items():
        print(f"   {C_BOLD}{k}{C_END}. {v}")
    print(f"\n   {C_DIM}(correct answer: {gold} = {opts[gold]}){C_END}\n")

    # ---- Stage 2: retrieval (from cache) ----
    hr("-")
    print(f"{C_BOLD}STAGE 2 — top-4 passages retrieved by FAISS (cached){C_END}")
    hr("-")
    for r in item["retrieved"][:4]:
        snippet = " ".join(r["text"].split()[:16])
        print(f"  rank {r['rank']}  cosine {C_CY}{r['score']:.3f}{C_END}  {r['source']}")
        print(f"       {C_DIM}{snippet}...{C_END}")
    print()

    # ---- Stages 3-5: generate Neutral then Strict ----
    for strat, label in [("neutral", "NEUTRAL prompt"), ("strict_citations", "STRICT prompt (cite + may refuse)")]:
        hr("-")
        print(f"{C_BOLD}{label}{C_END}")
        hr("-")
        prompt = R.build_prompt(strat, item["question"], opts, item["retrieved"][:4])
        ans, live = None, False
        if requests is not None:
            try:
                t0 = time.time()
                print(f"{C_DIM}  ...asking qwen2.5:3b (live)...{C_END}", flush=True)
                ans = R.ask_ollama(prompt)
                dt = time.time() - t0; live = True
            except Exception as e:
                print(f"{C_DIM}  Ollama not reachable ({type(e).__name__}) — showing the pre-recorded answer.{C_END}")
        if ans is None:
            ans, pred, abst = cached_answer(qid, strat); dt = 0.0
        else:
            pred = R.parse_answer_letter(ans)
            abst = any(c in ans.lower() for c in
                       ("does not contain", "cannot determine", "insufficient", "not provide",
                        "context does not", "cannot be determined", "not mention"))
        print(textwrap.fill(ans.replace("\n", " ").strip(), 76))
        correct = (pred == gold)
        tag = (f"{C_RED}ABSTAINED (refused){C_END}" if abst
               else (f"{C_GREEN}CORRECT{C_END}" if correct else f"{C_RED}WRONG{C_END}"))
        extra = f"  answer={pred}" if pred else ""
        speed = f"  ({dt:.1f}s live)" if live else "  (cached)"
        print(f"\n   -> {tag}{extra}{speed}\n")

    hr()
    print(f"{C_BOLD}Takeaway:{C_END} same evidence, different instruction. On this question the "
          f"strict prompt's behaviour differs from the neutral one — the whole point of the study.")
    hr()


def warm():
    import run_pipeline as R
    print("Warming qwen2.5:3b (loads it into memory so the live demo is fast)...")
    t0 = time.time()
    try:
        R.ask_ollama("Reply with the single word: ready.")
        print(f"Model warm and ready in {time.time()-t0:.1f}s. You can start the talk.")
    except Exception as e:
        print(f"Could NOT reach Ollama: {e}\nStart it first:  ollama serve   (and: ollama pull qwen2.5:3b)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=int, default=1)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--menu", action="store_true")
    args = ap.parse_args()
    if args.warm:
        warm(); sys.exit()
    if args.menu:
        print("Curated demo questions:")
        for q, d in CURATED.items():
            print(f"   {q:2d}  —  {d}")
        try:
            args.qid = int(input("\nEnter a QID (or just Enter for 1): ") or "1")
        except Exception:
            args.qid = 1
    run(args.qid)
