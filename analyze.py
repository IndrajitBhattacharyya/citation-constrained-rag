
import os, json, argparse, warnings, textwrap
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from rescore import score_df, load_retr, DEFAULT_THRESHOLD
from retrieval_audit import audit_at_k

plt.rcParams.update({"figure.dpi": 130, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})
# colour per strategy
COL = {"neutral": "#4C72B0", "chain_of_thought": "#DD8452", "strict_citations": "#C44E52"}
NAME = {"neutral": "Neutral", "chain_of_thought": "Chain-of-Thought",
        "strict_citations": "Strict (cite+refuse)"}
STRATS = ["neutral", "chain_of_thought", "strict_citations"]
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers ----
def boot_ci(x, n=3000, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) < 2:
        return stat(x), stat(x), stat(x)
    bs = [stat(RNG.choice(x, len(x), replace=True)) for _ in range(n)]
    return stat(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def mcnemar(a, b):
    """paired binary a vs b -> (b_disc, c_disc, p) two-sided exact."""
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    bd = int((a & ~b).sum()); cd = int((~a & b).sum())
    p = stats.binomtest(min(bd, cd), bd + cd, 0.5).pvalue if (bd + cd) else 1.0
    return bd, cd, p


def wilcoxon(a, b):
    try:
        w, p = stats.wilcoxon(a, b)
        return w, p
    except ValueError:
        return float("nan"), 1.0


# ------------------------------------------------------- RQ1: 3-way + CIs ----
def rq1(df, outdir):
    piv = {s: df[df.strategy == s].sort_values("qid").reset_index(drop=True) for s in STRATS}
    
    rows = []
    for s in STRATS:
        g = piv[s]
        answered = g[~g.abstained]
        derived = {
            "accuracy_overall": g.correct.mean(),
            "coverage": (~g.abstained).mean(),
            "accuracy_when_answered": answered.correct.mean() if len(answered) else np.nan,
            "abstention_rate": g.abstained.mean(),
            "faithfulness": g.faithfulness.mean(),
            "usr": g.usr.mean(),
            "answer_length": g.n_sentences.mean(),
            "knowledge_override": g.knowledge_override.mean(),
            "latency_s": g.latency_s.mean(),
        }
        for metric, arr in [("accuracy_overall", g.correct), ("coverage", ~g.abstained),
                            ("abstention_rate", g.abstained), ("faithfulness", g.faithfulness),
                            ("usr", g.usr), ("answer_length", g.n_sentences),
                            ("knowledge_override", g.knowledge_override),
                            ("latency_s", g.latency_s)]:
            m, lo, hi = boot_ci(np.asarray(arr, float))
            rows.append({"strategy": s, "metric": metric, "mean": round(m, 3),
                         "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
        m, lo, hi = boot_ci(np.asarray(answered.correct, float)) if len(answered) else (np.nan,)*3
        rows.append({"strategy": s, "metric": "accuracy_when_answered",
                     "mean": round(m, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(outdir, "rq1_summary.csv"), index=False)


    sig = []
    pairs = [("strict_citations", "neutral"), ("chain_of_thought", "neutral"),
             ("strict_citations", "chain_of_thought")]
    for a, b in pairs:
        ga, gb = piv[a], piv[b]
        bd, cd, p = mcnemar(ga.correct, gb.correct)
        sig.append({"pair": f"{a} vs {b}", "metric": "accuracy", "test": "McNemar",
                    "detail": f"b={bd},c={cd}", "p": round(p, 5)})
        bd, cd, p = mcnemar(ga.abstained, gb.abstained)
        sig.append({"pair": f"{a} vs {b}", "metric": "abstention", "test": "McNemar",
                    "detail": f"b={bd},c={cd}", "p": round(p, 6)})
        for metric in ["usr", "faithfulness", "n_sentences"]:
            w, p = wilcoxon(ga[metric], gb[metric])
            sig.append({"pair": f"{a} vs {b}", "metric": metric, "test": "Wilcoxon",
                        "detail": f"W={w:.1f}", "p": round(p, 5)})
    sigdf = pd.DataFrame(sig)
    sigdf.to_csv(os.path.join(outdir, "rq1_significance.csv"), index=False)

    # ---- figure: 3-way bars with bootstrap CI on 4 headline metrics ----
    def bar(ax, metric, title, ylim=(0, 1)):
        means, los, his, cols = [], [], [], []
        for s in STRATS:
            sub = summ[(summ.strategy == s) & (summ.metric == metric)].iloc[0]
            means.append(sub["mean"]); los.append(sub["mean"] - sub.ci_lo)
            his.append(sub.ci_hi - sub["mean"]); cols.append(COL[s])
        x = np.arange(len(STRATS))
        ax.bar(x, means, yerr=[los, his], capsize=5, color=cols)
        ax.set_xticks(x); ax.set_xticklabels([NAME[s].replace(" ", "\n", 1) for s in STRATS],
                                             fontsize=8)
        ax.set_title(title)
        if ylim: ax.set_ylim(*ylim)
        for i, v in enumerate(means):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    bar(ax[0], "accuracy_overall", "Accuracy (overall)")
    bar(ax[1], "abstention_rate", "Abstention rate")
    bar(ax[2], "faithfulness", "Faithfulness")
    bar(ax[3], "usr", "USR (unsupported)")
    fig.suptitle(f"Three prompting strategies (N={df.qid.nunique()} questions, bootstrap 95% CI)",
                 fontsize=13, y=1.03)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_rq1_main.png"), bbox_inches="tight")
    plt.close()

   
    strict = piv["strict_citations"]; neu = piv["neutral"].set_index("qid")
    ab_ids = strict[strict.abstained].qid
    good = int((neu.loc[ab_ids].correct == 0).sum())    # neutral wrong -> abstaining sensible
    bad = int((neu.loc[ab_ids].correct == 1).sum())     # neutral right -> abstaining costly
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.bar(["Sensible\n(neutral was wrong)", "Costly\n(neutral was right)"], [good, bad],
           color=["#55A868", "#C44E52"])
    ax.set_title(f"When Strict abstained (n={len(ab_ids)}):\nwould the Neutral answer have been right?")
    ax.set_ylabel("number of questions")
    for i, v in enumerate([good, bad]):
        ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_abstention_outcome.png"),
                                    bbox_inches="tight"); plt.close()

    print("RQ1 written: rq1_summary.csv, rq1_significance.csv, fig_rq1_main.png, "
          "fig_abstention_outcome.png")
    print(f"  Strict abstentions: {len(ab_ids)}  sensible={good}  costly={bad}")
    return summ, sigdf, {"good": good, "bad": bad, "n_abst": int(len(ab_ids))}


# ------------------------------------------- RQ2: retrieval audit + top-k ----
def rq2(df, retr_list, retr_by_qid, outdir, topk_path):
    # (a) retrieval audit at k=4 and failure split on the NEUTRAL arm (it answers most)
    aud4 = audit_at_k(retr_list, 4)
    neu = df[df.strategy == "neutral"].set_index("qid")
    aud4 = aud4.set_index("qid")
    joined = aud4.join(neu[["correct", "abstained", "usr"]])
    answered = joined[~joined.abstained.fillna(False)]
    # failure taxonomy among ANSWERED neutral questions
    retr_fail = int(((answered.correct == 0) & (~answered.relevant_hit)).sum())
    gen_fail = int(((answered.correct == 0) & (answered.relevant_hit)).sum())
    correct_hit = int(((answered.correct == 1) & (answered.relevant_hit)).sum())
    correct_nohit = int(((answered.correct == 1) & (~answered.relevant_hit)).sum())
    # external validation of the lexical hit as a relevance signal:
    p_corr_hit = float(answered[answered.relevant_hit].correct.mean()) if (answered.relevant_hit).any() else float("nan")
    p_corr_miss = float(answered[~answered.relevant_hit].correct.mean()) if (~answered.relevant_hit).any() else float("nan")
    audit = {
        "hit_rate_k4": round(float(aud4.relevant_hit.mean()), 3),   # lexical (validated)
        "mean_answer_semantic_max_k4": round(float(aud4.answer_semantic_max.mean()), 3),
        "p_correct_given_hit": round(p_corr_hit, 3),
        "p_correct_given_miss": round(p_corr_miss, 3),
        "answered_n": int(len(answered)),
        "retrieval_failures": retr_fail,   # wrong AND no relevant passage retrieved
        "generation_failures": gen_fail,   # wrong DESPITE relevant passage retrieved
        "correct_with_hit": correct_hit,
        "correct_without_hit": correct_nohit,
    }
    json.dump(audit, open(os.path.join(outdir, "rq2_audit.json"), "w"), indent=2)
    aud4.reset_index().to_csv(os.path.join(outdir, "rq2_retrieval_audit.csv"), index=False)

    # failure-split figure
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    ax.bar(["Retrieval failure\n(no evidence\nretrieved)", "Generation failure\n(evidence was\nretrieved)"],
           [retr_fail, gen_fail], color=["#C44E52", "#8172B3"])
    ax.set_title(f"Neutral errors on answered questions (n={gen_fail+retr_fail}):\n"
                 "retrieval vs generation")
    ax.set_ylabel("number of wrong answers")
    for i, v in enumerate([gen_fail, retr_fail]):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_rq2_failure_split.png"),
                                    bbox_inches="tight"); plt.close()

    # (b) correlation retrieval quality vs USR (kept from pilot, now with CI note)
    sub = df.dropna(subset=["retrieval_top_score", "usr"])
    r_rq2, p_rq2 = stats.pearsonr(sub.retrieval_top_score, sub.usr)
    fig, ax = plt.subplots(figsize=(6, 4.3))
    for s in STRATS:
        g = df[df.strategy == s]
        ax.scatter(g.retrieval_top_score, g.usr, c=COL[s], alpha=0.55, label=NAME[s], s=28)
    ax.set_xlabel("top-1 retrieval similarity"); ax.set_ylabel("USR")
    ax.set_title(f"RQ2: retrieval quality vs USR (r={r_rq2:.2f}, p={p_rq2:.3f})")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_rq2_retrieval.png"),
                                    bbox_inches="tight"); plt.close()

    # (c) Top-k sweep (if produced)
    topk_summary = None
    if os.path.exists(topk_path):
        tk = pd.read_csv(topk_path)
        # rescore each (already rescored upstream); summarise by k, strategy
        topk_summary = tk.groupby(["strategy", "topk"]).agg(
            accuracy=("correct", "mean"), abstention=("abstained", "mean"),
            usr=("usr", "mean"), faithfulness=("faithfulness", "mean"),
        ).round(3).reset_index()
        topk_summary.to_csv(os.path.join(outdir, "rq2_topk_summary.csv"), index=False)
        # also retrieval hit-rate vs k
        hit_by_k = pd.DataFrame([{"k": k, "hit_rate": audit_at_k(retr_list, k).relevant_hit.mean()}
                                 for k in sorted(tk.topk.unique())])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for s in tk.strategy.unique():
            gg = topk_summary[topk_summary.strategy == s]
            ax[0].plot(gg.topk, gg.usr, "o-", color=COL.get(s, "gray"), label=f"USR {NAME.get(s,s)}")
            ax[0].plot(gg.topk, gg.abstention, "s--", color=COL.get(s, "gray"), alpha=0.6,
                       label=f"abstain {NAME.get(s,s)}")
        ax[0].set_xlabel("Top-k passages"); ax[0].set_ylabel("rate")
        ax[0].set_title("RQ2: effect of retrieval depth"); ax[0].legend(fontsize=7)
        ax[1].plot(hit_by_k.k, hit_by_k.hit_rate, "o-", color="#55A868")
        ax[1].set_xlabel("Top-k passages"); ax[1].set_ylabel("retrieval hit-rate")
        ax[1].set_title("RQ2: relevant-evidence hit-rate vs depth"); ax[1].set_ylim(0, 1)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_rq2_topk.png"),
                                        bbox_inches="tight"); plt.close()

    print("RQ2 written: rq2_audit.json, rq2_retrieval_audit.csv, fig_rq2_failure_split.png, "
          "fig_rq2_retrieval.png" + (", fig_rq2_topk.png" if topk_summary is not None else ""))
    print(f"  hit-rate(k=4)={audit['hit_rate_k4']}  P(correct|hit)={audit['p_correct_given_hit']} "
          f"vs P(correct|miss)={audit['p_correct_given_miss']}  "
          f"retrieval_fail={retr_fail}  generation_fail={gen_fail}")
    return audit, r_rq2, p_rq2, topk_summary


# --------------------------------- RQ3: length vs USR, abstention controlled ----
def rq3(df, outdir):
    d = df.dropna(subset=["n_sentences", "usr"]).copy()
    d["abstained_i"] = d.abstained.astype(int)
    # raw correlation (what the pilot reported)
    r_raw, p_raw = stats.pearsonr(d.n_sentences, d.usr)
    # partial correlation of (length, usr) controlling for abstention
    def resid(y, X):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return y - X1 @ beta
    rx = resid(d.n_sentences.values.astype(float), d.abstained_i.values.astype(float))
    ry = resid(d.usr.values.astype(float), d.abstained_i.values.astype(float))
    r_partial, p_partial = stats.pearsonr(rx, ry)
    # OLS: usr ~ length + abstained  (report standardized-ish coefficients)
    X = np.column_stack([np.ones(len(d)), d.n_sentences.values, d.abstained_i.values]).astype(float)
    y = d.usr.values.astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    # t-stats
    n, kdim = X.shape
    sigma2 = ss_res / (n - kdim)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), n - kdim))
    # answered-only correlation
    ans = d[d.abstained_i == 0]
    r_ans, p_ans = stats.pearsonr(ans.n_sentences, ans.usr) if len(ans) > 2 else (np.nan, np.nan)

    out = pd.DataFrame([
        {"analysis": "raw Pearson (length vs USR)", "coef_or_r": round(r_raw, 3), "p": round(p_raw, 4)},
        {"analysis": "partial corr (control abstention)", "coef_or_r": round(r_partial, 3), "p": round(p_partial, 4)},
        {"analysis": "OLS beta[length]", "coef_or_r": round(beta[1], 4), "p": round(pvals[1], 4)},
        {"analysis": "OLS beta[abstained]", "coef_or_r": round(beta[2], 4), "p": round(pvals[2], 4)},
        {"analysis": "OLS R^2", "coef_or_r": round(r2, 3), "p": np.nan},
        {"analysis": "answered-only Pearson", "coef_or_r": round(r_ans, 3), "p": round(p_ans, 4)},
    ])
    out.to_csv(os.path.join(outdir, "rq3_confound.csv"), index=False)

    # figure: (A) scatter split by abstention with two fit lines,
    #         (B) how the correlation shrinks as abstention is controlled.
    abq = d[d.abstained_i == 1]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    axA.scatter(ans.n_sentences, ans.usr, s=42, c="#4C72B0", alpha=0.75,
                edgecolor="white", lw=0.5, label=f"Answered (n={len(ans)})", zorder=3)
    axA.scatter(abq.n_sentences, abq.usr, s=72, c="#C44E52", marker="X", alpha=0.9,
                edgecolor="white", lw=0.6, label=f"Abstained (n={len(abq)})", zorder=4)
    xs = np.linspace(0, d.n_sentences.max(), 50)
    m1, b1 = np.polyfit(d.n_sentences, d.usr, 1)
    m2, b2 = np.polyfit(ans.n_sentences, ans.usr, 1)
    axA.plot(xs, m1 * xs + b1, "--", color="#555", lw=2, label=f"fit: all data (r={r_raw:.2f})", zorder=2)
    axA.plot(xs, m2 * xs + b2, "-", color="#4C72B0", lw=2.5, label=f"fit: answered only (r={r_ans:.2f})", zorder=2)
    axA.annotate("Abstained answers tend to score\nHIGH-USR at any length — a refusal\nisn't \"grounded\" in medical prose.\n(Measurement artefact; explains\nonly PART of the slope.)",
                 xy=(3.0, 0.83), xytext=(6.6, 0.92), fontsize=9, color="#C44E52", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.6), ha="left", va="top")
    axA.set_xlabel("answer length (sentences)"); axA.set_ylabel("USR (unsupported-sentence ratio)")
    axA.set_title("(A) Length vs USR, split by abstention", fontsize=12, fontweight="bold")
    axA.grid(alpha=0.25); axA.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    axA.set_ylim(-0.05, 1.1)
    labels = ["Raw\n(all answers)", "Controlling for\nabstention\n(partial r)", "Answered\nquestions only"]
    vals = [r_raw, r_partial, r_ans]; cols = ["#999999", "#8172B3", "#4C72B0"]
    yy = np.arange(len(vals))[::-1]
    axB.barh(yy, vals, color=cols, height=0.55)
    for yi, v in zip(yy, vals):
        axB.text(v - 0.012, yi, f"{v:.2f}", va="center", ha="right", color="white",
                 fontweight="bold", fontsize=11)
    axB.axvline(0, color="k", lw=1)
    axB.set_yticks(yy); axB.set_yticklabels(labels, fontsize=9.5)
    axB.set_xlim(-0.42, 0.05); axB.set_xlabel("correlation (length vs USR)")
    axB.set_title("(B) Correlation barely changes when abstention\nis controlled — and never flips positive",
                  fontsize=11, fontweight="bold")
    axB.grid(axis="x", alpha=0.25)
    axB.text(-0.41, -0.78, "Naive expectation: longer -> MORE hallucination (positive).\n"
             "Reality: even answered-only it stays NEGATIVE.\n"
             "=> answer length is NOT a usable hallucination-risk signal.",
             fontsize=8.8, style="italic", color="#333", va="top")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_rq3_confound.png"),
                                    bbox_inches="tight"); plt.close()
    print("RQ3 written: rq3_confound.csv, fig_rq3_confound.png")
    print(f"  raw r={r_raw:.2f} | partial r={r_partial:.2f} (p={p_partial:.3f}) | "
          f"answered-only r={r_ans:.2f}")
    return out


# ------------------------------------------- USR-threshold sensitivity sweep ----
def threshold_sweep(main_csv, retr_by_qid, outdir, thresholds=(0.55, 0.60, 0.65, 0.70, 0.75)):
    base = pd.read_csv(main_csv)
    rows = []
    for th in thresholds:
        sc = score_df(base, retr_by_qid, threshold=th)
        for s in STRATS:
            g = sc[sc.strategy == s]
            rows.append({"threshold": th, "strategy": s,
                         "usr": round(g.usr.mean(), 3),
                         "faithfulness": round(g.faithfulness.mean(), 3)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(outdir, "threshold_sweep.csv"), index=False)
    # figure: USR vs threshold per strategy (abstention rate is threshold-independent)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for s in STRATS:
        g = sweep[sweep.strategy == s]
        ax.plot(g.threshold, g.usr, "o-", color=COL[s], label=NAME[s])
    ax.axvline(0.65, color="k", ls="--", lw=1, alpha=0.6, label="chosen 0.65")
    ax.set_xlabel("USR similarity threshold"); ax.set_ylabel("mean USR")
    ax.set_title("Sensitivity of USR to the grounding threshold"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_threshold_sweep.png"),
                                    bbox_inches="tight"); plt.close()
    print("Sensitivity written: threshold_sweep.csv, fig_threshold_sweep.png")
    return sweep


# ------------------------------------------------- second-seed stability ----
def seed_stability(main_df, seed2_csv, retr2_path, outdir):
    if not os.path.exists(seed2_csv):
        print("seed-2 results not found; skipping stability block")
        return None
    s2 = pd.read_csv(seed2_csv)
    retr2 = load_retr(retr2_path)
    s2 = score_df(s2, retr2, threshold=DEFAULT_THRESHOLD)
    rows = []
    for label, dd in [("seed42(main)", main_df), ("seed7", s2)]:
        for s in ["neutral", "strict_citations"]:
            g = dd[dd.strategy == s]
            rows.append({"run": label, "strategy": s,
                         "accuracy": round(g.correct.mean(), 3),
                         "abstention": round(g.abstained.mean(), 3),
                         "usr": round(g.usr.mean(), 3),
                         "faithfulness": round(g.faithfulness.mean(), 3),
                         "n": int(len(g))})
    stab = pd.DataFrame(rows)
    stab.to_csv(os.path.join(outdir, "seed_stability.csv"), index=False)
    print("Stability written: seed_stability.csv")
    print(stab.to_string(index=False))
    return stab


# ---------------------------------------------------- error-analysis export ----
def error_examples(df, detail_path, outdir, n_each=3):
    det = {(d["qid"], d["strategy"]): d for d in json.load(open(detail_path))}
    piv = {s: df[df.strategy == s].set_index("qid") for s in STRATS}
    strict = df[df.strategy == "strict_citations"]
    neu = piv["neutral"]

    def block(title, items):
        out = [f"## {title}\n"]
        for qid, note in items:
            d = det.get((qid, note["strategy"]))
            if not d:
                continue
            ans = textwrap.shorten(d["answer_text"].replace("\n", " "), 600)
            out.append(f"**Q{qid}** ({note['tag']}) — gold **{d['answer_idx']}** "
                       f"({d.get('answer_text_gold','')})\n")
            out.append(f"- {note['why']}\n")
            out.append(f"- Model: {ans}\n")
        return "\n".join(out) + "\n"

    sections = []
    # good abstentions: strict abstained AND neutral was wrong
    good = []
    for qid, r in strict[strict.abstained].set_index("qid").iterrows():
        if qid in neu.index and neu.loc[qid].correct == 0:
            good.append((qid, {"strategy": "strict_citations", "tag": "good abstention",
                               "why": "Strict declined; the neutral prompt answered this one WRONG, so refusing avoided an error."}))
    sections.append(block("Successful abstentions (declined where neutral erred)", good[:n_each]))
    # costly refusals: strict abstained AND neutral was right
    bad = []
    for qid, r in strict[strict.abstained].set_index("qid").iterrows():
        if qid in neu.index and neu.loc[qid].correct == 1:
            bad.append((qid, {"strategy": "strict_citations", "tag": "costly refusal",
                              "why": "Strict declined but the neutral prompt got it RIGHT, so refusing cost a correct answer."}))
    sections.append(block("Costly refusals (declined where neutral was right)", bad[:n_each]))
    # hallucinations: answered (not abstained), wrong, high USR
    hal = df[(~df.abstained) & (~df.correct) & (df.usr > 0.5)].sort_values("usr", ascending=False)
    hal_items = [(int(r.qid), {"strategy": r.strategy, "tag": f"hallucination ({NAME[r.strategy]}, USR={r.usr})",
                               "why": "Answered confidently, was wrong, and most sentences were ungrounded in the retrieved context."})
                 for _, r in hal.head(n_each).iterrows()]
    sections.append(block("Hallucinated answers (confident, wrong, ungrounded)", hal_items))

    md = "# Qualitative error analysis\n\n" + "\n".join(sections)
    open(os.path.join(outdir, "error_examples.md"), "w").write(md)
    print(f"Error analysis written: error_examples.md "
          f"(good={len(good)}, costly={len(bad)}, hallucinated={len(hal)})")


# ----------------------------------------------------------------- main ----
def run(outdir):
    main_csv = os.path.join(outdir, "results_main_seed42_k4.csv")
    retr_path = os.path.join(outdir, "retrievals_seed42.json")
    detail_path = os.path.join(outdir, "results_detail_main_seed42_k4.json")
    topk_path = os.path.join(outdir, "results_topk_sweep_scored.csv")
    seed2_csv = os.path.join(outdir, "results_seed7_k4.csv")
    retr2_path = os.path.join(outdir, "retrievals_seed7.json")

    retr_by_qid = load_retr(retr_path)
    retr_list = json.load(open(retr_path))

    # rescore main run (semantic USR/faithfulness/abstention at 0.65) and persist
    df = pd.read_csv(main_csv)
    df, detail = score_df(df, retr_by_qid, threshold=DEFAULT_THRESHOLD, return_detail=True)
    df.to_csv(os.path.join(outdir, "scored_main.csv"), index=False)
    # attach per-sentence detail for the distribution figure
    det_raw = json.load(open(detail_path))
    for d, sd in zip(det_raw, detail):
        d["sentence_detail"] = sd
    json.dump(det_raw, open(os.path.join(outdir, "scored_detail_main.json"), "w"), indent=2)
    print(f"Scored main run: {len(df)} rows, {df.qid.nunique()} questions\n")

    summ, sigdf, abst = rq1(df, outdir)
    audit, r2, p2, topk = rq2(df, retr_list, retr_by_qid, outdir, topk_path)
    rq3out = rq3(df, outdir)
    sweep = threshold_sweep(main_csv, retr_by_qid, outdir)
    stab = seed_stability(df, seed2_csv, retr2_path, outdir)
    error_examples(df, os.path.join(outdir, "scored_detail_main.json"), outdir)

    # per-sentence grounding distribution (uses scored detail)
    det = json.load(open(os.path.join(outdir, "scored_detail_main.json")))
    sims = {s: [] for s in STRATS}
    for d in det:
        for sd in d.get("sentence_detail", []):
            sims.setdefault(d["strategy"], []).append(sd["max_sim"])
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for s in STRATS:
        if sims[s]:
            ax.hist(sims[s], bins=22, alpha=0.5, color=COL[s], label=NAME[s], density=True)
    ax.axvline(0.65, color="k", ls="--", lw=1, label="threshold 0.65")
    ax.set_xlabel("per-sentence max similarity to context"); ax.set_ylabel("density")
    ax.set_title("Sentence grounding distribution"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_usr_distribution.png"),
                                    bbox_inches="tight"); plt.close()

    # headline numbers json
    def m(strat, metric):
        return float(summ[(summ.strategy == strat) & (summ.metric == metric)]["mean"].iloc[0])
    head = {
        "n_questions": int(df.qid.nunique()),
        "strategies": STRATS,
        "accuracy": {s: m(s, "accuracy_overall") for s in STRATS},
        "coverage": {s: m(s, "coverage") for s in STRATS},
        "accuracy_when_answered": {s: m(s, "accuracy_when_answered") for s in STRATS},
        "abstention": {s: m(s, "abstention_rate") for s in STRATS},
        "usr": {s: m(s, "usr") for s in STRATS},
        "faithfulness": {s: m(s, "faithfulness") for s in STRATS},
        "answer_length": {s: m(s, "answer_length") for s in STRATS},
        "abstention_outcome": abst,
        "rq2_retrieval": audit,
        "rq2_corr": round(float(r2), 3), "rq2_corr_p": round(float(p2), 4),
    }
    json.dump(head, open(os.path.join(outdir, "headline_numbers.json"), "w"), indent=2)
    print("\nheadline_numbers.json written.")
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="pilot_100Q")
    args = ap.parse_args()
    outdir = args.outdir if os.path.isabs(args.outdir) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    run(outdir)
