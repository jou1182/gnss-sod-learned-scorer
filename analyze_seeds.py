import json, numpy as np
import analyze
SEEDS = [0, 1, 2, 3, 4]
res = json.load(open("results/seeds_eval.json"))
out = {}
for kind in ("test", "ood"):
    base = analyze.load("test" if kind == "test" else "ood")
    R = [r for r in res if r["kind"] == kind]
    best = np.array([min(q["cost"] for q in base[r["i"]]["results"] if q["feasible"]) for r in R])
    ratio_be = np.array([next(q["cost"] for q in base[r["i"]]["results"] if q["method"] == "bdp-ratio") for r in R])
    for mth in ("greedy", "lns"):
        meds, wins = [], []
        for s in SEEDS:
            c = np.array([r["res"][str(s)][mth] for r in R], float)
            meds.append(float(np.nanmedian(c / best))); wins.append(float(np.mean(c < ratio_be)))
        out[f"{kind}-{mth}"] = dict(medians=meds, mean=float(np.mean(meds)), sd=float(np.std(meds, ddof=1)), lo=min(meds), hi=max(meds), win_vs_be=wins)
json.dump(out, open("results/seeds_summary.json", "w"), indent=1)
for k, v in out.items(): print(k, "median ratio per seed", np.round(v["medians"], 3), f"mean {v['mean']:.3f} sd {v['sd']:.3f}", "wins vs BE-ratio", np.round(v["win_vs_be"], 2))
