import json, numpy as np
from scipy import stats
import analyze


def load_pair(mode):
    new = json.load(open(f"results/strong_{mode}.json")); base = analyze.load("test" if mode == "test" else "ood")
    n = len(new); rows = []
    for i in range(n):
        d = {r["method"]: (r["cost"] if r["feasible"] else np.nan) for r in base[i]["results"]}
        for k, v in new[i]["costs"].items(): d[k] = np.nan if v is None else v
        rows.append(d)
    return rows


def summarize(mode, meths, ref):
    rows = load_pair(mode); n = len(rows)
    best = np.array([np.nanmin(list(d.values())) for d in rows])
    out = {}
    for m in meths:
        c = np.array([d[m] for d in rows]); out[m] = dict(med=float(np.nanmedian(c / best)), mean=float(np.nanmean(c / best)))
    tests = {}; ps = []
    for m in meths:
        if m == ref: continue
        a = np.array([d[ref] for d in rows]); b = np.array([d[m] for d in rows]); dd = a - b
        p = stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        tests[m] = dict(ref_cheaper=int((dd < 0).sum()), ref_costlier=int((dd > 0).sum()), med_diff_pct=float(np.median(dd / b) * 100), p=float(p)); ps.append(p)
    return n, out, tests


if __name__ == "__main__":
    M = ["bdp-ratio", "SA@20k", "GA@20k", "SA-repair@3s", "GA-repair@3s", "gbm-greedy", "gbm-LNS30", "SA-repair@26s", "GA-repair@26s", "GA@100k", "SA@200k"]
    for mode in ("test", "ood"):
        n, out, tests = summarize(mode, M, "gbm-LNS30")
        print(f"== {mode} (first {n}) ratio to best-known"); [print(f"  {m:14s} med {v['med']:.3f} mean {v['mean']:.3f}") for m, v in out.items()]
        print("  paired vs gbm-LNS30:"); [print(f"   {m:14s} LNS cheaper {t['ref_cheaper']:2d} costlier {t['ref_costlier']:2d} med diff {t['med_diff_pct']:+.2f}% p={t['p']:.3g}") for m, t in tests.items()]
