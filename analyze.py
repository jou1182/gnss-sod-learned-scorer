"""Statistics and tables from results/*.json (frozen protocol)."""
import json, sys
import numpy as np
from scipy import stats

METHODS_ORDER = ["conv-chain", "conv-ladder", "bdp-random(mean5)", "bdp-longest", "bdp-ratio",
                 "GA@20k", "BPSO@20k", "SA@20k", "GA@100k", "SA@200k", "gbm-greedy", "gbm-LNS30"]


def load(name):
    return json.load(open(f"results/{name}.json"))


def table(res):
    """returns dict method -> arrays over instances (nan if infeasible)"""
    meths = sorted({r["method"] for x in res for r in x["results"]}, key=lambda m: METHODS_ORDER.index(m) if m in METHODS_ORDER else 99)
    cost = {m: np.full(len(res), np.nan) for m in meths}
    feas = {m: np.zeros(len(res), bool) for m in meths}
    time_ = {m: np.zeros(len(res)) for m in meths}
    for i, x in enumerate(res):
        for r in x["results"]:
            feas[r["method"]][i] = r["feasible"]
            time_[r["method"]][i] = r["time"]
            if r["feasible"]:
                cost[r["method"]][i] = r["cost"]
    best = np.nanmin(np.stack([cost[m] for m in meths]), 0)         # best-known feasible per instance
    return meths, cost, feas, time_, best


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    m = [np.median(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(name, ref="gbm-greedy"):
    res = load(name)
    meths, cost, feas, time_, best = table(res)
    rows = []
    for m in meths:
        ratio = cost[m] / best
        rows.append(dict(method=m, feas=float(feas[m].mean()), med=float(np.nanmedian(ratio)), mean=float(np.nanmean(ratio)),
                         ci=boot_ci(ratio), tmed=float(np.median(time_[m]))))
    # paired tests vs policy
    tests = []
    pvals = []
    for m in meths:
        if m == ref: continue
        ok = feas[m] & feas[ref]
        if ok.sum() < 5: continue
        d = cost[ref][ok] - cost[m][ok]
        try:
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        tests.append(dict(vs=m, n=int(ok.sum()), median_diff_pct=float(np.median(d / cost[m][ok]) * 100),
                          wins=int((d < 0).sum()), ties=int((d == 0).sum()), losses=int((d > 0).sum()), p=float(p)))
        pvals.append(p)
    # Holm
    order = np.argsort(pvals); m_ = len(pvals); adj = np.zeros(m_); run = 0
    for rank, idx in enumerate(order):
        run = max(run, (m_ - rank) * pvals[idx]); adj[idx] = min(1.0, run)
    for t, a in zip(tests, adj): t["p_holm"] = float(a)
    return dict(rows=rows, tests=tests, n=len(res))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        s = summary(name)
        print(f"\n=== {name}  (n={s['n']}) ratio = cost / best-known feasible")
        print(f"{'method':22s} feas   median   mean   95%CI(median)      t_med(s)")
        for r in s["rows"]:
            print(f"{r['method']:22s} {r['feas']*100:5.1f}  {r['med']:.3f}  {r['mean']:.3f}  [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  {r['tmed']:.2f}")
        print("paired vs gbm-greedy:")
        for t in s["tests"]:
            print(f"  vs {t['vs']:20s} n={t['n']:3d} median diff {t['median_diff_pct']:+6.2f}%  W/T/L {t['wins']}/{t['ties']}/{t['losses']}  p_holm={t['p_holm']:.3g}")
