"""Feature importance (permutation AUC drop on held-out val states) + clean single-process timing."""
import sys, json, pickle, time, numpy as np
from sklearn.metrics import roc_auc_score
from sod.env import synth_corridor
from sod.gbm import GBMScorer, rollout, solve as gsolve, lns as glns
from sod import baselines as B

NAMES = ["cost", "length", "r_h", "r_u", "slack_h", "slack_u", "red_slack_h", "red_slack_u", "deg_min", "deg_max", "pos_along",
         "dist_ctrl", "pct(cost/damage)", "headroom_h", "headroom_u", "pct(length)", "pct(r_h)", "pct(slack_u)",
         "frac_removable", "frac_active", "active/n"]


def importance():
    clf = pickle.load(open("results/gbm.pkl", "rb")); sc = GBMScorer(clf)
    exp = {s: a for s, a, c, m, r in pickle.load(open("results/expert_val.pkl", "rb"))}
    X, Y = [], []
    rng = np.random.default_rng(0)
    for s in list(exp)[:30]:
        inst = synth_corridor(s)
        env, (x, y) = rollout(inst, sc, rng, eps=0.15, S=exp[s], collect=True, stride=4, max_rows=60)
        X.append(x); Y.append(y)
    X = np.concatenate(X); Y = np.concatenate(Y)
    base = roc_auc_score(Y, clf.predict_proba(X)[:, 1])
    imp = []
    for j in range(X.shape[1]):
        d = []
        for r in range(3):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); d.append(base - roc_auc_score(Y, clf.predict_proba(Xp)[:, 1]))
        imp.append(float(np.mean(d)))
    json.dump(dict(auc=float(base), names=NAMES, imp=imp, rows=int(len(Y))), open("results/importance.json", "w"))
    order = np.argsort(imp)[::-1]
    print("AUC", base, "rows", len(Y)); [print(f"  {NAMES[j]:18s} {imp[j]:.4f}") for j in order[:10]]


def timing():
    clf = pickle.load(open("results/gbm.pkl", "rb")); sc = GBMScorer(clf)
    rows = {}
    for s in range(30000, 30010):
        inst = synth_corridor(s)
        for name, fn in (("BE-ratio", lambda: B.backward(inst, "ratio")), ("GBM-greedy", lambda: gsolve(inst, sc)),
                         ("GBM-LNS30", lambda: glns(inst, sc, 30, seed=s)), ("GA@20k", lambda: B.ga(inst, 20000, s)),
                         ("SA@20k", lambda: B.sa(inst, 20000, s)), ("BPSO@20k", lambda: B.bpso(inst, 20000, s)), ("BE-longest", lambda: B.backward(inst, "longest")), ("GA@100k", lambda: B.ga(inst, 100000, s)), ("SA@200k", lambda: B.sa(inst, 200000, s))):
            t = time.perf_counter(); r = fn(); rows.setdefault(name, []).append(time.perf_counter() - t)
    out = {k: dict(median=float(np.median(v)), mean=float(np.mean(v))) for k, v in rows.items()}
    json.dump(out, open("results/timing.json", "w")); print(out)


if __name__ == "__main__":
    {"imp": importance, "timing": timing}[sys.argv[1]]()
