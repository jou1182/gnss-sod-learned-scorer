"""Gradient-boosting removal scorer: P(candidate is NOT in the expert plan | state features)."""
import time, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .bdp import BackwardEnv
from .baselines import _result


class GBMScorer:
    def __init__(self, clf=None):
        self.clf = clf

    def score(self, F, cost):
        if self.clf is None:
            return cost.copy()
        return self.clf.predict_proba(F)[:, 1] + 1e-3 * cost / (cost.max() + 1e-9)


def rollout(inst, scorer, rng=None, eps=0.0, active0=None, S=None, collect=False, stride=3, max_rows=40):
    """Backward elimination guided by scorer. If S (expert plan) given and collect, record (F,label)."""
    env = BackwardEnv(inst, active0)
    X, Y = [], []
    step = 0
    while True:
        gi, F = env.features()
        if len(gi) == 0:
            break
        if collect and S is not None and step % stride == 0:
            lab = ~S[gi]
            idx = np.arange(len(gi))
            if len(idx) > max_rows:
                idx = rng.choice(idx, max_rows, replace=False)
            X.append(F[idx]); Y.append(lab[idx])
        sc = scorer.score(F, inst.cost[gi])
        if eps > 0 and rng.random() < eps:
            if S is not None and rng.random() < 0.5 and (~S[gi]).any():
                a = int(rng.choice(np.where(~S[gi])[0]))
            else:
                a = int(rng.integers(len(gi)))
        else:
            a = int(np.argmax(sc))
        env.remove(int(gi[a]))
        step += 1
    return env, (np.concatenate(X) if X else np.zeros((0, F.shape[1] if len(gi) else 21), np.float32), np.concatenate(Y) if Y else np.zeros(0, bool))


def solve(inst, scorer, name="gbm-greedy"):
    t0 = time.time()
    env, _ = rollout(inst, scorer)
    return _result(inst, env.active, t0, 1, name)


def lns(inst, scorer, iters=30, k_add=40, eps=0.08, seed=0, name=None):
    """Scorer-guided ruin-and-recreate (large-neighbourhood search)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env, _ = rollout(inst, scorer)
    cur = env.active.copy(); cur_c = float(inst.cost[cur].sum())
    for _ in range(iters):
        out = np.where(~cur)[0]
        if len(out) == 0:
            break
        a0 = cur.copy(); a0[rng.choice(out, size=min(k_add, len(out)), replace=False)] = True
        env, _ = rollout(inst, scorer, rng, eps=eps, active0=a0)
        c = env.cost()
        if c < cur_c - 1e-9:
            cur, cur_c = env.active.copy(), c
    return _result(inst, cur, t0, iters, name or f"gbm-LNS{iters}")
