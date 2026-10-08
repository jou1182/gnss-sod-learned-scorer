"""DAgger-style training of the GBM removal scorer from expert plans."""
import sys, pickle, time, json, numpy as np
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sod.env import synth_corridor
from sod.gbm import GBMScorer, rollout
import sod.baselines as B

EXP = {}
_scorer = None


def _init(tr, va):
    for pkl in (tr, va):
        for seed, act, c, meth, ref in pickle.load(open(pkl, "rb")):
            EXP[seed] = (act, c, ref)


def _collect(args):
    seed, clf, rnd, rs = args
    rng = np.random.default_rng(rs + SEED * 1000003)
    inst = synth_corridor(seed)
    S = EXP[seed][0]
    sc = GBMScorer(clf)
    eps = 0.3 if rnd == 0 else 0.15
    env, (X, Y) = rollout(inst, sc, rng, eps=eps, S=S, collect=True)
    if rnd == 0:   # also pure expert-guided random-order trajectory
        env2, (X2, Y2) = rollout(inst, GBMScorer(None), rng, eps=1.0, S=S, collect=True)
        X, Y = np.concatenate([X, X2]), np.concatenate([Y, Y2])
    return X, Y


def _eval(args):
    seed, clf = args
    inst = synth_corridor(seed)
    env, _ = rollout(inst, GBMScorer(clf))
    ref = EXP[seed][2]; best = EXP[seed][1]
    return env.cost() / ref, env.cost() / best


if __name__ == "__main__":
    tr, va = "results/expert_train.pkl", "results/expert_val.pkl"
    tr_seeds = [x[0] for x in pickle.load(open(tr, "rb"))]
    va_seeds = [x[0] for x in pickle.load(open(va, "rb"))]
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    clf = None; Xs, Ys = [], []; log = []; t0 = time.time(); best = 9
    with Pool(30, initializer=_init, initargs=(tr, va)) as pool:
        for rnd in range(rounds):
            res = pool.map(_collect, [(s, clf, rnd, 1000 * rnd + i) for i, s in enumerate(tr_seeds)], chunksize=1)
            for X, Y in res:
                if len(Y): Xs.append(X); Ys.append(Y)
            X = np.concatenate(Xs); Y = np.concatenate(Ys)
            print(f"round {rnd}: rows {len(Y)} pos-rate {Y.mean():.3f} t={time.time()-t0:.0f}s", flush=True)
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=SEED)
            clf.fit(X, Y)
            ev = np.array(pool.map(_eval, [(s, clf) for s in va_seeds], chunksize=1))
            print(f"   val cost/ratio {ev[:,0].mean():.4f} (median {np.median(ev[:,0]):.4f})  cost/expert {ev[:,1].mean():.4f}  t={time.time()-t0:.0f}s", flush=True)
            log.append(dict(round=rnd, rows=int(len(Y)), val_vs_ratio=float(ev[:, 0].mean()), val_vs_expert=float(ev[:, 1].mean())))
            if ev[:, 0].mean() < best:
                best = ev[:, 0].mean(); pickle.dump(clf, open(f"results/gbm_seed{SEED}.pkl", "wb"))
    json.dump(log, open(f"results/gbm_log_seed{SEED}.json", "w"))
