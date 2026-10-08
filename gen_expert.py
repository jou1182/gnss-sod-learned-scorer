"""Generate expert (best-known) plans for training/validation corridors:
best of SA@200k, backward-ratio, backward-longest; then made irreducible by backward elimination."""
import sys, pickle, time, numpy as np
from multiprocessing import Pool
from sod.env import synth_corridor, verify
from sod import baselines as B
from sod.bdp import BackwardEnv


def irreducible(inst, active):
    env = BackwardEnv(inst, active)
    while True:
        c = env.removable()
        if len(c) == 0: break
        env.remove(int(c[np.argmax(inst.cost[c])]))
    return env.active.copy()


def task(seed):
    inst = synth_corridor(seed)
    cands = [B.backward(inst, "ratio"), B.backward(inst, "longest"), B.sa(inst, 200000, seed), B.sa(inst, 200000, seed + 7)]
    cands = [c for c in cands if c["feasible"]]
    best = min(cands, key=lambda c: c["cost"])
    act = irreducible(inst, best["active"])
    return seed, act, float(inst.cost[act].sum()), best["method"], B.backward(inst, "ratio")["cost"]


if __name__ == "__main__":
    lo, hi, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    with Pool(30) as p:
        res = p.map(task, range(lo, hi), chunksize=1)
    pickle.dump(res, open(out, "wb"))
    r = np.array([x[2] / x[4] for x in res])
    print(out, len(res), "mean expert/ratio %.4f" % r.mean(), "time %.0fs" % (time.time() - t0))
