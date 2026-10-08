"""python run_strong.py small|test|ood  -> results/strong_<mode>.json"""
import sys, json, time, pickle, os
import numpy as np
from multiprocessing import Pool
from sod.env import synth_corridor
from sod import baselines as B
from sod.gbm import GBMScorer, solve as gsolve, lns as glns
from sod.strong import sa_repair, ga_repair, bnb_exact, small_instance
import run_eval

_model = None


def _init():
    global _model
    _model = GBMScorer(pickle.load(open("results/gbm.pkl", "rb")))


def strip(r):
    r = dict(r); r.pop("active", None); return r


def small_task(i):
    N = 7 if i % 2 == 0 else 8; kappa = 1.5 if (i // 2) % 2 == 0 else 1.8
    inst = small_instance(60000 + i, N, 4.4 * N, kappa)
    ub = B.backward(inst, "ratio")
    ex = bnb_exact(inst, ub["active"], time_limit=300)
    out = {"BE-ratio": ub["cost"]}
    for name, r in (("GBM-greedy", gsolve(inst, _model)), ("GBM-LNS", glns(inst, _model, 30, seed=i)), ("GA-pen@20k", B.ga(inst, 20000, i)),
                    ("SA-pen@20k", B.sa(inst, 20000, i)), ("SA-repair@1s", sa_repair(inst, 1.0, i)), ("GA-repair@1s", ga_repair(inst, 1.0, i))):
        out[name] = r["cost"] if r["feasible"] else None
    return dict(i=i, N=N, kappa=kappa, M=inst.M, opt=ex["cost"], optimal=ex["optimal"], nodes=ex["nodes"], t_exact=ex["time"], costs=out)


def strong_task(args):
    mode, i = args
    inst = synth_corridor(30000 + i) if mode == "test" else run_eval.make(("ood", 40000 + i, {}))
    out = {}
    for b in (3.0, 26.0):
        for name, fn in ((f"SA-repair@{b:g}s", lambda: sa_repair(inst, b, i)), (f"GA-repair@{b:g}s", lambda: ga_repair(inst, b, i))):
            r = fn(); out[name] = r["cost"] if r["feasible"] else None
    return dict(i=i, costs=out)


if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time()
    with Pool(28, initializer=_init) as p:
        if mode == "small": res = p.map(small_task, range(60), chunksize=1)
        elif mode == "test": res = p.map(strong_task, [("test", i) for i in range(60)], chunksize=1)
        else: res = p.map(strong_task, [("ood", i) for i in range(20)], chunksize=1)
    json.dump(res, open(f"results/strong_{mode}.json", "w")); print(mode, "done", len(res), f"{time.time()-t0:.0f}s")
