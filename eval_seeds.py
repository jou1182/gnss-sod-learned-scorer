"""Evaluate scorers trained with different random seeds on the frozen test and OOD sets.
Usage: python eval_seeds.py   -> results/seeds_eval.json"""
import json, pickle, time, os
import numpy as np
from multiprocessing import Pool
from sod.env import synth_corridor
from sod.gbm import GBMScorer, solve as gsolve, lns as glns
import run_eval

SEEDS = [0, 1, 2, 3, 4]
_models = {}


def _init():
    for s in SEEDS:
        path = "results/gbm.pkl" if s == 0 else f"results/gbm_seed{s}.pkl"
        _models[s] = GBMScorer(pickle.load(open(path, "rb")))


def task(args):
    kind, i = args
    inst = synth_corridor(30000 + i) if kind == "test" else run_eval.make(("ood", 40000 + i, {}))
    out = {}
    for s in SEEDS:
        g = gsolve(inst, _models[s]); l = glns(inst, _models[s], 30, seed=i)
        out[s] = dict(greedy=g["cost"] if g["feasible"] else None, lns=l["cost"] if l["feasible"] else None)
    return dict(kind=kind, i=i, res=out)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("test", i) for i in range(200)] + [("ood", i) for i in range(50)]
    with Pool(28, initializer=_init) as p:
        res = p.map(task, jobs, chunksize=1)
    json.dump(res, open("results/seeds_eval.json", "w")); print("done", len(res), f"{time.time()-t0:.0f}s")
