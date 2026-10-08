"""Design-space exploration use case: 200 what-if scenarios on one corridor (accuracy limits x redundancy floor).
Usage: python run_usecase.py  -> results/usecase.json"""
import json, pickle, time
import numpy as np
from multiprocessing import Pool
from sod.env import geo_corridor, verify
from sod import baselines as B
from sod.gbm import GBMScorer, solve as gsolve, lns as glns
from sod.strong import sa_repair
import run_eval

WP, N = run_eval.SAUDI["HHR (Makkah-Jeddah-KAEC-Madinah)"]
EPS_H = [12.0, 14.0]
EPS_U = list(np.linspace(35.0, 50.0, 20))
RMIN = [0.05, 0.08, 0.10, 0.13, 0.16]
_model = None


def _init():
    global _model
    _model = GBMScorer(pickle.load(open("results/gbm.pkl", "rb")))


def task(k):
    eh = EPS_H[k % 2]; eu = EPS_U[(k // 2) % 20]; rm = RMIN[(k // 40) % 5]
    inst = geo_corridor(WP, N, seed=0, eps_h=eh, eps_u=float(eu), r_min=rm, name="HHR")
    if not verify(inst, np.ones(inst.M, bool))[0]:
        return dict(k=k, skipped=True, eps_h=eh, eps_u=float(eu), r_min=rm)
    out = {}
    t = time.process_time(); r = B.backward(inst, "ratio"); out["BE-ratio"] = (r["cost"] if r["feasible"] else None, time.process_time() - t)
    t = time.process_time(); r = gsolve(inst, _model); out["GBM-greedy"] = (r["cost"] if r["feasible"] else None, time.process_time() - t)
    t = time.process_time(); r = glns(inst, _model, 30, seed=k); out["GBM-LNS"] = (r["cost"] if r["feasible"] else None, time.process_time() - t)
    t = time.process_time(); r = sa_repair(inst, 3.0, k); out["SA-repair@3s"] = (r["cost"] if r["feasible"] else None, time.process_time() - t)
    t = time.process_time(); r = sa_repair(inst, 26.0, k); out["SA-repair@26s"] = (r["cost"] if r["feasible"] else None, time.process_time() - t)
    return dict(k=k, skipped=False, eps_h=eh, eps_u=float(eu), r_min=rm, M=inst.M, res=out)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(28, initializer=_init) as p:
        res = p.map(task, range(200), chunksize=1)
    json.dump(res, open("results/usecase.json", "w")); print("done", len(res), f"{time.time()-t0:.0f}s")
