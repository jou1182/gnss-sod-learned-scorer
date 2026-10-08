"""Evaluate all methods on a frozen instance set. Usage: python run_eval.py {test|ood|saudi|sens} [policy.pt]"""
import sys, json, time, os
import numpy as np
import pickle
from multiprocessing import Pool

from sod.env import synth_corridor, geo_corridor, verify, EPS_H, EPS_U
from sod import baselines as B
from sod.gbm import GBMScorer, solve as gsolve, lns as glns

SAUDI = {
    "HHR (Makkah-Jeddah-KAEC-Madinah)": ([(21.389, 39.858), (21.543, 39.173), (22.40, 39.08), (24.468, 39.614)], 45),
    "Riyadh-Hofuf-Dammam": ([(24.713, 46.675), (25.38, 49.59), (26.43, 50.10)], 43),
    "Jeddah-Riyadh (Landbridge, schematic)": ([(21.543, 39.173), (24.713, 46.675)], 70),
    "Linear city 170 km (schematic)": ([(28.07, 35.05), (27.72, 36.80)], 20),
}
_model = None


def _init(path):
    global _model
    _model = GBMScorer(pickle.load(open(path, "rb")))


def make(spec):
    kind, a, b = spec
    if kind == "synth":
        return synth_corridor(a, **b)
    if kind == "ood":
        for j in range(40):     # redraw (N, length) until the full network meets the accuracy limits
            rng = np.random.default_rng(a * 7 + j)
            try:
                return synth_corridor(a, N=int(rng.integers(45, 71)), length=float(rng.uniform(300, 600)), **b)
            except RuntimeError:
                continue
        raise RuntimeError("ood generation failed")
    if kind == "saudi":
        wp, N = SAUDI[a[0]]
        return geo_corridor(wp, N, seed=a[1], name=a[0], **b)


def strip(r):
    r = dict(r); r.pop("active", None); return r


def task(args):
    spec, big = args
    inst = make(spec)
    seed = int(spec[1] if spec[0] != "saudi" else spec[1][1])
    out = []
    out.append(B.conventional(inst, "chain")); out.append(B.conventional(inst, "ladder"))
    out.append(B.backward(inst, "longest")); out.append(B.backward(inst, "ratio"))
    rr = [B.backward(inst, "random", np.random.default_rng(s)) for s in range(5)]
    r0 = rr[0]; r0["cost"] = float(np.mean([x["cost"] for x in rr])); r0["method"] = "bdp-random(mean5)"; out.append(r0)
    out.append(gsolve(inst, _model, "gbm-greedy"))
    out.append(glns(inst, _model, 30, seed=seed, name="gbm-LNS30"))
    for name, fn in (("GA@20k", lambda: B.ga(inst, 20000, seed)), ("BPSO@20k", lambda: B.bpso(inst, 20000, seed)),
                     ("SA@20k", lambda: B.sa(inst, 20000, seed))):
        r = fn(); r["method"] = name; out.append(r)
    if big:
        for name, fn in (("GA@100k", lambda: B.ga(inst, 100000, seed)), ("SA@200k", lambda: B.sa(inst, 200000, seed))):
            r = fn(); r["method"] = name; out.append(r)
    ok, info = verify(inst, np.ones(inst.M, bool))
    meta = dict(name=inst.name or str(spec[1]), N=len(inst.xy), M=inst.M, n_ctrl=len(inst.ctrl),
                length_km=float(inst.s_along[-1]), cost_full=float(inst.cost.sum()), full_ok=ok,
                sig_h_full=info.get("sig_h"), sig_u_full=info.get("sig_u"), seed=seed)
    return dict(meta=meta, results=[strip(r) for r in out])


def main(which, path="results/gbm.pkl"):
    big = True
    b = {}
    if which == "test":
        specs = [("synth", 30000 + i, b) for i in range(200)]
    elif which == "ood":
        specs = [("ood", 40000 + i, b) for i in range(50)]
    elif which == "saudi":
        specs = [("saudi", (k, s), dict(eps_h=12.0, eps_u=35.0)) for k in SAUDI for s in range(10)]
    elif which.startswith("sens"):   # sensitivity: sens_eps08, sens_eps125, sens_r005, sens_r015
        cfg = {"sens_eps08": dict(eps_h=0.8 * EPS_H, eps_u=0.8 * EPS_U), "sens_eps125": dict(eps_h=1.25 * EPS_H, eps_u=1.25 * EPS_U),
               "sens_r005": dict(r_min=0.05), "sens_r015": dict(r_min=0.15)}[which]
        specs = [("synth", 30000 + i, cfg) for i in range(60)]
        big = False
    t0 = time.time()
    with Pool(28, initializer=_init, initargs=(path,)) as p:
        res = p.map(task, [(s, big) for s in specs], chunksize=1)
    json.dump(res, open(f"results/{which}.json", "w"))
    print(which, "done", len(res), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main(*sys.argv[1:])
