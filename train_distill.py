"""Distil expert (best-known) plans into the removal policy.

Loss (set-imitation): at every state of a trajectory, the admissible baselines that are NOT in
the expert plan are all 'correct' moves (removing them keeps the state a superset of the expert
plan, hence feasible - feasibility is monotone). We maximise the probability mass on them:
    L = -log sum_{i not in S*} softmax(score)_i
Trajectories mix expert-restricted sampling with the model's own choices restricted to correct moves."""
import sys, time, json, os, pickle
import numpy as np
import torch
from multiprocessing import Pool

from sod.env import synth_corridor
from sod.baselines import backward
from sod.bdp import BackwardEnv
from sod.policy import Scorer, rollout

_model = None
EXP = {}


def _init(train_pkl, val_pkl):
    global _model
    torch.set_num_threads(1)
    _model = Scorer()
    for pkl in (train_pkl, val_pkl):
        for seed, act, c, meth, ref in pickle.load(open(pkl, "rb")):
            EXP[seed] = (act, c, ref)


def _load(sd):
    _model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})


def _grads():
    return [p.grad.numpy().copy() for p in _model.parameters()]


def _task(args):
    sd, seed, rng_seed = args
    _load(sd); _model.zero_grad(set_to_none=False)
    rng = np.random.default_rng(rng_seed)
    inst = synth_corridor(seed)
    S, _, _ = EXP[seed]
    env = BackwardEnv(inst)
    n, tot = 0, 0.0
    follow_model = rng.random() < 0.6
    while True:
        gi, F = env.features()
        if len(gi) == 0:
            break
        good = ~S[gi]
        if not good.any():
            break
        logits = _model(torch.from_numpy(F))
        loss = -(torch.logsumexp(logits[torch.from_numpy(good)], 0) - torch.logsumexp(logits, 0))
        loss.backward(); n += 1; tot += loss.item()
        if follow_model:
            p = torch.softmax(logits.detach(), 0).numpy().astype(np.float64) * good
            p = p / p.sum() if p.sum() > 0 else good / good.sum()
            a = int(rng.choice(len(p), p=p))
        else:
            cand = np.where(good)[0]
            a = int(rng.choice(cand))
        env.remove(int(gi[a]))
    for p in _model.parameters():
        p.grad /= max(n, 1)
    return _grads(), tot / max(n, 1)


def _eval(args):
    sd, seed = args
    _load(sd)
    inst = synth_corridor(seed)
    env, _ = rollout(inst, _model, "greedy")
    S, c, ref = EXP[seed]
    return env.cost() / ref, env.cost() / c


def get_sd(m):
    return {k: v.detach().numpy().copy() for k, v in m.state_dict().items()}


def main(iters=200, workers=30, out="results/policy_distill.pt"):
    torch.manual_seed(0)
    train_pkl, val_pkl = "results/expert_train.pkl", "results/expert_val.pkl"
    tr_seeds = [x[0] for x in pickle.load(open(train_pkl, "rb"))]
    va_seeds = [x[0] for x in pickle.load(open(val_pkl, "rb"))]
    model = Scorer()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=2e-4)
    rng = np.random.default_rng(0)
    best, log, t0 = 1e9, [], time.time()
    with Pool(workers, initializer=_init, initargs=(train_pkl, val_pkl)) as pool:
        for it in range(iters):
            sd = get_sd(model)
            seeds = rng.choice(tr_seeds, workers, replace=False)
            res = pool.map(_task, [(sd, int(s), int(rng.integers(1 << 30))) for s in seeds])
            g = [np.mean([r[0][i] for r in res], 0) for i in range(len(res[0][0]))]
            for p, gi in zip(model.parameters(), g):
                p.grad = torch.from_numpy(gi.astype(np.float32))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if it % 10 == 0 or it == iters - 1:
                ev = np.array(pool.map(_eval, [(get_sd(model), s) for s in va_seeds]))
                v = float(ev[:, 0].mean())
                log.append(dict(it=it, loss=float(np.mean([r[1] for r in res])), val_vs_ratio=v, val_vs_expert=float(ev[:, 1].mean())))
                print(f"it {it} loss {log[-1]['loss']:.3f} val cost/ratio {v:.4f} cost/expert {ev[:,1].mean():.4f} t={time.time()-t0:.0f}s", flush=True)
                if v < best:
                    best = v; torch.save(model.state_dict(), out)
    json.dump(log, open("results/distill_log.json", "w"))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
