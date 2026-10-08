"""Learned removal policy (DeepSets scorer) + rollout utilities."""
import time
import numpy as np
import torch
import torch.nn as nn
from .bdp import BackwardEnv, N_FEAT
from .baselines import _result


class Scorer(nn.Module):
    """score_i = MLP([phi(f_i), mean_j phi(f_j), max_j phi(f_j)]) over removable candidates."""
    def __init__(self, d=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(N_FEAT, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, F):                       # F: (m,N_FEAT)
        h = self.phi(F)
        ctx = torch.cat([h, h.mean(0, keepdim=True).expand_as(h), h.max(0, keepdim=True)[0].expand_as(h)], 1)
        return self.head(ctx).squeeze(-1)


def rollout(inst, model, mode="sample", rng=None, temperature=1.0, store=False, active0=None):
    """Run one backward-elimination episode with the learned scorer.
    mode: 'greedy' (argmax) or 'sample'. If store, also returns the (features, action) list."""
    env = BackwardEnv(inst, active0)
    traj = []
    with torch.no_grad():
        while True:
            gi, F = env.features()
            if len(gi) == 0:
                break
            logits = model(torch.from_numpy(F)) / temperature
            if mode == "greedy":
                a = int(torch.argmax(logits))
            else:
                p = torch.softmax(logits, 0).numpy().astype(np.float64); p /= p.sum()
                a = int(rng.choice(len(p), p=p))
            if store:
                traj.append((F, a))
            env.remove(int(gi[a]))
    return env, traj


def solve(inst, model, mode="greedy", n_samples=1, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_samples):
        env, _ = rollout(inst, model, "greedy" if (mode == "greedy" and s == 0) else "sample", rng)
        c = env.cost()
        if best is None or c < best[0]:
            best = (c, env.active.copy())
    return _result(inst, best[1], t0, n_samples, f"policy-{mode}{n_samples if n_samples>1 else ''}")


def lns(inst, model, iters=30, k_add=12, seed=0, temperature=0.5, init=None):
    """Policy-guided ruin-and-recreate: re-insert k_add excluded baselines at random, then
    re-run (stochastic) policy-guided backward elimination; accept if cheaper."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if init is None:
        env, _ = rollout(inst, model, "greedy")
        cur = env.active.copy()
    else:
        cur = init.copy()
    cur_c = float(inst.cost[cur].sum())
    for _ in range(iters):
        out = np.where(~cur)[0]
        if len(out) == 0:
            break
        add = rng.choice(out, size=min(k_add, len(out)), replace=False)
        a0 = cur.copy(); a0[add] = True
        env, _ = rollout(inst, model, "sample", rng, temperature=temperature, active0=a0)
        c = env.cost()
        if c < cur_c - 1e-9:
            cur, cur_c = env.active.copy(), c
    return _result(inst, cur, t0, iters, f"policy-LNS{iters}")
