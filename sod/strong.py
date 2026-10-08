"""Stronger baselines and an exact solver for small networks.

* sa_repair / ga_repair : feasible-space metaheuristics, warm-started from the greedy rule, with a CPU-time budget.
* bnb_exact             : exact branch-and-bound (accuracy-based pruning is monotone; cost bound; stop when feasible).
"""
import time
import numpy as np
from .env import violation, verify, build_instance, _point_std
from . import baselines as B
from .baselines import _result


def _feas(inst, act):
    return violation(inst, act) == 0.0


def _polish(inst, act, order):
    """Make a feasible plan irreducible: try removing edges in the given order."""
    act = act.copy()
    for e in order:
        if act[e]:
            act[e] = False
            if not _feas(inst, act):
                act[e] = True
    return act


def sa_repair(inst, budget_s=3.0, seed=0, init=None):
    t0 = time.process_time(); rng = np.random.default_rng(seed); M = inst.M
    cur = (init if init is not None else B.backward(inst, "ratio")["active"]).copy()
    cc = float(inst.cost[cur].sum()); best, bc = cur.copy(), cc
    T0, T1 = 0.25 * inst.cost.mean(), 0.005 * inst.cost.mean()
    it = 0
    while True:
        el = time.process_time() - t0
        if el >= budget_s: break
        T = T0 * (T1 / T0) ** (el / budget_s)
        act = np.where(cur)[0]; out = np.where(~cur)[0]
        u = rng.random(); y = cur.copy()
        if u < 0.45 and len(act) > 1:                    # remove
            y[rng.choice(act)] = False
        elif u < 0.85 and len(out):                      # swap (remove one, add one)
            y[rng.choice(act)] = False; y[rng.choice(out)] = True
        elif len(out):                                   # add (uphill, allows reshaping)
            y[rng.choice(out)] = True
        else:
            continue
        it += 1
        cy = float(inst.cost[y].sum()); d = cy - cc
        if d > 0 and rng.random() >= np.exp(-d / T): continue
        if not _feas(inst, y): continue
        cur, cc = y, cy
        if cc < bc - 1e-9: best, bc = cur.copy(), cc
    best = _polish(inst, best, np.argsort(-inst.cost))
    return _result(inst, best, time.time(), it, f"SA-repair@{budget_s:g}s")


def ga_repair(inst, budget_s=3.0, seed=0, pop=14):
    t0 = time.process_time(); rng = np.random.default_rng(seed); M = inst.M

    def rand_polish(act):
        noise = rng.gumbel(size=M) * 0.15 * inst.cost.mean()
        return _polish(inst, act, np.argsort(-(inst.cost + noise)))

    P = [B.backward(inst, "ratio")["active"]]
    full = np.ones(M, bool)
    while len(P) < pop and (len(P) < 4 or time.process_time() - t0 < budget_s * 0.3):
        P.append(rand_polish(full))
    C = [float(inst.cost[p].sum()) for p in P]
    it = 0
    while time.process_time() - t0 < budget_s:
        i, j = rng.choice(len(P), 2, replace=False)
        child = P[i] | P[j]
        out = np.where(~child)[0]
        if len(out) and rng.random() < 0.5:
            child[rng.choice(out, size=min(len(out), int(rng.integers(1, 6))), replace=False)] = True
        child = rand_polish(child); cc = float(inst.cost[child].sum()); it += 1
        w = int(np.argmax(C))
        if cc < C[w] - 1e-9 and not any((child == p).all() for p in P):
            P[w], C[w] = child, cc
    b = int(np.argmin(C))
    return _result(inst, P[b], time.time(), it, f"GA-repair@{budget_s:g}s")


# ------------------------------------------------------------------ exact solver (small networks)
def _acc_ok(inst, act):
    A = inst.A[act]
    if A.shape[0] < inst.n: return False
    try:
        Qh = np.linalg.inv(A.T @ (inst.wh[act, None] * A)); Qu = np.linalg.inv(A.T @ (inst.wu[act, None] * A))
    except np.linalg.LinAlgError:
        return False
    return Qh.diagonal().max() <= inst.eps_h ** 2 + 1e-12 and Qu.diagonal().max() <= inst.eps_u ** 2 + 1e-12


def bnb_exact(inst, ub_plan, time_limit=300.0):
    """Exact minimum-cost feasible plan. Prunes on cost bound and on the (monotone) accuracy/connectivity
    condition of I + undecided; a node whose included set I is fully feasible is a candidate (no need to add more)."""
    t0 = time.time(); M = inst.M
    order = np.argsort(-inst.cost)                 # decide expensive baselines first
    best = ub_plan.copy(); bc = float(inst.cost[best].sum()); nodes = 0; timed_out = False
    inc = np.zeros(M, bool); und = np.ones(M, bool)
    stack = [(0, None)]

    def rec(k, ic):
        nonlocal best, bc, nodes, timed_out
        nodes += 1
        if (nodes & 255) == 0 and time.time() - t0 > time_limit:
            timed_out = True
        if timed_out or ic >= bc - 1e-9: return
        if _feas(inst, inc):
            best, bc = inc.copy(), ic; return
        if k == M: return
        if not _acc_ok(inst, inc | und): return
        e = order[k]
        und[e] = False                              # branch 1: exclude e
        rec(k + 1, ic)
        inc[e] = True                               # branch 2: include e
        rec(k + 1, ic + inst.cost[e])
        inc[e] = False; und[e] = True

    rec(0, 0.0)
    return dict(active=best, cost=bc, nodes=nodes, optimal=not timed_out, time=time.time() - t0)


def small_instance(seed, N=10, length=45.0, kappa=1.6, n_ctrl=2):
    """Small corridor with accuracy limits set relative to the full network (kappa x its precision)."""
    rng = np.random.default_rng(seed)
    gaps = rng.lognormal(0, 0.4, N - 1); s = np.concatenate([[0], np.cumsum(gaps)]); s = s / s[-1] * length
    y = rng.normal(0, 0.6, N); xy = np.stack([s, y], 1)
    ctrl = [0, N - 1] if n_ctrl == 2 else [0, N // 2, N - 1]
    inst = build_instance(xy, ctrl, eps_h=1e9, eps_u=1e9, name=f"small-{seed}", s_along=s)
    sh, su = _point_std(inst, np.ones(inst.M, bool))
    inst.eps_h, inst.eps_u = kappa * sh.max(), kappa * su.max()
    return inst
