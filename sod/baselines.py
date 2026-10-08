"""Reference solvers: conventional corridor layouts, backward-elimination heuristics,
and penalty-based metaheuristics (binary GA, binary PSO, SA) as used in the SOD literature."""
import time
import numpy as np
from .env import verify, violation
from .bdp import BackwardEnv


def _result(inst, active, t0, n_eval=0, name=""):
    ok, info = verify(inst, active)
    return dict(method=name, feasible=ok, cost=float(inst.cost[active].sum()),
                n_obs=int(np.sum(active)), length=float(inst.L[active].sum()),
                time=time.time() - t0, n_eval=n_eval, active=np.asarray(active, bool).copy(), **{k: v for k, v in info.items() if k in ("sig_h", "sig_u")})


# ---------------------------------------------------------------- conventional layouts
def conventional(inst, kind):
    """chain: consecutive stations only; ladder: consecutive + skip-one ties."""
    order = np.argsort(inst.s_along)
    rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    gap = np.abs(rank[inst.ei] - rank[inst.ej])
    t0 = time.time()
    if kind == "chain":
        act = gap == 1
    elif kind == "ladder":
        act = gap <= 2
    else:
        raise ValueError(kind)
    return _result(inst, act, t0, 1, f"conv-{kind}")


# ---------------------------------------------------------------- backward elimination
def backward(inst, rule="longest", rng=None, policy=None):
    t0 = time.time()
    env = BackwardEnv(inst)
    rng = rng or np.random.default_rng(0)
    steps = 0
    while True:
        cand = np.where(env.feas)[0]
        if len(cand) == 0:
            break
        if rule == "longest":
            k = cand[np.argmax(inst.cost[env.idx[cand]])]
        elif rule == "random":
            k = rng.choice(cand)
        elif rule == "ratio":      # cost saved per unit of accuracy slack consumed
            dmg = np.zeros(len(env.idx))
            for comp in ("h", "u"):
                dmg += 1.0 - env.slack[comp][0]
            sc = inst.cost[env.idx[cand]] / (dmg[cand] + 1e-9)
            k = cand[np.argmax(sc)]
        elif rule == "policy":
            k = policy(env, cand)
        else:
            raise ValueError(rule)
        env.remove(int(env.idx[k]))
        steps += 1
    return _result(inst, env.active, t0, steps, f"bdp-{rule}")


# ---------------------------------------------------------------- penalty metaheuristics
class _Fit:
    def __init__(self, inst, budget):
        self.inst, self.budget, self.n = inst, budget, 0
        self.P = 2.0 * inst.cost.sum()
        self.best, self.best_f, self.best_feas, self.best_cost = None, np.inf, None, np.inf
        self.best_feas_x = None

    def __call__(self, x):
        self.n += 1
        v = violation(self.inst, x)
        c = float(self.inst.cost[x].sum())
        f = c + self.P * v
        if f < self.best_f:
            self.best_f, self.best = f, x.copy()
        if v == 0.0 and c < self.best_cost:
            self.best_cost, self.best_feas_x = c, x.copy()
        return f

    @property
    def done(self): return self.n >= self.budget


def _finish(inst, fit, t0, name):
    x = fit.best_feas_x if fit.best_feas_x is not None else fit.best
    return _result(inst, x, t0, fit.n, name)


def ga(inst, budget=20000, seed=0, pop=36, pc=0.9, pm=None):
    rng = np.random.default_rng(seed); t0 = time.time(); M = inst.M
    pm = pm or 1.5 / M
    fit = _Fit(inst, budget)
    P = rng.random((pop, M)) < 0.7
    P[0] = True
    F = np.array([fit(p) for p in P])
    while not fit.done:
        newP = [P[np.argmin(F)].copy()]                     # elitism
        while len(newP) < pop:
            a = min(rng.integers(0, pop, 3), key=lambda i: F[i]); b = min(rng.integers(0, pop, 3), key=lambda i: F[i])
            c = np.where(rng.random(M) < 0.5, P[a], P[b]) if rng.random() < pc else P[a].copy()
            c ^= rng.random(M) < pm
            newP.append(c)
        P = np.array(newP); F = np.array([fit(p) for p in P])
    return _finish(inst, fit, t0, "GA")


def bpso(inst, budget=20000, seed=0, swarm=36, w=0.7, c1=1.5, c2=1.5, vmax=4.0):
    rng = np.random.default_rng(seed); t0 = time.time(); M = inst.M
    fit = _Fit(inst, budget)
    X = rng.random((swarm, M)) < 0.7; X[0] = True
    V = rng.normal(0, 1, (swarm, M))
    pb = X.copy(); pf = np.array([fit(x) for x in X])
    g = pb[np.argmin(pf)].copy(); gf = pf.min()
    while not fit.done:
        r1, r2 = rng.random((swarm, M)), rng.random((swarm, M))
        V = np.clip(w * V + c1 * r1 * (pb ^ X) * np.where(pb, 1, -1) + c2 * r2 * (g[None] ^ X) * np.where(g[None], 1, -1), -vmax, vmax)
        X = rng.random((swarm, M)) < 1 / (1 + np.exp(-V))
        for i in range(swarm):
            f = fit(X[i])
            if f < pf[i]:
                pf[i], pb[i] = f, X[i].copy()
                if f < gf: gf, g = f, X[i].copy()
            if fit.done: break
    return _finish(inst, fit, t0, "BPSO")


def sa(inst, budget=20000, seed=0, t_start=None, t_end=None):
    rng = np.random.default_rng(seed); t0 = time.time(); M = inst.M
    fit = _Fit(inst, budget)
    x = np.ones(M, bool); f = fit(x)
    T0 = t_start or 0.5; T1 = t_end or 0.005
    n = 0
    while not fit.done:
        T = T0 * (T1 / T0) ** (fit.n / budget)
        y = x.copy(); k = rng.integers(0, M); y[k] ^= True
        if rng.random() < 0.3:
            y[rng.integers(0, M)] ^= True
        fy = fit(y)
        if fy < f or rng.random() < np.exp(-(fy - f) / (T * inst.cost.mean())):
            x, f = y, fy
    return _finish(inst, fit, t0, "SA")
