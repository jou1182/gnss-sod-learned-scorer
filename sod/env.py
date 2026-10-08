"""Corridor GNSS control-network simulator for second-order design (SOD).

Model (documented assumptions, see paper section 3):
  * Observations are GNSS baseline vectors in a local ENU frame, independent between
    baselines, no inter-component correlation. Under this model the E, N and U components
    decouple: each is a graph (incidence-matrix) least-squares problem whose edge weight
    depends only on baseline length L:  sigma(L)^2 = a^2 + (b*L)^2.
  * Fixed (error-free) control stations model KSA-CORS-type reference points.
  * A plan = subset of candidate baselines. Feasible iff
        (i)   network connected to control (normal matrix non-singular),
        (ii)  max point std (E,N) <= eps_h  and  max point std (U) <= eps_u,
        (iii) every retained baseline has redundancy number r_i >= r_min (E,N and U).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# --- observation-error model (mm, mm/km) ------------------------------------------------
A_H, B_H = 5.0, 1.0     # horizontal  sigma = sqrt(5^2 + (1.0*L)^2)  mm,  L in km
A_U, B_U = 12.0, 3.0    # vertical    sigma = sqrt(12^2 + (3.0*L)^2) mm
L_CAP = 100.0           # candidate baselines: all pairs with L <= 100 km
COST_L0 = 25.0          # cost_i = 1 + L_i / 25  (session-equivalents: setup + travel/occupation)
R_MIN = 0.10            # minimum redundancy number of each retained baseline
EPS_H, EPS_U = 10.0, 25.0   # absolute 1-sigma point-accuracy limits (mm)
CTRL_SPACING = 100.0    # km, ~ KSA-CORS mean spacing (209 stations / 2.15e6 km^2)


@dataclass
class Instance:
    xy: np.ndarray          # (N,2) km
    ctrl: np.ndarray        # indices of fixed stations
    free: np.ndarray        # indices of free stations
    ei: np.ndarray          # (M,) candidate start station
    ej: np.ndarray          # (M,) candidate end station
    L: np.ndarray           # (M,) km
    cost: np.ndarray        # (M,)
    wh: np.ndarray          # (M,) weights horizontal (1/mm^2)
    wu: np.ndarray          # (M,) weights vertical
    A: np.ndarray           # (M,n_free) incidence restricted to free stations
    eps_h: float            # mm
    eps_u: float            # mm
    r_min: float
    s_along: np.ndarray     # (N,) along-track coordinate (km)
    name: str = ""

    @property
    def M(self): return len(self.L)
    @property
    def n(self): return len(self.free)


def build_instance(xy, ctrl, eps_h=EPS_H, eps_u=EPS_U, r_min=R_MIN, l_cap=L_CAP, name="", s_along=None):
    N = len(xy)
    ctrl = np.array(sorted(set(int(c) for c in ctrl)))
    free = np.array([i for i in range(N) if i not in set(ctrl.tolist())])
    ii, jj = np.triu_indices(N, 1)
    d = np.hypot(*(xy[ii] - xy[jj]).T)
    keep = d <= l_cap
    ei, ej, L = ii[keep], jj[keep], d[keep]
    # drop baselines between two control stations (carry no information)
    cset = np.zeros(N, bool); cset[ctrl] = True
    keep2 = ~(cset[ei] & cset[ej])
    ei, ej, L = ei[keep2], ej[keep2], L[keep2]
    M = len(L)
    pos = -np.ones(N, int); pos[free] = np.arange(len(free))
    A = np.zeros((M, len(free)))
    for k in range(M):
        if pos[ei[k]] >= 0: A[k, pos[ei[k]]] = -1.0
        if pos[ej[k]] >= 0: A[k, pos[ej[k]]] = +1.0
    wh = 1.0 / (A_H ** 2 + (B_H * L) ** 2)
    wu = 1.0 / (A_U ** 2 + (B_U * L) ** 2)
    cost = 1.0 + L / COST_L0
    if s_along is None:
        s_along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    inst = Instance(xy, ctrl, free, ei, ej, L, cost, wh, wu, A, 0.0, 0.0, r_min, s_along, name)
    inst.eps_h, inst.eps_u = eps_h, eps_u
    return inst


def _point_std(inst, active):
    A = inst.A[active]
    Qh = np.linalg.inv(A.T @ (inst.wh[active, None] * A))
    Qu = np.linalg.inv(A.T @ (inst.wu[active, None] * A))
    return np.sqrt(np.diag(Qh)), np.sqrt(np.diag(Qu))


def verify(inst, active):
    """Independent from-scratch verifier. Returns (feasible, info-dict)."""
    active = np.asarray(active, bool)
    if active.sum() < inst.n:
        return False, dict(reason="too_few")
    A = inst.A[active]
    Nh = A.T @ (inst.wh[active, None] * A)
    Nu = A.T @ (inst.wu[active, None] * A)
    try:
        np.linalg.cholesky(Nh); np.linalg.cholesky(Nu)
        Qh, Qu = np.linalg.inv(Nh), np.linalg.inv(Nu)
    except np.linalg.LinAlgError:
        return False, dict(reason="rank", sig_h=np.inf, sig_u=np.inf)
    sh, su = np.sqrt(np.diag(Qh)), np.sqrt(np.diag(Qu))
    rh = 1 - inst.wh[active] * np.einsum("ij,jk,ik->i", A, Qh, A)
    ru = 1 - inst.wu[active] * np.einsum("ij,jk,ik->i", A, Qu, A)
    ok = (sh.max() <= inst.eps_h + 1e-9 and su.max() <= inst.eps_u + 1e-9
          and rh.min() >= inst.r_min - 1e-9 and ru.min() >= inst.r_min - 1e-9)
    return bool(ok), dict(sig_h=float(sh.max()), sig_u=float(su.max()),
                          r_h=float(rh.min()), r_u=float(ru.min()),
                          cost=float(inst.cost[active].sum()), n_obs=int(active.sum()),
                          length=float(inst.L[active].sum()))


def violation(inst, active):
    """Scalar constraint violation (0 if feasible) used by penalty-based metaheuristics."""
    active = np.asarray(active, bool)
    if active.sum() < inst.n:
        return 1e3 + (inst.n - active.sum())
    A = inst.A[active]
    Nh = A.T @ (inst.wh[active, None] * A)
    Nu = A.T @ (inst.wu[active, None] * A)
    try:
        np.linalg.cholesky(Nh); np.linalg.cholesky(Nu)
        Qh, Qu = np.linalg.inv(Nh), np.linalg.inv(Nu)
    except np.linalg.LinAlgError:
        return 1e3
    sh, su = np.sqrt(Qh.diagonal()), np.sqrt(Qu.diagonal())
    v = max(0.0, sh.max() / inst.eps_h - 1) + max(0.0, su.max() / inst.eps_u - 1)
    rh = 1 - inst.wh[active] * np.einsum("ij,jk,ik->i", A, Qh, A)
    ru = 1 - inst.wu[active] * np.einsum("ij,jk,ik->i", A, Qu, A)
    v += max(0.0, inst.r_min - rh.min()) + max(0.0, inst.r_min - ru.min())
    return float(v)


# ------------------------------------------------------------------ instance generators
def _ctrl_along(s, spacing, rng):
    """Control stations at both ends + approximately every `spacing` km (nearest station)."""
    total = s[-1]
    k = int(np.floor(total / spacing))
    ctrl = {0, len(s) - 1}
    if k >= 2:
        for t in np.linspace(0, total, k + 1)[1:-1] + rng.normal(0, 0.1 * spacing, k - 1):
            ctrl.add(int(np.argmin(np.abs(s - t))))
    return sorted(ctrl)


def synth_corridor(seed, N=None, length=None, eps_h=EPS_H, eps_u=EPS_U, r_min=R_MIN, spacing=CTRL_SPACING):
    """Random corridor; redraws until the full network meets the accuracy limits."""
    for attempt in range(50):
        rng = np.random.default_rng(seed * 100 + attempt)
        n = N or int(rng.integers(16, 37))
        ln = length or float(rng.uniform(40, 300))
        gaps = rng.lognormal(0, 0.45, n - 1)
        s = np.concatenate([[0], np.cumsum(gaps)]); s = s / s[-1] * ln
        amp = rng.uniform(0.0, 0.08) * ln
        ph = rng.uniform(0, 2 * np.pi, 3)
        y = sum(amp / (k + 1) * np.sin(2 * np.pi * (k + 1) * s / ln * rng.uniform(0.5, 1.5) + ph[k]) for k in range(3))
        y = y + rng.normal(0, 0.8, n)
        xy = np.stack([s, y], 1)
        inst = build_instance(xy, _ctrl_along(s, spacing, rng), eps_h, eps_u, r_min, name=f"synth-{seed}")
        if verify(inst, np.ones(inst.M, bool))[0]:
            return inst
    raise RuntimeError("no feasible instance")


def geo_corridor(waypoints_latlon, N, seed, spacing=CTRL_SPACING, jitter_km=0.8,
                 eps_h=EPS_H, eps_u=EPS_U, r_min=R_MIN, name=""):
    """Schematic corridor along a polyline of (lat, lon) waypoints (approximate city
    coordinates, NOT survey alignments). Stations roughly equally spaced in arc length."""
    rng = np.random.default_rng(seed)
    pts = np.array(waypoints_latlon, float)
    lat0 = pts[:, 0].mean()
    kx = 111.32 * np.cos(np.radians(lat0)); ky = 110.57
    xy_w = np.stack([(pts[:, 1] - pts[0, 1]) * kx, (pts[:, 0] - pts[0, 0]) * ky], 1)
    seg = np.hypot(*np.diff(xy_w, axis=0).T); cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    s = np.linspace(0, total, N) + np.r_[0, rng.normal(0, 0.15 * total / N, N - 2), 0]
    s = np.clip(np.sort(s), 0, total)
    x = np.interp(s, cum, xy_w[:, 0]); y = np.interp(s, cum, xy_w[:, 1])
    xy = np.stack([x, y], 1) + rng.normal(0, jitter_km, (N, 2))
    return build_instance(xy, _ctrl_along(s, spacing, rng), eps_h, eps_u, r_min, name=name, s_along=s)
