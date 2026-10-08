"""Backward-elimination engine with exact rank-one feasibility masks.

State = set of active baselines. Removal of baseline i is *admissible* iff the plan without i
is still feasible (connected, accuracy limits, redundancy floor). All admissibility tests for
every active baseline are obtained in closed form (Sherman-Morrison downdate), so any removal
order yields a feasible, irreducible plan by construction.
"""
import numpy as np
from .env import Instance, CTRL_SPACING, L_CAP

EPS_R = 1e-7
N_FEAT = 21


class BackwardEnv:
    def __init__(self, inst: Instance, active=None):
        self.inst = inst
        self._active0 = active
        N = len(inst.xy)
        s = inst.s_along
        self.total = max(s[-1], 1.0)
        self.s_mid = 0.5 * (s[inst.ei] + s[inst.ej]) / self.total
        sc = s[inst.ctrl]
        mid = 0.5 * (s[inst.ei] + s[inst.ej])
        self.d_ctrl = np.abs(mid[:, None] - sc[None, :]).min(1) / CTRL_SPACING
        isc = np.zeros(N, bool); isc[inst.ctrl] = True
        self.n_ctrl_end = isc[inst.ei].astype(int) + isc[inst.ej].astype(int)
        self.N = N
        self.reset()

    def reset(self):
        self.active = np.ones(self.inst.M, bool) if self._active0 is None else self._active0.copy()
        self._refresh()
        return self

    def _refresh(self):
        I = self.inst
        idx = np.where(self.active)[0]
        A = I.A[idx]
        self.Qh = np.linalg.inv(A.T @ (I.wh[idx, None] * A))
        self.Qu = np.linalg.inv(A.T @ (I.wu[idx, None] * A))
        self._since = 0
        self._masks()

    def _masks(self):
        I = self.inst
        idx = np.where(self.active)[0]
        A = I.A[idx]
        feas = np.ones(len(idx), bool)
        slack, rcomp = {}, {}
        for comp, Q, w in (("h", self.Qh, I.wh[idx]), ("u", self.Qu, I.wu[idx])):
            QA = Q @ A.T
            G = A @ QA
            r = 1.0 - w * np.diag(G)
            rs = np.maximum(r, EPS_R)
            post = np.diag(Q)[:, None] + (w / rs)[None, :] * QA ** 2
            eps = I.eps_h if comp == "h" else I.eps_u
            s_pt = (eps ** 2 - post.max(0)) / eps ** 2
            rk = r[:, None] - (w[:, None] * w[None, :]) * G ** 2 / rs[None, :]
            np.fill_diagonal(rk, np.inf)
            s_red = rk.min(0) - I.r_min
            feas &= (r > EPS_R) & (s_pt >= 0) & (s_red >= 0)
            slack[comp] = (s_pt, s_red)
            rcomp[comp] = r
        self.idx, self.feas, self.slack, self.r = idx, feas, slack, rcomp

    def removable(self):
        return self.idx[self.feas]

    def remove(self, k):
        I = self.inst
        a = I.A[k]
        for comp in ("h", "u"):
            Q = self.Qh if comp == "h" else self.Qu
            w = I.wh[k] if comp == "h" else I.wu[k]
            qa = Q @ a
            Q += w * np.outer(qa, qa) / (1.0 - w * (a @ qa))
        self.active[k] = False
        self._since += 1
        if self._since >= 25:
            self._refresh()
        else:
            self._masks()

    def cost(self):
        return float(self.inst.cost[self.active].sum())

    # ------------------------------------------------------------------ features
    def features(self):
        """Features of the *removable* candidates. Returns (global idx, F[m,N_FEAT])."""
        I = self.inst
        sel = np.where(self.feas)[0]
        if len(sel) == 0:
            return np.zeros(0, int), np.zeros((0, N_FEAT), np.float32)
        gi = self.idx[sel]
        deg = np.bincount(np.concatenate([I.ei[self.idx], I.ej[self.idx]]), minlength=self.N)
        da, db = deg[I.ei[gi]], deg[I.ej[gi]]
        cost = I.cost[gi]
        dmg_h = 1.0 - self.slack["h"][0][sel]
        dmg_u = 1.0 - self.slack["u"][0][sel]

        def pct(x):
            return (np.argsort(np.argsort(x)) + 0.5) / len(x)

        hh = (I.eps_h ** 2 - np.diag(self.Qh).max()) / I.eps_h ** 2
        hu = (I.eps_u ** 2 - np.diag(self.Qu).max()) / I.eps_u ** 2
        F = np.stack([
            cost / I.cost.mean(),
            I.L[gi] / L_CAP,
            self.r["h"][sel], self.r["u"][sel],
            np.clip(self.slack["h"][0][sel], 0, 1), np.clip(self.slack["u"][0][sel], 0, 1),
            np.clip(self.slack["h"][1][sel] / 0.5, 0, 1), np.clip(self.slack["u"][1][sel] / 0.5, 0, 1),
            np.minimum(da, db) / 10.0, np.maximum(da, db) / 10.0,
            self.s_mid[gi], np.clip(self.d_ctrl[gi], 0, 2),
            pct(cost / (dmg_h + dmg_u + 1e-6)),
            np.full(len(gi), np.clip(hh, 0, 1)), np.full(len(gi), np.clip(hu, 0, 1)),
            pct(I.L[gi]), pct(self.r["h"][sel]), pct(self.slack["u"][0][sel]),
            np.full(len(gi), len(gi) / I.M), np.full(len(gi), self.active.sum() / I.M),
            np.full(len(gi), self.active.sum() / max(self.inst.n, 1) / 10.0),
        ], 1).astype(np.float32)
        return gi, F
