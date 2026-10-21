"""Activity decay under aging: fit, project, and say how far the projection reaches.

Activity a(t) is a pathway's rate constant after t hours of aging divided by its
fresh value, both measured at the pack's reference temperature. Two standard
forms are fitted and compared by AICc:

    exp:    a = a_eq + (1 - a_eq) exp(-kd t)                      first-order loss
    gple2:  a = a_eq + (1 - a_eq) / (1 + kd (1 - a_eq) t)         second-order (sintering-like)

`a_eq` is the activity the catalyst levels off at. Projections far beyond the
longest aging time measured are flagged; this is what lets a 6-week verdict
stand in for a 2,000-hour test, and also what keeps it honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from catalyst_engine.units import R


def _exp(t, kd, a_eq):
    return a_eq + (1.0 - a_eq) * np.exp(-kd * t)


def _gple2(t, kd, a_eq):
    return a_eq + (1.0 - a_eq) / (1.0 + kd * (1.0 - a_eq) * t)


MODELS = {"exp": _exp, "gple2": _gple2}


@dataclass
class DecayFit:
    model: str
    kd: float  # 1/h
    a_eq: float
    cov: np.ndarray  # covariance of (kd, a_eq)
    aicc: float
    t_max_h: float  # longest aging time measured
    n: int

    def activity(self, t_h) -> np.ndarray:
        return MODELS[self.model](np.asarray(t_h, dtype=float), self.kd, self.a_eq)

    def interval(self, t_h: float, level: float = 0.9, n_draws: int = 4000, seed: int = 0) -> tuple[float, float]:
        """Parameter-uncertainty interval for a(t), from draws of (kd, a_eq)."""
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal([self.kd, self.a_eq], self.cov, size=n_draws)
        kd = np.clip(draws[:, 0], 0.0, None)
        a_eq = np.clip(draws[:, 1], 0.0, 0.999)
        vals = MODELS[self.model](t_h, kd, a_eq)
        lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)

    def extrapolation_ratio(self, t_h: float) -> float:
        return t_h / self.t_max_h

    def hours_to(self, activity: float) -> float:
        """Aging hours until activity falls to `activity`; inf if the fit levels off above it."""
        if activity <= self.a_eq:
            return float("inf")
        if activity >= 1.0:
            return 0.0
        return float(brentq(lambda t: self.activity(t) - activity, 0.0, 1e9))


def fit_decay(t_h, activity, sigma=None, models: tuple[str, ...] = ("exp", "gple2")) -> DecayFit:
    """Fit aging data (t in hours, activity relative to fresh) and keep the model with the lowest AICc."""
    t = np.asarray(t_h, dtype=float)
    a = np.asarray(activity, dtype=float)
    s = np.full_like(a, 0.02) if sigma is None else np.broadcast_to(np.asarray(sigma, dtype=float), a.shape)
    if len(t) < 3:
        raise ValueError("need at least three aging points")
    best: DecayFit | None = None
    k = 2
    for name in models:
        f = MODELS[name]
        kd0 = 1.0 / max(t.max(), 1.0)
        res = least_squares(
            lambda x: (f(t, x[0], x[1]) - a) / s,
            [kd0, max(0.0, min(a.min() - 0.05, 0.9))],
            bounds=([0.0, 0.0], [10.0, 0.999]),
        )
        chi2 = float(2.0 * res.cost)
        n = len(t)
        aicc = chi2 + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
        dof = max(n - k, 1)
        cov = np.linalg.pinv(res.jac.T @ res.jac) * max(chi2 / dof, 1.0)
        fit = DecayFit(name, float(res.x[0]), float(res.x[1]), cov, aicc, float(t.max()), n)
        if best is None or fit.aicc < best.aicc:
            best = fit
    assert best is not None
    return best


def accelerate(kd: float, ea_kj: float, t_from_k: float, t_to_k: float) -> float:
    """Move a decay rate between aging temperatures with an Arrhenius activation energy."""
    return kd * float(np.exp(-ea_kj * 1000.0 / R * (1.0 / t_to_k - 1.0 / t_from_k)))
