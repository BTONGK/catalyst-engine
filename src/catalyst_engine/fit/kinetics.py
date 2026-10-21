"""Fit a catalyst's pathway parameters to steady-state bench data, for any pack.

Weighted least squares on outlet concentrations (wet ppm) with the protocol's
analyzer error model. Readings below the limit of detection are censored: they
only penalize a model that predicts more than the LOD. Uncertainty comes from
the Jacobian at the optimum, inflated by the reduced chi-square when the fit is
worse than the error model claims; parameters the protocol cannot pin down are
flagged so the protocol, not the catalyst, gets fixed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from catalyst_engine.chem.reactor import Conditions, outlet, solve
from catalyst_engine.packs import Pack, ProtocolSpec
from catalyst_engine.units import f_to_k

# Above these standard deviations a parameter is reported as poorly determined.
POORLY_DETERMINED = {"ln_k_ref": 0.7, "Ea_kJ": 30.0, "ln_K_ref": 2.0, "dH_kJ": 50.0}


@dataclass
class Observation:
    step: str
    t_k: float
    feed: np.ndarray  # wet mole fractions, network species order
    ghsv_per_h: float
    p_pa: float
    measured_ppm: dict[str, float]  # wet ppm at the outlet


def observations_from_table(pack: Pack, protocol: ProtocolSpec, rows: list[dict]) -> list[Observation]:
    """Rows carry `step`, a temperature (`T_F`, `T_C` or `T_K`) and `out_<species>_ppm` readings.
    Optional `in_<species>_ppm` columns override the protocol feed with the measured inlet."""
    species = pack.network.species
    obs = []
    for row in rows:
        step = protocol.step(str(row["step"]))
        if "T_K" in row and row["T_K"] not in ("", None):
            t_k = float(row["T_K"])
        elif "T_C" in row and row["T_C"] not in ("", None):
            t_k = float(row["T_C"]) + 273.15
        else:
            t_k = f_to_k(float(row["T_F"]))
        feed = step.feed_fractions(species)
        for s in species:
            v = row.get(f"in_{s}_ppm")
            if v not in ("", None):
                feed[species.index(s)] = float(v) * 1e-6
        measured = {}
        for s in protocol.analyzer.measured:
            v = row.get(f"out_{s}_ppm")
            if v not in ("", None):
                measured[s] = float(v)
        obs.append(Observation(step.id, t_k, feed, step.ghsv_per_h, step.pressure_kPa * 1000.0, measured))
    return obs


def read_observations(pack: Pack, protocol: ProtocolSpec, path: Path) -> list[Observation]:
    with Path(path).open(newline="") as fh:
        return observations_from_table(pack, protocol, list(csv.DictReader(fh)))


@dataclass
class FitResult:
    pack: str
    params: dict[str, float]  # full parameter set (fitted values plus fixed defaults)
    fitted: list[str]
    std: dict[str, float]
    correlation: np.ndarray
    chi2_reduced: float
    n_points: int
    poorly_determined: list[str] = field(default_factory=list)
    residuals: list[dict] = field(default_factory=list)

    def rate_constant_ref(self, pathway: str) -> tuple[float, float]:
        """k at the pack's reference temperature and its 1-sigma multiplicative factor."""
        ln_k = self.params[f"{pathway}.ln_k_ref"]
        return float(np.exp(ln_k)), float(np.exp(self.std.get(f"{pathway}.ln_k_ref", 0.0)))


def fit_kinetics(
    pack: Pack,
    protocol: ProtocolSpec,
    observations: list[Observation],
    *,
    fit: list[str] | None = None,
    fixed: dict[str, float] | None = None,
    n_starts: int = 6,
    seed: int = 0,
) -> FitResult:
    net = pack.network
    names = list(fit or net.fitted)
    base = net.parameters(fixed)
    unknown = set(names) - set(base)
    if unknown:
        raise KeyError(f"unknown parameters {sorted(unknown)}")
    an = protocol.analyzer
    cond = Conditions(
        np.array([o.t_k for o in observations]),
        np.array([o.p_pa for o in observations]),
        np.array([o.ghsv_per_h for o in observations]),
        np.array([o.feed for o in observations]),
    )
    # Flatten the measured points: (observation index, species index, value, sigma, lod)
    pts = []
    for i, o in enumerate(observations):
        for s, v in o.measured_ppm.items():
            sigma = float(np.hypot(an.sigma_abs_ppm, an.sigma_rel * v))
            pts.append((i, net.index(s), v, sigma, an.lod_ppm.get(s, 0.0)))
    if len(pts) <= len(names):
        raise ValueError(f"{len(pts)} measured points cannot determine {len(names)} parameters")
    obs_i = np.array([p[0] for p in pts])
    sp_i = np.array([p[1] for p in pts])
    meas = np.array([p[2] for p in pts])
    sig = np.array([p[3] for p in pts])
    lod = np.array([p[4] for p in pts])
    censored = meas < lod

    n_obs = cond.size
    frozen = {"grid": np.zeros((n_obs, 0))}

    def model_ppm_batch(xs: np.ndarray) -> np.ndarray:
        """Predicted ppm for several parameter vectors in one reactor call: (m, n_points)."""
        m = len(xs)
        tiled = Conditions(
            np.tile(cond.t_k, m), np.tile(cond.p_pa, m), np.tile(cond.ghsv_per_h, m), np.tile(cond.feed, (m, 1))
        )
        params = {k: np.full(m * n_obs, v) for k, v in base.items()}
        for j, n in enumerate(names):
            params[n] = np.repeat(xs[:, j], n_obs)
        with np.errstate(over="ignore", invalid="ignore"):
            out = outlet(net, params, tiled, grid=np.tile(frozen["grid"], (m, 1))).reshape(m, n_obs, -1)
        return out[:, obs_i, sp_i] * 1e6

    def to_residuals(pred: np.ndarray) -> np.ndarray:
        r = np.where(censored, np.maximum(pred - lod, 0.0) / sig, (pred - meas) / sig)
        return np.where(np.isfinite(r), r, 1e6)

    def residuals(x: np.ndarray) -> np.ndarray:
        return to_residuals(model_ppm_batch(x[None, :]))[0]

    def jacobian(x: np.ndarray) -> np.ndarray:
        h = 1e-5 * np.maximum(1.0, np.abs(x))
        xs = np.vstack([x, x + np.diag(h)])
        r = to_residuals(model_ppm_batch(xs))
        return ((r[1:] - r[0]) / h[:, None]).T

    lo = np.array([net.bounds(n)[0] for n in names])
    hi = np.array([net.bounds(n)[1] for n in names])
    x0 = np.clip(np.array([base[n] for n in names]), lo + 1e-6, hi - 1e-6)
    rng = np.random.default_rng(seed)
    starts = [x0]
    for _ in range(max(0, n_starts - 1)):
        jitter = np.array([rng.normal(0, 1.5) if n.endswith(("ln_k_ref", "ln_K_ref")) else rng.normal(0, 25) for n in names])
        starts.append(np.clip(x0 + jitter, lo + 1e-6, hi - 1e-6))

    def adapt_grid(x: np.ndarray) -> np.ndarray:
        return solve(net, base | dict(zip(names, x)), cond, rtol=1e-5, atol_ppm=1e-4).grid

    def run(start: np.ndarray):
        return least_squares(residuals, start, jac=jacobian, bounds=(lo, hi), x_scale="jac", method="trf", max_nfev=200)

    # Each start freezes the step grid its own parameters need, so the model is smooth in the
    # parameters during the fit. If the optimum's own grid disagrees, refit from it on that grid.
    best, best_grid = None, frozen["grid"]
    for start in starts:
        frozen["grid"] = adapt_grid(start)
        res = run(start)
        if best is None or res.cost < best.cost:
            best, best_grid = res, frozen["grid"]
    assert best is not None
    frozen["grid"] = best_grid
    on_frozen = model_ppm_batch(best.x[None, :])[0]
    frozen["grid"] = adapt_grid(best.x)
    if np.max(np.abs(model_ppm_batch(best.x[None, :])[0] - on_frozen)) > 1e-3:
        best = run(best.x)

    dof = max(len(pts) - len(names), 1)
    chi2_red = float(2.0 * best.cost / dof)
    jac = best.jac
    cov = np.linalg.pinv(jac.T @ jac) * max(chi2_red, 1.0)
    std_arr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std_arr, std_arr)
    std = dict(zip(names, map(float, std_arr)))
    poorly = [n for n in names if std[n] > POORLY_DETERMINED[n.rsplit(".", 1)[-1]]]
    params = base | dict(zip(names, map(float, best.x)))
    pred = model_ppm_batch(best.x[None, :])[0]
    resid_rows = [
        {
            "step": observations[i].step,
            "T_K": observations[i].t_k,
            "species": net.species[s],
            "measured_ppm": m,
            "model_ppm": float(p),
            "censored": bool(c),
        }
        for i, s, m, p, c in zip(obs_i, sp_i, meas, pred, censored)
    ]
    return FitResult(pack.id, params, names, std, corr, chi2_red, len(pts), poorly, resid_rows)
