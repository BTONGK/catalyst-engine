"""Isothermal plug-flow reactor for any compiled network, vectorized over catalysts and conditions.

Space velocity (GHSV) is normal m3 of gas per m3 of catalyst per hour at 0 C and 1 atm.
Optionally, an external mass-transfer coefficient caps each pathway's rate at the
flux of its key species to the wall (series resistance, exact for first-order
pathways), which is how a bench rate constant becomes a monolith design. Washcoat
diffusion scales each pathway's rate by the slab effectiveness factor tanh(phi)/phi,
with the Thiele modulus taken from the pathway's local first-order rate constant
(its rate over its key species' concentration; exact for first-order pathways).

Integration runs along the normalized bed coordinate z in [0, 1] with the ROS2
Rosenbrock method (second order, L-stable, so stiff kinetics cost accuracy steps,
not stability steps), step doubling for the error estimate and local Richardson
extrapolation. Each condition in a batch adapts its own steps: fast chemistry
can finish in a thin layer at the inlet while the rest of the bed takes long steps.

`solve` returns the accepted step grid. Passing it back as `grid` replays exactly
those steps, which makes the outlet a smooth function of the parameters; fitting
relies on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from catalyst_engine.chem.network import Network
from catalyst_engine.units import P_ATM, R, T_STD

_GAMMA = 1.0 + 1.0 / np.sqrt(2.0)  # ROS2, Verwer et al. (1999)


@dataclass
class Conditions:
    t_k: np.ndarray  # (B,)
    p_pa: np.ndarray  # (B,)
    ghsv_per_h: np.ndarray  # (B,)
    feed: np.ndarray  # (B, n_species) wet mole fractions
    km_a: np.ndarray | None = None  # (B, n_pathways) 1/s, external mass transfer for each key species
    # (B, n_pathways) s, washcoat diffusion for each key species: Thiele modulus squared per unit of the pathway's
    # first-order rate constant (1/s per m3 of catalyst), thickness^2 / (effective diffusivity x washcoat fraction)
    wc_phi2: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.feed = np.atleast_2d(np.asarray(self.feed, dtype=float))
        b = self.feed.shape[0]
        self.t_k = np.broadcast_to(np.asarray(self.t_k, dtype=float), (b,)).copy()
        self.p_pa = np.broadcast_to(np.asarray(self.p_pa, dtype=float), (b,)).copy()
        self.ghsv_per_h = np.broadcast_to(np.asarray(self.ghsv_per_h, dtype=float), (b,)).copy()
        if np.any(self.ghsv_per_h <= 0):
            raise ValueError("GHSV must be positive")

    @property
    def size(self) -> int:
        return self.feed.shape[0]


@dataclass
class Solution:
    outlet: np.ndarray  # (B, n_species) wet mole fractions
    grid: np.ndarray  # (B, n_steps) accepted step sizes along z, zero-padded


class _System:
    def __init__(self, network: Network, params: Mapping[str, float | np.ndarray], cond: Conditions):
        b = cond.size
        unknown = set(params) - set(network.defaults)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}")
        p = {k: np.broadcast_to(np.asarray(v, dtype=float), (b,)) for k, v in (network.defaults | dict(params)).items()}
        self.network = network
        self.n_s = len(network.species)
        self.consts = network.constants(p, cond.t_k)
        self.c_scale = (cond.p_pa / (R * cond.t_k))[:, None]
        self.km_a = cond.km_a
        self.wc_phi2 = cond.wc_phi2
        self.keys = np.array([pw.key for pw in network.pathways])
        self.f_tot0 = cond.ghsv_per_h / 3600.0 * P_ATM / (R * T_STD)  # mol/s per m3 of catalyst
        self.f0 = cond.feed * self.f_tot0[:, None]
        self.eye = np.eye(self.n_s)

    def rhs(self, f: np.ndarray, idx: np.ndarray) -> np.ndarray:
        consts = self.consts.take(idx)
        total = f.sum(axis=1, keepdims=True)
        conc = np.maximum(f, 0.0) / total * self.c_scale[idx]
        r = consts.key_rates(conc)
        if self.wc_phi2 is not None:
            key = conc[:, self.keys]
            k1 = np.where((r > 0.0) & (key > 0.0), r / np.maximum(key, 1e-300), 0.0)
            phi = np.sqrt(self.wc_phi2[idx] * k1)
            r = r * np.where(phi > 1e-4, np.tanh(phi) / np.maximum(phi, 1e-4), 1.0 - phi**2 / 3.0)
        if self.km_a is not None:
            flux = self.km_a[idx] * conc[:, self.keys]
            capped = r * flux / np.maximum(r + flux, 1e-300)
            r = np.where(r > 0.0, capped, r)
        return consts.species_rates(conc, r)

    def jacobian(self, f: np.ndarray, y0: np.ndarray, idx: np.ndarray) -> np.ndarray:
        n, s = f.shape
        delta = 1e-7 * np.maximum(np.abs(f), 1e-12 * self.f_tot0[idx, None])
        fp = f[:, None, :] + delta[:, :, None] * self.eye  # row i perturbs species i
        dy = self.rhs(fp.reshape(n * s, s), np.repeat(idx, s)).reshape(n, s, s) - y0[:, None, :]
        return (dy / delta[:, :, None]).transpose(0, 2, 1)

    def _ros2(self, f, h, idx, jac=None, y0=None):
        if y0 is None:
            y0 = self.rhs(f, idx)
        if jac is None:
            jac = self.jacobian(f, y0, idx)
        w = self.eye - _GAMMA * h[:, None, None] * jac
        k1 = np.linalg.solve(w, y0[..., None])[..., 0]
        k2 = np.linalg.solve(w, (self.rhs(f + h[:, None] * k1, idx) - 2.0 * k1)[..., None])[..., 0]
        return f + h[:, None] * (1.5 * k1 + 0.5 * k2)

    def step(self, f: np.ndarray, h: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One extrapolated step and its local error estimate."""
        y0 = self.rhs(f, idx)
        jac = self.jacobian(f, y0, idx)
        full = self._ros2(f, h, idx, jac, y0)
        half = self._ros2(f, 0.5 * h, idx, jac, y0)
        two = self._ros2(half, 0.5 * h, idx)
        err = (two - full) / 3.0
        return np.maximum(two + err, 0.0), err


def _finish(f: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 0.0)
    return f / f.sum(axis=1, keepdims=True)


def solve(
    network: Network,
    params: Mapping[str, float | np.ndarray],
    cond: Conditions,
    *,
    grid: np.ndarray | None = None,
    rtol: float = 1e-4,
    atol_ppm: float = 1e-3,
    h0: float = 1e-4,
    max_iter: int = 20000,
) -> Solution:
    """Integrate the bed. Without `grid`, each condition keeps |local error| <= rtol*|value| + atol_ppm."""
    sys = _System(network, params, cond)
    b = cond.size
    f = sys.f0.copy()
    if grid is not None:
        if grid.shape[0] != b:
            raise ValueError("grid must have one row per condition")
        for k in range(grid.shape[1]):
            idx = np.nonzero(grid[:, k] > 0.0)[0]
            if len(idx):
                f[idx] = sys.step(f[idx], grid[idx, k], idx)[0]
        return Solution(_finish(f), grid)

    z = np.zeros(b)
    h = np.full(b, h0)
    accepted: list[list[float]] = [[] for _ in range(b)]
    atol = atol_ppm * 1e-6 * sys.f_tot0
    active = np.arange(b)
    for _ in range(max_iter):
        if not len(active):
            break
        hh = np.minimum(h[active], 1.0 - z[active])
        with np.errstate(over="ignore", invalid="ignore"):
            new, err = sys.step(f[active], hh, active)
            norm = np.max(np.abs(err) / (rtol * np.abs(new) + atol[active, None]), axis=1)
        norm = np.where(np.isfinite(norm), norm, np.inf)
        ok = norm <= 1.0
        for i, step_h in zip(active[ok], hh[ok]):
            accepted[i].append(step_h)
        f[active[ok]] = new[ok]
        z[active[ok]] += hh[ok]
        with np.errstate(divide="ignore"):
            factor = np.clip(0.9 * norm ** (-1.0 / 3.0), 0.2, 4.0)
        h[active] = hh * factor
        active = active[z[active] < 1.0 - 1e-12]
    else:
        raise RuntimeError(f"reactor did not finish {len(active)} conditions in {max_iter} steps")
    width = max(len(a) for a in accepted)
    out_grid = np.zeros((b, width))
    for i, a in enumerate(accepted):
        out_grid[i, : len(a)] = a
    return Solution(_finish(f), out_grid)


def outlet(
    network: Network,
    params: Mapping[str, float | np.ndarray],
    cond: Conditions,
    *,
    grid: np.ndarray | None = None,
    rtol: float = 1e-4,
    atol_ppm: float = 1e-3,
) -> np.ndarray:
    """Outlet wet mole fractions, shape (B, n_species). `params` values are scalars or (B,) arrays."""
    return solve(network, params, cond, grid=grid, rtol=rtol, atol_ppm=atol_ppm).outlet


def outlet_rk4(network: Network, params: Mapping[str, float | np.ndarray], cond: Conditions, steps: int) -> np.ndarray:
    """Classical RK4 on a uniform grid: an independent reference for tests, not for stiff problems."""
    sys = _System(network, params, cond)
    idx = np.arange(cond.size)
    f, h = sys.f0.copy(), 1.0 / steps
    for _ in range(steps):
        k1 = sys.rhs(f, idx)
        k2 = sys.rhs(f + 0.5 * h * k1, idx)
        k3 = sys.rhs(f + 0.5 * h * k2, idx)
        k4 = sys.rhs(f + h * k3, idx)
        f = f + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return _finish(f)
