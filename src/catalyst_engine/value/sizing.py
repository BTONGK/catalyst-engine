"""Catalyst sizing for any pack: the highest space velocity (least catalyst) that meets metric targets."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.packs import Pack


class Bound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _one(self) -> "Bound":
        if (self.min is None) == (self.max is None):
            raise ValueError("give exactly one of min or max")
        return self

    def ok(self, values: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return values >= self.min if self.min is not None else values <= self.max


def _feasible(pack, params, feed, t_k, p_pa, targets, km_a, ghsv: np.ndarray) -> tuple[np.ndarray, dict]:
    b = len(ghsv)
    cond = Conditions(
        np.full(b, t_k), np.full(b, p_pa), ghsv, np.tile(feed, (b, 1)),
        km_a=None if km_a is None else np.tile(km_a, (b, 1)),
    )
    out = outlet(pack.network, pack.network.parameters(params), cond)
    metrics = pack.metrics(cond.feed, out)
    ok = np.ones(b, dtype=bool)
    for name, bound in targets.items():
        ok &= bound.ok(metrics[name])
    return ok, metrics


def max_space_velocity(
    pack: Pack,
    params: Mapping[str, float],
    feed: np.ndarray,
    t_k: float,
    p_pa: float,
    targets: Mapping[str, Bound],
    *,
    km_a: np.ndarray | None = None,
    lo: float = 2e3,
    hi: float = 2e6,
) -> tuple[float | None, dict[str, float]]:
    """Largest GHSV (1/h, normal) meeting every target, and the metrics there; None if no amount works."""
    unknown = set(targets) - set(pack.spec.metrics)
    if unknown:
        raise KeyError(f"targets name unknown metrics {sorted(unknown)}")
    grid = np.geomspace(lo, hi, 41)
    ok, _ = _feasible(pack, params, feed, t_k, p_pa, targets, km_a, grid)
    if not ok.any():
        return None, {}
    i = int(np.nonzero(ok)[0].max())
    a, b = grid[i], grid[min(i + 1, len(grid) - 1)]
    for _ in range(40):
        if b / a < 1.0005:
            break
        mid = np.sqrt(a * b)
        if _feasible(pack, params, feed, t_k, p_pa, targets, km_a, np.array([mid]))[0][0]:
            a = mid
        else:
            b = mid
    _, metrics = _feasible(pack, params, feed, t_k, p_pa, targets, km_a, np.array([a]))
    return float(a), {k: float(v[0]) for k, v in metrics.items()}
