"""Metrics computed from feed and outlet compositions, as declared by a pack."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from catalyst_engine.chem.network import Network
from catalyst_engine.units import Basis, from_wet


class BasisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dry: bool = False
    o2_ref_pct: float | None = None

    def to_basis(self) -> Basis:
        return Basis(dry=self.dry, o2_ref_pct=self.o2_ref_pct)


class TargetSpec(BaseModel):
    """A pack-level target for display and screening, e.g. a permit limit. Campaign gates live elsewhere."""

    model_config = ConfigDict(extra="forbid")
    min: float | None = None
    max: float | None = None
    label: str = ""

    @model_validator(mode="after")
    def _one(self) -> "TargetSpec":
        if (self.min is None) == (self.max is None):
            raise ValueError("a target takes exactly one of min or max")
        return self

    @property
    def value(self) -> float:
        return self.min if self.min is not None else self.max  # type: ignore[return-value]


class MetricSpec(BaseModel):
    """conversion: 1 - out/in of a species group. outlet: ppm of a group on a basis.
    consumption_ratio: moles of `numerator` consumed per mole of `denominator` consumed."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["conversion", "outlet", "consumption_ratio"]
    target: TargetSpec | None = None
    species: list[str] = []
    numerator: list[str] = []
    denominator: list[str] = []
    basis: BasisSpec = BasisSpec()
    unit: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "MetricSpec":
        if self.kind == "consumption_ratio":
            if not (self.numerator and self.denominator):
                raise ValueError("consumption_ratio needs numerator and denominator")
        elif not self.species:
            raise ValueError(f"{self.kind} needs species")
        if self.kind != "outlet" and (self.basis.dry or self.basis.o2_ref_pct is not None):
            raise ValueError("only outlet metrics take a basis")
        return self

    def label_unit(self) -> str:
        if self.unit:
            return self.unit
        if self.kind == "outlet":
            return self.basis.to_basis().label
        return "fraction" if self.kind == "conversion" else "mol/mol"

    def referenced(self) -> set[str]:
        return set(self.species) | set(self.numerator) | set(self.denominator)


def evaluate(spec: MetricSpec, network: Network, feed: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Evaluate one metric for arrays of feed and outlet wet mole fractions, shape (B, n_species)."""

    def total(y: np.ndarray, group: list[str]) -> np.ndarray:
        return y[:, [network.index(s) for s in group]].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        if spec.kind == "conversion":
            y_in = total(feed, spec.species)
            return np.where(y_in > 0, 1.0 - total(out, spec.species) / y_in, np.nan)
        if spec.kind == "consumption_ratio":
            num = total(feed, spec.numerator) - total(out, spec.numerator)
            den = total(feed, spec.denominator) - total(out, spec.denominator)
            return np.where(den > 0, num / den, np.nan)  # undefined unless the denominator is consumed
    ppm_wet = total(out, spec.species) * 1e6
    basis = spec.basis.to_basis()
    if not basis.dry:
        return ppm_wet
    water, oxygen = network.spec.water, network.spec.oxygen
    if water is None or (basis.o2_ref_pct is not None and oxygen is None):
        raise ValueError("dry or O2-corrected metrics need the pack to declare `water` and `oxygen`")
    h2o = out[:, network.index(water)]
    o2 = out[:, network.index(oxygen)] if oxygen else np.zeros(len(out))
    return from_wet(ppm_wet, basis, h2o_frac=h2o, o2_wet_frac=o2)
