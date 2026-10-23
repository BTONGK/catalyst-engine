"""Reaction packs: everything reaction-specific, declared as data.

A pack is a folder under `reactions/`:

    reactions/<id>/pack.yaml          species, pathways, rate-law forms, metrics, screening descriptor
    reactions/<id>/protocols/*.yaml   test protocols (feeds, temperatures, space velocities, analyzer)

Reactions built in the app go to `<data dir>/reactions/<id>/` in the same layout. The engine core never imports a
pack. Adding a reaction means adding a folder.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalyst_engine.chem.metrics import MetricSpec, evaluate
from catalyst_engine.chem.network import AdsorptionSpec, Network, NetworkSpec, PathwaySpec, SpeciesSpec
from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.design.spec import ScreeningSpec
from catalyst_engine.units import c_to_k, f_to_k, parse_fraction

PACKS_DIR = Path(os.environ.get("CE_PACKS_DIR", Path(__file__).resolve().parents[2] / "reactions"))


def user_packs_dir() -> Path:
    """Reactions built in the app: <data dir>/reactions, outside git like the lab data (CE_DATA_DIR moves it)."""
    return Path(os.environ.get("CE_DATA_DIR", Path(__file__).resolve().parents[2] / "data")) / "reactions"

__all__ = ["AdsorptionSpec", "Pack", "PackSpec", "ProtocolSpec", "StepSpec", "list_packs", "load_pack", "pack_path"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Temperatures(_Strict):
    temperatures_F: list[float] | None = None
    temperatures_C: list[float] | None = None
    temperatures_K: list[float] | None = None

    @model_validator(mode="after")
    def _one_scale(self) -> "_Temperatures":
        given = [t for t in (self.temperatures_F, self.temperatures_C, self.temperatures_K) if t is not None]
        if len(given) != 1 or not given[0]:
            raise ValueError("give exactly one of temperatures_F, temperatures_C, temperatures_K")
        return self

    def kelvin(self) -> list[float]:
        if self.temperatures_F is not None:
            return [f_to_k(t) for t in self.temperatures_F]
        if self.temperatures_C is not None:
            return [c_to_k(t) for t in self.temperatures_C]
        return list(self.temperatures_K or [])


class PackSpec(_Strict):
    id: str
    name: str
    version: int
    status: str
    description: str = ""
    summary: str = ""  # one plain-language sentence for the app's reaction cards
    category: str = "Other"
    reference_temperature_F: float | None = None
    reference_temperature_K: float | None = None
    water: str | None = None
    oxygen: str | None = None
    species: dict[str, SpeciesSpec]
    pathways: dict[str, PathwaySpec]
    metrics: dict[str, MetricSpec] = Field(default_factory=dict)
    screening: ScreeningSpec | None = None

    @model_validator(mode="after")
    def _check(self) -> "PackSpec":
        if (self.reference_temperature_F is None) == (self.reference_temperature_K is None):
            raise ValueError("give exactly one of reference_temperature_F, reference_temperature_K")
        for mid, m in self.metrics.items():
            missing = m.referenced() - set(self.species)
            if missing:
                raise ValueError(f"metric {mid}: undeclared species {sorted(missing)}")
        if self.screening is not None:
            unknown = set(self.screening.pathways) - set(self.pathways)
            if unknown:
                raise ValueError(f"screening names unknown pathways {sorted(unknown)}")
            if self.screening.metric not in self.metrics:
                raise ValueError(f"screening metric {self.screening.metric!r} is not a pack metric")
            unknown = set(self.screening.peak_gas) - set(self.species)
            if unknown:
                raise ValueError(f"screening peak_gas names undeclared species {sorted(unknown)}")
        return self

    def network_spec(self) -> NetworkSpec:
        t_ref = self.reference_temperature_K or f_to_k(self.reference_temperature_F or 0.0)
        return NetworkSpec(
            species=self.species,
            pathways=self.pathways,
            reference_temperature_K=t_ref,
            water=self.water,
            oxygen=self.oxygen,
        )


class AnalyzerSpec(_Strict):
    """Measurement error model: sigma = sqrt(abs^2 + (rel * value)^2); readings below LOD are censored."""

    name: str = "FTIR"
    measured: list[str]
    sigma_abs_ppm: float = 0.5
    sigma_rel: float = 0.02
    lod_ppm: dict[str, float] = Field(default_factory=dict)


class StepSpec(_Temperatures):
    id: str
    purpose: str = ""
    feed: dict[str, str | float]
    # wet: every fraction is a share of the whole gas, as the mass-flow controllers make it.
    # dry: every fraction except water is a share of the dry gas (the permit convention); water is a share of the whole.
    feed_basis: Literal["wet", "dry"] = "wet"
    ghsv_per_h: float
    pressure_kPa: float = 101.325

    def feed_fractions(self, species: list[str], water: str | None = None) -> np.ndarray:
        """Wet mole fractions in `species` order; `water` names the pack's water species (needed for a dry feed)."""
        dry = self.feed_basis == "dry"
        if dry and (water is None or self.feed.get(water, "balance") == "balance"):
            raise ValueError(f"step {self.id}: a dry-basis feed needs the water species and its share of the whole gas")
        y = np.zeros(len(species))
        balance = [s for s, v in self.feed.items() if v == "balance"]
        if len(balance) > 1:
            raise ValueError(f"step {self.id}: only one balance species allowed")
        for s, v in self.feed.items():
            if s not in species:
                raise ValueError(f"step {self.id}: feed species {s!r} is not in the pack")
            if v != "balance" and not (dry and s == water):
                y[species.index(s)] = parse_fraction(v)
        rest = 1.0 - y.sum()
        if rest < -1e-12:
            raise ValueError(f"step {self.id}: feed fractions sum above 1")
        if balance:
            y[species.index(balance[0])] = rest
        elif rest > 1e-9:
            raise ValueError(f"step {self.id}: feed sums to {1 - rest:.6f}; name a balance species")
        if dry:
            w = parse_fraction(self.feed[water])
            y *= 1.0 - w
            y[species.index(water)] = w
        return y


class ProtocolSpec(_Strict):
    id: str
    pack: str
    version: int
    status: str
    description: str = ""
    open_questions: list[str] = Field(default_factory=list)
    analyzer: AnalyzerSpec
    steps: list[StepSpec]

    @model_validator(mode="after")
    def _unique_steps(self) -> "ProtocolSpec":
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        return self

    def step(self, step_id: str) -> StepSpec:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"protocol {self.id} has no step {step_id!r}")


@dataclass
class ConditionRow:
    step: str
    t_k: float


@dataclass
class Pack:
    spec: PackSpec
    protocols: dict[str, ProtocolSpec]
    path: Path

    @cached_property
    def network(self) -> Network:
        return Network.compile(self.spec.network_spec())

    @property
    def id(self) -> str:
        return self.spec.id

    def protocol(self, protocol_id: str | None = None) -> ProtocolSpec:
        if protocol_id is None:
            if len(self.protocols) != 1:
                raise ValueError(f"pack {self.id} has {len(self.protocols)} protocols; name one")
            return next(iter(self.protocols.values()))
        if protocol_id not in self.protocols:
            raise KeyError(f"pack {self.id} has no protocol {protocol_id!r}")
        return self.protocols[protocol_id]

    def conditions(self, protocol: ProtocolSpec) -> tuple[Conditions, list[ConditionRow]]:
        rows, feeds, temps, pressures, ghsv = [], [], [], [], []
        for step in protocol.steps:
            y = step.feed_fractions(self.network.species, self.spec.water)
            for t in step.kelvin():
                rows.append(ConditionRow(step.id, t))
                feeds.append(y)
                temps.append(t)
                pressures.append(step.pressure_kPa * 1000.0)
                ghsv.append(step.ghsv_per_h)
        return Conditions(np.array(temps), np.array(pressures), np.array(ghsv), np.array(feeds)), rows

    def metrics(self, feed: np.ndarray, out: np.ndarray) -> dict[str, np.ndarray]:
        return {mid: evaluate(m, self.network, feed, out) for mid, m in self.spec.metrics.items()}

    def simulate(self, params: dict[str, float] | None = None, protocol_id: str | None = None) -> list[dict]:
        """Run every step of a protocol through the reactor; one row per (step, temperature)."""
        protocol = self.protocol(protocol_id)
        cond, rows = self.conditions(protocol)
        out = outlet(self.network, self.network.parameters(params), cond)
        metrics = self.metrics(cond.feed, out)
        table = []
        for i, row in enumerate(rows):
            entry = {"step": row.step, "T_K": row.t_k}
            entry.update({f"out_{s}_ppm": out[i, j] * 1e6 for j, s in enumerate(self.network.species)})
            entry.update({m: float(v[i]) for m, v in metrics.items()})
            table.append(entry)
        return table


class _ChemLoader(yaml.SafeLoader):
    """YAML 1.1 reads NO, ON, OFF, YES as booleans; here NO is nitric oxide. Only true/false are booleans."""


_ChemLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ChemLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def read_yaml(path: Path) -> dict:
    with path.open() as fh:
        data = yaml.load(fh, Loader=_ChemLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return data


def pack_path(pack_id: str) -> Path:
    """A pack's folder: the shipped library first, then the reactions built in the app."""
    for root in (PACKS_DIR, user_packs_dir()):
        if (root / pack_id / "pack.yaml").exists():
            return root / pack_id
    return PACKS_DIR / pack_id


def load_pack(pack: str | Path) -> Pack:
    path = Path(pack)
    if not path.exists():
        path = pack_path(str(pack))
    spec = PackSpec.model_validate(read_yaml(path / "pack.yaml"))
    if spec.id != path.name:
        raise ValueError(f"pack id {spec.id!r} must match its folder name {path.name!r}")
    protocols = {}
    for p in sorted((path / "protocols").glob("*.yaml")):
        proto = ProtocolSpec.model_validate(read_yaml(p))
        if proto.pack != spec.id:
            raise ValueError(f"{p}: protocol names pack {proto.pack!r}, expected {spec.id!r}")
        missing = set(proto.analyzer.measured) - set(spec.species)
        if missing:
            raise ValueError(f"{p}: analyzer measures undeclared species {sorted(missing)}")
        protocols[proto.id] = proto
    loaded = Pack(spec, protocols, path)
    _ = loaded.network  # compile now so a broken pack fails at load
    for proto in protocols.values():
        loaded.conditions(proto)
    return loaded


def list_packs(root: Path | None = None) -> list[str]:
    """Pack ids in `root`, or in the shipped library plus the reactions built in the app."""
    roots = [root] if root else [PACKS_DIR, user_packs_dir()]
    return sorted({p.name for r in roots if r.is_dir() for p in r.iterdir() if (p / "pack.yaml").exists()})
