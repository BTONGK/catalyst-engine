"""System-value scenarios: compare emission-control designs for one exhaust stream.

A scenario names a pack, an exhaust, metric targets and two or more designs
(a catalyst plus, optionally, tempering air). For each design it sizes the
catalyst bed and reports what the integrator pays for: catalyst volume, bed
depth, pressure drop, tempering-air flow and fan power, and their annual cost.

Rate parameters stand for the formed catalyst (honeycomb), not bench powder;
converting between the two is a measured factor that belongs in the scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from catalyst_engine.chem import thermo
from catalyst_engine.chem.formula import molar_mass
from catalyst_engine.packs import Pack, load_pack, read_yaml
from catalyst_engine.units import P_ATM, R, T_STD, f_to_k, parse_fraction
from catalyst_engine.value.monolith import Monolith
from catalyst_engine.value.sizing import Bound, max_space_velocity
from catalyst_engine.value.tempering import tempering_air


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExhaustSpec(_Strict):
    flow_kg_s: float
    temperature_F: float
    pressure_kPa: float = 101.325
    composition: dict[str, str | float]  # Cantera ids, wet mole fractions of the bulk gas


class ReagentSpec(_Strict):
    """Reagent dosed in proportion to a species group. A [min, max] ratio lets sizing pick the dose,
    as integrators tune ammonia-to-NOx ratio against slip."""

    species: str
    ratio: float | tuple[float, float]
    to: list[str]

    def ratios(self, n: int = 9) -> list[float]:
        if isinstance(self.ratio, tuple):
            return [float(r) for r in np.linspace(self.ratio[0], self.ratio[1], n)]
        return [float(self.ratio)]


class MonolithSpec(_Strict):
    cpsi: float
    wall_mm: float
    face_velocity_m_s: float


class FanSpec(_Strict):
    pressure_rise_kPa: float
    efficiency: float


class EconomicsSpec(_Strict):
    electricity_usd_per_mwh: float
    hours_per_year: float
    reagent_usd_per_tonne: float = 0.0


class CatalystSpec(_Strict):
    description: str = ""
    params: dict[str, float] = Field(default_factory=dict)


class DesignSpec(_Strict):
    name: str
    catalyst: str
    tempered_to_F: float | None = None  # omit to run the catalyst at full exhaust temperature


class ScenarioSpec(_Strict):
    name: str
    pack: str
    note: str = ""
    exhaust: ExhaustSpec
    pollutants: dict[str, str | float]  # pack species, wet, at the turbine exit
    reagent: ReagentSpec | None = None
    balance: str
    ambient_F: float
    targets: dict[str, Bound]
    monolith: MonolithSpec
    fan: FanSpec
    economics: EconomicsSpec
    catalysts: dict[str, CatalystSpec]
    designs: list[DesignSpec]


@dataclass
class DesignResult:
    name: str
    catalyst: str
    catalyst_T_F: float
    feasible: bool
    tempering_mass_ratio: float = 0.0
    reagent_ratio: float | None = None
    ghsv_per_h: float | None = None
    catalyst_volume_m3: float | None = None
    face_area_m2: float | None = None
    bed_depth_m: float | None = None
    pressure_drop_Pa: float | None = None
    fan_power_kW: float = 0.0
    fan_energy_usd_per_year: float = 0.0
    reagent_t_per_year: float = 0.0
    reagent_usd_per_year: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


def _pack_feed(pack: Pack, spec: ScenarioSpec, bulk: dict[str, float], dilution: float, ratio: float) -> np.ndarray:
    species = pack.network.species
    y = np.zeros(len(species))
    for s, sp in pack.spec.species.items():
        if sp.thermo_id in bulk and s != spec.balance:
            y[species.index(s)] = bulk[sp.thermo_id]
    for s, v in spec.pollutants.items():
        y[species.index(s)] = parse_fraction(v) * dilution
    if spec.reagent:
        group = sum(y[species.index(s)] for s in spec.reagent.to)
        y[species.index(spec.reagent.species)] = ratio * group
    y[species.index(spec.balance)] = 0.0
    y[species.index(spec.balance)] = 1.0 - y.sum()
    return y


def evaluate_scenario(spec: ScenarioSpec) -> list[DesignResult]:
    pack = load_pack(spec.pack)
    exhaust = {k: parse_fraction(v) for k, v in spec.exhaust.composition.items()}
    total = sum(exhaust.values())
    exhaust = {k: v / total for k, v in exhaust.items()}
    p_pa = spec.exhaust.pressure_kPa * 1000.0
    t_exh, t_amb = f_to_k(spec.exhaust.temperature_F), f_to_k(spec.ambient_F)
    mono = Monolith(spec.monolith.cpsi, spec.monolith.wall_mm)
    results = []
    for d in spec.designs:
        if d.catalyst not in spec.catalysts:
            raise KeyError(f"design {d.name}: unknown catalyst {d.catalyst!r}")
        t_cat = f_to_k(d.tempered_to_F) if d.tempered_to_F is not None else t_exh
        temper = tempering_air(exhaust, t_exh, t_cat, t_amb, p_pa=p_pa)
        bulk = temper.mixed_mole_fractions
        key_ids = [pack.spec.species[pack.network.species[pw.key]].thermo_id for pw in pack.network.pathways]
        km_a = np.array([mono.km_a(s, t_cat, p_pa, bulk) for s in key_ids])
        ghsv, metrics, ratio, chosen_feed = None, {}, None, None
        for r in spec.reagent.ratios() if spec.reagent else [0.0]:
            feed = _pack_feed(pack, spec, bulk, temper.dilution, r)
            g, m = max_space_velocity(pack, spec.catalysts[d.catalyst].params, feed, t_cat, p_pa, spec.targets, km_a=km_a)
            if g is not None and (ghsv is None or g > ghsv):
                ghsv, metrics, ratio, chosen_feed = g, m, (r if spec.reagent else None), feed
        m_air = temper.mass_ratio * spec.exhaust.flow_kg_s
        fan_w = 0.0
        if m_air > 0:
            q_air = m_air / thermo.density(t_amb, P_ATM, {"N2": 0.7808, "O2": 0.2095, "AR": 0.0093, "CO2": 0.0004})
            fan_w = q_air * spec.fan.pressure_rise_kPa * 1000.0 / spec.fan.efficiency
        res = DesignResult(
            d.name, d.catalyst, d.tempered_to_F if d.tempered_to_F is not None else spec.exhaust.temperature_F,
            ghsv is not None, temper.mass_ratio, ratio,
            fan_power_kW=fan_w / 1000.0,
            fan_energy_usd_per_year=fan_w / 1e6 * spec.economics.hours_per_year * spec.economics.electricity_usd_per_mwh,
            metrics=metrics,
        )
        if ghsv is not None:
            m_total = spec.exhaust.flow_kg_s + m_air
            n_dot = m_total / thermo.mean_molar_mass(bulk)
            q_normal_h = n_dot * R * T_STD / P_ATM * 3600.0
            q_actual = m_total / thermo.density(t_cat, p_pa, bulk)
            res.ghsv_per_h = ghsv
            res.catalyst_volume_m3 = q_normal_h / ghsv
            res.face_area_m2 = q_actual / spec.monolith.face_velocity_m_s
            res.bed_depth_m = res.catalyst_volume_m3 / res.face_area_m2
            res.pressure_drop_Pa = mono.pressure_drop_pa(res.bed_depth_m, spec.monolith.face_velocity_m_s, t_cat, p_pa, bulk)
            if spec.reagent and chosen_feed is not None:
                reagent = spec.reagent.species
                mol_s = n_dot * chosen_feed[pack.network.index(reagent)]
                kg_s = mol_s * molar_mass(pack.spec.species[reagent].formula) / 1000.0
                res.reagent_t_per_year = kg_s * 3600.0 * spec.economics.hours_per_year / 1000.0
                res.reagent_usd_per_year = res.reagent_t_per_year * spec.economics.reagent_usd_per_tonne
        results.append(res)
    return results


def load_scenario(path: str | Path) -> ScenarioSpec:
    return ScenarioSpec.model_validate(read_yaml(Path(path)))
