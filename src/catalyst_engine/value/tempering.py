"""Tempering air: how much ambient air it takes to cool exhaust to what a catalyst tolerates.

Solved as an adiabatic enthalpy balance with Cantera properties, so the answer
reflects real heat capacities and the exhaust's water and CO2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from scipy.optimize import brentq

from catalyst_engine.chem import thermo
from catalyst_engine.units import P_ATM

DRY_AIR = {"N2": 0.7808, "O2": 0.2095, "AR": 0.0093, "CO2": 0.0004}


@dataclass
class TemperingResult:
    mass_ratio: float  # kg air per kg exhaust
    mole_ratio: float  # mol air per mol exhaust
    dilution: float  # exhaust share of the mixed molar flow
    mixed_mole_fractions: dict[str, float]


def tempering_air(
    exhaust: Mapping[str, float],
    t_exhaust_k: float,
    t_target_k: float,
    t_ambient_k: float,
    *,
    air: Mapping[str, float] = DRY_AIR,
    p_pa: float = P_ATM,
) -> TemperingResult:
    """`exhaust` and `air` are mole fractions keyed by Cantera species id."""
    if t_exhaust_k <= t_target_k:
        return TemperingResult(0.0, 0.0, 1.0, dict(exhaust))
    if t_ambient_k >= t_target_k:
        raise ValueError("ambient air must be cooler than the target temperature")
    y_e = thermo.mole_to_mass_fractions(exhaust)
    y_a = thermo.mole_to_mass_fractions(air)
    h_e = thermo.enthalpy_mass(t_exhaust_k, p_pa, y_e)
    h_a = thermo.enthalpy_mass(t_ambient_k, p_pa, y_a)

    def mixed(r: float) -> dict[str, float]:
        keys = set(y_e) | set(y_a)
        return {k: (y_e.get(k, 0.0) + r * y_a.get(k, 0.0)) / (1.0 + r) for k in keys}

    def imbalance(r: float) -> float:
        return h_e + r * h_a - (1.0 + r) * thermo.enthalpy_mass(t_target_k, p_pa, mixed(r))

    r = float(brentq(imbalance, 0.0, 50.0))
    mole_ratio = r * thermo.mean_molar_mass(exhaust) / thermo.mean_molar_mass(air)
    keys = set(exhaust) | set(air)
    mix = {k: (exhaust.get(k, 0.0) + mole_ratio * air.get(k, 0.0)) / (1.0 + mole_ratio) for k in keys}
    return TemperingResult(r, mole_ratio, 1.0 / (1.0 + mole_ratio), mix)
