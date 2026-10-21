"""Thermodynamic and transport properties, backed by Cantera's GRI-Mech 3.0 data.

Packs name species by their Cantera id, so equilibrium constants, enthalpies
and diffusivities come from one vetted source instead of hand-typed tables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import cantera as ct
import numpy as np

from catalyst_engine.units import P_ATM, R

MECHANISM = "gri30.yaml"


@lru_cache(maxsize=1)
def _gas() -> ct.Solution:
    return ct.Solution(MECHANISM)


def has_species(cantera_id: str) -> bool:
    return cantera_id in _gas().species_names


def _index(cantera_id: str) -> int:
    if not has_species(cantera_id):
        raise KeyError(f"{cantera_id!r} is not in {MECHANISM}")
    return _gas().species_index(cantera_id)


def standard_gibbs_rt(cantera_ids: list[str], t_k: float) -> np.ndarray:
    """g°/RT for each species at T and the mechanism's reference pressure (1 atm)."""
    gas = _gas()
    gas.TP = t_k, gas.reference_pressure
    g = gas.standard_gibbs_RT
    return np.array([g[_index(s)] for s in cantera_ids])


def delta_g_kj(stoich: Mapping[str, float], t_k: float) -> float:
    """Standard Gibbs energy of reaction (kJ/mol) for {cantera_id: nu}, products positive."""
    ids = list(stoich)
    g_rt = standard_gibbs_rt(ids, t_k)
    return float(sum(stoich[s] * g for s, g in zip(ids, g_rt))) * R * t_k / 1000.0


def equilibrium_constant_c(stoich: Mapping[str, float], t_k: float) -> float:
    """K_c in (mol/m3)^dn for {cantera_id: nu}, consistent with concentrations in mol/m3."""
    ids = list(stoich)
    g_rt = standard_gibbs_rt(ids, t_k)
    ln_kp = -float(sum(stoich[s] * g for s, g in zip(ids, g_rt)))
    dn = float(sum(stoich.values()))
    p_ref = _gas().reference_pressure
    return float(np.exp(ln_kp) * (p_ref / (R * t_k)) ** dn)


def _set_mass_state(t_k: float, p_pa: float, mass_fractions: Mapping[str, float]) -> ct.Solution:
    gas = _gas()
    gas.TPY = t_k, p_pa, dict(mass_fractions)
    return gas


def mole_to_mass_fractions(mole_fractions: Mapping[str, float]) -> dict[str, float]:
    gas = _gas()
    gas.TPX = 300.0, P_ATM, dict(mole_fractions)
    return {s: float(y) for s, y in zip(gas.species_names, gas.Y) if y > 0}


def enthalpy_mass(t_k: float, p_pa: float, mass_fractions: Mapping[str, float]) -> float:
    """Specific enthalpy, J/kg."""
    return float(_set_mass_state(t_k, p_pa, mass_fractions).enthalpy_mass)


def density(t_k: float, p_pa: float, mole_fractions: Mapping[str, float]) -> float:
    gas = _gas()
    gas.TPX = t_k, p_pa, dict(mole_fractions)
    return float(gas.density)


def viscosity(t_k: float, p_pa: float, mole_fractions: Mapping[str, float]) -> float:
    """Dynamic viscosity, Pa s."""
    gas = _gas()
    gas.TPX = t_k, p_pa, dict(mole_fractions)
    return float(gas.viscosity)


def diffusivity(species: str, t_k: float, p_pa: float, mole_fractions: Mapping[str, float]) -> float:
    """Mixture-averaged diffusion coefficient of `species` (Cantera id), m2/s."""
    gas = _gas()
    gas.TPX = t_k, p_pa, dict(mole_fractions)
    return float(gas.mix_diff_coeffs[_index(species)])


def mean_molar_mass(mole_fractions: Mapping[str, float]) -> float:
    """kg/mol."""
    gas = _gas()
    gas.TPX = 300.0, P_ATM, dict(mole_fractions)
    return float(gas.mean_molecular_weight) / 1000.0
