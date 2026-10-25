"""Thermodynamic and transport properties, backed by Cantera's GRI-Mech 3.0 data, with the NASA Glenn
thermodynamic database (McBride, Zehe and Gordon 2002, Cantera's nasa_gas.yaml) for gases beyond it.

Packs name species by their Cantera id, so equilibrium constants, enthalpies and diffusivities come from vetted
sources instead of hand-typed tables. Each set of species takes all its data from one database: GRI-Mech when it
holds every one of them, else NASA Glenn. For the gases the reaction packs use that both databases hold, their
standard Gibbs energies agree within 0.1 kJ/mol from 300 to 1500 K for most, 0.9 kJ/mol for propane and 4.2 kJ/mol
at most (HCN, whose formation enthalpy is 130.8 kJ/mol in GRI-Mech and 135.0 in NASA Glenn); a few radicals outside
every pack differ further (NCO by 46 kJ/mol; experiments/design_method/model_checks.py prints them). Transport
(viscosity, diffusion) comes from GRI-Mech alone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import cantera as ct
import numpy as np

from catalyst_engine.units import P_ATM, R

MECHANISM = "gri30.yaml"
EXTRA = "nasa_gas.yaml"  # thermodynamic data only, both databases at 1 atm


@lru_cache(maxsize=1)
def _gas() -> ct.Solution:
    return ct.Solution(MECHANISM)


@lru_cache(maxsize=1)
def _extra() -> dict[str, ct.Species]:
    return {s.name: s for s in ct.Species.list_from_file(EXTRA)}


def has_species(cantera_id: str) -> bool:
    return cantera_id in _gas().species_names or cantera_id in _extra()


def has_transport(cantera_id: str) -> bool:
    """Whether the gas has viscosity and diffusion data (GRI-Mech's), which the honeycomb's film model needs."""
    return cantera_id in _gas().species_names


def _index(cantera_id: str) -> int:
    if not has_transport(cantera_id):
        raise KeyError(f"{cantera_id!r} is not in {MECHANISM}")
    return _gas().species_index(cantera_id)


def standard_gibbs_rt(cantera_ids: list[str], t_k: float) -> np.ndarray:
    """g°/RT for each species at T and 1 atm, all from GRI-Mech when it holds every one, else all from NASA Glenn."""
    gas = _gas()
    if all(s in gas.species_names for s in cantera_ids):
        gas.TP = t_k, gas.reference_pressure
        g = gas.standard_gibbs_RT
        return np.array([g[_index(s)] for s in cantera_ids])
    extra = _extra()
    missing = [s for s in cantera_ids if s not in extra]
    if missing:
        raise KeyError(f"{missing} sit outside {MECHANISM} and {EXTRA}; give the set from one database")
    return np.array([(extra[s].thermo.h(t_k) - t_k * extra[s].thermo.s(t_k)) / (ct.gas_constant * t_k)
                     for s in cantera_ids])


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
