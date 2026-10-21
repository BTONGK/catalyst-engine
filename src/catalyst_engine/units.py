"""Units, gas-concentration bases and standard conditions.

Every concentration that leaves this engine says which basis it is on. Bench
analyzers (FTIR) read wet gas; permits are written in ppm dry at a reference
O2. Mixing the two silently moves a 2.0 ppm limit by 11-55%, so conversions
are explicit functions here and nowhere else.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

R = 8.314462618  # J/(mol K)
K_B_EV = 8.617333262e-5  # eV/K
P_ATM = 101325.0  # Pa
T_STD = 273.15  # K, "normal" conditions used for GHSV (0 C, 1 atm)
AIR_O2_PCT = 20.9  # % O2 used in EPA corrections


def f_to_k(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0 + 273.15


def k_to_f(t_k: float) -> float:
    return (t_k - 273.15) * 9.0 / 5.0 + 32.0


def c_to_k(t_c: float) -> float:
    return t_c + 273.15


def rate_factor_for_energy_error(delta_ev: float, t_k: float) -> float:
    """How far a rate moves when a barrier is off by `delta_ev` (exp(dE/kT))."""
    return math.exp(delta_ev / (K_B_EV * t_k))


def molar_concentration(t_k: float, p_pa: float = P_ATM) -> float:
    """Total gas concentration in mol/m3 for an ideal gas."""
    return p_pa / (R * t_k)


_QTY = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(ppm|ppb|%|frac)?\s*$")


def parse_fraction(value: str | float | int) -> float:
    """Parse '25 ppm', '15 %', '0.1 frac' or a bare fraction (<= 1) to a mole fraction."""
    if isinstance(value, (int, float)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"bare number {value} is ambiguous; write it with ppm, % or frac")
        return float(value)
    m = _QTY.match(value)
    if not m:
        raise ValueError(f"cannot parse concentration {value!r}")
    number, unit = float(m.group(1)), m.group(2)
    scale = {"ppm": 1e-6, "ppb": 1e-9, "%": 1e-2, "frac": 1.0, None: 1.0}[unit]
    if unit is None and not 0.0 <= number <= 1.0:
        raise ValueError(f"bare number {value!r} is ambiguous; write it with ppm, % or frac")
    return number * scale


@dataclass(frozen=True)
class Basis:
    """A concentration basis: wet or dry gas, optionally corrected to a reference O2 %."""

    dry: bool = False
    o2_ref_pct: float | None = None

    def __post_init__(self) -> None:
        if self.o2_ref_pct is not None and not self.dry:
            raise ValueError("O2-corrected concentrations are defined on a dry basis")

    @property
    def label(self) -> str:
        if not self.dry:
            return "ppm wet"
        if self.o2_ref_pct is None:
            return "ppmvd"
        return f"ppmvd @ {self.o2_ref_pct:g}% O2"


WET = Basis()
DRY = Basis(dry=True)
PERMIT_15 = Basis(dry=True, o2_ref_pct=15.0)


def _o2_correction(basis: Basis, h2o_frac, o2_wet_frac):
    """Factor that takes ppm dry to ppm dry at the basis' reference O2 (1 when uncorrected)."""
    if basis.o2_ref_pct is None:
        return 1.0
    o2_dry_pct = 100.0 * np.asarray(o2_wet_frac) / (1.0 - np.asarray(h2o_frac))
    if np.any(o2_dry_pct >= AIR_O2_PCT):
        raise ValueError("dry O2 at or above 20.9% leaves no room for an O2 correction")
    return (AIR_O2_PCT - basis.o2_ref_pct) / (AIR_O2_PCT - o2_dry_pct)


def from_wet(value, dst: Basis, *, h2o_frac, o2_wet_frac):
    """Convert wet concentrations (scalars or arrays) to `dst`, given the gas's wet H2O and O2."""
    h2o = np.asarray(h2o_frac)
    if np.any((h2o < 0.0) | (h2o >= 1.0)):
        raise ValueError("h2o_frac must be in [0, 1)")
    if not dst.dry:
        return value
    return value / (1.0 - h2o) * _o2_correction(dst, h2o, o2_wet_frac)


def convert(value, src: Basis, dst: Basis, *, h2o_frac, o2_wet_frac):
    """Convert a concentration between any two bases, given the gas's wet H2O and wet O2 fractions."""
    wet = value
    if src.dry:
        wet = value / _o2_correction(src, h2o_frac, o2_wet_frac) * (1.0 - np.asarray(h2o_frac))
    return from_wet(wet, dst, h2o_frac=h2o_frac, o2_wet_frac=o2_wet_frac)
