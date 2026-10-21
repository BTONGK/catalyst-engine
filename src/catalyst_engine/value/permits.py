"""Permit arithmetic: ppmvd @ 15% O2 to lb/MMBtu, tons per year, and megawatts under a cap.

Uses EPA Method 19 for natural gas: F_d = 8,710 dscf/MMBtu (HHV basis, 68 F) and
385.3 scf per lb-mol at 68 F and 1 atm. Heat rates must therefore be HHV.
"""

from __future__ import annotations

from catalyst_engine.units import AIR_O2_PCT

F_D_NATURAL_GAS = 8710.0  # dscf/MMBtu
SCF_PER_LBMOL = 385.3
MOLAR_MASS = {"NOx": 46.01, "CO": 28.01, "NH3": 17.03, "HCHO": 30.03, "VOC_as_CH4": 16.04}


def lb_per_mmbtu(
    ppmvd: float, species: str, *, o2_ref_pct: float = 15.0, f_d: float = F_D_NATURAL_GAS
) -> float:
    """Emission rate per heat input. NOx is reported as NO2, as permits do."""
    lb_per_scf_per_ppm = MOLAR_MASS[species] / SCF_PER_LBMOL * 1e-6
    return ppmvd * lb_per_scf_per_ppm * f_d * AIR_O2_PCT / (AIR_O2_PCT - o2_ref_pct)


def tons_per_year(rate_lb_per_mmbtu: float, mw: float, heat_rate_btu_per_kwh_hhv: float, hours: float) -> float:
    heat_input_mmbtu_per_h = mw * heat_rate_btu_per_kwh_hhv / 1000.0
    return rate_lb_per_mmbtu * heat_input_mmbtu_per_h * hours / 2000.0


def max_mw_under_cap(
    cap_tons_per_year: float, rate_lb_per_mmbtu: float, heat_rate_btu_per_kwh_hhv: float, hours: float
) -> float:
    """Capacity that fits under an annual cap; it scales inversely with the emission rate."""
    return cap_tons_per_year / tons_per_year(rate_lb_per_mmbtu, 1.0, heat_rate_btu_per_kwh_hhv, hours)
