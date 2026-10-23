from pathlib import Path

import pytest

from catalyst_engine.chem import thermo
from catalyst_engine.units import f_to_k
from catalyst_engine.value.monolith import Monolith
from catalyst_engine.value.permits import lb_per_mmbtu, max_mw_under_cap
from catalyst_engine.value.scenario import evaluate_scenario, load_scenario
from catalyst_engine.value.tempering import tempering_air

EXHAUST = {"N2": 0.745, "O2": 0.125, "CO2": 0.039, "H2O": 0.083, "AR": 0.008}
ROOT = Path(__file__).resolve().parents[1]


def test_tempering_air_matches_the_analysis_range():
    # The analysis cites tempering air at 20-50% of exhaust flow for 1,000-1,247 F exhaust.
    r1150 = tempering_air(EXHAUST, f_to_k(1150), f_to_k(900), f_to_k(80)).mass_ratio
    r1247 = tempering_air(EXHAUST, f_to_k(1247), f_to_k(900), f_to_k(80)).mass_ratio
    assert 0.30 < r1150 < 0.38
    assert 0.42 < r1247 < 0.52
    assert tempering_air(EXHAUST, f_to_k(850), f_to_k(900), f_to_k(80)).mass_ratio == 0.0


def test_tempering_dilutes_ppm_but_not_the_permit_basis():
    r = tempering_air(EXHAUST, f_to_k(1150), f_to_k(900), f_to_k(80))
    assert r.mixed_mole_fractions["O2"] > EXHAUST["O2"]
    assert 0.0 < r.dilution < 1.0


def test_permit_arithmetic():
    # 2 ppmvd @ 15% O2 NOx (as NO2) on natural gas, EPA Method 19.
    assert lb_per_mmbtu(2.0, "NOx") == pytest.approx(0.00737, abs=2e-5)
    mw_full = max_mw_under_cap(100, lb_per_mmbtu(2.0, "NOx"), 9500, 8760)
    mw_half = max_mw_under_cap(100, lb_per_mmbtu(1.0, "NOx"), 9500, 8760)
    assert mw_half == pytest.approx(2 * mw_full)


def test_monolith_geometry_and_pressure_drop():
    m = Monolith(cpsi=64, wall_mm=0.6)
    assert m.pitch_m == pytest.approx(3.175e-3)
    assert m.hydraulic_diameter_m == pytest.approx(2.575e-3)
    assert 0.6 < m.open_frontal_area < 0.7
    gas = {"N2": 0.75, "O2": 0.14, "H2O": 0.08, "CO2": 0.03}
    dp1 = m.pressure_drop_pa(0.2, 6.0, 800.0, 101325.0, gas)
    assert m.pressure_drop_pa(0.4, 6.0, 800.0, 101325.0, gas) == pytest.approx(2 * dp1)


def test_washcoat_thiele_modulus_follows_the_coat():
    # phi^2 per unit rate constant = thickness / (effective diffusivity x surface per volume): doubling the coat doubles it
    gas = {"N2": 0.75, "O2": 0.14, "H2O": 0.08, "CO2": 0.03}
    thin = Monolith(cpsi=400, wall_mm=0.17, washcoat_um=50)
    thick = Monolith(cpsi=400, wall_mm=0.17, washcoat_um=100)
    phi2 = thin.washcoat_phi2("NO", 700.0, 101325.0, gas)
    d_eff = 0.01 * thermo.diffusivity("NO", 700.0, 101325.0, gas)
    assert phi2 == pytest.approx(50e-6 / (d_eff * thin.surface_per_volume))
    assert thick.washcoat_phi2("NO", 700.0, 101325.0, gas) == pytest.approx(2 * phi2)
    with pytest.raises(ValueError, match="thickness"):
        Monolith(cpsi=400, wall_mm=0.17).washcoat_phi2("NO", 700.0, 101325.0, gas)


def test_example_scenario_story():
    results = {r.name: r for r in evaluate_scenario(
        load_scenario(ROOT / "reactions/scr_hot/scenarios/example_frame_turbine.yaml"))}
    tempered = results["tempered to 900 F"]
    assert tempered.feasible and tempered.fan_power_kW > 0
    assert not results["conventional at 1150 F"].feasible
    hot = results["hot target at 1150 F"]
    assert hot.feasible and hot.fan_power_kW == 0
    assert hot.catalyst_volume_m3 < tempered.catalyst_volume_m3
    for r in (tempered, hot):
        assert r.metrics["nox_conversion"] >= 0.92 - 1e-6
        assert r.metrics["nh3_slip"] <= 2.0 + 1e-6
