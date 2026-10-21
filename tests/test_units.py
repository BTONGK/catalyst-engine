import pytest

from catalyst_engine.units import (
    DRY,
    PERMIT_15,
    WET,
    Basis,
    convert,
    f_to_k,
    k_to_f,
    parse_fraction,
    rate_factor_for_energy_error,
)


def test_temperatures():
    assert f_to_k(1050) == pytest.approx(838.706, abs=1e-3)
    assert k_to_f(f_to_k(1150)) == pytest.approx(1150)


def test_energy_error_matches_analysis():
    # The analysis: a 0.1 eV error moves a rate ~50x at room temperature, ~10x at 500 K, ~4x at 800 K.
    assert rate_factor_for_energy_error(0.1, 298.15) == pytest.approx(49, rel=0.02)
    assert rate_factor_for_energy_error(0.1, 500) == pytest.approx(10.2, rel=0.02)
    assert rate_factor_for_energy_error(0.1, 800) == pytest.approx(4.26, rel=0.01)


def test_permit_basis_depends_on_how_o2_is_specified():
    # 2.0 ppm read wet in gas with 10% H2O and 15% O2 (wet basis) is 3.1 ppmvd @ 15% O2...
    assert convert(2.0, WET, PERMIT_15, h2o_frac=0.10, o2_wet_frac=0.15) == pytest.approx(3.10, abs=0.01)
    # ...but 2.22 if the 15% O2 is on a dry basis (13.5% wet).
    assert convert(2.0, WET, PERMIT_15, h2o_frac=0.10, o2_wet_frac=0.135) == pytest.approx(2.22, abs=0.01)
    assert convert(2.0, WET, DRY, h2o_frac=0.10, o2_wet_frac=0.15) == pytest.approx(2.0 / 0.9)


def test_basis_round_trip():
    v = convert(1.7, PERMIT_15, WET, h2o_frac=0.08, o2_wet_frac=0.12)
    assert convert(v, WET, PERMIT_15, h2o_frac=0.08, o2_wet_frac=0.12) == pytest.approx(1.7)


def test_o2_correction_requires_dry_basis():
    with pytest.raises(ValueError):
        Basis(dry=False, o2_ref_pct=15)


def test_parse_fraction():
    assert parse_fraction("25 ppm") == pytest.approx(25e-6)
    assert parse_fraction("15 %") == pytest.approx(0.15)
    assert parse_fraction(0.5) == 0.5
    with pytest.raises(ValueError):
        parse_fraction(25)  # ambiguous without a unit
