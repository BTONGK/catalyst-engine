import pytest

from catalyst_engine.chem import thermo
from catalyst_engine.chem.formula import element_counts, molar_mass
from catalyst_engine.chem.network import NetworkSpec, parse_equation
from catalyst_engine.packs import read_yaml


def _spec(equation: str, key: str = "NH3") -> dict:
    return {
        "reference_temperature_K": 800.0,
        "species": {s: {"formula": s} for s in ["NH3", "NO", "O2", "N2", "H2O"]},
        "pathways": {"p": {"equation": equation, "role": "desired",
                           "rate": {"key": key, "orders": {key: 1}, "ln_k_ref": 1.0, "Ea_kJ": 50}}},
    }


def test_parse_equation():
    stoich, rev = parse_equation("4 NH3 + 4 NO + O2 -> 4 N2 + 6 H2O")
    assert stoich == {"NH3": -4, "NO": -4, "O2": -1, "N2": 4, "H2O": 6}
    assert not rev
    assert parse_equation("NO + 0.5 O2 <=> NO2")[1]


def test_formulas():
    assert element_counts("CH2O") == {"C": 1, "H": 2, "O": 1}
    assert element_counts("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert molar_mass("NH3") == pytest.approx(17.031, abs=1e-3)


def test_unbalanced_pathway_is_rejected():
    with pytest.raises(ValueError, match="not balanced"):
        NetworkSpec.model_validate(_spec("4 NH3 + 4 NO + O2 -> 4 N2 + 5 H2O"))


def test_key_must_be_a_reactant():
    with pytest.raises(ValueError, match="must be a reactant"):
        NetworkSpec.model_validate(_spec("4 NH3 + 4 NO + O2 -> 4 N2 + 6 H2O", key="N2"))


def test_yaml_reads_nitric_oxide_as_a_name(tmp_path):
    # YAML 1.1 turns a bare NO into False; packs must keep it as nitric oxide.
    f = tmp_path / "x.yaml"
    f.write_text("species:\n  NO: {formula: NO}\nflag: true\n")
    data = read_yaml(f)
    assert list(data["species"]) == ["NO"]
    assert data["flag"] is True


def test_thermo_against_standard_gibbs_energies():
    # NIST: dfG(NO) = 86.60, dfG(NO2) = 51.30, dfG(NH3) = -16.4 kJ/mol at 298.15 K.
    assert thermo.delta_g_kj({"NO": -1, "O2": -0.5, "NO2": 1}, 298.15) == pytest.approx(-35.3, abs=1.5)
    assert thermo.delta_g_kj({"NH3": -1, "N2": 0.5, "H2": 1.5}, 298.15) == pytest.approx(16.4, abs=1.0)
