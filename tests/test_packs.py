"""Conformance: every pack loads and runs, and the engine core stays reaction-agnostic."""

from pathlib import Path

import numpy as np
import pytest

from catalyst_engine.packs import PACKS_DIR, list_packs, load_pack

SRC = Path(__file__).resolve().parents[1] / "src" / "catalyst_engine"


def test_there_are_at_least_two_packs():
    assert {"scr_hot", "oxidation_engine"} <= set(list_packs())


@pytest.mark.parametrize("pack_id", list_packs())
def test_pack_runs_every_protocol(pack_id):
    p = load_pack(pack_id)
    assert p.protocols, "a pack needs at least one protocol"
    for proto in p.protocols.values():
        rows = p.simulate(None, proto.id)
        for r in rows:
            assert all(np.isfinite(r[f"out_{s}_ppm"]) for s in p.network.species)


@pytest.mark.parametrize("pack_id", list_packs())
def test_pack_parameters_have_bounds(pack_id):
    p = load_pack(pack_id)
    for name in p.network.defaults:
        lo, hi = p.network.bounds(name)
        assert lo <= p.network.defaults[name] <= hi, name


def test_core_never_names_a_pack():
    # Reaction-specific knowledge lives in reactions/; the engine must not special-case a pack.
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        offenders += [f"{path.name}: {pid}" for pid in list_packs(PACKS_DIR) if pid in text]
    assert not offenders, offenders


def test_a_pack_takes_every_gas_from_one_database():
    """GRI-Mech 3.0 when it holds all of a pack's gases, else NASA Glenn. The coated honeycomb's film model needs
    GRI-Mech's diffusion data for every gas, so a pack with a gas beyond it runs as powder."""
    from catalyst_engine.chem import thermo
    from catalyst_engine.design import screening

    nasa = [pid for pid in list_packs()
            if not all(thermo.has_transport(s.thermo_id) for s in load_pack(pid).spec.species.values())]
    assert nasa, "benzene, ethanol, formic acid and propane sit beyond GRI-Mech"
    for pid in list_packs():
        ids = [s.thermo_id for s in load_pack(pid).spec.species.values()]
        assert np.all(np.isfinite(thermo.standard_gibbs_rt(ids, 600.0)))
    p = load_pack(nasa[0])
    step = next(iter(p.protocols.values())).steps[0]
    with pytest.raises(ValueError, match="powder"):
        screening.wall_transfer(p, step, {"form": "monolith", "cpsi": 400.0, "wall_mm": 0.1})


def test_the_two_databases_agree_on_the_gases_the_packs_share():
    """For each gas a pack uses that both databases hold, the standard Gibbs energies from 300 to 1500 K agree within
    0.1 kJ/mol for most and 4.2 kJ/mol for every one (HCN), the figures the Methodology tab's loading walk-through and
    chem/thermo.py quote from experiments/design_method."""
    import json

    import cantera as ct

    from catalyst_engine.chem import thermo
    from catalyst_engine.units import R

    gri = ct.Solution("gri30.yaml")
    nasa = {s.name: s for s in ct.Species.list_from_file("nasa_gas.yaml")}
    used = {sp.thermo_id for pid in list_packs() for sp in load_pack(pid).spec.species.values()}
    gaps = {}
    for name in sorted(used & set(gri.species_names) & set(nasa)):
        worst = 0.0
        for t in np.arange(300.0, 1501.0, 100.0):
            gri.TP = t, gri.reference_pressure
            g_gri = gri.standard_gibbs_RT[gri.species_index(name)] * R * t / 1000.0
            g_nasa = (nasa[name].thermo.h(t) - t * nasa[name].thermo.s(t)) / 1e6
            worst = max(worst, abs(g_gri - g_nasa))
        gaps[name] = worst
    assert max(gaps.values()) < 4.3 and max(gaps, key=gaps.get) == "HCN"
    values = json.loads((SRC.parents[1] / "experiments" / "design_method" / "results" / "summary.json").read_text())[
        "values"]
    assert values["load_db_shared"] == len(gaps)
    assert values["load_db_within"] == sum(v <= 0.1 for v in gaps.values())
    # a set of gases GRI-Mech holds whole takes GRI-Mech's numbers; add benzene, beyond it, and all come from NASA Glenn
    gri.TP = 800.0, gri.reference_pressure
    h2 = nasa["H2"].thermo
    assert thermo.standard_gibbs_rt(["H2", "CO"], 800.0)[0] == pytest.approx(gri.standard_gibbs_RT[gri.species_index("H2")])
    assert thermo.standard_gibbs_rt(["H2", "C6H6"], 800.0)[0] == pytest.approx(
        (h2.h(800.0) - 800.0 * h2.s(800.0)) / (ct.gas_constant * 800.0))
