"""Electrocatalysis: the computational hydrogen electrode on the thermal screens' binding energies, the three
shipped reactions, the ranking, the worker's hydroperoxyl, and the electrochemistry side of the Design and Results
pages. No network and no atomistic model."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from catalyst_engine.design import electro, materials, screening, uma_worker

VIEWS = Path(__file__).resolve().parents[1] / "src" / "catalyst_engine" / "ui" / "views"


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CE_DATA_DIR", str(tmp_path / "data"))
    for var in ("MP_API_KEY", "PMG_MAPI_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


# ---------------------------------------------------------------- the method

def test_every_shipped_reaction_loads_with_one_intermediate_per_electron_transfer_but_the_last():
    ids = electro.list_electro()
    assert {"hydrogen_evolution", "oxygen_reduction", "oxygen_evolution"} <= set(ids)
    for sid in ids:
        s = electro.load_electro(sid)
        assert len(s.intermediates) == s.electrons - 1
        assert s.total_eV == pytest.approx(s.electrons * s.equilibrium_V)
        assert s.reference.formula


def test_hydrogen_evolution_overpotential_is_the_size_of_the_hydrogen_free_energy():
    """Nørskov et al. 2005: the exchange current peaks where H* sits at zero free energy, and either side the
    limiting potential falls by |dG_H|; the free energy is the binding energy plus 0.24 eV."""
    s = electro.load_electro("hydrogen_evolution")
    for e_bind in (-0.5, -0.24, 0.0, 0.3):
        g = s.free_energies({"H": e_bind})
        assert g["H"] == pytest.approx(e_bind + 0.24)
        assert s.overpotential(g) == pytest.approx(abs(g["H"]))
        assert s.limiting_potential(g)[0] == pytest.approx(-abs(g["H"]))
        # at a potential past its limiting potential every step runs downhill
        assert s.uphill(g, s.limiting_potential(g)[0] - 0.01) == 0.0
    assert s.volcano_limit(0.0) == 0.0 and s.volcano_limit(0.3) == pytest.approx(-0.3)


def test_oxygen_reduction_on_platinum_follows_norskov_2004():
    """With the paper's Pt(111) binding energies (Table 2: dE_OH 1.05, dE_O 1.57 eV against H2O and H2) and the
    app's corrections, the free energies climb from the products to O2 + 2 H2 at 4.92 eV, every step runs
    downhill at 0 V, the equilibrium potential levels the ends, and the O* to OH* step limits the potential."""
    s = electro.load_electro("oxygen_reduction")
    g = s.free_energies({"OH": 1.05, "O": 1.57, "OOH": 1.05 + 3.2})
    levels = s.levels(g)
    assert levels[0] == pytest.approx(4.92) and levels[-1] == 0.0
    assert all(step < 0 for step in s.steps(g))
    u_l, k = s.limiting_potential(g)
    assert s.step_names()[k] == "O → OH"
    assert 0.5 < u_l < 0.6 and s.overpotential(g) == pytest.approx(1.23 - u_l)
    at_eq = electro.diagram(s, g, 1.23)
    assert at_eq[0]["level"] == pytest.approx(0.0) and at_eq[-1]["level"] == 0.0
    # the free-energy diagram's steps at a potential are the steps shifted by it
    assert [b["level"] - a["level"] for a, b in zip(at_eq, at_eq[1:])] == pytest.approx(s.steps_at(g, 1.23))
    # the ideal catalyst: every level a quarter of the way down, zero overpotential
    ideal = {"OOH": 3.69, "O": 2.46, "OH": 1.23}
    assert s.overpotential(ideal) == pytest.approx(0.0, abs=1e-9)
    assert s.uphill(ideal, 1.23) == pytest.approx(0.0, abs=1e-9) and s.uphill(ideal, 1.33) == pytest.approx(0.1)


def test_oxygen_evolution_volcano_peaks_where_man_2011_puts_it():
    """The O* minus OH* gap is the descriptor; with OOH* 3.2 eV above OH* the two middle steps cross at 1.6 eV,
    where the overpotential floor is 0.37 V."""
    s = electro.load_electro("oxygen_evolution")
    assert s.volcano_limit(1.6) == pytest.approx(1.6)
    assert s.volcano_limit(1.6) - s.equilibrium_V == pytest.approx(0.37)
    assert s.volcano_limit(1.2) == pytest.approx(2.0) and s.volcano_limit(2.0) == pytest.approx(2.0)
    g = s.free_energies({"OH": 0.8, "O": 2.4, "OOH": 4.0})
    assert s.descriptor == {"O": 1, "OH": -1}
    assert s.levels(g)[-1] == pytest.approx(4.92)
    assert s.limiting_potential(g)[0] == pytest.approx(max(s.steps(g)))
    assert s.overpotential(g) > 0


# ---------------------------------------------------------------- predictions

def _electro_screen(spec: electro.ElectroSpec, bindings: dict[str, dict[str, float]], broken: dict[str, str] | None = None,
                    compare_V: float | None = None) -> Path:
    """A finished screen whose results come from the given binding energies; `broken` names a material and the
    intermediate that split on every site."""
    entries = []
    for i, sym in enumerate(bindings):
        e = materials.element_entry(sym)
        if i == 0:
            e.update(is_reference=True, label=f"{sym} (catalyst to beat)")
        entries.append(e)
    folder = electro.create_screen(spec, "test electro screen", {"compare_V": compare_V if compare_V is not None
                                                                 else spec.compare_V}, entries, "quick")
    lines = []
    for sym, bind in bindings.items():
        mid = f"element:{sym}"
        lines.append({"material": mid, "adsorbate": "*", "site": "clean", "facet": [1, 1, 1], "n_atoms": 36,
                      "energy": -100.0, "structure": "structures/x.cif"})
        for ads, e in bind.items():
            split = (broken or {}).get(sym) == ads
            lines.append({"material": mid, "adsorbate": ads, "site": "top", "E_bind": e,
                          "anomaly": "dissociated" if split else "", "n_ads": 1, "converged": True,
                          "structure": "structures/x.cif"})
    (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    slab = uma_worker.build_slab(uma_worker.bulk_from(entries[0]), (1, 1, 1))
    (folder / "structures").mkdir(exist_ok=True)
    (folder / "structures" / "x.cif").write_text(materials.cif_text(slab))
    (folder / "status.json").write_text(json.dumps({"state": "finished", "done": 1, "total": 1, "started": 0,
                                                    "finished": 60}))
    return folder


def test_predictions_rank_by_overpotential_and_set_speeds_against_the_catalyst_to_beat():
    s = electro.load_electro("hydrogen_evolution")
    folder = _electro_screen(s, {"Pt": {"H": -0.3}, "Pd": {"H": -0.24}, "Au": {"H": 0.3}, "Ni": {"H": -0.6}},
                             compare_V=-0.1)
    job = screening.read_job(folder)
    assert electro.is_electro(job) and job["adsorbates"] == ["H"]
    preds, curves = electro.predict(s, job, screening.read_results(folder))
    assert [p.formula for p in preds] == ["Pd", "Pt", "Ni", "Au"]
    pd_, pt, ni, au = preds
    assert pd_.overpotential_V == pytest.approx(0.0, abs=1e-9) and pt.overpotential_V == pytest.approx(0.06)
    assert pt.is_reference and pt.rate_ratio == pytest.approx(1.0) and pt.rank == 2
    assert pd_.rate_ratio == pytest.approx(1.0)  # at -0.1 V both run every step downhill: the same top speed
    assert au.rate_ratio < 1e-3 and au.rate_range[0] <= au.rate_ratio <= au.rate_range[1]
    assert ni.limiting_range[0] <= ni.limiting_V <= ni.limiting_range[1]
    assert set(curves["materials"]) == {p.material for p in preds} and curves["reference"] == pt.material
    assert curves["ideal"][0] == 0.0  # the ideal runs downhill at every potential below equilibrium
    assert all(p.complete for p in preds)


def test_an_intermediate_that_splits_everywhere_leaves_a_material_without_a_limiting_potential():
    s = electro.load_electro("oxygen_reduction")
    folder = _electro_screen(s, {"Pt": {"OOH": 4.1, "O": 1.6, "OH": 1.0}, "Ru": {"OOH": 3.0, "O": 0.0, "OH": 0.0}},
                             broken={"Ru": "OOH"})
    preds, _ = electro.predict(s, screening.read_job(folder), screening.read_results(folder))
    ru = next(p for p in preds if p.formula == "Ru")
    assert ru.missing == ["OOH"] and ru.rank is None and ru.limiting_V is None
    assert next(p for p in preds if p.formula == "Pt").rank == 1


# ---------------------------------------------------------------- the worker

def test_the_worker_places_hydroperoxyl_whole_and_reads_its_bonds():
    bulk = uma_worker.bulk_from(materials.element_entry("Pt"))
    slab = uma_worker.build_slab(bulk, uma_worker.densest_plane(bulk))
    sites = uma_worker.choose_sites(uma_worker.surface_sites(slab), "OOH", 3)
    assert sites[0]["kind"] == "top"
    adslab = uma_worker.place(slab, "OOH", sites[0])
    ads = adslab[len(slab):]
    d = ads.get_all_distances()
    assert d[0, 1] == pytest.approx(1.40, abs=0.01) and d[1, 2] == pytest.approx(0.98, abs=0.01)
    assert uma_worker.anomaly(adslab, len(slab), "OOH") == ""
    stretched = adslab.copy()
    stretched.positions[len(slab) + 1] += [0.0, 0.0, 1.5]  # the O-O bond broken
    assert uma_worker.anomaly(stretched, len(slab), "OOH") == "dissociated"
    assert uma_worker.REFERENCE["OOH"] == {"H2O": 2.0, "H2": -1.5}
    nh3 = uma_worker.place(slab, "NH3", uma_worker.choose_sites(uma_worker.surface_sites(slab), "NH3", 1)[0])
    assert uma_worker.anomaly(nh3, len(slab), "NH3") == ""


# ---------------------------------------------------------------- the pages

def test_design_page_switches_to_electrocatalysis_and_adds_metals():
    at = AppTest.from_file(str(VIEWS / "design.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.segmented_control(key="design_kind").set_value("electro").run()
    assert not at.exception, at.exception
    assert any("What should the electrode do?" in m.value for m in at.markdown)
    assert any(s.label == "Compare catalysts at (V vs RHE)" for s in at.slider)
    next(b for b in at.button if b.label == "Add these metals").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["design_electro_candidates"]) > 0
    assert "design_candidates" not in at.session_state or not at.session_state["design_candidates"]
    at.selectbox(key="electro_pack").set_value("oxygen_evolution").run()
    assert not at.exception, at.exception
    assert any("Iridium oxide" in m.value for m in at.markdown)


def test_results_page_shows_an_electrochemical_screen():
    s = electro.load_electro("oxygen_reduction")
    _electro_screen(s, {"Pt": {"OOH": 4.1, "O": 1.6, "OH": 1.0}, "Pd": {"OOH": 4.2, "O": 1.7, "OH": 1.1},
                        "Au": {"OOH": 4.8, "O": 2.8, "OH": 1.5}})
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any(m.label == "Overpotential" for m in at.metric)
    assert [t.label for t in at.tabs][:4] == ["Ranking", "Activity vs potential", "Free-energy diagram", "Volcano"]
    table = at.dataframe[0].value
    assert "Overpotential (V)" in table.columns and "OH* (eV)" in table.columns
    assert any("the catalyst to beat, leads" in m.value or "Best candidate" in m.value for m in at.markdown)
