"""Electrocatalysis: the computational hydrogen electrode on the worker's binding energies, the three shipped
reactions, the ranking, the worker's hydroperoxyl, and the section's own pages. No network and no atomistic model."""

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
        # each chemical step adds a step of the path that moves zero electrons
        assert len(s.intermediates) == s.electrons - 1 + len(s.chemical_steps)
        assert len(s.electron_steps) == s.electrons
        assert s.total_eV == pytest.approx(s.electrons * s.equilibrium_V)
        assert s.reference.formula
        # a reaction kept to the playground says what it waits for
        assert s.on_design_page or s.design_note


def test_a_file_whose_electron_count_and_path_disagree_stops_the_load():
    from pydantic import ValidationError

    raw = electro.load_electro("co2_to_co").model_dump(by_alias=True)
    with pytest.raises(ValidationError, match="electron steps"):
        electro.ElectroSpec.model_validate({**raw, "chemical_steps": []})  # three steps would move three electrons
    with pytest.raises(ValidationError, match="distinct step"):
        electro.ElectroSpec.model_validate({**raw, "chemical_steps": [{"step": 9, "label": "past the end"}]})
    with pytest.raises(ValidationError, match="Extra inputs"):
        electro.ElectroSpec.model_validate({**raw, "chemical_step": raw["chemical_steps"]})  # a misspelt key


def test_a_chemical_step_stays_put_as_the_potential_moves():
    """CO2 to CO: two proton-electron steps, then CO leaves the surface, a step that moves zero electrons. The
    potential tilts the first two and leaves the third; the limiting potential reads the electron steps alone, and
    the hurdle at a potential counts an uphill chemical step at every potential."""
    s = electro.load_electro("co2_to_co")
    assert [c.step for c in s.chemical_steps] == [2] and s.electron_steps == [0, 1]
    g = {"COOH": -0.6, "CO": -0.5}  # CO held 0.5 eV too tightly
    at0, at1 = s.steps_at(g, 0.0), s.steps_at(g, -0.3)
    assert at1[0] == pytest.approx(at0[0] - 0.3) and at1[1] == pytest.approx(at0[1] - 0.3)
    assert at1[2] == pytest.approx(at0[2]) == pytest.approx(0.5)  # CO leaving: uphill at every potential
    u_l, k = s.limiting_potential(g)
    assert k == 1 and u_l == pytest.approx(-s.steps(g)[1])
    assert s.chemical_hurdles(g) == {2: pytest.approx(0.5)}
    for u in (u_l - 0.5, u_l, 0.0):
        assert s.hurdle(g, u) >= 0.5 - 1e-9  # the chemical step holds the surface back at any potential
    # electron steps that carry less than their share leave the rest to the chemical step: a negative overpotential
    g2 = {"COOH": -0.2, "CO": -0.3}
    assert s.overpotential(g2) < 0 and sum(s.steps(g2)[k] for k in s.electron_steps) < -s.total_eV
    # the ideal catalyst keeps the chemical step level
    assert s.chemical_hurdles(s.ideal()) == {2: pytest.approx(0.0, abs=1e-12)}
    assert s.overpotential(s.ideal()) == pytest.approx(0.0, abs=1e-12)


def test_the_likely_ranges_hold_with_chemical_steps():
    """Ammonia oxidation ends in a step that moves zero electrons (two N* pair up and leave as N2); a grid over the
    error box stays inside the exact ranges and reaches within one spacing of their ends."""
    import itertools

    import numpy as np
    for sid, potential in (("ammonia_oxidation", 0.5), ("co2_to_co", -0.6)):
        s = electro.load_electro(sid)
        g = {a: v + 0.07 * (i + 1) for i, (a, v) in enumerate(s.ideal().items())}
        lo, hi = electro.limiting_range(s, g)
        h_lo, h_hi = electro.hurdle_range(s, g, potential)
        n = 13 if len(s.adsorbates) > 2 else 41
        spacing = 2 * electro.SIGMA_EV / (n - 1)
        axes = [np.linspace(g[a] - electro.SIGMA_EV, g[a] + electro.SIGMA_EV, n) for a in s.adsorbates]
        ul, hu = [], []
        for v in itertools.product(*axes):
            gg = dict(zip(s.adsorbates, v))
            ul.append(s.limiting_potential(gg)[0])
            hu.append(s.hurdle(gg, potential))
        tol = spacing + 1e-9
        assert lo - 1e-9 <= min(ul) <= lo + tol and hi - tol <= max(ul) <= hi + 1e-9, sid
        assert h_lo - 1e-9 <= min(hu) <= h_lo + tol and h_hi - tol <= max(hu) <= h_hi + 1e-9, sid


def test_a_shift_moves_the_reaction_to_its_own_zero():
    """Hydrogen peroxide production: OOH*'s binding starts on two waters less 1.5 H2; the file adds Nørskov 2004's
    +0.18 eV and a -3.526 eV shift (H2O2(aq) and H2 sit 3.526 eV above two waters), so the path runs from 1.39 eV
    (2 x 0.695 V) to zero. The walk-through in the electro Methodology tab follows these same numbers."""
    s = electro.load_electro("h2o2_production")
    assert s.total_eV == pytest.approx(1.39)
    assert s.free_energies({"OOH": 4.052})["OOH"] == pytest.approx(4.052 + 0.18 - 3.526)
    g = s.free_energies({"OOH": 4.052})
    assert s.steps(g) == pytest.approx([0.706 - 1.39, -0.706])
    assert s.limiting_potential(g) == (pytest.approx(0.684), 0)
    assert s.ideal()["OOH"] == pytest.approx(0.695)
    assert electro.limiting_range(s, g) == pytest.approx((0.534, 0.695), abs=1e-9)


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
    downhill at 0 V, and the equilibrium potential levels the ends. O2 splits freely on platinum (barrier below
    zero), so the splitting route sets the limiting potential: with the OH layer's hydrogen bonds, OH* to water
    limits at 0.80 V. With the paper's own +0.01 eV for O* in place of the app's +0.05, the same arithmetic gives
    the paper's 0.78 V with O* to OH* limiting (Table 2: dG1 0.45, dG2 0.43 eV at 1.23 V)."""
    s = electro.load_electro("oxygen_reduction")
    g = s.free_energies({"OH": 1.05, "O": 1.57, "OOH": 1.05 + 3.2})
    levels = s.levels(g)
    assert levels[0] == pytest.approx(4.92) and levels[-1] == 0.0
    assert all(step < 0 for step in s.steps(g))
    u_l, k = s.limiting_potential(g)
    assert s.step_names()[k] == "OH → *" and s.limiting_route(g) == "O2 splits first"
    assert u_l == pytest.approx(0.80, abs=1e-9) and s.overpotential(g) == pytest.approx(1.23 - u_l)
    # the OOH route alone stops where OOH* forms: 4.92 - (4.25 + 0.18) eV
    assert -max(s.steps(g)) == pytest.approx(4.92 - 4.43)
    # O2 splitting on platinum runs below every step: its barrier, 1.8 x 1.57 - 2.89 eV less 0.13, is negative
    assert s.activation.barrier({"O": 1.57}) == pytest.approx(1.8 * 1.57 - 2.89)
    assert s.barrier(g) == pytest.approx(1.8 * 1.57 - 2.89 - 0.13) and s.barrier(g) < 0
    hurdles = s.route_hurdles(g, 0.9)
    assert s.hurdle(g, 0.9) == pytest.approx(hurdles["O2 splits first"]) == pytest.approx(0.90 - 0.80)
    assert s.route_at(g, 0.9) == "O2 splits first" and hurdles["OOH* forms first"] == pytest.approx(0.90 - 0.49)
    paper = {"OOH": g["OOH"], "O": 1.57 + 0.01, "OH": 1.05 - 0.25}  # the Table 2 footnote's adjustments
    st = s.steps_at(paper, 1.23)
    assert st[2] == pytest.approx(0.45, abs=0.005) and st[3] == pytest.approx(0.43, abs=0.005)
    u_paper, k_paper = s.limiting_potential(paper)
    assert u_paper == pytest.approx(0.78, abs=0.005) and s.step_names()[k_paper] == "O → OH"
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
    # its catalysts are oxides, where UMA misses DFT beyond the error bar: the playground keeps it, the Design page waits
    assert not s.on_design_page and electro.load_electro("oxygen_reduction").on_design_page


def test_the_paper_s_table_2_comes_out_of_the_app_s_arithmetic():
    """Nørskov 2004, Table 2, with its footnote's adjustments (G_OH = dE_OH - 0.25, G_O = dE_O + 0.01 eV): the app's
    last two steps at 1.23 V are the table's dG1 and dG2, and its splitting barrier the table's E_a."""
    s = electro.load_electro("oxygen_reduction")
    table = {"Pt": (1.05, 1.57, 0.45, 0.43, -0.06), "Au": (1.49, 2.75, -0.29, -0.01, 2.06),
             "Ni": (0.13, 0.34, 0.76, 1.35, -2.28), "W": (-0.80, -2.06, 2.23, 2.28, -6.60)}
    for d_oh, d_o, dg1, dg2, ea in table.values():
        g = {"OOH": 4.0, "O": d_o + 0.01, "OH": d_oh - 0.25}
        st = s.steps_at(g, 1.23)
        assert st[2] == pytest.approx(dg1, abs=0.006) and st[3] == pytest.approx(dg2, abs=0.006)
        assert s.activation.barrier({"O": d_o}) == pytest.approx(ea, abs=0.006)


def test_the_volcano_follows_the_paper_s_scaling_and_both_routes():
    """With dE_OH = 0.5 dE_O + 0.05 and dE_OOH = 0.5 dE_O + 3.18 eV (Nørskov 2004, Fig. 8) and the app's corrections,
    the line peaks at 0.68 V where OH* sits at 0.68 eV; to its left O2 splits freely, to its right OOH* formation
    limits. Each line reproduces the full arithmetic on a surface built from those relations."""
    s = electro.load_electro("oxygen_reduction")
    assert s.volcano_limit(0.68) == pytest.approx(0.68)
    assert s.volcano_limit(0.40) == pytest.approx(0.40) and s.volcano_limit(0.90) == pytest.approx(1.36 - 0.90)
    for x in (0.2, 0.5, 0.64, 0.7, 1.0):
        d_oh = x + 0.25
        d_o = 2 * (d_oh - 0.05)
        g = s.free_energies({"OH": d_oh, "O": d_o, "OOH": 0.5 * d_o + 3.18})
        assert g["OH"] == pytest.approx(x)
        assert s.volcano_limit(x) == pytest.approx(s.limiting_potential(g)[0], abs=1e-9)
        assert (s.volcano_barrier[0] * x + s.volcano_barrier[1]) == pytest.approx(s.barrier(g), abs=1e-9)


def test_the_ideal_catalyst_levels_every_step_at_the_equilibrium_potential():
    """A reduction path runs down from the reactants, an oxidation path up; either way the ideal staircase has every
    electron step the same size, every chemical step level, and zero overpotential."""
    for pid in electro.list_electro():
        s = electro.load_electro(pid)
        ideal = s.ideal()
        steps = s.steps(ideal)
        assert [steps[k] for k in s.electron_steps] == pytest.approx([steps[s.electron_steps[0]]] * s.electrons,
                                                                     abs=1e-12)
        assert [steps[c.step] for c in s.chemical_steps] == pytest.approx([0.0] * len(s.chemical_steps), abs=1e-12)
        assert s.overpotential(ideal) == pytest.approx(0.0, abs=1e-12)
    assert electro.load_electro("oxygen_reduction").ideal() == pytest.approx({"OOH": 3.69, "O": 2.46, "OH": 1.23})
    assert electro.load_electro("oxygen_evolution").ideal() == pytest.approx({"OH": 1.23, "O": 2.46, "OOH": 3.69})


def test_the_likely_ranges_are_the_exact_ends_over_the_error_box():
    """Every intermediate moves within ±0.15 eV at once: a fine grid over that box stays inside the computed ranges
    and comes within one grid spacing of their ends, the volcano top and the switch between routes included. The
    exact ends can sit between grid points, at the potential where the O2-splitting barrier crosses zero."""
    import itertools

    import numpy as np
    s = electro.load_electro("oxygen_reduction")
    g = s.free_energies({"OH": 1.04, "O": 1.58, "OOH": 4.05})  # platinum-like: its barrier crosses zero in the box
    lo, hi = electro.limiting_range(s, g)
    h_lo, h_hi = electro.hurdle_range(s, g, 0.9)
    n = 31
    spacing = 2 * electro.SIGMA_EV / (n - 1)
    axes = [np.linspace(g[a] - electro.SIGMA_EV, g[a] + electro.SIGMA_EV, n) for a in s.adsorbates]
    ul, hu = [], []
    for v in itertools.product(*axes):
        gg = dict(zip(s.adsorbates, v))
        ul.append(s.limiting_potential(gg)[0])
        hu.append(s.hurdle(gg, 0.9))
    tol = spacing + 1e-9
    assert lo - 1e-9 <= min(ul) <= lo + tol and hi - tol <= max(ul) <= hi + 1e-9
    assert h_lo - 1e-9 <= min(hu) <= h_lo + tol and h_hi - tol <= max(hu) <= h_hi + 1e-9
    # hydrogen evolution: a surface 0.08 eV from the top reaches it within its error bar
    her = electro.load_electro("hydrogen_evolution")
    assert electro.limiting_range(her, {"H": -0.08}) == pytest.approx((-0.23, 0.0), abs=1e-9)


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


def test_the_faster_route_carries_the_current():
    """Nørskov 2004, Fig. 8: the routes run side by side and the faster one dominates. Platinum splits O2 freely and
    goes through O*; a silver-like surface binds O too weakly to split O2 (barrier 0.67 eV), so it goes through OOH*
    and skips that barrier, at its OOH* step's cost; gold pays more on the same route."""
    s = electro.load_electro("oxygen_reduction")
    folder = _electro_screen(s, {"Pt": {"OOH": 4.25, "O": 1.57, "OH": 1.05},
                                 "Ag": {"OOH": 4.1, "O": 2.05, "OH": 1.25},
                                 "Au": {"OOH": 4.69, "O": 2.75, "OH": 1.49}})
    preds, curves = electro.predict(s, screening.read_job(folder), screening.read_results(folder))
    assert [p.formula for p in preds] == ["Pt", "Ag", "Au"]
    pt, ag, au = preds
    assert pt.route == "O2 splits first" and pt.hurdle_eV == pytest.approx(0.90 - 0.80)
    assert ag.barrier_eV == pytest.approx(1.8 * 2.05 - 2.89 - 0.13) and ag.route == "OOH* forms first"
    assert ag.hurdle_eV == pytest.approx(4.1 + 0.18 - 4.92 + 0.90)  # its OOH* step, below the barrier
    assert ag.limiting_route == "OOH* forms first" and ag.limiting_V == pytest.approx(4.92 - 4.28)
    assert au.barrier_eV > ag.barrier_eV and pt.barrier_eV < 0 and au.hurdle_eV > ag.hurdle_eV
    assert ag.rate_range[0] <= ag.rate_ratio <= ag.rate_range[1]
    # at low potential every route of every surface runs downhill: the same top speed
    assert curves["materials"][ag.material]["ratio"][0] == pytest.approx(1.0)


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

def test_the_electro_design_page_adds_metals_and_switches_reactions():
    at = AppTest.from_file(str(VIEWS / "electro_design.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("What should the electrode do?" in m.value for m in at.markdown)
    assert any(s.label == "Compare catalysts at (V vs RHE)" for s in at.slider)
    next(b for b in at.button if b.label == "Add these metals").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["electro_candidates"]) > 0
    shown = at.selectbox(key="electro_pack").options  # the reactions on the Design page, by their names
    on_page = [i for i in electro.list_electro() if electro.load_electro(i).on_design_page]
    assert len(shown) == len(on_page) and {"hydrogen_evolution", "oxygen_reduction"} <= set(on_page)
    assert all(electro.load_electro(i).name in " ".join(shown) for i in on_page)
    kept = [i for i in electro.list_electro() if i not in on_page]  # the playground keeps the rest
    assert kept and all(electro.load_electro(i).name not in " ".join(shown) for i in kept)
    at.selectbox(key="electro_pack").set_value("hydrogen_evolution").run()
    assert not at.exception, at.exception
    assert any("Platinum" in m.value for m in at.markdown)
    at.toggle(key="electro_advanced").set_value(True).run()
    assert not at.exception, at.exception
    assert at.selectbox(key="electro_task_adv_hydrogen_evolution").value == "oc20"


def test_the_electro_results_page_shows_a_screen():
    s = electro.load_electro("oxygen_reduction")
    _electro_screen(s, {"Pt": {"OOH": 4.1, "O": 1.6, "OH": 1.0}, "Pd": {"OOH": 4.2, "O": 1.7, "OH": 1.1},
                        "Au": {"OOH": 4.8, "O": 2.8, "OH": 1.5}})
    assert electro.list_screens() and electro.screens_dir().name == "screens"
    assert electro.screens_dir().parent.name == "electro"  # apart from the thermal screens
    at = AppTest.from_file(str(VIEWS / "electro_results.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any(m.label == "Overpotential" for m in at.metric)
    assert [t.label for t in at.tabs][:4] == ["Ranking", "Activity vs potential", "Free-energy diagram", "Volcano"]
    table = at.dataframe[0].value
    assert "Overpotential (V)" in table.columns and "OH* (eV)" in table.columns
    assert "Faster route at 0.90 V" in table.columns and set(table["Faster route at 0.90 V"]) <= {
        "O2 splits first", "OOH* forms first", electro.EITHER}
    assert any("the catalyst to beat, leads" in m.value or "Best candidate" in m.value for m in at.markdown)


def test_the_shipped_electro_examples_land_apart_from_the_thermal_screens():
    electro.add_examples()
    screens = electro.list_screens()
    assert len(screens) == len([p for p in electro.EXAMPLES.iterdir() if (p / "job.json").exists()]) >= 2
    assert not screening.list_screens()  # the thermal screens folder stays empty
    for folder in screens:
        job = screening.read_job(folder)
        s = electro.load_electro(job["request"]["pack"])
        preds, _ = electro.predict(s, job, screening.read_results(folder))
        assert all(p.complete for p in preds) and next(p for p in preds if p.is_reference).formula == "Pt"


def test_the_playground_and_manual_render_and_the_ideal_catalyst_has_zero_overpotential():
    at = AppTest.from_file(str(VIEWS / "electro_playground.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Set the ideal catalyst").click().run()
    assert not at.exception, at.exception
    assert next(m.value for m in at.metric if m.label == "Overpotential") == "0.00 V"
    # every reaction file runs in the playground, oxygen evolution included
    assert set(at.selectbox(key="epg_pack").options) == {electro.load_electro(i).name for i in electro.list_electro()}
    for pid in ("oxygen_reduction", "oxygen_evolution"):
        at.selectbox(key="epg_pack").set_value(pid).run()
        assert not at.exception, at.exception
        next(b for b in at.button if b.label == "Set the ideal catalyst").click().run()
        assert next(m.value for m in at.metric if m.label == "Overpotential") == "0.00 V"
    at = AppTest.from_file(str(VIEWS / "electro_manual.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Electrochemistry basics" in t.label for t in at.tabs)
    assert any("computational hydrogen electrode" in m.value.lower() for m in at.markdown)


def test_the_playground_reads_a_chemical_step_apart_from_the_electron_steps():
    """CO2 to CO with CO held 0.5 eV too tightly: the electron steps reach the equilibrium potential, and the page
    says the CO-leaving step stays uphill at every potential. Hold COOH* and CO* so the electron steps carry less than
    their share, and the page says the chemical step holds the rest."""
    at = AppTest.from_file(str(VIEWS / "electro_playground.py"), default_timeout=120).run()
    at.selectbox(key="epg_pack").set_value("co2_to_co").run()
    for ads, g in (("COOH", -0.6), ("CO", -0.5)):
        at.slider(key=f"epg_g_co2_to_co_{ads}").set_value(g).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Every electron step runs downhill by the equilibrium potential." in text
    assert "(CO leaves the surface) **+0.50 eV**" in text and "holds the surface back at every potential" in text
    assert next(m.value for m in at.metric if m.label == "Overpotential") == "0.00 V"
    at.slider(key="epg_g_co2_to_co_COOH").set_value(-0.2).run()
    at.slider(key="epg_g_co2_to_co_CO").set_value(-0.3).run()
    text = " ".join(m.value for m in at.markdown)
    assert "before the equilibrium potential, because the chemical step holds the rest" in text
