"""Catalyst design: the volcano, surface building, Materials Project helpers, predictions from binding energies,
the reaction builder, and the design pages. No network and no atomistic model: synthetic results stand in for the
worker's output."""

import json
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes
from streamlit.testing.v1 import AppTest

from catalyst_engine.design import builder, elements, materials, screening, uma_worker, zeolites
from catalyst_engine.design.spec import ReferenceSpec, ScreeningSpec
from catalyst_engine.packs import PACKS_DIR, list_packs, load_pack
from catalyst_engine.ui import common

VIEWS = Path(__file__).resolve().parents[1] / "src" / "catalyst_engine" / "ui" / "views"
SCREENABLE = [pid for pid in list_packs(PACKS_DIR) if load_pack(pid).spec.screening is not None]
CO = "co_oxidation" if "co_oxidation" in SCREENABLE else SCREENABLE[0]  # its volcano top sits off the reference


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    # Each test gets its own screens and reactions folders, and the Materials Project key stays out of reach.
    monkeypatch.setenv("CE_DATA_DIR", str(tmp_path / "data"))
    for var in ("MP_API_KEY", "PMG_MAPI_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


# ---------------------------------------------------------------- the volcano

def test_barrier_shift_is_zero_at_the_reference_and_lowest_at_the_volcano_top():
    spec = ScreeningSpec(descriptor={"O": 1.0}, pathways=["p"], metric="m", reference=ReferenceSpec(formula="Pt"),
                         optimum_shift_eV=0.2, slope_weak=0.9, slope_strong=1.0)
    ref = 1.5
    assert spec.barrier_shift(ref, ref) == pytest.approx(0.0)
    assert spec.barrier_shift(ref + 0.2, ref) == pytest.approx(-0.2)  # the top beats the reference
    assert spec.barrier_shift(ref - 0.5, ref) == pytest.approx(0.5)  # stronger binding, slope 1.0
    assert spec.barrier_shift(ref + 0.7, ref) == pytest.approx(0.9 * 0.5 - 0.2)  # weaker, past the top


def test_two_adsorbate_descriptor_widens_its_uncertainty_by_the_weights():
    spec = ScreeningSpec(descriptor={"C": 0.5, "O": 0.5}, pathways=["p"], metric="m",
                         reference=ReferenceSpec(formula="Ni"), sigma_eV=0.2)
    assert spec.sigma_descriptor == pytest.approx(0.2 * np.sqrt(0.5))


def test_calibration_puts_the_reference_at_half_the_best():
    assert len(SCREENABLE) >= 8
    pack = load_pack(CO)
    step = screening.design_step(pack)
    t_cal = float(np.median([t - 273.15 for t in step.kelvin()]))
    params = {**pack.network.parameters(), **screening.calibrate(pack, step, t_cal)}
    one = step.model_copy(update={"temperatures_C": [t_cal], "temperatures_F": None, "temperatures_K": None})
    metric = pack.spec.screening.metric
    best, ref = screening.metric_curves(pack, one, [screening.fast_params(pack), params], metric)[:, 0]
    assert ref == pytest.approx(0.5 * best, rel=0.02)


# ---------------------------------------------------------------- surfaces

def test_densest_plane_of_a_primitive_fcc_cell_is_its_close_packed_face():
    bulk = uma_worker.bulk_from(materials.element_entry("Pt"))
    hkl = uma_worker.densest_plane(bulk)
    recip = np.linalg.inv(bulk.cell.array).T
    d = 1.0 / np.linalg.norm(np.dot(hkl, recip))
    assert d == pytest.approx(3.92 / np.sqrt(3), abs=0.02)  # the (111) spacing of fcc Pt, a = 3.92 A


def test_ordered_alloys_show_their_close_packed_face_with_the_bulk_composition():
    from ase import Atoms

    a = 3.75  # L1_2 Cu3Au: the close-packed {111} layers hold one Au for every three Cu
    l12 = Atoms("AuCu3", scaled_positions=[(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)], cell=[a, a, a], pbc=True)
    density, mismatch = uma_worker.plane_layers(l12, uma_worker.densest_plane(l12))
    assert density == pytest.approx(4 / (np.sqrt(3) * a * a), rel=1e-3) and mismatch == pytest.approx(0)
    b = 2.99  # B2 CuPd: the densest {110} layers hold both elements
    b2 = Atoms("CuPd", scaled_positions=[(0, 0, 0), (.5, .5, .5)], cell=[b, b, b], pbc=True)
    hkl = uma_worker.densest_plane(b2)
    assert sorted(map(abs, hkl)) == [0, 1, 1]
    assert uma_worker.plane_layers(b2, hkl)[0] == pytest.approx(np.sqrt(2) / b ** 2, rel=1e-3)


@pytest.mark.parametrize("symbol", ["Pt", "Ru", "Fe"])
def test_slab_has_vacuum_a_fixed_bottom_and_every_site_kind(symbol):
    bulk = uma_worker.bulk_from(materials.element_entry(symbol))
    slab = uma_worker.build_slab(bulk, uma_worker.densest_plane(bulk))
    z = slab.positions[:, 2]
    assert 8 <= len(slab) <= 90
    assert slab.cell.lengths()[2] - np.ptp(z) == pytest.approx(20.0, abs=0.01)
    fixed = slab.constraints[0].get_indices()
    assert 0 < len(fixed) < len(slab) and z[fixed].max() < z.max()
    kinds = {s["kind"] for s in uma_worker.surface_sites(slab)}
    assert {"top", "bridge", "hollow"} <= kinds


def test_a_placed_atom_that_stays_put_reads_as_bound():
    bulk = uma_worker.bulk_from(materials.element_entry("Pt"))
    slab = uma_worker.build_slab(bulk, uma_worker.densest_plane(bulk))
    site = uma_worker.choose_sites(uma_worker.surface_sites(slab), "O", 1)[0]
    adslab = uma_worker.place(slab, "O", site)
    assert len(adslab) == len(slab) + 1
    assert uma_worker.anomaly(adslab, len(slab), "O") == ""
    adslab.positions[-1, 2] += 5.0
    assert uma_worker.anomaly(adslab, len(slab), "O") == "desorbed"


class _DoubleWell(Calculator):
    """Each atom sits in a double well along x around where it started: zero force at the start, which lies
    0.0625 eV above the two minima 0.5 A to either side. A slab built there balances on a symmetric point."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, x0):
        super().__init__()
        self.x0 = x0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = self.atoms.positions[:, 0] - self.x0
        forces = np.zeros((len(x), 3))
        forces[:, 0] = -4 * x * (x ** 2 - 0.25)
        self.results = {"energy": float(np.sum((x ** 2 - 0.25) ** 2)), "forces": forces}


def test_the_nudge_moves_only_the_free_atoms_a_little_and_the_same_way_every_time():
    bulk = uma_worker.bulk_from(materials.element_entry("Pt"))
    slab = uma_worker.build_slab(bulk, uma_worker.densest_plane(bulk))
    fixed = slab.constraints[0].get_indices()
    a, b = uma_worker.nudge(slab.copy()), uma_worker.nudge(slab.copy())
    move = np.linalg.norm(a.positions - slab.positions, axis=1)
    assert np.allclose(a.positions, b.positions)
    assert move[fixed].max() == 0 and 0 < move.max() < 0.2
    assert uma_worker.largest_move(slab.positions, a) == pytest.approx(move.max())


def test_a_surface_balanced_on_a_symmetric_point_settles_once_nudged():
    from ase.optimize import LBFGS

    bulk = uma_worker.bulk_from(materials.element_entry("Pt"))
    slab = uma_worker.build_slab(bulk, uma_worker.densest_plane(bulk))
    n_fixed = len(slab.constraints[0].get_indices())
    calc = _DoubleWell(slab.positions[:, 0].copy())

    def relaxed(atoms):
        return uma_worker.relax_with(atoms, calc, fmax=0.01, steps=300)[0]

    assert relaxed(slab.copy()) == pytest.approx(0.0625 * len(slab))  # zero force: it stays on the balance point
    assert relaxed(uma_worker.nudge(slab.copy())) == pytest.approx(0.0625 * n_fixed, abs=1e-3)  # every free atom settled
    tipped = uma_worker.nudge(slab.copy())  # LBFGS walks back up onto the balance point; BFGS is why the worker uses it
    tipped.calc = calc
    LBFGS(tipped, logfile=None).run(fmax=0.01, steps=300)
    assert tipped.get_potential_energy() == pytest.approx(0.0625 * len(slab), abs=1e-3)


def test_a_binding_energy_is_set_against_the_surface_beneath_when_that_surface_settled_lower():
    bulk = uma_worker.bulk_from(materials.element_entry("Pt"))
    slab = uma_worker.build_slab(bulk, uma_worker.densest_plane(bulk))
    adslab = uma_worker.place(slab, "NH3", uma_worker.choose_sites(uma_worker.surface_sites(slab), "NH3", 1)[0])
    relaxed = []

    def relax(atoms):
        relaxed.append((len(atoms), len(atoms.constraints[0].get_indices())))
        return -104.5

    # the usual case: the molecule strains the surface beneath it, which then holds more energy than the clean slab
    assert uma_worker.clean_reference(adslab, len(slab), -100.0, lambda a: -99.8, relax) == (-100.0, 0.0)
    assert relaxed == []
    # the surface settled 4 eV lower under the molecule: relax it with the molecule lifted off and set against that
    assert uma_worker.clean_reference(adslab, len(slab), -100.0, lambda a: -104.0, relax) == (-104.5, 4.5)
    assert relaxed == [(len(slab), len(slab.constraints[0].get_indices()))]
    rows = [{"material": "m", "adsorbate": "*", "site": "clean"},
            {"material": "m", "adsorbate": "NH3", "site": "top", "E_bind": -0.7, "slab_drop_eV": 4.5,
             "structure": "s.cif", "anomaly": ""}]
    b = screening.binding_summary(rows)["m"]["binding"]["NH3"]
    assert b["E"] == -0.7 and b["kept"] == 1 and "settled 4.50 eV lower" in b["sites"][0]["note"]


def test_a_cubic_metal_read_in_its_cubic_cell_gives_the_textbook_faces():
    primitive = uma_worker.bulk_from(materials.element_entry("Pt"))
    cubic = uma_worker.bulk_from({**materials.element_entry("Pt"), "cubic": True})
    close = uma_worker.plane_layers(primitive, uma_worker.densest_plane(primitive))[0]
    assert uma_worker.plane_layers(cubic, (1, 1, 1))[0] == pytest.approx(close, rel=1e-3)
    assert uma_worker.plane_layers(cubic, (1, 0, 0))[0] < close
    assert common.cubic_metal(materials.element_entry("Pt")) and not common.cubic_metal(materials.element_entry("Ru"))


def test_faces_read_from_plain_text():
    assert common.parse_face("auto") is None and common.parse_face(" ") is None
    assert common.parse_face("1 1 1") == common.parse_face("(111)") == [1, 1, 1]
    assert common.parse_face("1,-1,0") == [1, -1, 0]
    for bad in ("1 1", "0 0 0", "a b c"):
        with pytest.raises(ValueError):
            common.parse_face(bad)


# ---------------------------------------------------------------- zeolites

CHA = Path(__file__).resolve().parents[1] / "experiments" / "zeolite_check" / "data" / "cha_mp-aaaabamv.json"


def _cha():
    from ase import Atoms

    d = json.loads(CHA.read_text())
    return Atoms(d["species"], scaled_positions=d["frac"], cell=d["lattice"], pbc=True)


def test_chabazite_shows_the_rings_and_cage_of_its_framework_type():
    z = _cha()
    assert zeolites.is_framework(z) and zeolites.code("mp-aaaabamv") == "CHA"
    assert len(zeolites.windows(z, 6)) == 2  # the double six-ring
    assert zeolites.clearance(z, zeolites.ring_xyz(z, zeolites.windows(z, 8)[0]).mean(axis=0)) > 3.0  # open 8-ring
    assert zeolites.clearance(z, zeolites.pore_center(z)) == pytest.approx(4.9, abs=0.1)  # the cha cage


@pytest.mark.parametrize("form, n_al, cation", [("H", 1, ["H"]), ("M+", 1, ["Cu"]), ("M2+", 2, ["Cu"]),
                                                ("MOH+", 1, ["Cu", "O", "H"])])
def test_each_zeolite_site_puts_aluminium_in_the_framework_and_its_cation_beside_it(form, n_al, cation):
    z = _cha()
    variants, site = zeolites.build(z, form, "Cu")
    assert len(variants) == (4 if form == "H" else 1)  # the acid form tries its proton on each O around the Al
    for v in variants:
        symbols = v.get_chemical_symbols()
        assert symbols.count("Al") == n_al and symbols.count("Si") == 12 - n_al
        assert symbols[len(z):] == cation and v[site].symbol == cation[0]
        d = v.get_all_distances(mic=True)
        np.fill_diagonal(d, 9.0)
        assert d.min() > 0.9
        al = [i for i, s in enumerate(symbols) if s == "Al"]
        assert min(v.get_distances(site, al, mic=True)) < 3.6  # the cation sits by the charge it balances
    if form == "M2+":
        assert v.get_distance(*al, mic=True) > 5.0  # across the six-ring, two silicons apart either way


def test_a_molecule_binds_to_a_zeolite_site_at_its_reach_in_open_space():
    z = _cha()
    for form in ("H", "M2+", "MOH+"):
        variants, site = zeolites.build(z, form, "Cu")
        s = variants[0]
        out = zeolites.place_at(s, site, uma_worker.ADSORBATE_ATOMS["NH3"], 2.0)
        n = len(s)
        assert out.get_distance(site, n, mic=True) == pytest.approx(2.0)
        assert zeolites.clearance(s, out.positions[n], skip=site) > 2.4
        assert all(out.get_distance(site, n + k, mic=True) > 2.0 for k in (1, 2, 3))  # its H atoms point away


def test_the_worker_records_a_zeolite_site_the_way_it_records_a_surface(tmp_path):
    import io

    z = _cha()
    variants, site = zeolites.build(z, "H", "Cu")
    entry = {"id": "zeolite:test", "label": "test", "lattice": z.cell.array.tolist(), "site": site,
             "variants": [{"species": v.get_chemical_symbols(), "frac": v.get_scaled_positions().tolist()}
                          for v in variants]}
    (tmp_path / "structures").mkdir()
    out = io.StringIO()
    done, total = uma_worker._framework(entry, "t", "test", ["NH3"], {"NH3": -5.0}, lambda a: (-1.0 * len(a), 3, True),
                                        lambda a: -1.0 * len(a), out, tmp_path, 0, 0)
    rows = [json.loads(line) for line in out.getvalue().splitlines()]
    assert (done, total) == (8, 8)
    assert rows[0]["site"] == "clean" and rows[0]["kind"] == "framework" and rows[0]["energy"] == -37.0
    assert [r["site"] for r in rows[1:]] == ["H 1", "H 2", "H 3", "H 4"]
    assert all(r["E_bind"] == pytest.approx(-41.0 + 37.0 + 5.0) and r["slab_drop_eV"] == 0 for r in rows[1:])
    assert screening.binding_summary(rows)["zeolite:test"]["binding"]["NH3"]["kept"] == 4


# ---------------------------------------------------------------- materials and elements

def test_chemical_systems_cover_every_mix_up_to_the_limit():
    assert materials.chemical_systems(["Pd", "Cu"], 2, False) == ["Cu", "Pd", "Cu-Pd"]
    assert materials.chemical_systems(["Cu", "O"], 1, True) == ["Cu-O"]
    assert len(materials.chemical_systems(["Cu", "Ni", "Fe"], 3, False)) == 7


def test_element_entries_build_their_reference_crystal_without_the_api():
    entry = materials.element_entry("Ni")
    atoms = materials.atoms_of(entry)
    assert entry["id"] == "element:Ni" and set(atoms.get_chemical_symbols()) == {"Ni"}
    assert "_cell_length_a" in materials.cif_text(atoms)
    assert all(materials.atoms_of(materials.element_entry(e)) is not None for e in elements.METALS)


def test_supply_labels_follow_the_scarcest_element():
    assert elements.supply_label(["Pt"]) == "Precious metal"
    assert elements.supply_label(["Fe", "O"]) == "Abundant"
    assert elements.supply_label(["Cu", "Pd"]) == "Precious metal"


# ---------------------------------------------------------------- predictions

def _screen(pack, bindings: dict[str, float], t_design: float, temps: list[float] | None = None) -> Path:
    """A finished screen with given O binding energies; the first material is the reference."""
    entries = [{**materials.element_entry(sym), **({"is_reference": True} if i == 0 else {})}
               for i, sym in enumerate(bindings)]
    step = screening.design_step(pack)
    temps = temps or [t - 273.15 for t in step.kelvin()]
    cond = {"temperatures_C": temps, "pressure_bar": step.pressure_kPa / 100, "ghsv_per_h": step.ghsv_per_h,
            "feed": dict(step.feed)}
    folder = screening.create_screen(pack, "test screen", cond, entries, "quick", "oc20", t_design)
    lines = []
    for sym, e in bindings.items():
        mid = f"element:{sym}"
        lines.append({"material": mid, "adsorbate": "*", "site": "clean", "facet": [0, 0, 1], "n_atoms": 36,
                      "energy": -100.0, "structure": "structures/x.cif"})
        for ads in pack.spec.screening.descriptor:
            lines.append({"material": mid, "adsorbate": ads, "site": "hollow 1", "E_bind": e, "anomaly": "",
                          "n_ads": 1, "converged": True, "structure": "structures/x.cif"})
            lines.append({"material": mid, "adsorbate": ads, "site": "top", "E_bind": e + 2.0, "anomaly": "desorbed",
                          "n_ads": 1, "converged": True, "structure": "structures/y.cif"})
    (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    slab = uma_worker.build_slab(uma_worker.bulk_from(entries[0]), (1, 1, 1))
    (folder / "structures").mkdir(exist_ok=True)
    for name in ("x.cif", "y.cif"):
        (folder / "structures" / name).write_text(materials.cif_text(slab))
    (folder / "status.json").write_text(json.dumps({"state": "finished", "done": 1, "total": 1, "started": 0,
                                                    "finished": 60}))
    return folder


def test_predictions_rank_by_distance_from_the_volcano_top():
    pack = load_pack(CO)
    spec = pack.spec.screening
    ref = 1.5
    top = ref + spec.optimum_shift_eV
    folder = _screen(pack, {"Pt": ref, "Pd": top, "Au": ref + 1.5, "Ru": ref - 1.0}, t_design=200.0)
    preds, curves = screening.predict(pack, screening.read_job(folder), screening.read_results(folder))
    order = [p.formula for p in preds]
    assert order[0] == "Pd" and order[-1] in ("Au", "Ru")
    best = preds[0]
    assert best.rate_ratio > 1 and best.rate_range[0] <= best.rate_ratio <= best.rate_range[1]
    ref_pred = next(p for p in preds if p.is_reference)
    assert ref_pred.rate_ratio == pytest.approx(1.0)
    # the desorbed spot is kept on file and left out of the binding energy
    assert ref_pred.binding["O"]["tried"] == 2 and ref_pred.binding["O"]["E"] == pytest.approx(ref)
    assert set(curves["materials"]) == {p.material for p in preds}
    assert all(m <= b + 1e-6 for m, b in zip(curves["materials"][best.material]["mid"], curves["best_possible"]))


def test_the_volcano_top_follows_the_free_energy_of_its_gas():
    import cantera as ct

    pack = load_pack(CO)
    assert pack.spec.screening.peak_gas == {"O2": 0.5}
    step = screening.design_step(pack)
    t0 = pack.network.t_ref
    moves = screening.peak_moves(pack, step, [t0 - 100, t0, t0 + 100])
    assert moves[1] == pytest.approx(0.0, abs=1e-12)  # where the pack sets it: its reference temperature and feed
    gas = ct.Solution("gri30.yaml")
    p_o2 = step.feed_fractions(pack.network.species, pack.spec.water)[pack.network.species.index("O2")]
    p_o2 *= step.pressure_kPa * 1000 / ct.one_atm

    def mu_half_o2(t):  # eV per O atom
        gas.TP = t, ct.one_atm
        return 0.5 * screening.K_B * t * (gas.standard_gibbs_RT[gas.species_index("O2")] + np.log(p_o2))

    assert moves[2] == pytest.approx(mu_half_o2(t0 + 100) - mu_half_o2(t0), rel=1e-6)
    assert moves[2] < -0.05 < 0.05 < moves[0]  # heat moves the top toward stronger binding, cooling toward weaker
    leaner = step.model_copy(update={"feed": {**step.feed, "O2": "0.5 %"}})
    assert screening.peak_moves(pack, leaner, [t0])[0] < 0  # less oxygen: the surface empties more easily


def test_the_leader_can_change_with_temperature_and_the_anchor_keeps_the_fixed_volcano():
    pack = load_pack(CO)
    spec = pack.spec.screening
    ref = 1.5
    top = ref + spec.optimum_shift_eV
    anchor = pack.network.t_ref - 273.15
    folder = _screen(pack, {"Pt": ref, "Ag": top, "Pd": top - 0.4}, t_design=anchor,
                     temps=list(np.linspace(100.0, 500.0, 9)))
    preds, curves = screening.predict(pack, screening.read_job(folder), screening.read_results(folder))
    by = {p.formula: p for p in preds}
    # at the anchor the top sits where the pack puts it, so the ranking matches the fixed volcano
    assert by["Pd"].shift == pytest.approx(spec.barrier_shift(top - 0.4, ref))
    assert [p.formula for p in preds][0] == "Ag"
    # the stronger binder takes over as the top moves toward it with heat
    assert curves["leaders"][0] == by["Ag"].material and curves["leaders"][-1] == by["Pd"].material
    spans = screening.leader_spans(curves, {p.material: p.formula for p in preds})
    # the top sweeps from Ag past Pt to Pd, so the lead passes in order of binding strength
    assert [s[0] for s in spans] == ["Ag", "Pt", "Pd"] and spans[0][1] == 100.0 and spans[-1][2] == 500.0
    assert curves["peak"][0] > top > curves["peak"][-1]
    # the likely range holds the model and widens away from the anchor
    hot = curves["materials"][by["Pd"].material]
    assert all(lo <= mid <= hi for lo, mid, hi in zip(hot["lo"], hot["mid"], hot["hi"]))


def test_the_likely_range_holds_every_case_of_the_error_bar_and_the_peak_move_exactly():
    # brute force over the descriptor's error bar and every share of the peak's move in the span: the grid stays
    # inside the exact ends and reaches each within the grid's own spacing
    n = 41
    for pid in SCREENABLE:
        spec = load_pack(pid).spec.screening
        s = spec.sigma_descriptor
        for ref in (0.0, 1.5):
            for d in (ref - 0.6, ref - 0.1, ref + spec.optimum_shift_eV, ref + 0.5):
                for move in (-0.3, 0.0, 0.2):
                    model, lo, hi = screening.shift_cases(load_pack(pid), d, ref, move)
                    grid = [spec.barrier_shift(x, ref, f * move) for x in np.linspace(d - s, d + s, n)
                            for f in np.linspace(*screening.PEAK_FOLLOW_SPAN, n)]
                    tol = (max(spec.slope_weak, spec.slope_strong) * 2 * s
                           + (spec.slope_weak + spec.slope_strong) * abs(move) * np.ptp(screening.PEAK_FOLLOW_SPAN)) / (n - 1)
                    assert lo - 1e-12 <= min(grid) <= lo + tol and hi - tol <= max(grid) <= hi + 1e-12
                    assert lo <= model <= hi


def test_a_barrier_shift_per_temperature_matches_one_shift_for_all():
    pack = load_pack(CO)
    step = screening.design_step(pack)
    n = len(step.kelvin())
    one, each = (screening.metric_curves(pack, step, [screening.shifted_params(pack, s)], pack.spec.screening.metric)
                 for s in (0.1, np.full(n, 0.1)))
    assert np.allclose(one, each)


def test_screens_list_newest_first_and_report_their_state():
    pack = load_pack(SCREENABLE[0])
    folder = _screen(pack, {"Pt": 1.5, "Pd": 1.4}, t_design=300.0)
    assert screening.list_screens()[0] == folder
    assert screening.read_status(folder)["state"] == "finished"


def test_the_shipped_example_lands_in_an_empty_screens_folder_and_ranks_like_the_app():
    screening.add_examples()
    screens = screening.list_screens()
    assert len(screens) == len(list(screening.EXAMPLES.iterdir())) >= 1
    folder = screens[0]
    job = screening.read_job(folder)
    preds, _ = screening.predict(load_pack(job["request"]["pack"]), job, screening.read_results(folder))
    assert all(p.complete for p in preds)
    assert next(p for p in preds if p.is_reference).rate_ratio == pytest.approx(1.0)
    assert [p.formula for p in preds[:2]] == ["Pt", "PdPt"]  # platinum leads; the alloy sits within its error bar
    screening.add_examples()  # a folder that already holds screens stays as it is
    assert screening.list_screens() == screens


# ---------------------------------------------------------------- the reaction builder

def _new_reaction(tmp_path, name="Dry reforming at home", **over):
    args = dict(name=name, category="Hydrogen and energy", summary="Turns methane and CO2 into syngas.",
                wanted="CH4 + CO2 <=> 2 CO + 2 H2", side=["CO2 + H2 <=> CO + H2O"],
                metric={"kind": "conversion", "species": ["CH4"]}, descriptor=["C", "O"], reference_formula="Ni",
                reference_name="Nickel", suggested=["Ni", "Co", "Ru"], feed={"CH4": "20 %", "CO2": "20 %", "N2": "balance"},
                pressure_bar=1.0, ghsv_per_h=20000, temperatures_C=list(np.linspace(400, 900, 11)), half_C=650.0,
                root=tmp_path)
    return builder.create_reaction(**{**args, **over})


def test_builder_writes_a_calibrated_pack_that_loads(tmp_path):
    pid = _new_reaction(tmp_path)
    pack = load_pack(tmp_path / pid)
    assert pack.spec.screening.descriptor == {"C": 0.5, "O": 0.5}
    # carbon from the methane it feeds (leaving hydrogen), oxygen from its CO2 (leaving CO)
    assert pack.spec.screening.peak_gas == {"CH4": 0.5, "H2": -1.0, "CO2": 0.5, "CO": -0.5}
    assert set(pack.spec.species) == {"CH4", "CO2", "CO", "H2", "H2O", "N2"}
    step = pack.protocols["design-v1"].steps[0]
    one = step.model_copy(update={"temperatures_C": [650.0], "temperatures_F": None, "temperatures_K": None})
    best, ref = screening.metric_curves(pack, one, [screening.fast_params(pack), pack.network.parameters()], "main")[:, 0]
    assert ref == pytest.approx(0.5 * best, rel=0.05)
    builder.delete_reaction(pid, root=tmp_path)
    assert not (tmp_path / pid).exists()


def test_builder_reports_plain_problems_and_leaves_no_folder(tmp_path):
    with pytest.raises(ValueError, match="outside the app's gas data"):
        _new_reaction(tmp_path, name="Benzene", wanted="C6H6 + 3 H2 -> C6H12", side=[],
                      metric={"kind": "conversion", "species": ["C6H6"]}, feed={"C6H6": "1 %", "N2": "balance"})
    with pytest.raises(ValueError, match="halfway temperature"):
        _new_reaction(tmp_path, name="Too hot", half_C=1200.0)
    with pytest.raises(ValueError, match="already exists"):
        _new_reaction(tmp_path, name=SCREENABLE[0].replace("_", " "))  # the same id as a library reaction
    assert list(tmp_path.iterdir()) == []


def test_descriptor_suggestions_follow_the_chemistry():
    assert builder.suggest_descriptor("2 NH3 <=> N2 + 3 H2") == ["N"]
    assert builder.suggest_descriptor("2 CO + O2 -> 2 CO2") == ["O"]
    assert builder.suggest_descriptor("CO2 + 4 H2 <=> CH4 + 2 H2O") == ["C", "O"]
    assert builder.suggest_descriptor("C2H4 + H2 -> C2H6") == ["C"]


# ---------------------------------------------------------------- pages

def test_results_page_shows_a_finished_screen():
    pack = load_pack(CO)
    _screen(pack, {"Pt": 1.5, "Pd": 1.7, "Au": 2.8}, t_design=200.0)
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Best candidate" in m.value for m in at.markdown)
    assert len(at.dataframe) >= 1


def test_results_page_shows_the_example_with_its_reference_leading_and_no_rounding_alarm():
    screening.add_examples()
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("the catalyst to beat, leads" in m.value for m in at.markdown)
    assert len(at.warning) == 0  # its slowest curves touch zero to within 1e-16


def test_design_page_adds_metals_to_the_list():
    at = AppTest.from_file(str(VIEWS / "design.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Add these metals").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["design_candidates"]) > 0


def test_a_honeycomb_caps_each_rate_at_the_gas_reaching_the_walls():
    """On a coated honeycomb a first-order reaction runs at k km / (k + km): the reactor matches the exact plug-flow
    answer with that rate, and a faster catalyst gains less than it would as powder."""
    import math

    pack = load_pack("co_oxidation")
    step = screening.design_step(pack)
    assert screening.wall_transfer(pack, step, None) is None
    assert screening.wall_transfer(pack, step, {"form": "powder"}) is None
    km = screening.wall_transfer(pack, step, screening.MONOLITH)
    assert km.shape == (len(step.kelvin()), len(pack.network.pathways)) and (km > 0).all()
    params = pack.network.parameters()
    powder, mono = screening.metric_curves(pack, step, [params], "co_conversion"), \
        screening.metric_curves(pack, step, [params], "co_conversion", screening.MONOLITH)
    assert (mono <= powder + 1e-9).all() and (mono < powder - 1e-3).any()
    fast = screening.fast_params(pack)
    assert screening.metric_curves(pack, step, [fast], "co_conversion", screening.MONOLITH).min() \
        <= screening.metric_curves(pack, step, [fast], "co_conversion").min()
    # the exact answer for CO + 1/2 O2 -> CO2, first order in CO, with the film in series and the gas shrinking
    pw = pack.spec.screening.pathways[0]
    t = np.array(step.kelvin())
    y = step.feed_fractions(pack.network.species, pack.spec.water)
    eps = -0.5 * y[pack.network.index("CO")]
    k = np.exp(params[f"{pw}.ln_k_ref"] - params[f"{pw}.Ea_kJ"] * 1000 / 8.314462618 * (1 / t - 1 / pack.network.t_ref))
    k_wall = km[:, [p.id for p in pack.network.pathways].index(pw)]
    q = step.ghsv_per_h / 3600 * t / 273.15 * 101325 / (step.pressure_kPa * 1000)
    for kt, x in zip(k * k_wall / (k + k_wall) / q, mono[0]):
        lo, hi = 0.0, 1.0 - 1e-15
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if -(1 + eps) * math.log(1 - mid) - eps * mid < kt else (lo, mid)
        assert x == pytest.approx(0.5 * (lo + hi), abs=1e-3)


def test_design_page_advanced_mode_sets_the_catalyst_form():
    at = AppTest.from_file(str(VIEWS / "design.py"), default_timeout=120).run()
    assert not any(r.label == "Catalyst form" for r in at.radio)
    at.toggle(key="design_advanced").set_value(True).run()
    form = next(r for r in at.radio if r.label == "Catalyst form")
    assert form.value == "powder"
    form.set_value("monolith").run()
    assert not at.exception, at.exception
    assert any("Open channel 1.10 mm across" in c.value for c in at.caption)
    assert any("channel walls sets the limit" in c.value for c in at.caption)
    next(n for n in at.number_input if n.label == "Wall and coat (mm)").set_value(1.4).run()
    assert any("fill the whole cell" in e.value for e in at.error)


def test_design_page_advanced_mode_adds_faces_and_model_choices():
    """Advanced adds faces, the UMA mode and the catalyst to beat; zeolites stay out of the app until a UMA mode
    matches published DFT for them (experiments/zeolite_check/)."""
    at = AppTest.from_file(str(VIEWS / "design.py"), default_timeout=120).run()
    at.toggle(key="design_advanced").set_value(True).run()
    assert not at.exception, at.exception
    assert "Zeolites" not in [t.label for t in at.tabs]
    assert list(at.selectbox(key="design_task_adv").options) == ["Catalysis (trained on surfaces with molecules)",
                                                                 "Materials (trained on bulk crystals)"]
    next(b for b in at.button if b.label == "Add these metals").click().run()
    assert not at.exception, at.exception
    assert any("Face" in d.value.columns for d in at.dataframe)  # the list now takes a face for each crystal


def test_results_page_names_a_chosen_catalyst_to_beat_and_shows_its_zeolite_site():
    pack = load_pack(CO)
    z = _cha()
    variants, site = zeolites.build(z, "M2+", "Cu")
    zeo = {"id": "zeolite:mp-aaaabamv:M2+:Cu", "label": "Cu2+ Chabazite (CHA), Si/Al 5", "formula": "Cu2+-CHA",
           "kind": "framework", "elements": ["Al", "Cu", "O", "Si"], "source": "test", "e_hull": 0.02,
           "lattice": z.cell.array.tolist(), "site": site, "is_reference": True, "ref_name": "Cu2+-CHA",
           "variants": [{"species": variants[0].get_chemical_symbols(),
                         "frac": variants[0].get_scaled_positions().tolist()}]}
    step = screening.design_step(pack)
    cond = {"temperatures_C": [t - 273.15 for t in step.kelvin()], "pressure_bar": step.pressure_kPa / 100,
            "ghsv_per_h": step.ghsv_per_h, "feed": dict(step.feed), "shape": screening.MONOLITH}
    folder = screening.create_screen(pack, "zeolite screen", cond, [zeo, materials.element_entry("Pt")], "quick",
                                     "odac", 200.0)
    assert "variants" not in screening.read_job(folder)["request"]["materials"][0]
    lines = [{"material": zeo["id"], "adsorbate": "*", "site": "clean", "kind": "framework", "n_atoms": 37,
              "energy": -300.0, "structure": "structures/z.cif"},
             {"material": zeo["id"], "adsorbate": "O", "site": "Cu", "E_bind": 1.2, "anomaly": "", "n_ads": 1,
              "converged": True, "structure": "structures/z.cif"},
             {"material": "element:Pt", "adsorbate": "*", "site": "clean", "facet": [0, 0, 1], "n_atoms": 36,
              "energy": -100.0, "structure": "structures/z.cif"},
             {"material": "element:Pt", "adsorbate": "O", "site": "hollow 1", "E_bind": 1.5, "anomaly": "",
              "n_ads": 1, "converged": True, "structure": "structures/z.cif"}]
    (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    (folder / "structures").mkdir()
    (folder / "structures" / "z.cif").write_text(materials.cif_text(variants[0]))
    (folder / "status.json").write_text(json.dumps({"state": "finished", "done": 1, "total": 1, "started": 0,
                                                    "finished": 60}))
    at = AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert "Cu2+-CHA (to beat)" in list(at.dataframe[0].value["Material"])
    next(b for b in at.selectbox if b.label == "Material").set_value(zeo["id"]).run()
    assert not at.exception, at.exception
    assert any("Zeolite site" in c.value for c in at.caption)
    assert any("coated honeycomb, 400 cells per square inch" in m.value for m in at.markdown)
