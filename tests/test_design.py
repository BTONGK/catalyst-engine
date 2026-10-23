"""Catalyst design: the volcano, surface building, Materials Project helpers, predictions from binding energies,
the reaction builder, and the design pages. No network and no atomistic model: synthetic results stand in for the
worker's output."""

import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from catalyst_engine.design import builder, elements, materials, screening, uma_worker
from catalyst_engine.design.spec import ReferenceSpec, ScreeningSpec
from catalyst_engine.packs import PACKS_DIR, list_packs, load_pack

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

def _screen(pack, bindings: dict[str, float], t_design: float) -> Path:
    """A finished screen with given O binding energies; the first material is the reference."""
    entries = [{**materials.element_entry(sym), **({"is_reference": True} if i == 0 else {})}
               for i, sym in enumerate(bindings)]
    step = screening.design_step(pack)
    temps = [t - 273.15 for t in step.kelvin()]
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

def _new_reaction(tmp_path, name="Dry reforming", **over):
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
