import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from catalyst_engine.data.demo import build_demo, demo_packs
from catalyst_engine.data.ingest import fit_rows, ingest_run, issues, settled
from catalyst_engine.data.ledger import Ledger
from catalyst_engine.data.records import add_campaign, add_formulation, add_sample
from catalyst_engine.data.scoring import score_campaign
from catalyst_engine.data.store import Store
from catalyst_engine.data.synthetic import write_simulated_log
from catalyst_engine.packs import list_packs, load_pack
from catalyst_engine.units import k_to_f

FE = [{"material": "Fe", "role": "active", "amount": 1.5, "unit": "wt%"}]


@pytest.fixture(scope="module")
def pack():
    # Any pack with a protocol works; the tests use the first one that has a demo.
    return load_pack(demo_packs()[0])


@pytest.fixture(scope="module")
def demo(pack, tmp_path_factory):
    return build_demo(pack, tmp_path_factory.mktemp("demo") / "simulated")


def campaign_store(pack, root, kind="simulated"):
    store = Store.create(root, kind)
    proto = next(iter(pack.protocols.values()))
    step = proto.steps[0]
    ref = add_formulation(store, "reference", FE, role="reference")
    cand = add_formulation(store, "candidate", FE)
    t = round(k_to_f(step.kelvin()[-1]))
    metric = next(m for m, spec in pack.spec.metrics.items() if spec.target is not None)
    camp = add_campaign(store, pack, "test", proto.id, "rig-1", ref["id"],
                        {"metric": metric, "step": step.id, "temperatures_F": [t], "rule": "beats_reference"})
    return store, proto, ref, cand, camp


def test_every_table_is_append_only(tmp_path):
    store = Store.create(tmp_path / "s", "lab")
    rec = add_formulation(store, "reference", FE, role="reference")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.con.execute("UPDATE formulations SET name = 'other' WHERE id = ?", (rec["id"],))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.con.execute("DELETE FROM formulations")
    fixed = add_formulation(store, "reference, renamed", FE, role="reference", supersedes=rec["id"])
    assert [r["id"] for r in store.current("formulations")] == [fixed["id"]]
    assert len(store.rows("formulations")) == 2  # the corrected record stays on file


def test_lab_store_refuses_simulated_files(pack, tmp_path):
    store = Store.create(tmp_path / "lab", "lab")
    path = write_simulated_log(tmp_path / "sim.csv", pack, next(iter(pack.protocols.values())))
    with pytest.raises(ValueError, match="simulated store"):
        store.add_raw_file(path)


def test_raw_files_are_content_addressed_and_read_only(pack, tmp_path):
    store = Store.create(tmp_path / "s", "simulated")
    path = write_simulated_log(tmp_path / "a.csv", pack, next(iter(pack.protocols.values())))
    digest = store.add_raw_file(path)
    assert store.add_raw_file(path) == digest and len(store.rows("raw_files")) == 1
    assert store.verify_raw(digest)
    assert not store.raw_path(digest).stat().st_mode & 0o222


def test_ledger_detects_an_edited_entry(tmp_path):
    store, _, _, cand, camp = campaign_store(load_pack(demo_packs()[0]), tmp_path / "s")
    ledger = Ledger(store)
    ledger.pick(camp["id"], cand["id"], "model")
    ledger.append("note", {"text": "second entry"})
    assert ledger.verify().ok and ledger.verify().entries == 2
    # Someone with direct database access drops the guard and rewrites the pick's arm.
    store.con.execute("DROP TRIGGER ledger_no_update")
    store.con.execute("UPDATE ledger SET payload = replace(payload, 'model', 'expert') WHERE seq = 1")
    check = ledger.verify()
    assert not check.ok and "entry 1 changed" in check.problem


def test_points_recover_the_steady_state(pack, tmp_path):
    store, proto, ref, _, camp = campaign_store(pack, tmp_path / "s")
    sample = add_sample(store, ref["id"], "monolith core")
    path = write_simulated_log(tmp_path / "run.csv", pack, proto, seed=3, rig="rig-1", sample=sample["id"])
    run = ingest_run(store, pack, path, campaign_id=camp["id"])
    assert run["status"] == "complete"
    truth = {(r["step"], round(k_to_f(r["T_K"]), 1)): r for r in pack.simulate(None, proto.id)}
    points = store.points(run["id"])
    assert len(points) == len(truth)
    an = proto.analyzer
    for p in points:
        assert settled(p) and not [f for f in issues(p) if not f.startswith("inlet")], p["flags"]
        for s, v in p["outlet"].items():
            want = truth[(p["step"], p["setpoint_F"])][f"out_{s}_ppm"]
            sigma = np.hypot(an.sigma_abs_ppm, an.sigma_rel * want) / np.sqrt(p["n"])
            assert abs(v - want) < 5 * sigma + 0.05, (p["step"], p["setpoint_F"], s, v, want)
    assert len(fit_rows(pack, proto, points)) == len(points)


def test_failed_and_unsettled_runs_stay_on_file(pack, tmp_path):
    store, proto, ref, _, camp = campaign_store(pack, tmp_path / "s")
    sample = add_sample(store, ref["id"], "monolith core")
    short = write_simulated_log(tmp_path / "short.csv", pack, proto, rig="rig-1", sample=sample["id"], stop_after_points=2)
    failed = ingest_run(store, pack, short, campaign_id=camp["id"])
    assert failed["status"] == "failed"
    assert any(f.startswith("missing") for f in store.latest_derivation(failed["id"])["run_flags"])
    step = proto.steps[-1]
    t = round(k_to_f(step.kelvin()[-1]), 1)
    cut = write_simulated_log(tmp_path / "cut.csv", pack, proto, rig="rig-1", sample=sample["id"],
                              short_holds={(step.id, t): 120.0})
    run = ingest_run(store, pack, cut, campaign_id=camp["id"])
    point = next(p for p in store.points(run["id"]) if p["step"] == step.id and p["setpoint_F"] == t)
    assert not settled(point)
    assert {r["id"] for r in store.rows("runs")} == {failed["id"], run["id"]}


def test_campaign_runs_share_one_protocol_version_and_rig(pack, tmp_path):
    store, proto, ref, _, camp = campaign_store(pack, tmp_path / "s")
    sample = add_sample(store, ref["id"], "monolith core")
    path = write_simulated_log(tmp_path / "run.csv", pack, proto, sample=sample["id"], rig="rig-2")
    with pytest.raises(ValueError, match="runs on rig-1"):
        ingest_run(store, pack, path, campaign_id=camp["id"])
    edited = proto.model_copy(update={"steps": [proto.steps[0].model_copy(update={"ghsv_per_h": 12345.0}),
                                                *proto.steps[1:]]})
    changed = replace(pack, protocols={**pack.protocols, proto.id: edited})
    with pytest.raises(ValueError, match="different version"):
        ingest_run(store, changed, path, campaign_id=camp["id"], rig="rig-1")


def test_runs_logged_before_the_pick_never_count(pack, tmp_path):
    store, proto, ref, cand, camp = campaign_store(pack, tmp_path / "s")
    ledger = Ledger(store)
    for f in (ref, cand):
        sample = add_sample(store, f["id"], "monolith core")
        path = write_simulated_log(tmp_path / f"{f['id']}.csv", pack, proto, rig="rig-1", sample=sample["id"])
        ingest_run(store, pack, path, campaign_id=camp["id"])
    ledger.pick(camp["id"], cand["id"], "model")
    score = score_campaign(store, pack, camp["id"])
    (result,) = score.picks
    assert result.state == "awaiting test" and len(result.early_runs) == 1
    assert score.blinded and result.arm is None and not score.arms


def test_demo_scores_each_arm_from_the_log(demo, pack):
    camp = demo.rows("campaigns")[0]
    score = score_campaign(demo, pack, camp["id"])
    assert score.simulated and not score.blinded and score.ready
    assert len(score.reference_runs) == 2
    picked = {a.arm: a.picked for a in score.arms}
    assert sum(picked.values()) == len(Ledger(demo).picks(camp["id"]))
    assert all(p.state in ("hit", "miss") for p in score.picks)
    assert any(p.failed_runs for p in score.picks) and any(p.unsettled_runs for p in score.picks)
    assert score.predictions and score.coverage is not None
    assert Ledger(demo).verify().ok
    assert all(demo.verify_raw(r["sha256"]) for r in demo.rows("raw_files"))


def test_anchor_records_the_lab_log_only(demo, tmp_path):
    with pytest.raises(ValueError, match="lab log"):
        Ledger(demo).anchor(tmp_path / "anchors.md")
    lab = Store.create(tmp_path / "lab", "lab")
    Ledger(lab).append("note", {"text": "first"})
    line = Ledger(lab).anchor(tmp_path / "anchors.md")
    assert Ledger(lab).head() in line and (tmp_path / "anchors.md").read_text().endswith(line)


def test_every_demo_pack_builds(tmp_path):
    for pid in demo_packs():
        assert pid in list_packs()
        store = build_demo(load_pack(pid), tmp_path / pid)
        assert store.simulated and store.counts()["runs"] > 0


def test_stored_points_feed_the_fitter(pack, tmp_path):
    from catalyst_engine.fit.kinetics import fit_kinetics, observations_from_table

    store, proto, ref, _, camp = campaign_store(pack, tmp_path / "s")
    sample = add_sample(store, ref["id"], "monolith core")
    name = pack.network.fitted[0]
    true = pack.network.defaults[name] + 0.3
    path = write_simulated_log(tmp_path / "run.csv", pack, proto, {name: true}, seed=7, rig="rig-1", sample=sample["id"])
    run = ingest_run(store, pack, path, campaign_id=camp["id"])
    obs = observations_from_table(pack, proto, fit_rows(pack, proto, store.points(run["id"])))
    res = fit_kinetics(pack, proto, obs, fit=[name], n_starts=1)
    assert abs(res.params[name] - true) < 4 * res.std[name] + 0.02


def test_inlet_noise_on_an_unfed_species_leaves_conversion_undefined(pack, tmp_path):
    # A step that feeds none of a conversion metric's species: the analyzer's inlet reading is noise around zero,
    # so the conversion there is undefined rather than a large negative number.
    from catalyst_engine.data.ingest import point_metrics

    store, proto, ref, _, camp = campaign_store(pack, tmp_path / "s")
    sample = add_sample(store, ref["id"], "monolith core")
    path = write_simulated_log(tmp_path / "run.csv", pack, proto, seed=5, rig="rig-1", sample=sample["id"])
    run = ingest_run(store, pack, path, campaign_id=camp["id"])
    points = store.points(run["id"])
    values = point_metrics(pack, proto, points)
    for mid, spec in pack.spec.metrics.items():
        if spec.kind != "conversion":
            continue
        for p, v in zip(points, values):
            feed = proto.step(p["step"]).feed_fractions(pack.network.species, pack.spec.water)
            fed = sum(feed[pack.network.index(s)] for s in spec.species)
            assert np.isnan(v[mid]) if fed == 0 else 0.0 <= v[mid] <= 1.0, (p["step"], p["setpoint_F"], v[mid])
