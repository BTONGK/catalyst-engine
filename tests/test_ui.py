"""Every app page renders against the real packs, and the interactive pages respond."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

VIEWS = Path(__file__).resolve().parents[1] / "src" / "catalyst_engine" / "ui" / "views"


def _metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


@pytest.mark.parametrize("page", sorted(p.name for p in VIEWS.glob("*.py")))
def test_page_renders(page):
    at = AppTest.from_file(str(VIEWS / page), default_timeout=60).run()
    assert not at.exception, at.exception


def test_playground_rewards_less_ammonia_burn():
    at = AppTest.from_file(str(VIEWS / "playground.py"), default_timeout=60).run()
    at.selectbox(key="pg_pack").set_value("scr_hot").run()
    before = _metric(at, "Meets every target")
    for s in at.select_slider:
        if s.key.endswith(":mult") and ":nh3_ox" in s.key:
            s.set_value(0.1)
    at.run()
    assert not at.exception, at.exception
    after = _metric(at, "Meets every target")
    assert int(after.split()[0]) > int(before.split()[0])


def test_fit_lab_fits_the_demo_catalyst():
    at = AppTest.from_file(str(VIEWS / "fitlab.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Fit").click().run()
    assert not at.exception, at.exception
    n_well, n_total = _metric(at, "Measured well").split(" of ")
    assert n_well == n_total


def test_fit_lab_flags_aliasing_when_a_step_is_dropped():
    at = AppTest.from_file(str(VIEWS / "fitlab.py"), default_timeout=180).run()
    steps = at.multiselect[0]
    steps.set_value([s for s in steps.value if s != "nh3_oxidation"]).run()
    next(b for b in at.button if b.label == "Fit").click().run()
    assert not at.exception, at.exception
    assert any("confounds" in w.value for w in at.warning)


def test_lab_data_shows_the_demo_campaign():
    at = AppTest.from_file(str(VIEWS / "labdata.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert _metric(at, "Prediction log").endswith("entries")
    assert any(m.label == "Model picks" for m in at.metric)
    at.selectbox(key="lab_run").set_value("R-0003").run()  # the demo's failed run
    assert not at.exception, at.exception
    assert any("Still to run" in w.value for w in at.warning)


def test_lab_store_starts_empty_and_takes_records():
    from catalyst_engine.data.records import add_formulation
    from catalyst_engine.data.store import Store, store_path

    at = AppTest.from_file(str(VIEWS / "labdata.py"), default_timeout=120).run()
    at.segmented_control(key="lab_store").set_value("Lab store").run()
    assert not at.exception, at.exception
    if not (store_path("lab") / "store.db").exists():
        next(b for b in at.button if b.label == "Create the lab store").click().run()
    store = Store(store_path("lab"))
    add_formulation(store, "Reference catalyst", [{"material": "Fe", "amount": 1.5, "unit": "wt%"}], role="reference")
    at.run()
    assert not at.exception, at.exception
    assert _metric(at, "Catalysts") == "1" and store.kind == "lab"


def test_literature_check_shows_both_fits():
    at = AppTest.from_file(str(VIEWS / "literature.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any(m.label.endswith(", fitted") for m in at.metric)
    for control in at.segmented_control:
        control.set_value("holdout")
    at.run()
    assert not at.exception, at.exception
    assert any("held-out points" in c.value for c in at.caption)


def test_atomistic_page_keeps_to_atomistic_reports():
    from catalyst_engine.ui import common

    assert all(r.get("kind", "atomistic") == "atomistic" for r in common.experiment_reports().values())
    assert all(r["kind"] == "literature" for r in common.experiment_reports("literature").values())
