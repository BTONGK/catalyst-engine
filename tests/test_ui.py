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
