"""The reaction check behind on_design_page: its exact p-value, the link between each verdict and the pages a
reaction appears on, and the Methodology tabs that quote its numbers. Reads the committed results; no atomistic
model and no network."""

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from catalyst_engine.design import electro
from catalyst_engine.packs import list_packs, load_pack

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "experiments" / "reaction_check"
VIEWS = REPO / "src" / "catalyst_engine" / "ui" / "views"


def _check_module():
    spec = importlib.util.spec_from_file_location("reaction_check", CHECK / "reaction_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary() -> dict:
    return json.loads((CHECK / "results" / "summary.json").read_text())


@pytest.mark.parametrize("seed", range(6))
def test_the_exact_p_value_counts_every_ordering(seed):
    """The dynamic program's p-value equals the share of all n! orderings of the second ranking that agree with the
    first at least as well, ties included, counted one by one."""
    rc = _check_module()
    rng = np.random.default_rng(seed)
    n = 5 + seed % 3
    a = rc.ranks(list(rng.integers(0, 4 if seed % 2 else 99, n)), True)  # odd seeds carry ties
    b = rc.ranks(list(rng.normal(size=n)), True)
    rho, p = rc.spearman_exact(a, b)
    observed = np.rint(2 * a) @ np.rint(2 * b)
    agree = sum(np.rint(2 * a) @ np.rint(2 * np.array(perm)) >= observed - 1e-9 for perm in itertools.permutations(b))
    assert p == pytest.approx(agree / math.factorial(n), rel=1e-12)
    assert rho == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_every_thermal_verdict_sets_the_pages_its_reaction_appears_on():
    """A pack scored against its primary measured table sits on the Design page exactly when its check passes."""
    scored = [r for r in _summary()["thermal"] if r["role"] == "primary"]
    assert len(scored) >= 5
    for row in scored:
        screening = load_pack(row["pack"]).spec.screening
        assert screening.on_design_page == row["passes"], row["pack"]
    # a passing reaction meets every part of the rule written before the results
    rc = _check_module()
    for row in scored:
        if row["passes"]:
            assert row["n"] >= rc.MIN_METALS and row["rho"] >= rc.MIN_RHO and row["p"] <= rc.MAX_P
            assert row["best_in_top"]


def test_every_electro_verdict_sets_the_pages_its_reaction_appears_on():
    """Each newer electrochemical reaction joins the Design page exactly when its check passes; the rest keep to the
    playground with a note saying what they wait for."""
    rows = _summary()["electro_reactions"]
    assert {r["reaction"] for r in rows} <= set(electro.list_electro())
    for row in rows:
        spec = electro.load_electro(row["reaction"])
        assert spec.on_design_page == row["passes"], row["reaction"]
        assert spec.on_design_page or spec.design_note
        if row["passes"]:
            assert not row["intermediates_failing"] and not row["corrections_missing"] and row["ranking_passes"]


def test_the_methodology_tabs_quote_the_check_and_number_their_sections():
    """Both Methodology tabs fill every number from their own summary plus the reaction check's (`also`), and point
    at their sections by the numbers the page shows."""
    from catalyst_engine.ui import common

    for kind in ("methodology", "electro"):
        (report,) = common.experiment_reports(kind).values()
        assert "../reaction_check/results/summary.json" in report["also"]
        assert "rc_run_metals" in report["data"]["values"]  # merged from the reaction check
    at = AppTest.from_file(str(VIEWS / "electro_manual.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    shown = [m.value for m in at.markdown]
    heads = ["#### 1. What the app does when it loads an electrochemical reaction", "#### 2. The method, step by step",
             "#### 3. Each reaction against published calculations", "#### 6. What this section leaves out, and why",
             "#### 8. Reproduce this page"]
    at_ = [shown.index(h) for h in heads]  # every one shown, in page order
    assert at_ == sorted(at_)
    text = " ".join(m.value for m in at.markdown)
    assert "(section 3)" in text and "{" not in text.replace("{\\rm", "")
    at = AppTest.from_file(str(VIEWS / "manual.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("What the app does when it loads a reaction package" in m.value for m in at.markdown)
    assert any("Each reaction against published measurements" in m.value for m in at.markdown)
