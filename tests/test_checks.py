"""The reaction check behind each reaction's verdict: its exact p-value, the link between each verdict and the
reaction's file (check, check_note), the Design pages that list every reaction with its verdict, and the Methodology
tabs that quote its numbers. Reads the committed results; no atomistic model and no network."""

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


def _rho(note: str) -> str:
    return " ".join(note.split()).replace("−", "-")


def test_every_thermal_verdict_matches_its_pack():
    """A pack scored against its primary measured table carries that table's verdict, passed or missed, and quotes
    its rho; a pack without one awaits a table. Every pack says what its check found."""
    rows = _summary()["thermal"]
    scored = {r["pack"]: r for r in rows if r["role"] == "primary"}
    assert len(scored) >= 5
    for pid in list_packs():
        screening = load_pack(pid).spec.screening
        if screening is None:
            continue
        row = scored.get(pid)
        assert screening.check == ("awaiting" if row is None else "passed" if row["passes"] else "missed"), pid
        assert screening.check_note, pid
        for r in (r for r in rows if r["pack"] == pid):  # every table the note names, its own numbers
            assert f"rho {r['rho']:.2f} over {r['n']} metals" in _rho(screening.check_note) or r["role"] == "secondary", pid
    # a passing reaction meets every part of the rule written before the results
    rc = _check_module()
    for row in scored.values():
        if row["passes"]:
            assert row["n"] >= rc.MIN_METALS and row["rho"] >= rc.MIN_RHO and row["p"] <= rc.MAX_P
            assert row["best_in_top"]


def test_every_electro_verdict_matches_its_file():
    """Each newer electrochemical reaction carries its check's verdict, quotes its ranking's rho, and marks the very
    intermediates whose corrections rest on UMA alone; the first three take their verdicts from their own checks."""
    rows = {r["reaction"]: r for r in _summary()["electro_reactions"]}
    ec = json.loads((REPO / "experiments" / "electro_check" / "results" / "summary.json").read_text())["values"]
    for rid in electro.list_electro():
        spec = electro.load_electro(rid)
        uma = {i.name for i in spec.intermediates if i.correction_from == "uma"}
        assert spec.check_note, rid
        if rid not in rows:  # checked in experiments/electro_check, with published corrections throughout
            assert rid in {"hydrogen_evolution", "oxygen_reduction", "oxygen_evolution"} and not uma, rid
            continue
        row = rows[rid]
        assert spec.check == row["check"], rid
        assert uma == set(row["corrections_missing"]), rid
        if row["rho"] is not None:
            assert f"rho {row['rho']:.2f}" in _rho(spec.check_note), rid
        if row["passes"]:
            assert not row["intermediates_failing"] and not row["corrections_missing"] and row["ranking_passes"]
    her, orr, oer = (electro.load_electro(i) for i in ("hydrogen_evolution", "oxygen_reduction", "oxygen_evolution"))
    assert her.check == orr.check == "passed" and oer.check == "missed"
    assert f"{ec['her_mae_rpbe']:.2f} eV" in her.check_note and f"rho {ec['her_spearman']:.2f}" in her.check_note
    assert f"{ec['her_n']} metals" in her.check_note and f"rho {ec['her_spearman']:.2f} over {ec['her_j0_n']}" in her.check_note
    assert f"{ec['pt_orr_limit']:.2f} V" in orr.check_note and f"rho {ec['orr_spearman_paper']:.2f}" in orr.check_note
    orr_note = " ".join(orr.check_note.split())
    assert f"OH* sits {ec['orr_oh_mae']:.2f} eV and its O* {ec['orr_o_mae']:.2f} eV" in orr_note
    assert f"over {ec['orr_n']} metals" in orr_note and f"binding {ec['orr_o_words']}" in orr_note
    assert f"{ec['oxide_omat_lo']:.2f}–{ec['oxide_omat_hi']:.2f} eV" in oer.check_note


def test_both_design_pages_list_every_reaction_with_its_verdict():
    """Every reaction with a descriptor appears on its Design page, its verdict in the list and beside the choice."""
    at = AppTest.from_file(str(VIEWS / "design.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    box = at.selectbox(key="design_pack")
    packs = [pid for pid in list_packs() if load_pack(pid).spec.screening is not None]
    assert sorted(box.options) == sorted(f"{load_pack(p).spec.name}  ·  {load_pack(p).spec.category}  ·  "
                                         f"{ {'passed': 'checked', 'missed': 'missed its check', 'awaiting': 'awaiting its check'}[load_pack(p).spec.screening.check]}"
                                         for p in packs)
    box.set_value("propane_combustion").run()
    assert not at.exception, at.exception
    assert any("Ranking missed its check" in m.value for m in at.markdown)
    assert any("rho 0.17 over 6 metals" in c.value for c in at.caption)
    at = AppTest.from_file(str(VIEWS / "electro_design.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    box = at.selectbox(key="electro_pack")
    assert len(box.options) == len(electro.list_electro())
    for rid in ("co2_to_methane", "oxygen_evolution", "h2o2_oxidation"):
        at.selectbox(key="electro_pack").set_value(rid).run()
        assert not at.exception, (rid, at.exception)
        state = electro.load_electro(rid).check
        label = {"passed": "Ranking checked", "missed": "Ranking missed its check", "awaiting": "Ranking awaits its check"}
        assert any(label[state] in m.value for m in at.markdown), rid


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
