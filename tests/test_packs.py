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
