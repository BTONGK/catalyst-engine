"""Build the simulated demo store from a pack's `demo.yaml`, so every part of the data platform has data to show.

The demo walks one campaign end to end: catalysts and samples, picks and predictions logged before any run,
simulated analyzer logs ingested like real ones (one run fails and is retested, one hold ends early), then the
arms are revealed and scored. All of it lives in the simulated store.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from catalyst_engine.data.ingest import ingest_run
from catalyst_engine.data.ledger import Ledger
from catalyst_engine.data.records import add_campaign, add_formulation, add_sample
from catalyst_engine.data.store import Store, store_path
from catalyst_engine.data.synthetic import write_simulated_log
from catalyst_engine.packs import Pack, list_packs, load_pack, read_yaml
from catalyst_engine.units import f_to_k


def demo_packs() -> list[str]:
    return [p for p in list_packs() if (load_pack(p).path / "demo.yaml").exists()]


def _shifted(pack: Pack, shifts: dict[str, float] | None) -> dict[str, float]:
    params = dict(pack.network.defaults)
    for name, delta in (shifts or {}).items():
        if name not in params:
            raise KeyError(f"demo shifts unknown parameter {name!r}")
        params[name] += float(delta)
    return params


def _prediction(pack: Pack, protocol_id: str, params: dict[str, float], metric: str, step: str,
                temperature_F: float) -> float:
    rows = pack.simulate(params, protocol_id)
    t_k = f_to_k(temperature_F)
    return next(r[metric] for r in rows if r["step"] == step and abs(r["T_K"] - t_k) < 0.01)


def build_demo(pack: Pack, root: Path | None = None) -> Store:
    spec = read_yaml(pack.path / "demo.yaml")
    root = Path(root or store_path("simulated"))
    if (root / "store.db").exists():
        if Store(root).kind != "simulated":
            raise ValueError(f"{root} holds a lab store; the demo only replaces a simulated store")
        shutil.rmtree(root)
    store = Store.create(root, "simulated")
    ledger = Ledger(store)

    camp = spec["campaign"]
    ref = spec["reference"]
    ref_f = add_formulation(store, ref["name"], ref["components"], role="reference",
                            preparation=ref.get("preparation", ""), notes=ref.get("notes", ""))
    campaign = add_campaign(store, pack, camp["name"], camp["protocol"], camp["rig"], ref_f["id"], camp["hit_rule"],
                            notes=camp.get("notes", ""))
    rule = campaign["hit_rule"]
    model_name = spec.get("model", "demo model")

    candidates = []
    for c in spec["candidates"]:
        f = add_formulation(store, c["name"], c["components"], preparation=c.get("preparation", ""),
                            notes=c.get("notes", ""))
        candidates.append((c, f))
    for c, f in candidates:
        ledger.pick(campaign["id"], f["id"], c["arm"], c.get("rationale", ""))
    for c, f in candidates:
        if "believed" not in c:
            continue
        believed = _shifted(pack, c["believed"])
        for t in rule["temperatures_F"]:
            v = _prediction(pack, camp["protocol"], believed, rule["metric"], rule["step"], t)
            half = float(c.get("interval", 0.05))
            ledger.predict(campaign["id"], f["id"], rule["metric"], rule["step"], t, v, v - half, v + half, model_name)

    sample_kw = spec.get("sample", {"form": "monolith core"})
    protocol = pack.protocol(camp["protocol"])
    seed = int(spec.get("seed", 0))
    with tempfile.TemporaryDirectory() as tmp:
        def run(formulation: dict, params: dict[str, float], notes: str = "", **log_kw) -> None:
            nonlocal seed
            sample = add_sample(store, formulation["id"], **sample_kw)
            seed += 1
            path = write_simulated_log(Path(tmp) / f"demo-{sample['id']}.csv", pack, protocol, params, seed=seed,
                                       rig=camp["rig"], sample=sample["id"], **log_kw)
            ingest_run(store, pack, path, campaign_id=campaign["id"], notes=notes)

        ref_params = _shifted(pack, ref.get("truth"))
        run(ref_f, ref_params, "Reference run at the start of the batch.")
        for c, f in candidates:
            params = _shifted(pack, c.get("truth"))
            for extra in c.get("runs", [{}]):
                log_kw = {}
                if "stop_after_points" in extra:
                    log_kw["stop_after_points"] = int(extra["stop_after_points"])
                if "short_hold" in extra:
                    h = extra["short_hold"]
                    log_kw["short_holds"] = {(h["step"], float(h["temperature_F"])): float(h["seconds"])}
                run(f, params, extra.get("notes", ""), **log_kw)
        run(ref_f, ref_params, "Reference run at the end of the batch.")

    if spec.get("unblind", True):
        ledger.unblind(campaign["id"])
    return store
