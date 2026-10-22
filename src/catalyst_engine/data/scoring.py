"""Score a campaign from the prediction log and the stored runs, and nothing else.

A pick counts once: its first complete run logged after the pick whose rule points all settled. The reference value is the mean of the
reference catalyst's complete runs in the same campaign, which shares one protocol version and one rig.
Arms stay hidden until the campaign is unblinded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import fisher_exact

from catalyst_engine.data.ingest import run_points, settled
from catalyst_engine.data.ledger import ARMS, Ledger
from catalyst_engine.data.records import direction
from catalyst_engine.data.store import Store
from catalyst_engine.packs import Pack


@dataclass
class PickResult:
    formulation: str
    name: str
    arm: str | None  # None while the campaign is blinded
    pick_seq: int
    state: str  # hit | miss | awaiting test | awaiting reference | needs retest
    run: str | None = None
    values: dict[float, float | None] = field(default_factory=dict)
    failed_runs: list[str] = field(default_factory=list)
    unsettled_runs: list[str] = field(default_factory=list)  # complete, with a rule point that never settled
    early_runs: list[str] = field(default_factory=list)  # logged before the pick; never counted


@dataclass
class ArmResult:
    arm: str
    picked: int
    tested: int
    hits: int

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.tested if self.tested else None


@dataclass
class CampaignScore:
    campaign: dict
    blinded: bool
    simulated: bool
    direction: str
    reference: dict[float, float | None]
    reference_runs: list[str]
    picks: list[PickResult]
    arms: list[ArmResult]
    predictions: list[dict]
    comparisons: list[dict]  # model arm against each other arm, once unblinded

    @property
    def ready(self) -> bool:
        return all(v is not None for v in self.reference.values())

    @property
    def coverage(self) -> float | None:
        scored = [p for p in self.predictions if p["inside"] is not None]
        return sum(p["inside"] for p in scored) / len(scored) if scored else None


def _values(store: Store, pack: Pack, run_id: str, rule: dict) -> dict[float, float | None]:
    """The rule's metric at each rule temperature, from settled points of the run's latest derivation."""
    _, points, metrics = run_points(store, pack, run_id)
    out: dict[float, float | None] = {}
    for t in rule["temperatures_F"]:
        found = [m[rule["metric"]] for p, m in zip(points, metrics)
                 if p["step"] == rule["step"] and round(p["setpoint_F"]) == round(t) and settled(p)]
        out[t] = float(found[0]) if found and np.isfinite(found[0]) else None
    return out


def _passes(value: float, ref: float, rule: dict, better: str, target: float | None) -> bool:
    if rule["rule"] == "meets_target":
        return value >= target if better == "higher" else value <= target
    return value >= ref + rule["margin"] if better == "higher" else value <= ref - rule["margin"]


def score_campaign(store: Store, pack: Pack, campaign_id: str) -> CampaignScore:
    ledger = Ledger(store)
    campaign = store.get("campaigns", campaign_id)
    rule = campaign["hit_rule"]
    better = direction(pack, rule)
    target_spec = pack.spec.metrics[rule["metric"]].target
    target = target_spec.value if target_spec else None
    blinded = not ledger.unblinded(campaign_id)
    run_seq = ledger.run_seq()
    samples = {s["id"]: s for s in store.rows("samples")}
    formulations = {f["id"]: f for f in store.rows("formulations")}
    runs = [r for r in store.current("runs") if r["campaign_id"] == campaign_id
            and r["protocol_hash"] == campaign["protocol_hash"] and r["rig"] == campaign["rig"]]
    runs.sort(key=lambda r: run_seq.get(r["id"], 0))

    def formulation_of(run: dict) -> str:
        return samples[run["sample_id"]]["formulation_id"]

    ref_runs = [r for r in runs if formulation_of(r) == campaign["reference_formulation_id"] and r["status"] == "complete"]
    ref_values = [_values(store, pack, r["id"], rule) for r in ref_runs]
    reference = {}
    for t in rule["temperatures_F"]:
        vals = [v[t] for v in ref_values if v[t] is not None]
        reference[t] = float(np.mean(vals)) if vals else None

    picks = []
    for e in ledger.picks(campaign_id):
        fid = e["payload"]["formulation"]
        mine = [r for r in runs if formulation_of(r) == fid]
        result = PickResult(fid, formulations[fid]["name"], None if blinded else e["payload"]["arm"], e["seq"],
                            "awaiting test")
        result.early_runs = [r["id"] for r in mine if run_seq.get(r["id"], 0) < e["seq"]]
        after = [r for r in mine if run_seq.get(r["id"], 0) > e["seq"]]
        result.failed_runs = [r["id"] for r in after if r["status"] != "complete"]
        for r in (r for r in after if r["status"] == "complete"):
            values = _values(store, pack, r["id"], rule)
            if any(v is None for v in values.values()):
                result.unsettled_runs.append(r["id"])
                continue
            result.run, result.values = r["id"], values
            break
        if result.run:
            if all(reference[t] is not None for t in rule["temperatures_F"]):
                hit = all(_passes(result.values[t], reference[t], rule, better, target) for t in rule["temperatures_F"])
                result.state = "hit" if hit else "miss"
            else:
                result.state = "awaiting reference"
        elif result.unsettled_runs:
            result.state = "needs retest"
        picks.append(result)

    counted_runs = {p.formulation: p.run for p in picks if p.run}
    predictions = []
    for e in ledger.predictions(campaign_id):
        pr = e["payload"]
        measured = None
        run_id = counted_runs.get(pr["formulation"])
        if run_id:
            _, points, metrics = run_points(store, pack, run_id)
            found = [m[pr["metric"]] for p, m in zip(points, metrics)
                     if p["step"] == pr["step"] and round(p["setpoint_F"]) == round(pr["temperature_F"]) and settled(p)]
            measured = float(found[0]) if found else None
        inside = None
        if measured is not None and pr.get("low") is not None and pr.get("high") is not None:
            inside = pr["low"] <= measured <= pr["high"]
        predictions.append({"seq": e["seq"], "formulation": pr["formulation"], "metric": pr["metric"],
                            "step": pr["step"], "temperature_F": pr["temperature_F"], "predicted": pr["value"],
                            "low": pr.get("low"), "high": pr.get("high"), "measured": measured, "inside": inside,
                            "model": pr.get("model", ""), "logged_at": e["created_at"]})

    arms, comparisons = [], []
    if not blinded:
        for arm in ARMS:
            mine = [p for p in picks if p.arm == arm]
            if mine:
                tested = [p for p in mine if p.state in ("hit", "miss")]
                arms.append(ArmResult(arm, len(mine), len(tested), sum(p.state == "hit" for p in tested)))
        model = next((a for a in arms if a.arm == "model"), None)
        for other in arms:
            if model and other.arm != "model" and model.tested and other.tested:
                table = [[model.hits, model.tested - model.hits], [other.hits, other.tested - other.hits]]
                p_value = float(fisher_exact(table, alternative="greater")[1])
                comparisons.append({"against": other.arm, "p_value": p_value})

    return CampaignScore(campaign, blinded, store.simulated, better, reference, [r["id"] for r in ref_runs], picks,
                         arms, predictions, comparisons)
