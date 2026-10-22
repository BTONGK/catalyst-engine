"""Adding catalysts, samples, protocols and campaigns, with the checks the bench relies on."""

from __future__ import annotations

import json

from catalyst_engine.data.store import Store, canonical, sha256_bytes
from catalyst_engine.packs import Pack, ProtocolSpec
from catalyst_engine.units import k_to_f

FORMULATION_ROLES = ("reference", "candidate")
HIT_RULES = ("beats_reference", "meets_target")


def protocol_hash(pack: Pack, protocol: ProtocolSpec) -> str:
    return sha256_bytes(canonical({"pack": pack.id, "protocol": protocol.model_dump(mode="json")}).encode())


def register_protocol(store: Store, pack: Pack, protocol: ProtocolSpec) -> str:
    """Snapshot the protocol as it stands; a later edit to the YAML gets its own hash."""
    digest = protocol_hash(pack, protocol)
    if not store.rows("protocols", "hash = ?", (digest,)):
        store.insert("protocols", {"hash": digest, "pack": pack.id, "protocol_id": protocol.id,
                                   "version": protocol.version, "body": canonical(protocol.model_dump(mode="json"))})
    return digest


def stored_protocol(store: Store, digest: str) -> ProtocolSpec:
    """The protocol exactly as it was when registered, whatever the YAML says today."""
    return ProtocolSpec.model_validate(json.loads(store.get("protocols", digest)["body"]))


def add_formulation(store: Store, name: str, components: list[dict], role: str = "candidate", preparation: str = "",
                    notes: str = "", supersedes: str | None = None) -> dict:
    """components: [{material, role, amount, unit}], e.g. {material: V2O5, role: active, amount: 1.5, unit: wt%}."""
    if role not in FORMULATION_ROLES:
        raise ValueError(f"role must be one of {FORMULATION_ROLES}")
    for c in components:
        if not {"material", "amount", "unit"} <= c.keys():
            raise ValueError(f"each component needs material, amount and unit: {c}")
    if supersedes:
        store.get("formulations", supersedes)
    return store.insert("formulations", {"name": name, "role": role, "components": components,
                                         "preparation": preparation, "notes": notes, "supersedes": supersedes})


def add_sample(store: Store, formulation_id: str, form: str, mass_g: float | None = None,
               volume_mL: float | None = None, aged_from: str | None = None, aging: dict | None = None,
               notes: str = "", supersedes: str | None = None) -> dict:
    store.get("formulations", formulation_id)
    if aged_from:
        store.get("samples", aged_from)
    return store.insert("samples", {"formulation_id": formulation_id, "form": form, "mass_g": mass_g,
                                    "volume_mL": volume_mL, "aged_from": aged_from, "aging": aging, "notes": notes,
                                    "supersedes": supersedes})


def add_campaign(store: Store, pack: Pack, name: str, protocol_id: str, rig: str, reference_formulation_id: str,
                 hit_rule: dict, notes: str = "") -> dict:
    """hit_rule: {metric, step, temperatures_F: [..], rule: beats_reference | meets_target, margin}.
    beats_reference: better than the reference by `margin` at every listed temperature, on the same rig.
    meets_target: meets the pack's target for the metric at every listed temperature."""
    protocol = pack.protocol(protocol_id)
    if hit_rule.get("rule") not in HIT_RULES:
        raise ValueError(f"hit_rule.rule must be one of {HIT_RULES}")
    metric = pack.spec.metrics.get(hit_rule.get("metric", ""))
    if metric is None:
        raise ValueError(f"hit_rule.metric must be one of {sorted(pack.spec.metrics)}")
    if metric.target is None and "direction" not in hit_rule:
        raise ValueError("this metric has no target; give hit_rule.direction (higher or lower)")
    if hit_rule["rule"] == "meets_target" and metric.target is None:
        raise ValueError("meets_target needs a metric with a target")
    step = protocol.step(hit_rule.get("step", ""))
    planned = {round(k_to_f(t)) for t in step.kelvin()}
    temps = [float(t) for t in hit_rule.get("temperatures_F", [])]
    if not temps or not {round(t) for t in temps} <= planned:
        raise ValueError(f"hit_rule.temperatures_F must come from step {step.id}'s temperatures {sorted(planned)}")
    ref = store.get("formulations", reference_formulation_id)
    if ref["role"] != "reference":
        raise ValueError(f"{reference_formulation_id} is a candidate; the campaign reference has role reference")
    rule = {"metric": hit_rule["metric"], "step": step.id, "temperatures_F": temps, "rule": hit_rule["rule"],
            "margin": float(hit_rule.get("margin", 0.0))}
    if "direction" in hit_rule:
        rule["direction"] = hit_rule["direction"]
    digest = register_protocol(store, pack, protocol)
    return store.insert("campaigns", {"name": name, "pack": pack.id, "protocol_hash": digest, "rig": rig,
                                      "reference_formulation_id": reference_formulation_id, "hit_rule": rule,
                                      "notes": notes})


def direction(pack: Pack, rule: dict) -> str:
    """'higher' or 'lower': which way is better for the rule's metric."""
    if "direction" in rule:
        return rule["direction"]
    target = pack.spec.metrics[rule["metric"]].target
    return "higher" if target is not None and target.min is not None else "lower"
