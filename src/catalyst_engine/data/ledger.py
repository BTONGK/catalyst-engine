"""The prediction log: a hash chain inside the store.

Each entry's hash covers its sequence number, time, kind, payload and the previous entry's hash, so changing or
removing any past entry breaks every hash after it. Picks and predictions go in before the bench sees the catalyst;
each ingested run goes in too, so the chain itself proves which came first. Anchoring writes the head hash into a
tracked file, and a git commit of that file timestamps the whole log outside this machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from catalyst_engine.data.store import Store, canonical, now, sha256_bytes

GENESIS = "0" * 64
ARMS = ("model", "expert", "random")
KINDS = ("pick", "prediction", "run", "unblind", "note")
ANCHORS = Path(__file__).resolve().parents[3] / "docs" / "prediction_anchors.md"


def entry_hash(seq: int, created_at: str, kind: str, payload: str, prev_hash: str) -> str:
    return sha256_bytes(f"{seq}|{created_at}|{kind}|{payload}|{prev_hash}".encode())


@dataclass
class Check:
    ok: bool
    entries: int
    head: str
    problem: str = ""


class Ledger:
    def __init__(self, store: Store):
        self.store = store

    def entries(self, kind: str | None = None) -> list[dict]:
        rows = self.store.rows("ledger", "kind = ?" if kind else "", (kind,) if kind else (), order="seq")
        return rows

    def head(self) -> str:
        last = self.store.con.execute("SELECT hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return last[0] if last else GENESIS

    def append(self, kind: str, payload: dict) -> dict:
        if kind not in KINDS:
            raise ValueError(f"ledger kind must be one of {KINDS}")
        con = self.store.con
        with con:  # one transaction, so two writers cannot both chain onto the same head
            con.execute("BEGIN IMMEDIATE")
            last = con.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
            created_at = now()
            body = canonical(payload | {"by": self.store.user})
            digest = entry_hash(seq, created_at, kind, body, prev)
            con.execute("INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?)", (seq, created_at, kind, body, prev, digest))
        return {"seq": seq, "created_at": created_at, "kind": kind, "payload": payload, "prev_hash": prev, "hash": digest}

    def verify(self) -> Check:
        prev = GENESIS
        rows = self.store.con.execute("SELECT seq, created_at, kind, payload, prev_hash, hash FROM ledger ORDER BY seq")
        n = 0
        for n, (seq, created_at, kind, payload, prev_hash, digest) in enumerate(rows, start=1):
            if seq != n:
                return Check(False, n, prev, f"entry {n} is missing (next entry is {seq})")
            if prev_hash != prev:
                return Check(False, n, prev, f"entry {seq} points to a different previous entry")
            if entry_hash(seq, created_at, kind, payload, prev_hash) != digest:
                return Check(False, n, prev, f"entry {seq} changed after it was written")
            prev = digest
        return Check(True, n, prev)

    # entries with meaning -------------------------------------------------------------------------------------------

    def pick(self, campaign_id: str, formulation_id: str, arm: str, rationale: str = "") -> dict:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.store.get("campaigns", campaign_id)
        self.store.get("formulations", formulation_id)
        if self.store.get("campaigns", campaign_id)["reference_formulation_id"] == formulation_id:
            raise ValueError("the reference catalyst runs in every campaign; picks are candidates")
        if any(e["payload"]["formulation"] == formulation_id for e in self.picks(campaign_id)):
            raise ValueError(f"{formulation_id} is already picked in {campaign_id}")
        if self.unblinded(campaign_id):
            raise ValueError(f"{campaign_id} is unblinded; picks close when the arms are revealed")
        return self.append("pick", {"campaign": campaign_id, "formulation": formulation_id, "arm": arm,
                                    "rationale": rationale})

    def predict(self, campaign_id: str, formulation_id: str, metric: str, step: str, temperature_F: float,
                value: float, low: float | None = None, high: float | None = None, model: str = "") -> dict:
        self.store.get("formulations", formulation_id)
        if low is not None and high is not None and not low <= value <= high:
            raise ValueError("the predicted value must sit inside its interval")
        return self.append("prediction", {"campaign": campaign_id, "formulation": formulation_id, "metric": metric,
                                          "step": step, "temperature_F": float(temperature_F), "value": float(value),
                                          "low": low, "high": high, "model": model})

    def record_run(self, run: dict) -> dict:
        return self.append("run", {"run": run["id"], "raw_sha256": run["raw_sha256"], "sample": run["sample_id"],
                                   "campaign": run["campaign_id"], "status": run["status"]})

    def unblind(self, campaign_id: str) -> dict:
        self.store.get("campaigns", campaign_id)
        if self.unblinded(campaign_id):
            raise ValueError(f"{campaign_id} is already unblinded")
        return self.append("unblind", {"campaign": campaign_id})

    def picks(self, campaign_id: str) -> list[dict]:
        return [e for e in self.entries("pick") if e["payload"]["campaign"] == campaign_id]

    def predictions(self, campaign_id: str) -> list[dict]:
        return [e for e in self.entries("prediction") if e["payload"]["campaign"] == campaign_id]

    def run_seq(self) -> dict[str, int]:
        """Ledger position of every ingested run."""
        return {e["payload"]["run"]: e["seq"] for e in self.entries("run")}

    def unblinded(self, campaign_id: str) -> bool:
        return any(e["payload"]["campaign"] == campaign_id for e in self.entries("unblind"))

    def anchor(self, path: Path = ANCHORS) -> str:
        """Append the head hash to a tracked file; committing that file timestamps the log."""
        if self.store.simulated:
            raise ValueError("anchors record the lab log; the simulated store stays out of the repository")
        check = self.verify()
        if not check.ok:
            raise ValueError(f"the log fails its check: {check.problem}")
        line = f"| {now()} | {check.entries} | `{check.head}` |\n"
        if not path.exists():
            path.write_text("# Prediction log anchors\n\nEach row is the lab prediction log's head hash at that time. "
                            "The git commit that adds a row timestamps every entry up to it.\n\n"
                            "| Time (UTC) | Entries | Head hash |\n|---|---|---|\n")
        with path.open("a") as fh:
            fh.write(line)
        return line
