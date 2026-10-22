"""An append-only SQLite store with a content-addressed folder of raw files.

Every table refuses UPDATE and DELETE; a correction is a new record that names the one it `supersedes`, so the
history stays whole and failed runs stay on file. Every record carries who added it, when, and the engine version.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal

import catalyst_engine

REPO = Path(__file__).resolve().parents[3]
SIMULATED_MARKER = "# simulated by catalyst-engine"  # first line of every file a simulator writes
Kind = Literal["lab", "simulated"]

_PROVENANCE = "created_at TEXT NOT NULL, created_by TEXT NOT NULL, code_version TEXT NOT NULL"
SCHEMA = {
    "meta": "key TEXT PRIMARY KEY, value TEXT NOT NULL",
    "formulations": f"""id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, components TEXT NOT NULL,
        preparation TEXT NOT NULL, notes TEXT NOT NULL, supersedes TEXT, {_PROVENANCE}""",
    "samples": f"""id TEXT PRIMARY KEY, formulation_id TEXT NOT NULL REFERENCES formulations(id), form TEXT NOT NULL,
        mass_g REAL, volume_mL REAL, aged_from TEXT REFERENCES samples(id), aging TEXT, notes TEXT NOT NULL,
        supersedes TEXT, {_PROVENANCE}""",
    "protocols": f"hash TEXT PRIMARY KEY, pack TEXT NOT NULL, protocol_id TEXT NOT NULL, version INTEGER NOT NULL, "
                 f"body TEXT NOT NULL, {_PROVENANCE}",
    "campaigns": f"""id TEXT PRIMARY KEY, name TEXT NOT NULL, pack TEXT NOT NULL,
        protocol_hash TEXT NOT NULL REFERENCES protocols(hash), rig TEXT NOT NULL,
        reference_formulation_id TEXT NOT NULL REFERENCES formulations(id), hit_rule TEXT NOT NULL,
        notes TEXT NOT NULL, {_PROVENANCE}""",
    "raw_files": f"sha256 TEXT PRIMARY KEY, original_name TEXT NOT NULL, size INTEGER NOT NULL, simulated INTEGER NOT NULL, "
                 f"{_PROVENANCE}",
    "runs": f"""id TEXT PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES samples(id), campaign_id TEXT REFERENCES campaigns(id),
        protocol_hash TEXT NOT NULL REFERENCES protocols(hash), rig TEXT NOT NULL, operator TEXT NOT NULL,
        started_at TEXT NOT NULL, status TEXT NOT NULL, raw_sha256 TEXT NOT NULL REFERENCES raw_files(sha256),
        notes TEXT NOT NULL, supersedes TEXT, {_PROVENANCE}""",
    "derivations": f"""id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), method TEXT NOT NULL,
        settings TEXT NOT NULL, run_flags TEXT NOT NULL, {_PROVENANCE}""",
    "points": """derivation_id TEXT NOT NULL REFERENCES derivations(id), step TEXT NOT NULL, setpoint_F REAL NOT NULL,
        T_F REAL NOT NULL, outlet TEXT NOT NULL, outlet_std TEXT NOT NULL, inlet TEXT NOT NULL, n INTEGER NOT NULL,
        window_start_s REAL NOT NULL, window_end_s REAL NOT NULL, flags TEXT NOT NULL,
        PRIMARY KEY (derivation_id, step, setpoint_F)""",
    "notes": f"id TEXT PRIMARY KEY, record_id TEXT NOT NULL, text TEXT NOT NULL, {_PROVENANCE}",
    "ledger": """seq INTEGER PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
        prev_hash TEXT NOT NULL, hash TEXT NOT NULL UNIQUE""",
}
ID_PREFIX = {"formulations": "F", "samples": "S", "campaigns": "C", "runs": "R", "derivations": "D", "notes": "N"}
JSON_COLUMNS = {"components", "aging", "hit_rule", "settings", "run_flags", "outlet", "outlet_std", "inlet", "flags",
                "payload"}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@cache
def code_version() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "describe", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = ""
    return f"{catalyst_engine.__version__}+{sha or 'unknown'}"


def canonical(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace. The ledger hashes this form."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AppendOnlyError(RuntimeError):
    pass


class Store:
    """One store folder: `store.db` plus `raw/` (files named by their SHA-256)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db_path = self.root / "store.db"
        if not self.db_path.exists():
            raise FileNotFoundError(f"no store at {self.root}; create one with Store.create")
        self.con = sqlite3.connect(self.db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.user = os.environ.get("CE_USER") or getpass.getuser()

    @classmethod
    def create(cls, root: Path, kind: Kind) -> "Store":
        root = Path(root)
        if (root / "store.db").exists():
            raise FileExistsError(f"a store already exists at {root}")
        (root / "raw").mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(root / "store.db")
        for table, columns in SCHEMA.items():
            con.execute(f"CREATE TABLE {table} ({columns})")
            for op in ("UPDATE", "DELETE"):
                con.execute(f"CREATE TRIGGER {table}_no_{op.lower()} BEFORE {op} ON {table} "
                            f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END")
        con.executemany("INSERT INTO meta VALUES (?, ?)",
                        [("kind", kind), ("created_at", now()), ("schema_version", "1"), ("code_version", code_version())])
        con.commit()
        con.close()
        return cls(root)

    @classmethod
    def open_or_create(cls, root: Path, kind: Kind) -> "Store":
        store = cls(root) if (Path(root) / "store.db").exists() else cls.create(root, kind)
        if store.kind != kind:
            raise ValueError(f"the store at {root} is a {store.kind} store, expected {kind}")
        return store

    @property
    def kind(self) -> Kind:
        return self.con.execute("SELECT value FROM meta WHERE key = 'kind'").fetchone()[0]

    @property
    def simulated(self) -> bool:
        return self.kind == "simulated"

    # writing -------------------------------------------------------------------------------------------------------

    def _provenance(self) -> dict:
        return {"created_at": now(), "created_by": self.user, "code_version": code_version()}

    def next_id(self, table: str) -> str:
        n = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return f"{ID_PREFIX[table]}-{n + 1:04d}"

    def insert(self, table: str, record: dict, *, provenance: bool = True) -> dict:
        row = dict(record)
        if table in ID_PREFIX and "id" not in row:
            row["id"] = self.next_id(table)
        if provenance:
            row.update(self._provenance())
        values = {k: canonical(v) if k in JSON_COLUMNS and v is not None else v for k, v in row.items()}
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        try:
            with self.con:
                self.con.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"{table}: {exc}") from exc
        return row

    def add_raw_file(self, path: Path, original_name: str | None = None) -> str:
        """Copy a file into raw/ under its SHA-256 (read-only). Adding the same bytes twice returns the same hash."""
        data = Path(path).read_bytes()
        simulated = data.startswith(SIMULATED_MARKER.encode())
        if simulated and not self.simulated:
            raise ValueError("this file was written by a simulator; simulated files belong in the simulated store")
        digest = sha256_bytes(data)
        if self.con.execute("SELECT 1 FROM raw_files WHERE sha256 = ?", (digest,)).fetchone():
            return digest
        dest = self.raw_path(digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        dest.chmod(0o444)
        self.insert("raw_files", {"sha256": digest, "original_name": original_name or Path(path).name,
                                  "size": len(data), "simulated": int(simulated)})
        return digest

    def raw_path(self, digest: str) -> Path:
        return self.root / "raw" / digest[:2] / digest

    def verify_raw(self, digest: str) -> bool:
        path = self.raw_path(digest)
        return path.exists() and sha256_bytes(path.read_bytes()) == digest

    def add_note(self, record_id: str, text: str) -> dict:
        return self.insert("notes", {"record_id": record_id, "text": text})

    # reading -------------------------------------------------------------------------------------------------------

    def rows(self, table: str, where: str = "", params: tuple = (), order: str = "") -> list[dict]:
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + (f" ORDER BY {order}" if order else "")
        out = []
        for r in self.con.execute(sql, params):
            d = dict(r)
            for k in JSON_COLUMNS & d.keys():
                if d[k] is not None:
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def current(self, table: str) -> list[dict]:
        """Records that no later record supersedes."""
        rows = self.rows(table, order="id")
        replaced = {r["supersedes"] for r in rows if r.get("supersedes")}
        return [r for r in rows if r["id"] not in replaced]

    def get(self, table: str, record_id: str) -> dict:
        key = "hash" if table == "protocols" else "id"
        found = self.rows(table, f"{key} = ?", (record_id,))
        if not found:
            raise KeyError(f"{table} has no {record_id!r}")
        return found[0]

    def latest_derivation(self, run_id: str) -> dict | None:
        found = self.rows("derivations", "run_id = ?", (run_id,), order="id DESC")
        return found[0] if found else None

    def points(self, run_id: str) -> list[dict]:
        d = self.latest_derivation(run_id)
        return self.rows("points", "derivation_id = ?", (d["id"],), order="step, setpoint_F") if d else []

    def counts(self) -> dict[str, int]:
        return {t: self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in SCHEMA if t != "meta"}

    def close(self) -> None:
        self.con.close()


def data_dir() -> Path:
    """Where the stores live: CE_DATA_DIR, or data/ at the repository root (ignored by git)."""
    return Path(os.environ.get("CE_DATA_DIR", REPO / "data"))


def store_path(kind: Kind) -> Path:
    return data_dir() / kind
