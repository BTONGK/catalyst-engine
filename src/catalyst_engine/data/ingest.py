"""Turn a raw analyzer log into steady-state points, keeping the raw file untouched.

The rig logger writes one CSV row per analyzer reading:

    # rig: bench-1                 header lines (optional) fill run details the command line leaves out
    # protocol: <protocol id>
    # sample: S-0002
    # operator: <name>
    # started_at: 2026-11-03T14:05:00+00:00
    time_s,step,setpoint_F,mode,T_F,NO (ppm),NO2 (ppm),NH3 (ppm),N2O (ppm)

`mode` is `bypass` while the gas skips the catalyst (the analyzer then reads the inlet) and `reactor` otherwise;
it defaults to reactor. `T_C` works in place of `T_F`. Species columns match the protocol analyzer's species,
with or without a ` (ppm)` or `_ppm` suffix.

Each (step, setpoint) becomes one point: the mean over the last `window_s` of readings, checked for drift,
scatter and temperature. Points that miss a check carry flags; they stay on file and the fit and scoring
treat them as unsettled.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from catalyst_engine.data.ledger import Ledger
from catalyst_engine.data.records import register_protocol, stored_protocol
from catalyst_engine.data.store import SIMULATED_MARKER, Store
from catalyst_engine.packs import Pack, ProtocolSpec
from catalyst_engine.units import k_to_f

METHOD = "steady-window-v1"
SETTINGS = {
    "window_s": 120.0,  # average the last two minutes of each hold
    "drift_sigma": 3.0,  # largest change across the window, in analyzer sigmas
    "scatter_sigma": 2.0,  # largest standard deviation inside the window, in analyzer sigmas
    "temperature_range_F": 5.0,  # largest temperature swing inside the window
    "temperature_offset_F": 10.0,  # largest gap between the mean temperature and the setpoint
    "inlet_tolerance": 0.10,  # largest relative gap between the measured inlet and the protocol feed
}
UNSETTLED = ("unsteady", "temperature_unsteady", "temperature_off", "short_hold")
RUN_STATUSES = ("complete", "failed", "aborted")


@dataclass
class AnalyzerFile:
    meta: dict[str, str]
    species: list[str]
    rows: list[dict] = field(default_factory=list)  # time_s, step, setpoint_F, mode, T_F, <species>: ppm
    simulated: bool = False


def _species_name(column: str) -> str:
    return re.sub(r"(\s*\(ppm\)|_ppm)$", "", column.strip())


def parse_analyzer_text(text: str, measured: list[str]) -> AnalyzerFile:
    meta, body = {}, []
    simulated = text.startswith(SIMULATED_MARKER)
    for line in text.splitlines():
        if line.startswith("#"):
            key, sep, value = line[1:].partition(":")
            if sep and not body:
                meta[key.strip()] = value.strip()
        elif line.strip():
            body.append(line)
    reader = csv.DictReader(io.StringIO("\n".join(body)))
    columns = {c: _species_name(c) for c in reader.fieldnames or []}
    required = {"time_s", "step", "setpoint_F"}
    if not required <= set(columns.values()):
        raise ValueError(f"the file needs columns {sorted(required)}; it has {list(columns)}")
    if not {"T_F", "T_C"} & set(columns.values()):
        raise ValueError("the file needs a T_F or T_C column")
    species = [s for s in measured if s in columns.values()]
    if not species:
        raise ValueError(f"the file has no columns for the analyzer species {measured}")
    rows = []
    for raw in reader:
        r = {columns[k]: v for k, v in raw.items() if k in columns}
        t_f = float(r["T_F"]) if r.get("T_F") not in (None, "") else k_to_f(float(r["T_C"]) + 273.15)
        row = {"time_s": float(r["time_s"]), "step": r["step"].strip(), "setpoint_F": float(r["setpoint_F"]),
               "mode": (r.get("mode") or "reactor").strip().lower(), "T_F": t_f}
        for s in species:
            row[s] = float(r[s]) if r.get(s) not in (None, "") else np.nan
        rows.append(row)
    if not rows:
        raise ValueError("the file has no readings")
    return AnalyzerFile(meta, species, sorted(rows, key=lambda x: x["time_s"]), simulated)


def read_analyzer_file(path: Path, measured: list[str]) -> AnalyzerFile:
    return parse_analyzer_text(Path(path).read_text(), measured)


def _sigma(protocol: ProtocolSpec, value: float) -> float:
    an = protocol.analyzer
    return float(np.hypot(an.sigma_abs_ppm, an.sigma_rel * abs(value)))


def _window(rows: list[dict], window_s: float) -> list[dict]:
    end = rows[-1]["time_s"]
    return [r for r in rows if r["time_s"] >= end - window_s]


def derive_points(pack: Pack, protocol: ProtocolSpec, data: AnalyzerFile,
                  settings: dict | None = None) -> tuple[list[dict], list[str]]:
    """Steady-state points and run-level flags for one analyzer file."""
    cfg = SETTINGS | (settings or {})
    species = data.species
    lod = protocol.analyzer.lod_ppm
    run_flags: list[str] = []

    inlets: dict[str, dict[str, float]] = {}
    for step in protocol.steps:
        bypass = [r for r in data.rows if r["mode"] == "bypass" and r["step"] == step.id]
        expected = step.feed_fractions(pack.network.species, pack.spec.water) * 1e6
        if not bypass:
            continue
        win = _window(bypass, cfg["window_s"])
        inlet = {s: float(np.nanmean([r[s] for r in win])) for s in species}
        inlets[step.id] = inlet
        for s in species:
            want = expected[pack.network.index(s)]
            if want > 0 and abs(inlet[s] - want) > cfg["inlet_tolerance"] * want:
                run_flags.append(f"inlet_mismatch: {step.id} {s} reads {inlet[s]:.3g} ppm, protocol {want:.3g}")

    segments: dict[tuple[str, float], list[dict]] = {}
    for r in data.rows:
        if r["mode"] == "reactor":
            segments.setdefault((r["step"], round(r["setpoint_F"], 1)), []).append(r)

    planned = {(s.id, round(k_to_f(t), 1)) for s in protocol.steps for t in s.kelvin()}
    step_ids = {s.id for s in protocol.steps}
    points = []
    for (step_id, setpoint), rows in segments.items():
        flags: list[str] = []
        if (step_id, setpoint) not in planned:
            flags.append("off_protocol")
        win = _window(rows, cfg["window_s"])
        times = np.array([r["time_s"] for r in win])
        temps = np.array([r["T_F"] for r in win])
        if len(win) < 3 or times[-1] - times[0] < 0.5 * cfg["window_s"]:
            flags.append("short_hold")
        outlet, outlet_std = {}, {}
        for s in species:
            y = np.array([r[s] for r in win], dtype=float)
            ok = np.isfinite(y)
            mean = float(np.mean(y[ok])) if ok.any() else float("nan")
            std = float(np.std(y[ok], ddof=1)) if ok.sum() > 1 else 0.0
            outlet[s], outlet_std[s] = mean, std
            sig = _sigma(protocol, mean)
            if ok.sum() >= 3 and np.ptp(times[ok]) > 0:
                drift = abs(np.polyfit(times[ok], y[ok], 1)[0]) * cfg["window_s"]
                if drift > cfg["drift_sigma"] * sig or std > cfg["scatter_sigma"] * sig:
                    flags.append(f"unsteady: {s}")
            if mean < lod.get(s, 0.0):
                flags.append(f"below_lod: {s}")
            if mean < 0:
                flags.append(f"negative: {s}")
        t_mean = float(np.mean(temps))
        if np.ptp(temps) > cfg["temperature_range_F"]:
            flags.append("temperature_unsteady")
        if abs(t_mean - setpoint) > cfg["temperature_offset_F"]:
            flags.append("temperature_off")
        inlet = inlets.get(step_id)
        if inlet is None:
            if step_id in step_ids:
                step = protocol.step(step_id)
                y_in = step.feed_fractions(pack.network.species, pack.spec.water) * 1e6
                inlet = {s: float(y_in[pack.network.index(s)]) for s in species}
            else:
                inlet = {}
            flags.append("inlet_from_protocol")
        points.append({"step": step_id, "setpoint_F": setpoint, "T_F": t_mean, "outlet": outlet,
                       "outlet_std": outlet_std, "inlet": inlet, "n": len(win),
                       "window_start_s": float(times[0]), "window_end_s": float(times[-1]),
                       "flags": sorted(set(flags), key=flags.index)})

    for step_id, setpoint in sorted(planned - set(segments), key=lambda p: (p[0], p[1])):
        run_flags.append(f"missing: {step_id} at {setpoint:,.0f} F")
    order = {s.id: i for i, s in enumerate(protocol.steps)}
    points.sort(key=lambda p: (order.get(p["step"], len(order)), p["setpoint_F"]))
    return points, run_flags


def settled(point: dict) -> bool:
    return not any(f.split(":")[0] in UNSETTLED for f in point["flags"])


def issues(point: dict) -> list[str]:
    """Flags that call for a look. Readings below the detection limit are routine and left out."""
    return [f for f in point["flags"] if not f.startswith("below_lod")]


def inlet_used(pack: Pack, protocol: ProtocolSpec, point: dict) -> dict[str, float]:
    """The inlet (wet ppm) the point's metrics and fits use. A measured inlet counts for species the step feeds,
    and for any other species read at or above its detection limit; below that, the step's feed stands."""
    feed = protocol.step(point["step"]).feed_fractions(pack.network.species, pack.spec.water) * 1e6
    lod = protocol.analyzer.lod_ppm
    out = {}
    for s, v in point["inlet"].items():
        planned = float(feed[pack.network.index(s)])
        measured = "inlet_from_protocol" not in point["flags"] and np.isfinite(v)
        out[s] = max(v, 0.0) if measured and (planned > 0 or v >= lod.get(s, 0.0)) else planned
    return out


def fit_rows(pack: Pack, protocol: ProtocolSpec, points: list[dict], settled_only: bool = True) -> list[dict]:
    """Points in the row form `fit.kinetics.observations_from_table` reads."""
    steps = {s.id for s in protocol.steps}
    rows = []
    for p in points:
        if p["step"] not in steps or (settled_only and not settled(p)):
            continue
        row = {"step": p["step"], "T_F": p["T_F"]}
        row.update({f"out_{s}_ppm": max(v, 0.0) for s, v in p["outlet"].items() if np.isfinite(v)})
        row.update({f"in_{s}_ppm": v for s, v in inlet_used(pack, protocol, p).items()})
        rows.append(row)
    return rows


def compositions(pack: Pack, protocol: ProtocolSpec, points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Feed and outlet wet mole fractions for each point, shaped for `Pack.metrics`. Species the analyzer
    reads take the measured values; the rest (water, oxygen, the carrier) keep their feed values."""
    species = pack.network.species
    feeds, outs = [], []
    for p in points:
        feed = protocol.step(p["step"]).feed_fractions(species, pack.spec.water)
        for s, v in inlet_used(pack, protocol, p).items():
            feed[species.index(s)] = v * 1e-6
        out = feed.copy()
        for s, v in p["outlet"].items():
            if np.isfinite(v):
                out[species.index(s)] = max(v, 0.0) * 1e-6
        feeds.append(feed)
        outs.append(out)
    return np.array(feeds), np.array(outs)


def point_metrics(pack: Pack, protocol: ProtocolSpec, points: list[dict]) -> list[dict[str, float]]:
    """The pack's metrics at each point; points on a step outside the protocol get NaN."""
    steps = {s.id for s in protocol.steps}
    known = [p for p in points if p["step"] in steps]
    values = pack.metrics(*compositions(pack, protocol, known)) if known else {}
    out, i = [], 0
    for p in points:
        if p["step"] in steps:
            out.append({m: float(v[i]) for m, v in values.items()})
            i += 1
        else:
            out.append({m: float("nan") for m in pack.spec.metrics})
    return out


def derive(store: Store, pack: Pack, run_id: str, settings: dict | None = None) -> dict:
    """Derive (or re-derive) a run's points from its raw file. Earlier derivations stay on file."""
    run = store.get("runs", run_id)
    protocol = stored_protocol(store, run["protocol_hash"])
    data = read_analyzer_file(store.raw_path(run["raw_sha256"]), protocol.analyzer.measured)
    points, run_flags = derive_points(pack, protocol, data, settings)
    derivation = store.insert("derivations", {"run_id": run_id, "method": METHOD,
                                              "settings": SETTINGS | (settings or {}), "run_flags": run_flags})
    for p in points:
        store.insert("points", {"derivation_id": derivation["id"], **p}, provenance=False)
    return derivation


def ingest_run(store: Store, pack: Pack, path: Path, *, sample_id: str | None = None, campaign_id: str | None = None,
               protocol_id: str | None = None, rig: str | None = None, operator: str | None = None,
               started_at: str | None = None, status: str = "auto", notes: str = "",
               supersedes: str | None = None, original_name: str | None = None) -> dict:
    """Store the raw file, record the run, derive its points and log it in the ledger.

    Details come from the arguments first, then the file header. A campaign run must use the campaign's
    protocol version and rig, so every candidate is compared with a reference measured the same way."""
    path = Path(path)
    text = path.read_text()
    meta = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        key, sep, value = line[1:].partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    campaign = store.get("campaigns", campaign_id) if campaign_id else None
    if campaign:
        proto_row = store.get("protocols", campaign["protocol_hash"])
        protocol_id = protocol_id or proto_row["protocol_id"]
    protocol = pack.protocol(protocol_id or meta.get("protocol"))
    digest = register_protocol(store, pack, protocol)
    rig = rig or meta.get("rig")
    sample_id = sample_id or meta.get("sample")
    if not rig or not sample_id:
        raise ValueError("give the rig and the sample (as arguments or # rig: / # sample: header lines)")
    store.get("samples", sample_id)
    if campaign:
        if digest != campaign["protocol_hash"]:
            raise ValueError(f"{campaign['id']} runs a different version of {protocol.id}; the protocol file has "
                             "changed since the campaign began. Start a new campaign for the new version.")
        if rig != campaign["rig"]:
            raise ValueError(f"{campaign['id']} runs on {campaign['rig']}; this run is on {rig}")
    data = parse_analyzer_text(text, protocol.analyzer.measured)
    if status == "auto":
        _, run_flags = derive_points(pack, protocol, data)
        status = "failed" if any(f.startswith("missing") for f in run_flags) else "complete"
    if status not in RUN_STATUSES:
        raise ValueError(f"status must be one of {RUN_STATUSES} or auto")
    raw = store.add_raw_file(path, original_name)
    run = store.insert("runs", {"sample_id": sample_id, "campaign_id": campaign_id, "protocol_hash": digest, "rig": rig,
                                "operator": operator or meta.get("operator") or store.user,
                                "started_at": started_at or meta.get("started_at") or "", "status": status,
                                "raw_sha256": raw, "notes": notes, "supersedes": supersedes})
    derive(store, pack, run["id"])
    Ledger(store).record_run(run)
    return run


def run_points(store: Store, pack: Pack, run_id: str) -> tuple[ProtocolSpec, list[dict], list[dict[str, float]]]:
    """A run's protocol snapshot, its latest points, and the pack metrics at each point."""
    run = store.get("runs", run_id)
    protocol = stored_protocol(store, run["protocol_hash"])
    points = store.points(run_id)
    return protocol, points, point_metrics(pack, protocol, points)

