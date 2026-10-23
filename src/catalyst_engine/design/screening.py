"""Catalyst screens: run the atomistic worker on candidate materials, then predict how each would perform.

A screen is a folder under <data dir>/screens/<id>/ holding job.json (what to compute, plus the user's request:
reaction, conditions, materials), and what the worker writes: status.json, results.jsonl, structures/, log.txt.

Prediction, per material:
  1. descriptor  = weighted sum of its lowest binding energies (the pack's screening descriptor)
  2. barrier shift = the volcano's height difference from the reference catalyst (ScreeningSpec.barrier_shift)
  3. the pack's screened pathways run with that extra barrier: Ea + shift, and k(T_ref) x exp(-shift / kT_ref)
  4. the reactor model turns those rates into the pack's metric across the user's temperatures
The descriptor's uncertainty gives each prediction a low and a high case.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.design import elements
from catalyst_engine.packs import Pack, StepSpec
from catalyst_engine.units import R

WORKER = Path(__file__).with_name("uma_worker.py")
EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "screens"  # finished screens shipped with the app
EV_KJ = 96.485  # kJ/mol per eV
K_B = 8.617333e-5  # eV/K
RUNNING = ("queued", "loading", "running")
NONMETALS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "Te"}


# ---------------------------------------------------------------- setup and storage

def uma_setup() -> dict:
    """Where the UMA Python and checkpoint are (CE_UMA_PYTHON, CE_UMA_CHECKPOINT) and whether both exist."""
    python = Path(os.environ.get("CE_UMA_PYTHON", "/opt/miniconda3/bin/python3"))
    checkpoint = Path(os.environ.get("CE_UMA_CHECKPOINT", Path.home() / "uma" / "checkpoints" / "uma-s-1p1.pt"))
    return {"python": str(python), "checkpoint": str(checkpoint), "ready": python.exists() and checkpoint.exists()}


def screens_dir() -> Path:
    return Path(os.environ.get("CE_DATA_DIR", Path(__file__).resolve().parents[3] / "data")) / "screens"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "screen"


def create_screen(pack: Pack, name: str, conditions: dict, materials: list[dict], thoroughness: str = "standard",
                  task: str = "oc20", design_temperature_C: float | None = None) -> Path:
    """Write a screen folder. `materials` are job entries (materials.job_entry / element_entry); the pack's
    reference catalyst must be one of them, flagged is_reference."""
    spec = pack.spec.screening
    if spec is None:
        raise ValueError(f"{pack.spec.name} has no screening descriptor")
    if not any(m.get("is_reference") for m in materials):
        raise ValueError("include the reference catalyst")
    stamp = datetime.now()
    folder = screens_dir() / f"{stamp:%Y%m%d-%H%M%S}-{_slug(name)}"
    folder.mkdir(parents=True)
    temps = conditions["temperatures_C"]
    request = {"name": name, "pack": pack.id, "pack_name": pack.spec.name, "created": stamp.isoformat(timespec="seconds"),
               "conditions": conditions, "thoroughness": thoroughness, "task": task,
               "design_temperature_C": design_temperature_C if design_temperature_C is not None
               else float(np.median(temps)),
               "materials": [{k: v for k, v in m.items() if k not in ("species", "frac", "lattice")} for m in materials]}
    job = {"checkpoint": uma_setup()["checkpoint"], "task": task, "thoroughness": thoroughness,
           "adsorbates": list(spec.descriptor), "materials": materials, "request": request}
    (folder / "job.json").write_text(json.dumps(job, indent=1))
    (folder / "status.json").write_text(json.dumps({"state": "queued", "done": 0, "total": 0,
                                                     "message": "Waiting to start", "updated": time.time()}))
    return folder


def launch(folder: Path) -> int:
    """Start the worker in the background; it keeps running when the page closes."""
    setup = uma_setup()
    env = dict(os.environ, TORCHDYNAMO_DISABLE="1", PYTHONUNBUFFERED="1")
    with (folder / "log.txt").open("ab") as log:
        proc = subprocess.Popen([setup["python"], str(WORKER), str(folder)], stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=folder, start_new_session=True)
    status = json.loads((folder / "status.json").read_text())
    status.update(pid=proc.pid, state="queued", message="Starting the UMA model", updated=time.time())
    (folder / "status.json").write_text(json.dumps(status))
    return proc.pid


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:  # a worker this app started stays a zombie after it exits until reaped here
        if os.waitpid(pid, os.WNOHANG)[0]:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cancel(folder: Path) -> None:
    status = read_status(folder)
    if status.get("pid") and _alive(status["pid"]):
        try:
            os.killpg(os.getpgid(status["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
    status.update(state="cancelled", message="Stopped by you", updated=time.time())
    (folder / "status.json").write_text(json.dumps(status))


def read_job(folder: Path) -> dict:
    return json.loads((folder / "job.json").read_text())


def read_status(folder: Path) -> dict:
    path = folder / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {"state": "queued"}
    if status.get("state") in RUNNING and status.get("pid") and not _alive(status["pid"]):
        status.update(state="stopped", message="The worker stopped early; see log.txt in the screen folder")
    return status


def read_results(folder: Path) -> list[dict]:
    path = folder / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add_examples() -> None:
    """Copy the shipped example screens into an empty screens folder, so Results opens on a finished one."""
    root = screens_dir()
    if EXAMPLES.is_dir() and not (root.is_dir() and any(root.iterdir())):
        for src in sorted(EXAMPLES.iterdir()):
            if (src / "job.json").exists():
                shutil.copytree(src, root / src.name)


def list_screens() -> list[Path]:
    root = screens_dir()
    return sorted((p for p in root.iterdir() if (p / "job.json").exists()), reverse=True) if root.is_dir() else []


def seconds_estimate(materials: list[dict], n_adsorbates: int, thoroughness: str) -> float:
    """Rough run time on a laptop CPU: about 10 s per relaxation on a metal, 50 s on a compound."""
    sites = {"quick": 2, "standard": 3, "thorough": 5}[thoroughness]
    total = 20.0
    for m in materials:
        per = 50.0 if set(m.get("elements", [])) & NONMETALS else 10.0
        total += per * (1 + sites * n_adsorbates)
    return total


# ---------------------------------------------------------------- reactor runs

def design_step(pack: Pack) -> StepSpec:
    """The pack's first protocol step: the default conditions on the design page."""
    proto = next(iter(pack.protocols.values()))
    return proto.steps[0]


def step_for(conditions: dict) -> StepSpec:
    return StepSpec(id="design", feed=conditions["feed"], ghsv_per_h=conditions["ghsv_per_h"],
                    pressure_kPa=conditions["pressure_bar"] * 100.0, temperatures_C=conditions["temperatures_C"])


def shifted_params(pack: Pack, shift_eV: float, base: dict[str, float] | None = None) -> dict[str, float]:
    """The pack's parameters with the screened pathways' barrier raised by `shift_eV` (negative lowers it)."""
    p = dict(base or pack.network.parameters())
    t_ref = pack.network.t_ref
    for pw in pack.spec.screening.pathways:
        ea = p[f"{pw}.Ea_kJ"]
        new = float(np.clip(ea + shift_eV * EV_KJ, 5.0, 300.0))
        p[f"{pw}.Ea_kJ"] = new
        p[f"{pw}.ln_k_ref"] = float(np.clip(p[f"{pw}.ln_k_ref"] - (new - ea) * 1000.0 / (R * t_ref), -30.0, 30.0))
    return p


def fast_params(pack: Pack, base: dict[str, float] | None = None, boost: float = 9.0) -> dict[str, float]:
    """Screened pathways about 8,000 times faster than the reference at its reference temperature and barely
    slowed by cooling: the outlet sits at equilibrium (or full conversion), the best a catalyst can do."""
    p = dict(base or pack.network.parameters())
    for pw in pack.spec.screening.pathways:
        p[f"{pw}.ln_k_ref"] = min(30.0, p[f"{pw}.ln_k_ref"] + boost)
        p[f"{pw}.Ea_kJ"] = 5.0
    return p


def metric_curves(pack: Pack, step: StepSpec, param_sets: list[dict[str, float]], metric: str) -> np.ndarray:
    """The metric for each parameter set at each of the step's temperatures: (sets, temperatures)."""
    y = step.feed_fractions(pack.network.species, pack.spec.water)
    t = np.array(step.kelvin())
    m, b = len(param_sets), len(t)
    cond = Conditions(np.tile(t, m), np.full(m * b, step.pressure_kPa * 1000.0), np.full(m * b, step.ghsv_per_h),
                      np.tile(y, (m * b, 1)))
    params = {k: np.repeat([ps[k] for ps in param_sets], b) for k in param_sets[0]}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            out = outlet(pack.network, params, cond)
        except (np.linalg.LinAlgError, RuntimeError):
            if m == 1:
                return np.full((1, b), np.nan)
            # one stiff case stops a whole batch; run the sets one at a time so only that one is lost
            return np.vstack([metric_curves(pack, step, [ps], metric) for ps in param_sets])
        values = pack.metrics(cond.feed, out)[metric]
    return np.asarray(values, dtype=float).reshape(m, b)


def calibrate(pack: Pack, step: StepSpec, t_cal_C: float, share: float = 0.5) -> dict[str, float]:
    """Parameters that put the reference catalyst at `share` of the best possible metric at `t_cal_C`: the
    screened pathways' rate constants move together until it gets there."""
    one = step.model_copy(update={"temperatures_C": [t_cal_C], "temperatures_F": None, "temperatures_K": None})
    metric = pack.spec.screening.metric
    best = next(v for boost in (9.0, 6.0, 4.0)
                if np.isfinite(v := metric_curves(pack, one, [fast_params(pack, boost=boost)], metric)[0, 0]))
    base = pack.network.parameters()

    def value(offset: float) -> float:
        p = dict(base)
        for pw in pack.spec.screening.pathways:
            p[f"{pw}.ln_k_ref"] = float(np.clip(base[f"{pw}.ln_k_ref"] + offset, -30.0, 30.0))
        return metric_curves(pack, one, [p], metric)[0, 0]

    lo, hi = -40.0, 40.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if value(mid) < share * best:
            lo = mid
        else:
            hi = mid
    off = 0.5 * (lo + hi)
    return {f"{pw}.ln_k_ref": float(np.clip(base[f"{pw}.ln_k_ref"] + off, -30.0, 30.0))
            for pw in pack.spec.screening.pathways}


# ---------------------------------------------------------------- predictions

@dataclass
class Prediction:
    material: str
    label: str
    formula: str
    is_reference: bool
    elements: list[str]
    e_hull: float | None
    source: str
    binding: dict[str, dict] = field(default_factory=dict)  # adsorbate -> {E, site, structure, tried, kept}
    slab: dict | None = None
    error: str = ""
    descriptor: float | None = None
    shift: float | None = None  # extra barrier over the reference, eV
    shift_range: tuple[float, float] | None = None
    rate_ratio: float | None = None  # rate over the reference's at the design temperature
    rate_range: tuple[float, float] | None = None
    metric_at_design: float | None = None
    metric_range: tuple[float, float] | None = None
    t_half_C: float | None = None
    rank: int | None = None

    @property
    def supply(self) -> str:
        return elements.supply_label(self.elements)

    @property
    def complete(self) -> bool:
        return self.descriptor is not None


def binding_summary(results: list[dict]) -> dict[str, dict]:
    """Per material: its clean slab, its lowest binding energy per adsorbate over the sites that stayed put."""
    out: dict[str, dict] = {}
    for r in results:
        m = out.setdefault(r["material"], {"slab": None, "binding": {}, "error": ""})
        if r["site"] == "error":
            m["error"] = r.get("error", "failed")
        elif r["adsorbate"] == "*":
            m["slab"] = r
        else:
            b = m["binding"].setdefault(r["adsorbate"], {"E": None, "site": "", "structure": "", "tried": 0, "kept": 0,
                                                         "sites": []})
            b["tried"] += 1
            b["sites"].append({"site": r["site"], "E": r["E_bind"], "note": r.get("anomaly", ""),
                               "structure": r["structure"], "converged": r.get("converged", True),
                               "n_ads": r.get("n_ads", 1)})
            if not r.get("anomaly"):
                b["kept"] += 1
                if b["E"] is None or r["E_bind"] < b["E"]:
                    b.update(E=r["E_bind"], site=r["site"], structure=r["structure"])
    return out


def predict(pack: Pack, job: dict, results: list[dict], n_temperatures: int = 31) -> tuple[list[Prediction], dict]:
    """Predictions for every material in a screen, best first, and the curves behind them."""
    spec = pack.spec.screening
    req = job["request"]
    summary = binding_summary(results)
    preds = []
    for m in req["materials"]:
        s = summary.get(m["id"], {"slab": None, "binding": {}, "error": ""})
        p = Prediction(material=m["id"], label=m.get("label", m["id"]), formula=m.get("formula", m["id"]),
                       is_reference=bool(m.get("is_reference")), elements=m.get("elements", []),
                       e_hull=m.get("e_hull"), source=m.get("source", ""), binding=s["binding"], slab=s["slab"],
                       error=s["error"])
        if all(s["binding"].get(a, {}).get("E") is not None for a in spec.descriptor):
            p.descriptor = float(sum(w * s["binding"][a]["E"] for a, w in spec.descriptor.items()))
        preds.append(p)

    ref = next((p for p in preds if p.is_reference), None)
    t_lo, t_hi = min(req["conditions"]["temperatures_C"]), max(req["conditions"]["temperatures_C"])
    grid = list(np.linspace(t_lo, t_hi, n_temperatures))
    curves = {"T_C": grid, "materials": {}, "reference": None, "best_possible": None}
    if ref is None or ref.descriptor is None:
        return preds, curves

    t_design = req["design_temperature_C"] + 273.15
    sigma = spec.sigma_descriptor
    for p in preds:
        if p.descriptor is None:
            continue
        cases = [spec.barrier_shift(d, ref.descriptor) for d in (p.descriptor - sigma, p.descriptor, p.descriptor + sigma)]
        top = ref.descriptor + spec.optimum_shift_eV
        if p.descriptor - sigma < top < p.descriptor + sigma:
            cases.append(spec.barrier_shift(top, ref.descriptor))
        p.shift = 0.0 if p.is_reference else cases[1]
        p.shift_range = (0.0, 0.0) if p.is_reference else (min(cases), max(cases))
        p.rate_ratio = float(np.exp(-p.shift / (K_B * t_design)))
        p.rate_range = (float(np.exp(-p.shift_range[1] / (K_B * t_design))),
                        float(np.exp(-p.shift_range[0] / (K_B * t_design))))

    step = step_for({**req["conditions"], "temperatures_C": grid})
    design_step_ = step_for({**req["conditions"], "temperatures_C": [req["design_temperature_C"]]})
    done = [p for p in preds if p.shift is not None]
    sets = [fast_params(pack)] + [shifted_params(pack, s) for p in done
                                  for s in (p.shift, p.shift_range[0], p.shift_range[1])]
    values = metric_curves(pack, step, sets, spec.metric)
    at_design = metric_curves(pack, design_step_, sets, spec.metric)[:, 0]
    best = values[0]
    curves["best_possible"] = best.tolist()
    for i, p in enumerate(done):
        mid, lo, hi = values[1 + 3 * i: 4 + 3 * i]
        curves["materials"][p.material] = {"mid": mid.tolist(), "lo": np.minimum(lo, hi).tolist(),
                                           "hi": np.maximum(lo, hi).tolist()}
        d_mid, d_a, d_b = at_design[1 + 3 * i: 4 + 3 * i]
        p.metric_at_design = float(d_mid)
        p.metric_range = (float(min(d_a, d_b)), float(max(d_a, d_b)))
        reached = [t for t, v, b in zip(grid, mid, best) if b > 0 and v >= 0.5 * b]
        p.t_half_C = float(reached[0]) if reached else None
    curves["reference"] = ref.material
    ranked = sorted(done, key=lambda p: p.shift)
    for n, p in enumerate(ranked, start=1):
        p.rank = n
    preds.sort(key=lambda p: (p.rank is None, p.rank or 0, p.label))
    return preds, curves


def volcano_line(pack: Pack, reference_descriptor: float, span: float = 1.5, n: int = 61,
                 t_design_C: float = 300.0) -> tuple[np.ndarray, np.ndarray]:
    """The volcano itself: rate over the reference's against the descriptor, for the chart."""
    spec = pack.spec.screening
    top = reference_descriptor + spec.optimum_shift_eV
    x = np.linspace(top - span, top + span, n)
    y = np.array([np.exp(-spec.barrier_shift(v, reference_descriptor) / (K_B * (t_design_C + 273.15))) for v in x])
    return x, y
