"""Electrocatalyst screens: the computational hydrogen electrode on the surfaces and binding energies the thermal
screens compute.

An electrochemical reaction is a file in reactions/electro/<id>.yaml: its intermediates in order along the path,
each one's free-energy corrections, the descriptor, the catalyst to beat and the lines of its volcano. The engine
core never names one.

The method (Nørskov et al., J. Phys. Chem. B 2004):
  1. each intermediate's free energy against liquid water and H2 at U = 0 V vs RHE is its binding energy from the
     worker (set against H2O and H2 gas) plus a zero-point and entropy correction, plus the hydrogen bonds from
     water where the reaction declares them;
  2. every step transfers one proton and electron, so at potential U its free energy shifts by -eU (reduction) or
     +eU (oxidation); a proton's free energy at U = 0 is half an H2 molecule on the RHE scale, at any pH;
  3. the limiting potential is where the last step turns downhill: U_L = -max(dG_i) for a reduction,
     +max(dG_i) for an oxidation; the overpotential is its distance from the equilibrium potential;
  4. the rate at a potential follows the largest step still uphill there (Sabatier analysis, Nørskov 2004,
     eqs. 9-11): rate ~ exp(-max(0, dG_i(U)) / kT), so a material's speed against the catalyst to beat is
     exp((dG_max,ref(U) - dG_max(U)) / kT) at 298 K.
Barriers between the steps, the water layer's own structure and the surface's change of coverage with potential
stay outside: the Manual's Methodology tab lists them.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalyst_engine.design import elements, screening
from catalyst_engine.design.spec import ADSORBATES
from catalyst_engine.packs import PACKS_DIR, read_yaml

ELECTRO_DIR = PACKS_DIR / "electro"
K_T = 8.617333e-5 * 298.15  # eV at 25 C
SIGMA_EV = 0.15  # uncertainty of each binding energy, as the thermal screens take it


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Intermediate(_Strict):
    name: str  # an adsorbate the worker can place
    label: str = ""
    correction_eV: float = 0.0  # zero-point energy minus T x entropy at 298 K
    water_eV: float = 0.0  # stabilization by hydrogen bonds from liquid water


class ElectroReference(_Strict):
    formula: str
    name: str = ""


class ElectroSpec(_Strict):
    id: str
    name: str
    category: str = "Hydrogen and energy"
    summary: str = ""
    equation: str
    direction: Literal["reduction", "oxidation"]
    electrons: int = Field(gt=0)
    equilibrium_V: float
    compare_V: float
    intermediates: list[Intermediate]
    descriptor: dict[str, float]
    volcano_steps: list[list[float]] = Field(default_factory=list)  # each step's dG as a x + b in the descriptor
    reference: ElectroReference
    task: str = "oc20"
    suggested_elements: list[str] = Field(default_factory=list)
    basis: str = ""

    @model_validator(mode="after")
    def _check(self) -> "ElectroSpec":
        names = [i.name for i in self.intermediates]
        unknown = set(names) | set(self.descriptor)
        unknown -= set(ADSORBATES)
        if unknown:
            raise ValueError(f"{self.id} names unknown adsorbates {sorted(unknown)}")
        if not names or len(set(names)) != len(names):
            raise ValueError(f"{self.id} needs a list of distinct intermediates")
        if len(names) != self.electrons - 1:
            raise ValueError(f"{self.id}: {self.electrons} electrons need {self.electrons - 1} intermediates")
        if any(len(ab) != 2 for ab in self.volcano_steps):
            raise ValueError(f"{self.id}: each volcano step is [slope, intercept]")
        return self

    @property
    def adsorbates(self) -> list[str]:
        return [i.name for i in self.intermediates]

    @property
    def total_eV(self) -> float:
        """Free energy released by the whole reaction at U = 0: electrons x equilibrium potential."""
        return self.electrons * self.equilibrium_V

    @property
    def sigma_descriptor(self) -> float:
        return SIGMA_EV * sum(w * w for w in self.descriptor.values()) ** 0.5

    def free_energies(self, binding: dict[str, float]) -> dict[str, float]:
        """Each intermediate's free energy at U = 0 vs RHE from its binding energy (eV)."""
        return {i.name: binding[i.name] + i.correction_eV + i.water_eV for i in self.intermediates}

    def levels(self, g: dict[str, float]) -> list[float]:
        """Free energy along the path at U = 0: the start, each intermediate, the end."""
        inner = [g[i.name] for i in self.intermediates]
        return [self.total_eV, *inner, 0.0] if self.direction == "reduction" else [0.0, *inner, self.total_eV]

    def steps(self, g: dict[str, float]) -> list[float]:
        lv = self.levels(g)
        return [b - a for a, b in zip(lv, lv[1:])]

    def steps_at(self, g: dict[str, float], potential_V: float) -> list[float]:
        """Each step's free energy at a potential: every step moves one electron."""
        shift = potential_V if self.direction == "reduction" else -potential_V
        return [s + shift for s in self.steps(g)]

    def limiting_potential(self, g: dict[str, float]) -> tuple[float, int]:
        """The potential where the last step turns downhill, and which step that is."""
        st = self.steps(g)
        k = int(np.argmax(st))
        return (-st[k], k) if self.direction == "reduction" else (st[k], k)

    def overpotential(self, g: dict[str, float]) -> float:
        u, _ = self.limiting_potential(g)
        return (self.equilibrium_V - u) if self.direction == "reduction" else (u - self.equilibrium_V)

    def uphill(self, g: dict[str, float], potential_V: float) -> float:
        """The largest step still uphill at a potential (eV); zero once every step runs downhill."""
        return max(0.0, max(self.steps_at(g, potential_V)))

    def volcano_limit(self, x: float) -> float:
        """The limiting potential of the drawn volcano at descriptor x, through the reaction's scaling lines."""
        top = max(a * x + b for a, b in self.volcano_steps)
        return -top if self.direction == "reduction" else top

    def step_names(self) -> list[str]:
        names = ["*"] + self.adsorbates + ["*"]
        if self.direction == "reduction":
            names[0] = "start"
        else:
            names[-1] = "end"
        return [f"{a} → {b}" for a, b in zip(names, names[1:])]


def list_electro() -> list[str]:
    return sorted(p.stem for p in ELECTRO_DIR.glob("*.yaml")) if ELECTRO_DIR.is_dir() else []


def load_electro(spec_id: str) -> ElectroSpec:
    spec = ElectroSpec.model_validate(read_yaml(ELECTRO_DIR / f"{spec_id}.yaml"))
    if spec.id != spec_id:
        raise ValueError(f"electro reaction id {spec.id!r} must match its file name {spec_id!r}")
    return spec


def is_electro(job: dict) -> bool:
    return job.get("request", {}).get("kind") == "electro"


def create_screen(spec: ElectroSpec, name: str, conditions: dict, materials: list[dict],
                  thoroughness: str = "standard", task: str | None = None, slab: dict | None = None) -> Path:
    """Write an electrochemical screen folder; the worker runs it exactly as a thermal one. `conditions` holds
    compare_V, the operating potential vs RHE for the headline numbers."""
    if not any(m.get("is_reference") for m in materials):
        raise ValueError("include the reference catalyst")
    task = task or spec.task
    stamp = datetime.now()
    folder = screening.screens_dir() / f"{stamp:%Y%m%d-%H%M%S}-{screening._slug(name)}"
    folder.mkdir(parents=True)
    request = {"name": name, "kind": "electro", "pack": spec.id, "pack_name": spec.name,
               "created": stamp.isoformat(timespec="seconds"), "conditions": conditions, "thoroughness": thoroughness,
               "task": task, "materials": [{k: v for k, v in m.items() if k not in ("species", "frac", "lattice")}
                                           for m in materials], **({"slab": slab} if slab else {})}
    job = {"checkpoint": screening.uma_setup()["checkpoint"], "task": task, "thoroughness": thoroughness,
           "adsorbates": spec.adsorbates, "materials": materials, "request": request, **({"slab": slab} if slab else {})}
    (folder / "job.json").write_text(json.dumps(job, indent=1))
    (folder / "status.json").write_text(json.dumps({"state": "queued", "done": 0, "total": 0,
                                                     "message": "Waiting to start", "updated": time.time()}))
    return folder


@dataclass
class ElectroPrediction:
    material: str
    label: str
    formula: str
    is_reference: bool
    elements: list[str]
    e_hull: float | None
    source: str
    binding: dict[str, dict] = field(default_factory=dict)
    slab: dict | None = None
    error: str = ""
    missing: list[str] = field(default_factory=list)  # intermediates that broke apart or drifted off at every site
    free_energies: dict[str, float] = field(default_factory=dict)  # at U = 0 vs RHE
    steps: list[float] = field(default_factory=list)
    limiting_step: int | None = None
    limiting_V: float | None = None
    limiting_range: tuple[float, float] | None = None
    overpotential_V: float | None = None
    descriptor: float | None = None
    rate_ratio: float | None = None  # speed over the catalyst to beat at the comparison potential
    rate_range: tuple[float, float] | None = None
    rank: int | None = None

    @property
    def supply(self) -> str:
        return elements.supply_label(self.elements)

    @property
    def complete(self) -> bool:
        return self.limiting_V is not None


def _cases(spec: ElectroSpec, g: dict[str, float]) -> list[dict[str, float]]:
    """Every combination of each intermediate's free energy at, below and above its value by the error bar."""
    names = spec.adsorbates
    return [{n: g[n] + d for n, d in zip(names, deltas)} for deltas in itertools.product((-SIGMA_EV, 0.0, SIGMA_EV),
                                                                                       repeat=len(names))]


def predict(spec: ElectroSpec, job: dict, results: list[dict], n_potentials: int = 61
            ) -> tuple[list[ElectroPrediction], dict]:
    """Predictions for every material of an electrochemical screen, best first, and the curves behind them."""
    req = job["request"]
    summary = screening.binding_summary(results)
    preds: list[ElectroPrediction] = []
    for m in req["materials"]:
        s = summary.get(m["id"], {"slab": None, "binding": {}, "error": ""})
        p = ElectroPrediction(material=m["id"], label=m.get("label", m["id"]), formula=m.get("formula", m["id"]),
                              is_reference=bool(m.get("is_reference")), elements=m.get("elements", []),
                              e_hull=m.get("e_hull"), source=m.get("source", ""), binding=s["binding"], slab=s["slab"],
                              error=s["error"])
        bind = {a: s["binding"].get(a, {}).get("E") for a in spec.adsorbates}
        p.missing = [a for a in spec.adsorbates if bind[a] is None and s["binding"].get(a, {}).get("tried")]
        if all(v is not None for v in bind.values()):
            g = spec.free_energies(bind)
            p.free_energies = g
            p.steps = spec.steps(g)
            p.limiting_V, p.limiting_step = spec.limiting_potential(g)
            p.overpotential_V = spec.overpotential(g)
            limits = [spec.limiting_potential(c)[0] for c in _cases(spec, g)]
            p.limiting_range = (min(limits), max(limits))
            p.descriptor = float(sum(w * g[a] for a, w in spec.descriptor.items()))
        preds.append(p)

    ref = next((p for p in preds if p.is_reference), None)
    u_cmp = float(req["conditions"]["compare_V"])
    span = (spec.equilibrium_V - 1.0, spec.equilibrium_V + 0.3) if spec.direction == "reduction" \
        else (spec.equilibrium_V - 0.3, spec.equilibrium_V + 1.0)
    grid = np.linspace(*span, n_potentials)
    curves = {"U_V": grid.tolist(), "materials": {}, "reference": None, "ideal": None}
    if ref is None or not ref.complete:
        return preds, curves
    done = [p for p in preds if p.complete]
    # the ideal catalyst: every step equal, so all of them turn downhill at the equilibrium potential
    ideal_step = spec.total_eV / spec.electrons * (-1 if spec.direction == "reduction" else 1)
    shift = grid if spec.direction == "reduction" else -grid
    curves["ideal"] = np.maximum(0.0, ideal_step + shift).tolist()  # its largest uphill step (eV)
    ref_up = np.array([spec.uphill(ref.free_energies, u) for u in grid])
    ref_cmp = spec.uphill(ref.free_energies, u_cmp)
    for p in done:
        up = np.array([spec.uphill(p.free_energies, u) for u in grid])
        cases = [spec.uphill(c, u_cmp) for c in _cases(spec, p.free_energies)]
        p.rate_ratio = float(np.exp((ref_cmp - spec.uphill(p.free_energies, u_cmp)) / K_T))
        p.rate_range = (float(np.exp((ref_cmp - max(cases)) / K_T)), float(np.exp((ref_cmp - min(cases)) / K_T)))
        curves["materials"][p.material] = {"uphill": up.tolist(), "ratio": np.exp((ref_up - up) / K_T).tolist()}
    curves["reference"] = ref.material
    ranked = sorted(done, key=lambda p: (p.overpotential_V, p.label))
    for n, p in enumerate(ranked, start=1):
        p.rank = n
    preds.sort(key=lambda p: (p.rank is None, p.rank or 0, p.label))
    return preds, curves


def volcano_line(spec: ElectroSpec, x_lo: float, x_hi: float, n: int = 161) -> tuple[np.ndarray, np.ndarray]:
    """The drawn volcano: the limiting potential against the descriptor, through the reaction's scaling lines."""
    x = np.linspace(x_lo, x_hi, n)
    return x, np.array([spec.volcano_limit(v) for v in x])


def diagram(spec: ElectroSpec, g: dict[str, float], potential_V: float) -> list[dict]:
    """The free-energy staircase at a potential: one row per state along the path with its level (eV). The products
    of a reduction, and the clean surface an oxidation starts from, stay at zero; each electron still to transfer
    (reduction) or already transferred (oxidation) moves a level by the potential."""
    lv = spec.levels(g)
    n = spec.electrons
    names = (["start"] if spec.direction == "reduction" else ["*"]) + spec.adsorbates + \
        (["*"] if spec.direction == "reduction" else ["end"])
    moved = [v - (n - k) * potential_V for k, v in enumerate(lv)] if spec.direction == "reduction" else \
        [v - k * potential_V for k, v in enumerate(lv)]
    return [{"state": s, "level": float(v), "step": k} for k, (s, v) in enumerate(zip(names, moved))]
