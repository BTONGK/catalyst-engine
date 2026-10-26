"""Electrocatalyst screens: the computational hydrogen electrode on the surfaces and binding energies the thermal
screens compute.

An electrochemical reaction is a file in reactions/electro/<id>.yaml: its intermediates in order along the path,
each one's free-energy corrections, the descriptor, the catalyst to beat and the lines of its volcano. The engine
core never names one. Electrochemical screens live in their own folder, <data dir>/electro/screens/, apart from
the thermal screens, and the shipped examples in examples/electro/screens/.

The method (Nørskov et al., J. Phys. Chem. B 2004):
  1. each intermediate's free energy at U = 0 V vs RHE is its binding energy from the worker (set against the
     worker's gases: H2O, H2, N2, CO, NO, NH3) plus a zero-point and entropy correction, plus the hydrogen bonds
     from water where the reaction declares them, plus a fixed shift from tabulated thermochemistry that moves the
     worker's gases to the reaction's own zero (its products for a reduction, its reactants for an oxidation);
  2. each electron step transfers one proton and electron, so at potential U its free energy shifts by +eU
     (reduction) or -eU (oxidation); a proton's free energy at U = 0 is half an H2 molecule on the RHE scale, at
     any pH. A chemical step moves no electron (a molecule landing or leaving, two adsorbates pairing up), so the
     potential leaves it alone; it may carry a barrier that follows a binding energy;
  3. a reaction may declare a chemical step (O2 splitting for oxygen reduction) that opens a second route: it makes
     one intermediate straight from the reactants, with a barrier that follows a binding energy, and the route
     goes on along the path's own steps from there (the paper's eqs. 18-22);
  4. the limiting potential is the best an open route reaches, where its last electron step turns downhill:
     U_L = -max(dG_i) over the route's electron steps for a reduction, +max(dG_i) for an oxidation; the chemical
     route is open once its barrier is gone; the overpotential is the distance from the equilibrium potential;
  5. the rate at a potential follows each route's largest hurdle there, its largest step still uphill, a chemical
     step's free energy or barrier, or the route's own barrier, and the faster route carries it (Sabatier analysis,
     the paper's eq. 16 and Fig. 8):
     rate ~ exp(-h(U) / kT), so a material's speed against the catalyst to beat is exp((h_ref(U) - h(U)) / kT)
     at 298 K;
  6. the likely ranges move every intermediate's free energy anywhere within the error bar at once and take the
     exact extremes by linear programming over that box, the barrier moving with its binding energy.
Barriers of the proton transfers, the water layer's own structure and the surface's change of coverage with
potential stay outside: the Manual's Methodology tab lists them.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import linprog

from catalyst_engine.design import elements, screening
from catalyst_engine.design.spec import ADSORBATES as THERMAL_ADSORBATES
from catalyst_engine.packs import PACKS_DIR, read_yaml

ELECTRO_DIR = PACKS_DIR / "electro"
EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "electro" / "screens"
# the intermediates the worker can place, in plain words: the thermal set plus the electrochemical ones
ADSORBATES = {**THERMAL_ADSORBATES, "OOH": "hydroperoxyl", "COOH": "carboxyl", "OCHO": "formate", "NH": "imide",
              "NH2": "amide", "NNH": "diazenyl (N2H)", "NNH2": "hydrazinylidene (NNH2)", "HNNH": "diazene (HNNH)",
              "HNNH2": "hydrazinyl (N2H3)", "NOH": "NOH", "CHO": "formyl", "COH": "hydroxymethylidyne (COH)",
              "CH2O": "formaldehyde", "CHOH": "hydroxymethylene (CHOH)", "CH3O": "methoxy",
              "CH2OH": "hydroxymethyl"}
K_T = 8.617333e-5 * 298.15  # eV at 25 C
SIGMA_EV = 0.15  # uncertainty of each binding energy, as the thermal screens take it
EITHER = "Either route"  # both routes reach the same value: the step that limits lies on both
Affine = tuple[np.ndarray, float]  # c . g + d, over the intermediates' free energies in path order


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Intermediate(_Strict):
    name: str  # an adsorbate the worker can place
    label: str = ""
    correction_eV: float = 0.0  # zero-point energy minus T x entropy at 298 K
    water_eV: float = 0.0  # stabilization by hydrogen bonds from liquid water
    layer_eV: float = 0.0  # stabilization by hydrogen bonds within the adsorbate layer at the working coverage
    # moves the worker's gas reference to the reaction's own zero, from tabulated formation free energies
    shift_eV: float = 0.0

    @property
    def total_correction(self) -> float:
        return self.correction_eV + self.water_eV + self.layer_eV


class Barrier(_Strict):
    """A chemical step's barrier from one intermediate's binding energy, E_a = slope x dE + intercept (eV)."""
    from_: str = Field(alias="from")
    slope: float
    intercept: float
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ChemicalStep(_Strict):
    """A step of the path that moves no electron, so the potential leaves it alone: a molecule landing on the
    surface or leaving it, or two adsorbates pairing up. `step` counts along the path from 0 (start to the first
    intermediate) to the number of intermediates (the last one to the end). Its hurdle is its free energy, or its
    barrier when that is higher."""
    step: int = Field(ge=0)
    label: str
    barrier: Barrier | None = None


class Activation(_Strict):
    """A chemical step that opens a second route: it makes `joins_at` straight from the reactants, and the route
    goes on along the path's own proton-electron steps from there. Its barrier follows one intermediate's binding
    energy, E_a = slope x dE + intercept (a Brønsted-Evans-Polanyi relation), and holds at every potential;
    `bonus_eV` is what its looser transition state earns over a proton transfer (Nørskov 2004: 0.13 eV for O2
    splitting at 1 bar and 300 K)."""
    label: str  # the chemical step, in plain words
    route: str  # the route it opens, in plain words
    from_: str = Field(alias="from")
    joins_at: str
    slope: float
    intercept: float
    bonus_eV: float = 0.0
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def barrier(self, binding: dict[str, float]) -> float:
        return self.slope * binding[self.from_] + self.intercept


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
    route: str = "The electron steps"  # the path's own route, in plain words, beside a chemical one
    descriptor: dict[str, float]
    volcano_steps: list[list[float]] = Field(default_factory=list)  # each step's dG as a x + b in the descriptor
    volcano_barrier: list[float] | None = None  # the chemical barrier, less its bonus, as [a, b] in the descriptor
    activation: Activation | None = None  # a chemical step and the route it opens
    chemical_steps: list[ChemicalStep] = Field(default_factory=list)  # steps along the path that move no electron
    reference: ElectroReference
    task: str = "oc20"
    suggested_elements: list[str] = Field(default_factory=list)
    on_design_page: bool = True  # False keeps a reaction to the playground until its check passes
    design_note: str = ""  # what it waits for before the Design page, in plain words
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
        chem = [c.step for c in self.chemical_steps]
        if len(set(chem)) != len(chem) or any(k > len(names) for k in chem):
            raise ValueError(f"{self.id}: each chemical step names a distinct step of the path, 0 to {len(names)}")
        if len(names) + 1 - len(chem) != self.electrons:
            raise ValueError(f"{self.id}: {self.electrons} electrons need {self.electrons} electron steps, so "
                             f"{self.electrons - 1 + len(chem)} intermediates beside {len(chem)} chemical steps")
        if {c.barrier.from_ for c in self.chemical_steps if c.barrier} - set(names):
            raise ValueError(f"{self.id}: a chemical step's barrier follows an intermediate of the path")
        if chem and self.activation is not None:
            raise ValueError(f"{self.id}: a second route and chemical steps along the path stay in separate reactions")
        if any(len(ab) != 2 for ab in self.volcano_steps):
            raise ValueError(f"{self.id}: each volcano step is [slope, intercept]")
        if self.activation is not None and not {self.activation.from_, self.activation.joins_at} <= set(names):
            raise ValueError(f"{self.id}: the chemical step follows, and joins, intermediates of the path")
        if self.volcano_barrier is not None and (self.activation is None or len(self.volcano_barrier) != 2):
            raise ValueError(f"{self.id}: the volcano's barrier line is [slope, intercept] for a chemical step")
        return self

    @property
    def adsorbates(self) -> list[str]:
        return [i.name for i in self.intermediates]

    @property
    def n_steps(self) -> int:
        return len(self.intermediates) + 1

    @property
    def electron_steps(self) -> list[int]:
        """The steps that move a proton and electron, in path order."""
        chem = {c.step for c in self.chemical_steps}
        return [k for k in range(self.n_steps) if k not in chem]

    def chemical(self, k: int) -> ChemicalStep | None:
        return next((c for c in self.chemical_steps if c.step == k), None)

    @property
    def total_eV(self) -> float:
        """Free energy released by the whole reaction at U = 0: electrons x equilibrium potential."""
        return self.electrons * self.equilibrium_V

    def ideal(self) -> dict[str, float]:
        """The ideal catalyst's free energies at U = 0: every step the same size, so all of them turn downhill
        together at the equilibrium potential. A reduction path runs down from the reactants, an oxidation path up."""
        step = self.total_eV / self.electrons * (-1 if self.direction == "reduction" else 1)
        level = self.total_eV if self.direction == "reduction" else 0.0
        out, electron = {}, set(self.electron_steps)
        for k, a in enumerate(self.adsorbates):
            level += step if k in electron else 0.0  # a chemical step stays level on the ideal catalyst
            out[a] = level
        return out

    @property
    def sigma_descriptor(self) -> float:
        return SIGMA_EV * sum(w * w for w in self.descriptor.values()) ** 0.5

    @property
    def join_step(self) -> int:
        """The chemical route's first proton-electron step: the one that leaves the intermediate it makes."""
        return 1 + self.adsorbates.index(self.activation.joins_at) if self.activation is not None else 0

    def routes(self) -> list[tuple[str, list[int], bool]]:
        """Each route: its name, the steps it takes along the path, and whether it carries the chemical barrier. The
        path's own route takes every step, a chemical step's included."""
        path = list(range(self.n_steps))
        if self.activation is None:
            return [(self.route, path, False)]
        return [(self.route, path, False), (self.activation.route, path[self.join_step:], True)]

    def _binding_of(self, g: dict[str, float], name: str) -> float:
        """An intermediate's binding energy back from its free energy: less its corrections and its shift."""
        i = self.intermediates[self.adsorbates.index(name)]
        return g[name] - i.total_correction - i.shift_eV

    def free_energies(self, binding: dict[str, float]) -> dict[str, float]:
        """Each intermediate's free energy at U = 0 vs RHE from its binding energy (eV), on the reaction's own scale."""
        return {i.name: binding[i.name] + i.total_correction + i.shift_eV for i in self.intermediates}

    def levels(self, g: dict[str, float]) -> list[float]:
        """Free energy along the path at U = 0: the start, each intermediate, the end."""
        inner = [g[i.name] for i in self.intermediates]
        return [self.total_eV, *inner, 0.0] if self.direction == "reduction" else [0.0, *inner, self.total_eV]

    def steps(self, g: dict[str, float]) -> list[float]:
        lv = self.levels(g)
        return [b - a for a, b in zip(lv, lv[1:])]

    def steps_at(self, g: dict[str, float], potential_V: float) -> list[float]:
        """Each step's free energy at a potential: an electron step moves by one electron's worth, a chemical step
        stays put."""
        shift = potential_V if self.direction == "reduction" else -potential_V
        electron = set(self.electron_steps)
        return [s + shift if k in electron else s for k, s in enumerate(self.steps(g))]

    def step_lines(self, potential_V: float = 0.0) -> list[Affine]:
        """Each step's free energy at a potential as an affine function of the intermediates' free energies."""
        n = len(self.intermediates)
        start, end = (self.total_eV, 0.0) if self.direction == "reduction" else (0.0, self.total_eV)
        shift = potential_V if self.direction == "reduction" else -potential_V
        electron = set(self.electron_steps)
        lines = []
        for k in range(n + 1):
            c, d = np.zeros(n), shift if k in electron else 0.0
            if k < n:
                c[k] += 1.0
            else:
                d += end
            if k > 0:
                c[k - 1] -= 1.0
            else:
                d -= start
            lines.append((c, d))
        return lines

    def barrier_line(self) -> Affine | None:
        """The chemical barrier, less its bonus, as an affine function of the free energies: its binding energy is
        that intermediate's free energy less its corrections."""
        a = self.activation
        if a is None:
            return None
        j = self.adsorbates.index(a.from_)
        c = np.zeros(len(self.intermediates))
        c[j] = a.slope
        i = self.intermediates[j]
        return c, a.intercept - a.bonus_eV - a.slope * (i.total_correction + i.shift_eV)

    def chemical_barrier_line(self, k: int) -> Affine | None:
        """A chemical step's barrier as an affine function of the free energies; None without one."""
        c_step = self.chemical(k)
        if c_step is None or c_step.barrier is None:
            return None
        b = c_step.barrier
        j = self.adsorbates.index(b.from_)
        c = np.zeros(len(self.intermediates))
        c[j] = b.slope
        i = self.intermediates[j]
        return c, b.intercept - b.slope * (i.total_correction + i.shift_eV)

    def chemical_hurdles(self, g: dict[str, float]) -> dict[int, float]:
        """Each chemical step's hurdle (eV): its free energy, or its barrier when that is higher. The potential
        leaves both alone."""
        st = self.steps(g)
        out = {}
        for c in self.chemical_steps:
            barrier = None if c.barrier is None else c.barrier.slope * self._binding_of(g, c.barrier.from_) + c.barrier.intercept
            out[c.step] = st[c.step] if barrier is None else max(st[c.step], barrier)
        return out

    def barrier(self, g: dict[str, float]) -> float | None:
        """The chemical step's barrier on a surface (eV) from its free energies at U = 0, with its prefactor bonus
        taken off; None without one."""
        line = self.barrier_line()
        return None if line is None else float(line[0] @ np.array([g[n] for n in self.adsorbates]) + line[1])

    def limiting_potential(self, g: dict[str, float]) -> tuple[float, int]:
        """The best potential an open route reaches, where its last step turns downhill (the highest for a
        reduction, the lowest for an oxidation), and which step that is. The chemical route counts once its barrier
        is gone; it takes part of the path's steps, so it then reaches at least as far as the path."""
        st = self.steps(g)
        take = self.routes()[-1][1] if self.activation is not None and self.barrier(g) <= 0 else self.electron_steps
        k = max(take, key=lambda i: st[i])
        return (-st[k], k) if self.direction == "reduction" else (st[k], k)

    def limiting_route(self, g: dict[str, float]) -> str:
        """The route that reaches the limiting potential; EITHER when both reach it alike."""
        if self.activation is None or self.barrier(g) > 0:
            return self.route
        st = self.steps(g)
        return EITHER if max(st[self.join_step:]) >= max(st) - 1e-9 else self.activation.route

    def overpotential(self, g: dict[str, float]) -> float:
        u, _ = self.limiting_potential(g)
        return (self.equilibrium_V - u) if self.direction == "reduction" else (u - self.equilibrium_V)

    def uphill(self, g: dict[str, float], potential_V: float) -> float:
        """The largest step of the path still uphill at a potential (eV); zero once every step runs downhill. A
        chemical step uphill stays so at every potential."""
        return max(0.0, max(self.steps_at(g, potential_V)))

    def route_hurdles(self, g: dict[str, float], potential_V: float) -> dict[str, float]:
        """Each route's largest hurdle at a potential (eV): its largest step still uphill, or its chemical barrier
        when that is higher; zero once the route runs downhill throughout."""
        st, b, ch = self.steps_at(g, potential_V), self.barrier(g), self.chemical_hurdles(g)
        return {name: max(0.0, *(ch.get(k, st[k]) for k in take), *((b,) if chem else ()))
                for name, take, chem in self.routes()}

    def hurdle(self, g: dict[str, float], potential_V: float) -> float:
        """The hurdle that sets the rate at a potential: the faster route's, since the routes run side by side
        (Sabatier analysis, Nørskov 2004, eq. 16 and Fig. 8: the higher of the two activities dominates)."""
        return min(self.route_hurdles(g, potential_V).values())

    def route_at(self, g: dict[str, float], potential_V: float) -> str:
        """The faster route at a potential; EITHER when both run alike."""
        h = self.route_hurdles(g, potential_V)
        best = min(h.values())
        fast = [name for name, v in h.items() if v <= best + 1e-9]
        return fast[0] if len(fast) == 1 else EITHER

    def volcano_limit(self, x: float) -> float:
        """The limiting potential of the drawn volcano at descriptor x, through the reaction's scaling lines: the
        best open route, the chemical one where its barrier line sits at or below zero."""
        lines = [a * x + b for a, b in self.volcano_steps]  # the electron steps that can limit
        top = max(lines)
        if self.volcano_barrier is not None and self.volcano_barrier[0] * x + self.volcano_barrier[1] <= 0:
            top = max(lines[self.join_step:])
        return -top if self.direction == "reduction" else top

    def step_names(self) -> list[str]:
        names = ["*"] + self.adsorbates + ["*"]
        if self.direction == "reduction":
            names[0] = "start"
        else:
            names[-1] = "end"
        out = []
        for k, (a, b) in enumerate(zip(names, names[1:])):
            c = self.chemical(k)
            out.append(f"{a} → {b}" + (f" ({c.label})" if c else ""))
        return out


def list_electro() -> list[str]:
    """Every electrochemical reaction file, including those kept to the playground."""
    return sorted(p.stem for p in ELECTRO_DIR.glob("*.yaml")) if ELECTRO_DIR.is_dir() else []


def load_electro(spec_id: str) -> ElectroSpec:
    spec = ElectroSpec.model_validate(read_yaml(ELECTRO_DIR / f"{spec_id}.yaml"))
    if spec.id != spec_id:
        raise ValueError(f"electro reaction id {spec.id!r} must match its file name {spec_id!r}")
    return spec


def is_electro(job: dict) -> bool:
    return job.get("request", {}).get("kind") == "electro"


def screens_dir() -> Path:
    return Path(os.environ.get("CE_DATA_DIR", Path(__file__).resolve().parents[3] / "data")) / "electro" / "screens"


def list_screens() -> list[Path]:
    root = screens_dir()
    return sorted((p for p in root.iterdir() if (p / "job.json").exists()), reverse=True) if root.is_dir() else []


def add_examples() -> None:
    """Copy the shipped example screens into an empty electrochemistry screens folder."""
    root = screens_dir()
    if EXAMPLES.is_dir() and not (root.is_dir() and any(root.iterdir())):
        for src in sorted(EXAMPLES.iterdir()):
            if (src / "job.json").exists():
                shutil.copytree(src, root / src.name)


def create_screen(spec: ElectroSpec, name: str, conditions: dict, materials: list[dict],
                  thoroughness: str = "standard", task: str | None = None, slab: dict | None = None) -> Path:
    """Write an electrochemical screen folder; the worker runs it exactly as a thermal one. `conditions` holds
    compare_V, the operating potential vs RHE for the headline numbers."""
    if not any(m.get("is_reference") for m in materials):
        raise ValueError("include the reference catalyst")
    task = task or spec.task
    stamp = datetime.now()
    folder = screens_dir() / f"{stamp:%Y%m%d-%H%M%S}-{screening._slug(name)}"
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
    limiting_route: str = ""
    limiting_range: tuple[float, float] | None = None
    overpotential_V: float | None = None
    descriptor: float | None = None
    barrier_eV: float | None = None  # the chemical step's barrier, where the reaction has one
    chemical_eV: dict[int, float] = field(default_factory=dict)  # each chemical step's hurdle along the path
    hurdle_eV: float | None = None  # the faster route's largest hurdle at the comparison potential
    route: str = ""  # the faster route at the comparison potential
    rate_ratio: float | None = None  # speed over the catalyst to beat at the comparison potential
    rate_range: tuple[float, float] | None = None
    rank: int | None = None

    @property
    def supply(self) -> str:
        return elements.supply_label(self.elements)

    @property
    def complete(self) -> bool:
        return self.limiting_V is not None


# ---------------------------------------------------------------- likely ranges: exact extremes over the error box

def _box(spec: ElectroSpec, g: dict[str, float], sigma: float) -> list[tuple[float, float]]:
    return [(g[n] - sigma, g[n] + sigma) for n in spec.adsorbates]


def _min_of_max(terms: list[Affine], box: list[tuple[float, float]], extra: list[Affine] = ()) -> float | None:
    """The least that the largest of some affine terms reaches over a box, within extra half-spaces (each term
    of `extra` at or below zero): a linear program in the free energies and one more variable. None when the
    half-spaces leave nothing of the box."""
    n = len(box)
    a_ub = [[*c, -1.0] for c, _ in terms] + [[*c, 0.0] for c, _ in extra]
    b_ub = [-d for _, d in terms] + [-d for _, d in extra]
    res = linprog(np.r_[np.zeros(n), 1.0], A_ub=a_ub, b_ub=b_ub, bounds=[*box, (None, None)], method="highs")
    return float(res.fun) if res.status == 0 else None


def _max_of_min(terms: list[Affine], box: list[tuple[float, float]], extra: list[Affine] = ()) -> float | None:
    """The most that the smallest of some affine terms reaches over a box: the mirror image of _min_of_max."""
    n = len(box)
    a_ub = [[*(-c), 1.0] for c, _ in terms] + [[*c, 0.0] for c, _ in extra]
    b_ub = [d for _, d in terms] + [-d for _, d in extra]
    res = linprog(np.r_[np.zeros(n), -1.0], A_ub=a_ub, b_ub=b_ub, bounds=[*box, (None, None)], method="highs")
    return float(-res.fun) if res.status == 0 else None


def limiting_range(spec: ElectroSpec, g: dict[str, float], sigma: float = SIGMA_EV) -> tuple[float, float]:
    """The exact spread of the limiting potential while every intermediate's free energy moves anywhere within
    ±sigma at once. The box splits where the chemical barrier reaches zero: on one side the path's own steps
    count, on the other the chemical route's."""
    box, lines = _box(spec, g, sigma), spec.step_lines()
    steps = [lines[k] for k in spec.electron_steps]  # the limiting potential follows the electron steps
    parts = [(steps, [])]
    line = spec.barrier_line()
    if line is not None:
        chem = steps[spec.join_step:]
        parts = []
        if _max_of_min([line], box) > 0:  # somewhere the barrier stands
            parts.append((steps, [(-line[0], -line[1])]))
        if _min_of_max([line], box) <= 0:  # somewhere it is gone
            parts.append((chem, [line]))
    lo, hi = [], []
    for terms, extra in parts:
        least_top = _min_of_max(terms, box, extra)
        most_top = max(_max_of_min([t], box, extra) for t in terms)
        if spec.direction == "reduction":  # U_L = -(the largest step)
            lo.append(-most_top)
            hi.append(-least_top)
        else:  # U_L = the largest step
            lo.append(least_top)
            hi.append(most_top)
    return min(lo) + 0.0, max(hi) + 0.0  # + 0.0 turns a negated zero into a plain one


def hurdle_range(spec: ElectroSpec, g: dict[str, float], potential_V: float,
                 sigma: float = SIGMA_EV) -> tuple[float, float]:
    """The exact spread of the hurdle at a potential over the same box: the least any route's hurdle reaches, and
    the most the faster route's reaches. The barrier moves with its binding energy. The faster route's hurdle is
    the smallest of the routes' largest terms, so its maximum is the largest, over one term per route, of the most
    the smallest of them reaches."""
    box, steps = _box(spec, g, sigma), spec.step_lines(potential_V)
    zero = (np.zeros(len(box)), 0.0)
    walls = {c.step: spec.chemical_barrier_line(c.step) for c in spec.chemical_steps if c.barrier}
    per_route = [[zero, *(steps[k] for k in take), *(walls[k] for k in take if k in walls),
                  *((spec.barrier_line(),) if chem else ())] for _, take, chem in spec.routes()]
    lo = min(_min_of_max(terms, box) for terms in per_route)
    hi = max(_max_of_min(list(pick), box) for pick in itertools.product(*per_route))
    return lo, hi


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
            p.limiting_route = spec.limiting_route(g)
            p.overpotential_V = spec.overpotential(g)
            p.limiting_range = limiting_range(spec, g)
            p.descriptor = float(sum(w * g[a] for a, w in spec.descriptor.items()))
            p.barrier_eV = spec.barrier(g)
            p.chemical_eV = spec.chemical_hurdles(g)
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
    ref_up = np.array([spec.hurdle(ref.free_energies, u) for u in grid])
    ref_cmp = spec.hurdle(ref.free_energies, u_cmp)
    for p in done:
        up = np.array([spec.hurdle(p.free_energies, u) for u in grid])
        p.hurdle_eV = spec.hurdle(p.free_energies, u_cmp)
        p.route = spec.route_at(p.free_energies, u_cmp)
        lo, hi = hurdle_range(spec, p.free_energies, u_cmp)
        p.rate_ratio = float(np.exp((ref_cmp - p.hurdle_eV) / K_T))
        p.rate_range = (float(np.exp((ref_cmp - hi) / K_T)), float(np.exp((ref_cmp - lo) / K_T)))
        curves["materials"][p.material] = {"uphill": up.tolist(), "ratio": np.exp((ref_up - up) / K_T).tolist()}
    curves["reference"] = ref.material
    # best first: the smallest hurdle at the comparison potential, then the smallest overpotential
    ranked = sorted(done, key=lambda p: (round(p.hurdle_eV, 6), p.overpotential_V, p.label))
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
    electron = set(spec.electron_steps)
    moved_by = [sum(1 for j in range(k) if j in electron) for k in range(len(lv))]  # electrons moved so far
    moved = [v - (n - e) * potential_V for e, v in zip(moved_by, lv)] if spec.direction == "reduction" else \
        [v - e * potential_V for e, v in zip(moved_by, lv)]
    return [{"state": s, "level": float(v), "step": k} for k, (s, v) in enumerate(zip(names, moved))]
