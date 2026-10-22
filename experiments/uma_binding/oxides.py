"""UMA test run, part 2: NH3 and NO on the oxides in commercial SCR catalysts (anatase TiO2, V2O5).

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/uma_binding/oxides.py --task oc20 --out out.csv

UMA has no mode trained on adsorption at oxide surfaces, so each system runs in the catalysis mode (oc20,
RPBE total energies) and the materials mode (omat, PBE/PBE+U total energies) to see which tracks experiment.

Binding energy: E_bind = E(slab + molecule) - E(slab) - E(gas molecule); negative is bound.
Gas references:
  oc20: OC20 atomic references plus the experimental formation energy of the molecule, converted to an
        electronic energy with zero-point energies (NIST-JANAF enthalpies at 0 K; harmonic ZPEs).
  omat: the molecule in a 15 A box, computed in the same mode.
Structures: COD 9015929 (anatase, Howard et al. 1991) and COD 2020756 (V2O5, Enjalbert and Galy 1986).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import surface
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).parent
CHECKPOINT = "/Users/bt/uma/checkpoints/uma-s-1p1.pt"
OC20_REFS = {"H": -3.477, "O": -7.204, "N": -8.083}
# Electronic formation energies from the OC20 reference species (eV):
#   NH3: 1/2 N2 + 3/2 H2 -> NH3        dH0 = -0.403, dZPE = +0.412  -> dE = -0.815
#   NO:  1/2 N2 + H2O - H2 -> NO       dH0 = +3.407, dZPE = -0.241  -> dE = +3.647
FORMATION = {"NH3": -0.815, "NO": 3.647}


def nh3(n_at=(0.0, 0.0, 0.0)) -> Atoms:
    h = [(0.94 * np.cos(t), 0.94 * np.sin(t), 0.38) for t in np.radians([0, 120, 240])]
    a = Atoms("NH3", positions=[(0, 0, 0), *h])
    a.translate(n_at)
    return a


def no(n_at=(0.0, 0.0, 0.0)) -> Atoms:
    a = Atoms("NO", positions=[(0, 0, 0), (0, 0, 1.17)])
    a.translate(n_at)
    return a


def coordination(a: Atoms, cutoff: float = 2.3) -> np.ndarray:
    i, _ = neighbor_list("ij", a, cutoff)
    return np.bincount(i, minlength=len(a))


def cut_slab(bulk: Atoms, miller, thickness: float, valid) -> Atoms:
    """First window of a thick periodic block that satisfies `valid` on both faces."""
    big = surface(bulk, miller, layers=8)
    z = big.positions[:, 2]
    for z0 in np.unique(np.round(z, 2)):
        a = big[(z >= z0 - 0.05) & (z <= z0 + thickness)]
        a.pbc = [True, True, False]
        if valid(a):
            a.center(vacuum=10.0, axis=2)
            return a
    raise ValueError("no valid termination found")


def anatase_101() -> Atoms:
    bulk = read(HERE / "structures" / "anatase_cod9015929.cif")

    def valid(a: Atoms) -> bool:
        s, cn = np.array(a.get_chemical_symbols()), coordination(a)
        return np.sum(s == "O") == 2 * np.sum(s == "Ti") and not np.any((s == "O") & (cn < 2)) and not np.any((s == "Ti") & (cn < 5))

    slab = cut_slab(bulk, (1, 0, 1), 9.9, valid).repeat((1, 3, 1))
    z = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(mask=z < z.min() + 2.5))
    return slab


def v2o5_001() -> Atoms:
    """The van der Waals layer surface (often called (010) in other settings), two whole layers.
    The layers interleave in height, so they are picked by bond connectivity, one complete layer at a time."""
    bulk = read(HERE / "structures" / "V2O5_cod2020756.cif")
    big = surface(bulk, (0, 0, 1), layers=6)
    big.pbc = [True, True, False]
    i, j = neighbor_list("ij", big, 2.3)
    _, labels = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(big), len(big))))
    s, cn = np.array(big.get_chemical_symbols()), coordination(big)
    whole = [lab for lab in np.unique(labels)
             if 2 * np.sum((labels == lab) & (s == "O")) == 5 * np.sum((labels == lab) & (s == "V"))
             and np.all(cn[(labels == lab) & (s == "V")] == 5)]
    whole.sort(key=lambda lab: big.positions[labels == lab, 2].mean())
    slab = big[np.isin(labels, whole[:2])]
    slab.center(vacuum=10.0, axis=2)
    slab = slab.repeat((1, 3, 1))
    z = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(mask=z < z.min() + 2.0))
    return slab


def top_sites(slab: Atoms, symbol: str, cn_wanted: int) -> list[int]:
    s, cn, z = np.array(slab.get_chemical_symbols()), coordination(slab), slab.positions[:, 2]
    idx = [i for i in range(len(slab)) if s[i] == symbol and cn[i] == cn_wanted and z[i] > z.max() - 2.5]
    return sorted(idx, key=lambda i: (slab.positions[i, 1], slab.positions[i, 0]))


def v_lewis_sites(slab: Atoms) -> list[int]:
    """Top-layer V atoms whose own vanadyl points down, leaving the open side facing the gas."""
    s, z = np.array(slab.get_chemical_symbols()), slab.positions[:, 2]
    top_v = [i for i in range(len(slab)) if s[i] == "V" and z[i] > z.max() - 3.0]
    out = []
    for i in top_v:
        d = slab.get_distances(i, list(range(len(slab))), mic=True)
        vanadyl = min((j for j in range(len(slab)) if s[j] == "O" and d[j] < 1.7), key=lambda j: d[j])
        if slab.positions[vanadyl, 2] < z[i]:
            out.append(i)
    return out


def up_vanadyl(slab: Atoms) -> int:
    s, z = np.array(slab.get_chemical_symbols()), slab.positions[:, 2]
    o_top = [i for i in range(len(slab)) if s[i] == "O" and z[i] > z.max() - 0.3]
    return o_top[0]


class Runner:
    def __init__(self, task: str):
        self.task = task
        self.calc = FAIRChemCalculator(pretrained_mlip.load_predict_unit(CHECKPOINT, device="cpu"), task_name=task)
        self.gas: dict[str, float] = {}

    def relax(self, atoms: Atoms, label: str, fmax: float = 0.05, steps: int = 500) -> tuple[float, int, bool]:
        atoms.calc = self.calc
        opt = LBFGS(atoms, logfile=None)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        write(HERE / "results" / "structures" / f"{self.task}_{label}.extxyz", atoms)
        return float(atoms.get_potential_energy()), opt.get_number_of_steps(), converged

    def box(self, mol: Atoms, label: str) -> float:
        m = mol.copy()
        m.cell = [15.0, 15.1, 15.2]
        m.center()
        m.pbc = True
        return self.relax(m, f"gas_{label}")[0]

    def gas_reference(self, name: str, mol: Atoms) -> float:
        if name not in self.gas:
            if self.task == "oc20":
                self.gas[name] = sum(OC20_REFS[s] for s in mol.get_chemical_symbols()) + FORMATION[name]
            else:
                self.gas[name] = self.box(mol, name)
        return self.gas[name]


def place_above(slab: Atoms, site: int, mol: Atoms, height: float) -> Atoms:
    a = slab.copy()
    a.set_constraint(slab.constraints)
    m = mol.copy()
    m.translate(slab.positions[site] + np.array([0.0, 0.0, height]))
    return a + m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["oc20", "omat"], required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    (HERE / "results" / "structures").mkdir(parents=True, exist_ok=True)
    run = Runner(args.task)
    rows = []

    def record(system, adsorbate, site, e_bind, steps, conv, t0, note=""):
        rows.append({"task": args.task, "system": system, "adsorbate": adsorbate, "site": site, "E_bind_eV": f"{e_bind:.4f}",
                     "steps": steps, "converged": conv, "seconds": f"{time.time() - t0:.0f}", "note": note})
        print(f"[{args.task}] {system} {adsorbate}@{site}: {e_bind:+.3f} eV ({steps} steps) {note}", flush=True)

    if args.task == "omat":  # how the materials mode handles gas molecules: NH3 formation energy vs experiment (-0.81)
        t0 = time.time()
        e_n2 = run.box(Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.10)]), "N2")
        e_h2 = run.box(Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)]), "H2")
        e_nh3 = run.gas_reference("NH3", nh3())
        record("gas", "NH3", "formation from N2+H2", e_nh3 - 0.5 * e_n2 - 1.5 * e_h2, 0, True, t0, "experiment -0.81")

    # Anatase TiO2(101): NH3 and NO on the five-fold Ti site; then a full monolayer of NH3
    t0 = time.time()
    slab = anatase_101()
    e_slab, *_ = run.relax(slab, "anatase_clean")
    ti5 = top_sites(slab, "Ti", 5)
    for name, builder, h in (("NH3", nh3, 2.2), ("NO", no, 2.2)):
        t0 = time.time()
        a = place_above(slab, ti5[0], builder(), h)
        e, steps, conv = run.relax(a, f"anatase_{name}")
        record("anatase TiO2(101)", name, "Ti5c", e - e_slab - run.gas_reference(name, builder()), steps, conv, t0,
               f"1/{len(ti5)} ML")
    t0 = time.time()
    a = slab.copy()
    a.set_constraint(slab.constraints)
    for i in ti5:
        a += nh3(slab.positions[i] + np.array([0.0, 0.0, 2.2]))
    e, steps, conv = run.relax(a, "anatase_NH3_fullML")
    record("anatase TiO2(101)", "NH3", "Ti5c, full monolayer (per molecule)",
           (e - e_slab) / len(ti5) - run.gas_reference("NH3", nh3()), steps, conv, t0, f"{len(ti5)} NH3")

    # V2O5 layer surface: Lewis V site (NH3, NO) and a Bronsted V-OH site (NH3)
    t0 = time.time()
    slab = v2o5_001()
    e_slab, *_ = run.relax(slab, "v2o5_clean")
    lewis = v_lewis_sites(slab)
    for name, builder, h in (("NH3", nh3, 2.2), ("NO", no, 2.2)):
        t0 = time.time()
        a = place_above(slab, lewis[0], builder(), h)
        e, steps, conv = run.relax(a, f"v2o5_{name}_lewis")
        record("V2O5(001)", name, "V Lewis", e - e_slab - run.gas_reference(name, builder()), steps, conv, t0)
    t0 = time.time()
    oh = slab.copy()
    oh.set_constraint(slab.constraints)
    o_site = up_vanadyl(slab)
    oh += Atoms("H", positions=[slab.positions[o_site] + [0.0, 0.0, 0.98]])
    e_oh, *_ = run.relax(oh, "v2o5_OH")
    a = oh.copy()
    a.set_constraint(oh.constraints)
    a += nh3(oh.positions[-1] + np.array([0.0, 0.0, 1.7]))
    e, steps, conv = run.relax(a, "v2o5_NH3_bronsted")
    n_idx = len(oh)
    d_nh = a.get_distances(n_idx, list(range(len(oh) - 1, len(a))), mic=True)
    note = "proton moved to N (NH4+ formed)" if d_nh[0] < 1.2 else "proton stayed on O"
    record("V2O5(001)", "NH3", "V-OH Bronsted", e - e_oh - run.gas_reference("NH3", nh3()), steps, conv, t0, note)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
