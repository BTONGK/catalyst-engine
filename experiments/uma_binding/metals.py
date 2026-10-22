"""UMA test run, part 1: nitrogen and oxygen adsorbates on close-packed metal surfaces.

Run with the conda Python that has fairchem (see CLAUDE.md):

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/uma_binding/metals.py --metals Pt,Cu --out out.csv

Binding energies follow the OC20 convention fairchem uses for UMA's catalysis task (total energies,
referenced to gas-phase H2, H2O and N2):

    E_ads = E(slab + adsorbate) - E(slab) - sum of atomic references

with H = 1/2 H2, O = H2O - H2, N = 1/2 N2 (DFT values from the OC20 paper, as in fairchem's AdsorbML recipe).
Each adsorbate starts at several sites; relaxed structures that desorb or break apart are flagged and left
out of the minimum.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import add_adsorbate, fcc111, hcp0001
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

CHECKPOINT = "/Users/bt/uma/checkpoints/uma-s-1p1.pt"
REFS = {"H": -3.477, "O": -7.204, "N": -8.083}  # eV, OC20 atomic references
METALS = {"Pt": "fcc", "Pd": "fcc", "Rh": "fcc", "Ir": "fcc", "Ni": "fcc", "Cu": "fcc", "Ag": "fcc", "Au": "fcc", "Ru": "hcp"}


def _nh3() -> Atoms:
    h = [(0.94 * np.cos(t), 0.94 * np.sin(t), 0.38) for t in np.radians([0, 120, 240])]
    return Atoms("NH3", positions=[(0, 0, 0), *h])


# Binding atom first, at the origin; (sites, height above the surface in Angstrom)
ADSORBATES = {
    "N": (Atoms("N"), ["fcc", "hcp"], 1.1),
    "O": (Atoms("O"), ["fcc", "hcp"], 1.2),
    "NH": (Atoms("NH", positions=[(0, 0, 0), (0, 0, 1.03)]), ["fcc", "hcp"], 1.1),
    "NH2": (Atoms("NH2", positions=[(0, 0, 0), (0.82, 0, 0.62), (-0.82, 0, 0.62)]), ["bridge", "fcc"], 1.5),
    "NH3": (_nh3(), ["ontop"], 2.1),
    "OH": (Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.98)]), ["fcc", "bridge", "ontop"], 1.4),
    "NO": (Atoms("NO", positions=[(0, 0, 0), (0, 0, 1.17)]), ["fcc", "hcp", "ontop"], 1.4),
}


def slab_for(metal: str) -> Atoms:
    build = fcc111 if METALS[metal] == "fcc" else hcp0001
    slab = build(metal, size=(3, 3, 4), vacuum=10.0, periodic=True)
    z = slab.positions[:, 2]
    layers = np.unique(np.round(z, 2))
    slab.set_constraint(FixAtoms(indices=[i for i, zi in enumerate(z) if round(zi, 2) <= layers[1]]))
    return slab


def relax(atoms: Atoms, calc, fmax: float = 0.05, steps: int = 300) -> tuple[float, int, bool]:
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return float(atoms.get_potential_energy()), opt.get_number_of_steps(), converged


def anomaly(adslab: Atoms, n_slab: int, ads0: Atoms) -> str:
    ads = adslab[n_slab:]
    slab_pos = adslab.positions[:n_slab]
    d_surf = np.min(np.linalg.norm(slab_pos[None, :, :] - ads.positions[:, None, :], axis=-1))
    if d_surf > 3.0:
        return "desorbed"
    if len(ads0) > 1:
        d0 = ads0.get_all_distances()
        d1 = ads.get_all_distances(mic=True)
        bonded = (d0 > 0) & (d0 < 1.3)
        if np.any(d1[bonded] > 1.5 * d0[bonded]):
            return "dissociated"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", default=",".join(METALS))
    ap.add_argument("--adsorbates", default=",".join(ADSORBATES))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    calc = FAIRChemCalculator(pretrained_mlip.load_predict_unit(CHECKPOINT, device="cpu"), task_name="oc20")
    fields = ["metal", "adsorbate", "site", "E_ads_eV", "steps", "converged", "anomaly", "seconds"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for metal in args.metals.split(","):
            t0 = time.time()
            slab = slab_for(metal)
            e_slab, steps, conv = relax(slab, calc)
            w.writerow({"metal": metal, "adsorbate": "*", "site": "clean", "E_ads_eV": f"{e_slab:.4f}", "steps": steps,
                        "converged": conv, "anomaly": "", "seconds": f"{time.time() - t0:.1f}"})
            fh.flush()
            for name in args.adsorbates.split(","):
                ads0, sites, height = ADSORBATES[name]
                for site in sites:
                    t0 = time.time()
                    adslab = slab.copy()
                    adslab.set_constraint(slab.constraints)
                    add_adsorbate(adslab, ads0.copy(), height, position=site, mol_index=0)
                    e, steps, conv = relax(adslab, calc)
                    e_ads = e - e_slab - sum(REFS[s] for s in ads0.get_chemical_symbols())
                    w.writerow({"metal": metal, "adsorbate": name, "site": site, "E_ads_eV": f"{e_ads:.4f}",
                                "steps": steps, "converged": conv, "anomaly": anomaly(adslab, len(slab), ads0),
                                "seconds": f"{time.time() - t0:.1f}"})
                    fh.flush()
                    print(f"{metal} {name}@{site}: {e_ads:+.3f} eV ({steps} steps, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
