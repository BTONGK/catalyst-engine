"""UMA test run, gas check: each mode's own energies for the isolated gas molecules.

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/uma_binding/gas.py --out results/gas.csv

Compares each mode's H2, N2 and H2O with the OC20 references and its NH3 and NO formation energies with
experiment, which bounds how much the gas reference can move a binding energy.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from ase import Atoms
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

CHECKPOINT = "/Users/bt/uma/checkpoints/uma-s-1p1.pt"
OC20_MOLECULES = {"H2": 2 * -3.477, "N2": 2 * -8.083, "H2O": -7.204 + 2 * -3.477}
FORMATION = {"NH3": -0.815, "NO": 3.647}  # electronic, from NIST-JANAF at 0 K with ZPEs removed (see oxides.py)

_h = [(0.94 * np.cos(t), 0.94 * np.sin(t), 0.38) for t in np.radians([0, 120, 240])]
MOLECULES = {
    "H2": Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)]),
    "N2": Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.10)]),
    "H2O": Atoms("OH2", positions=[(0, 0, 0), (0.76, 0, 0.59), (-0.76, 0, 0.59)]),
    "NH3": Atoms("NH3", positions=[(0, 0, 0), *_h]),
    "NO": Atoms("NO", positions=[(0, 0, 0), (0, 0, 1.15)]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    unit = pretrained_mlip.load_predict_unit(CHECKPOINT, device="cpu")
    rows = []
    for task in ("oc20", "omat"):
        calc = FAIRChemCalculator(unit, task_name=task)
        e = {}
        for name, mol in MOLECULES.items():
            m = mol.copy()
            m.cell = [15.0, 15.1, 15.2]
            m.center()
            m.pbc = True
            m.calc = calc
            LBFGS(m, logfile=None).run(fmax=0.02, steps=200)
            e[name] = float(m.get_potential_energy())
        for name, ref in OC20_MOLECULES.items():
            rows.append({"task": task, "quantity": f"{name} energy", "uma_eV": f"{e[name]:.3f}", "reference_eV": f"{ref:.3f}",
                         "reference": "OC20 reference"})
        computed = {"NH3": e["NH3"] - 0.5 * e["N2"] - 1.5 * e["H2"], "NO": e["NO"] - 0.5 * e["N2"] - e["H2O"] + e["H2"]}
        for name, value in computed.items():
            rows.append({"task": task, "quantity": f"{name} formation", "uma_eV": f"{value:.3f}",
                         "reference_eV": f"{FORMATION[name]:.3f}", "reference": "experiment"})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"[{r['task']}] {r['quantity']}: {float(r['uma_eV']):+.3f} eV ({r['reference']} {float(r['reference_eV']):+.3f})")


if __name__ == "__main__":
    main()
