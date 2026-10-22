"""UMA test run, lattice check: relax the bulk oxide crystals in each mode and compare with experiment.

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/uma_binding/lattice.py --out results/lattice.csv
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import torch
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip

HERE = Path(__file__).parent
CHECKPOINT = "/Users/bt/uma/checkpoints/uma-s-1p1.pt"
CRYSTALS = ["anatase_cod9015929.cif", "V2O5_cod2020756.cif"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    warnings.filterwarnings("ignore", message="crystal system")  # ASE's setting note for Pmmn; bond lengths check out
    unit = pretrained_mlip.load_predict_unit(CHECKPOINT, device="cpu")
    rows = []
    for cif in CRYSTALS:
        for task in ("oc20", "omat"):
            a = read(HERE / "structures" / cif)
            before = a.cell.cellpar()[:3]
            a.calc = FAIRChemCalculator(unit, task_name=task)
            LBFGS(FrechetCellFilter(a), logfile=None).run(fmax=0.01, steps=300)
            after = a.cell.cellpar()[:3]
            for axis, x, y in zip("abc", before, after):
                rows.append({"crystal": cif.split("_")[0], "task": task, "axis": axis, "experiment_A": f"{x:.3f}",
                             "uma_A": f"{y:.3f}", "change_pct": f"{100 * (y / x - 1):.1f}"})
            print(f"{cif} [{task}] " + "  ".join(f"{x:.3f} -> {y:.3f} Å ({100 * (y / x - 1):+.1f}%)" for x, y in zip(before, after)))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
