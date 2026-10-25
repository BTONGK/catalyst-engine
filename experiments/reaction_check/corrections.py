"""Zero-point and entropy corrections for the electrochemical intermediates, from UMA's own vibrations.

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/reaction_check/corrections.py

Runs in the UMA Python (numpy, ASE and fairchem). The rule was written down before this ran (PASS_RULE.md, the
corrections section). For each intermediate on Pt(111), and on Pd(111) and Ru(0001) to show how much the numbers
move from metal to metal: take the lowest-energy site of the reaction-check run (runs/metals-thermal and
runs/metals-electro, the same choice the app makes), refine that minimum with the metal atoms held still (BFGS to
0.01 eV/A, so the finite differences start from a resting point), build the adsorbate's force-constant matrix by
central differences of 0.01 A, and take its harmonic frequencies, any below 50 cm-1 raised to 50 cm-1. Then

    correction = (ZPE - T S)_adsorbate - sum over its reference gases of (ZPE - T S)_gas

at 298.15 K, with the gases' zero-point energies from their measured fundamentals and their standard entropies from
NIST-JANAF (water at 0.035 bar). Writes results/corrections.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS

HERE = Path(__file__).resolve().parent
RUNS = [HERE / "runs" / "metals-thermal", HERE / "runs" / "metals-electro"]
OUT = HERE / "results" / "corrections.json"
sys.path.insert(0, str(HERE.parents[1] / "src" / "catalyst_engine" / "design"))
from uma_worker import REFERENCE  # noqa: E402  the worker's gas reference for each adsorbate

T = 298.15
K_B = 8.617333262e-5  # eV/K
CM = 1.239841984e-4  # eV per cm-1
J_PER_EV = 96485.33212  # J/mol per eV
DELTA, FLOOR, FMAX = 0.01, 50.0, 0.01
METALS = ["Pt", "Pd", "Ru"]  # Pt is the rule's; Pd and Ru show the spread
ADSORBATES = ["H", "O", "OH", "OOH", "N", "NH", "NH2", "NNH", "NNH2", "HNNH", "HNNH2", "NO", "NOH", "CO", "COOH", "OCHO"]

# Measured fundamentals (cm-1, NIST CCCBDB) and standard entropies at 298.15 K and 1 bar (J/mol/K, NIST-JANAF)
GAS = {
    "H2": {"fundamentals": [4161.2], "S": 130.680},
    "N2": {"fundamentals": [2329.9], "S": 191.609},
    "H2O": {"fundamentals": [3657.1, 1594.7, 3755.9], "S": 188.834, "bar": 0.035},
    "CO": {"fundamentals": [2143.3], "S": 197.653},
    "NO": {"fundamentals": [1876.0], "S": 210.758},
    "NH3": {"fundamentals": [3337.2, 950.0, 3443.6, 3443.6, 1626.1, 1626.1], "S": 192.774},
}
# Published corrections the rule checks against (eV), and Skúlason 2012's constants shown beside them
PUBLISHED = {"H": 0.24, "O": 0.05, "OH": 0.35, "OOH": 0.40}
SKULASON = {"N": 0.26, "NH": 0.57, "NH2": 0.88, "NNH": 0.80, "NNH2": 1.11}


def gas_g(name: str) -> dict:
    g = GAS[name]
    zpe = 0.5 * sum(g["fundamentals"]) * CM
    s = g["S"] - 8.314462618 * np.log(g.get("bar", 1.0))  # J/mol/K at its pressure
    ts = T * s / J_PER_EV
    return {"ZPE": zpe, "TS": ts, "G": zpe - ts}


def harmonic(freqs_cm: np.ndarray) -> tuple[float, float]:
    """ZPE and T x S (eV) of harmonic oscillators at T."""
    e = freqs_cm * CM
    x = e / (K_B * T)
    zpe = 0.5 * float(e.sum())
    s = float(np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x))))  # in units of k
    return zpe, K_B * T * s


def lowest_site(metal: str, adsorbate: str) -> dict | None:
    """The app's choice: the lowest binding energy among the sites that stayed put (no anomaly)."""
    best, folder = None, None
    for run in RUNS:
        path = run / "results.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("material") != f"element:{metal}" or r.get("adsorbate") != adsorbate or r.get("anomaly"):
                continue
            if best is None or r["E_bind"] < best["E_bind"]:
                best, folder = r, run
    return None if best is None else {**best, "folder": folder}


def frequencies(atoms, calc, n_ads: int) -> np.ndarray:
    """Harmonic frequencies (cm-1) of the last n_ads atoms, the rest held still, by central differences."""
    idx = list(range(len(atoms) - n_ads, len(atoms)))
    atoms.set_constraint(FixAtoms(indices=list(range(len(atoms) - n_ads))))
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=FMAX, steps=400)
    pos0 = atoms.positions.copy()
    n = 3 * n_ads
    hess = np.zeros((n, n))
    for a, i in enumerate(idx):
        for c in range(3):
            forces = []
            for sign in (1, -1):
                atoms.positions = pos0
                atoms.positions[i, c] += sign * DELTA
                forces.append(atoms.get_forces(apply_constraint=False)[idx].ravel())
            hess[3 * a + c] = -(forces[0] - forces[1]) / (2 * DELTA)
    atoms.positions = pos0
    hess = 0.5 * (hess + hess.T)
    masses = np.repeat(atoms.get_masses()[idx], 3)
    w2 = np.linalg.eigvalsh(hess / np.sqrt(np.outer(masses, masses)))  # eV/(A^2 amu)
    # sqrt(eV/(A^2 amu)) in rad/s, then cm-1: 1 eV/(A^2 amu) = 9.648533e27 s^-2
    omega = np.sign(w2) * np.sqrt(np.abs(w2) * 9.648533212e27)
    return omega / (2 * np.pi * 2.99792458e10)


def main() -> None:
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    torch.set_num_threads(4)
    unit = pretrained_mlip.load_predict_unit(os.path.expanduser("~/uma/checkpoints/uma-s-1p1.pt"), device="cpu")
    calc = FAIRChemCalculator(unit, task_name="oc20")
    gases = {g: gas_g(g) for g in GAS}
    out = {"rule": {"T_K": T, "delta_A": DELTA, "floor_cm": FLOOR, "fmax_eV_A": FMAX, "model": "UMA-S 1.1, oc20 mode"},
           "gases": {g: {k: round(v, 4) for k, v in d.items()} for g, d in gases.items()}, "metals": {}}
    for metal in METALS:
        rows = {}
        for a in ADSORBATES:
            site = lowest_site(metal, a)
            if site is None:
                continue
            atoms = read(site["folder"] / site["structure"])
            raw = frequencies(atoms, calc, site["n_ads"])
            used = np.where(raw < FLOOR, FLOOR, raw)
            zpe, ts = harmonic(used)
            ref = sum(w * gases[g]["G"] for g, w in REFERENCE[a].items())
            rows[a] = {"site": site["site"], "frequencies_cm": [round(float(f), 1) for f in raw],
                       "raised": int(np.sum(raw < FLOOR)), "ZPE": round(zpe, 4), "TS": round(ts, 4),
                       "reference": REFERENCE[a], "correction": round(zpe - ts - ref, 4)}
            print(metal, a, site["site"], rows[a]["correction"], flush=True)
        out["metals"][metal] = rows
    pt = out["metals"]["Pt"]
    misses = {a: round(pt[a]["correction"] - v, 4) for a, v in PUBLISHED.items()}
    out["check"] = {"published": PUBLISHED, "misses": misses, "max_miss": max(abs(v) for v in misses.values()),
                    "median_miss": float(np.median([abs(v) for v in misses.values()])),
                    "skulason": {a: {"published": v, "uma": pt[a]["correction"]} for a, v in SKULASON.items()}}
    out["check"]["passes"] = bool(out["check"]["max_miss"] <= 0.10 and out["check"]["median_miss"] <= 0.05)
    spread = {a: round(max(out["metals"][m][a]["correction"] for m in METALS if a in out["metals"][m]) -
                       min(out["metals"][m][a]["correction"] for m in METALS if a in out["metals"][m]), 4)
              for a in ADSORBATES if a in pt}
    out["spread"] = spread
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["check"], indent=1))
    print("spread", spread)


if __name__ == "__main__":
    main()
