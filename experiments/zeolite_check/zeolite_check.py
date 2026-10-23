"""Does UMA bind ammonia in copper chabazite the way published DFT does?

Run with the Python that has fairchem:

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/zeolite_check/zeolite_check.py

The framework is the Materials Project's all-silica chabazite (mp-aaaabamv, the 12-T rhombohedral cell, Si12O24;
data/cha_mp-aaaabamv.json). On it the script builds the sites SCR catalysts use and binds one NH3 to each:

  H-CHA   one Al, its proton on each of the four O around it (Bronsted acid site); NH3 takes the proton (NH4+)
  ZCu     one Al in the six-membered ring, Cu+ over the ring
  Z2Cu    two Al across the six-membered ring (third neighbours), Cu2+ in the ring
  ZCuOH   one Al in the eight-membered ring, [CuOH]+ in the ring

Sites come from the app's builder (src/catalyst_engine/design/zeolites.py) with copper as the metal, and NH3 goes
on as the app places it. All atoms relax in the fixed cell (BFGS to 0.05 eV/A) in each UMA mode: catalysis (oc20,
RPBE), materials (omat, PBE/PBE+U) and porous frameworks (odac, PBE-D3). E_bind = E(site + NH3) - E(site) - E(NH3 gas), the most stable
arrangement of each. Writes results/zeolite.json and the relaxed structures to structures/.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parents[1] / "src" / "catalyst_engine" / "design"
TASKS = ("oc20", "omat", "odac")
FMAX, STEPS = 0.05, 400
SITES = {"H-CHA": "H", "ZCu": "M+", "Z2Cu": "M2+", "ZCuOH": "MOH+"}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, DESIGN / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zl = _load("zeolites")
w = _load("uma_worker")


def framework() -> Atoms:
    d = json.loads((HERE / "data" / "cha_mp-aaaabamv.json").read_text())
    return Atoms(d["species"], scaled_positions=d["frac"], cell=d["lattice"], pbc=True)


def main() -> None:
    import torch
    from ase.optimize import BFGS
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    torch.set_num_threads(int(os.environ.get("CE_THREADS", max(1, min(8, (os.cpu_count() or 4) - 2)))))
    checkpoint = os.environ.get("CE_UMA_CHECKPOINT", str(Path.home() / "uma" / "checkpoints" / "uma-s-1p1.pt"))
    unit = pretrained_mlip.load_predict_unit(checkpoint, device="cpu")
    tasks = sys.argv[1:] or list(TASKS)
    z = framework()
    starts = {name: zl.build(z, form, "Cu") for name, form in SITES.items()}
    (HERE / "structures").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    out_path = HERE / "results" / "zeolite.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.0f} s] {msg}", flush=True)

    for task in tasks:
        calc = FAIRChemCalculator(unit, task_name=task)

        def relax(atoms):
            atoms.calc = calc
            opt = BFGS(atoms, logfile=None)
            conv = bool(opt.run(fmax=FMAX, steps=STEPS))
            return float(atoms.get_potential_energy()), opt.get_number_of_steps(), conv

        gas = w.ADSORBATE_ATOMS["NH3"].copy()
        gas.cell = [15.0, 15.1, 15.2]
        gas.center()
        gas.pbc = True
        e_gas = relax(gas)[0]
        rows = {}
        for name, (variants, site) in starts.items():
            bare = []
            for k, s in enumerate(variants):
                s = s.copy()
                e, n, c = relax(s)
                bare.append((e, k, s, n, c))
                write(HERE / "structures" / f"{task}__{name}__{k}.cif", s)
            e0, k0, s0, n0, c0 = min(bare, key=lambda r: r[0])
            bound = []
            for k, (e_b, _, s_b, _, _) in enumerate(bare):
                start = zl.place_at(s_b, site, w.ADSORBATE_ATOMS["NH3"], 1.6 if name == "H-CHA" else 2.0)
                e, n, c = relax(start)
                bound.append((e, k, start, n, c))
                write(HERE / "structures" / f"{task}__{name}__{k}__NH3.cif", start)
            e1, k1, s1, n1, c1 = min(bound, key=lambda r: r[0])
            n_idx = len(s1) - 4
            others = [i for i in range(len(s1) - 4)]
            d = s1.get_distances(n_idx, others, mic=True)
            near = int(others[int(np.argmin(d))])
            hs = s1.get_distances(n_idx, list(range(len(s1))), mic=True)
            n_h = int(sum(1 for i, x in enumerate(hs) if s1[i].symbol == "H" and x < 1.25))
            rows[name] = {"E_bind": e1 - e0 - e_gas, "E_site_variants": [r[0] - e0 for r in bare],
                          "E_bound_variants": [r[0] - e0 - e_gas for r in bound],
                          "N_partner": s1[near].symbol, "N_partner_A": float(d.min()), "N_H_count": n_h,
                          "converged": bool(c0 and c1), "steps": [n0, n1]}
            log(f"{task} {name}: E_bind {rows[name]['E_bind']:+.3f} eV; N next to {rows[name]['N_partner']} at "
                f"{rows[name]['N_partner_A']:.2f} A; N holds {n_h} H; converged {rows[name]['converged']}")
        results[task] = {"E_NH3_gas": e_gas, "sites": rows}
        out_path.write_text(json.dumps(results, indent=1))
    results["generated"] = time.strftime("%Y-%m-%d")
    results["seconds"] = round(time.time() - t_start)
    out_path.write_text(json.dumps(results, indent=1))
    log("wrote results/zeolite.json")


if __name__ == "__main__":
    main()
