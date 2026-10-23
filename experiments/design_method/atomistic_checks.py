"""Checks of the design worker's surface calculations, one choice at a time, with the same UMA model and settings.

Run with the Python that has fairchem (see report.yaml, rerun):

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 atomistic_checks.py [section ...]

With section names, only those run and the rest of results/atomistic.json stays as it is.

Materials: the fourteen of the shipped example screen (examples/screens/), oxygen as the adsorbate, the catalysis
(oc20) mode, the standard setting (LBFGS to 0.05 eV/A, up to 250 steps). Writes results/atomistic.json:

  lattice      each crystal's own UMA lattice (catalysis and materials modes) against the input crystal
  strain       oxygen binding on Pt and Cu with the lattice stretched or squeezed by up to 2%
  slab_size    oxygen binding on Pt and Ru against slab thickness and width
  sites        every surface site on every material, against the standard setting's first three
  tolerance    the best structure re-relaxed to 0.01 eV/A
  segregation  each alloy with its bulk-terminated top layer, and with the top layer made of one element
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLE = next((REPO / "examples" / "screens").iterdir())

spec = importlib.util.spec_from_file_location("uma_worker", REPO / "src" / "catalyst_engine" / "design" / "uma_worker.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

import torch  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402
from ase.filters import FrechetCellFilter  # noqa: E402
from ase.optimize import LBFGS  # noqa: E402
from fairchem.core import FAIRChemCalculator, pretrained_mlip  # noqa: E402

CHECKPOINT = os.environ.get("CE_UMA_CHECKPOINT", str(Path.home() / "uma" / "checkpoints" / "uma-s-1p1.pt"))
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 2)))
unit = pretrained_mlip.load_predict_unit(CHECKPOINT, device="cpu")
CALC = {task: FAIRChemCalculator(unit, task_name=task) for task in ("oc20", "omat")}
STANDARD = w.SETTINGS["standard"]


def relax(atoms, fmax=STANDARD["fmax"], steps=STANDARD["steps"], task="oc20"):
    atoms.calc = CALC[task]
    opt = LBFGS(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return float(atoms.get_potential_energy()), converged


def gas_reference() -> float:
    """Oxygen's reference, H2O - H2, as the worker computes it."""
    energy = {}
    for name in ("H2O", "H2"):
        mol = w.GAS[name].copy()
        mol.cell = [15.0, 15.1, 15.2]
        mol.center()
        mol.pbc = True
        energy[name] = relax(mol)[0]
    return energy["H2O"] - energy["H2"]


E_O = gas_reference()


def binding(slab, sites, fmax=STANDARD["fmax"]):
    """Relaxed clean slab, then oxygen on each site: [{site, E, anomaly, converged, atoms}]."""
    slab = slab.copy()
    slab.set_constraint(FixAtoms(mask=slab.positions[:, 2] < slab.positions[:, 2].min()
                                 + 0.5 * np.ptp(slab.positions[:, 2])))
    e_slab, _ = relax(slab, fmax=fmax)
    out = []
    for site in sites:
        ads = w.place(slab, "O", site)
        e, conv = relax(ads, fmax=fmax)
        out.append({"site": site["label"], "E": e - e_slab - E_O, "anomaly": w.anomaly(ads, len(slab), "O"),
                    "converged": conv, "atoms": ads, "e_slab": e_slab, "slab": slab})
    return out


def best(rows):
    kept = [r for r in rows if not r["anomaly"]]
    return min(kept, key=lambda r: r["E"]) if kept else None


def flipped(slab):
    """The same slab upside down: the bottom termination becomes the top."""
    s = slab.copy()
    z = s.positions[:, 2]
    s.positions[:, 2] = z.max() + z.min() - z
    s.set_constraint()
    return s


def top_layer(slab, depth=0.6):
    z = slab.positions[:, 2]
    syms = [s for s, zz in zip(slab.get_chemical_symbols(), z) if zz > z.max() - depth]
    return {e: syms.count(e) for e in sorted(set(syms))}


job = json.loads((EXAMPLE / "job.json").read_text())
materials = job["materials"]
OUT = HERE / "results" / "atomistic.json"
SECTIONS = ("lattice", "strain", "slab_size", "sites", "segregation")
wanted = sys.argv[1:] or list(SECTIONS)
results: dict = json.loads(OUT.read_text()) if OUT.exists() and sys.argv[1:] else {}
results.update(generated=time.strftime("%Y-%m-%d"), checkpoint=Path(CHECKPOINT).name, example=EXAMPLE.name,
               gas_reference_O_eV=E_O)
seconds = results.get("section_seconds", {})
t_start = time.time()
t_section = t_start


def log(msg):
    print(f"[{time.time() - t_start:6.0f} s] {msg}", flush=True)


# ---------------------------------------------------------------- lattice
if "lattice" in wanted:
    rows = []
    for m in materials:
        bulk = w.bulk_from(m)
        a0 = (bulk.get_volume() / len(bulk)) ** (1 / 3)
        row = {"material": m["formula"], "source": m.get("source", "")}
        for task in ("oc20", "omat"):
            b = bulk.copy()
            b.calc = CALC[task]
            opt = LBFGS(FrechetCellFilter(b), logfile=None)
            opt.run(fmax=0.01, steps=300)
            row[f"{task}_pct"] = 100 * ((b.get_volume() / len(b)) ** (1 / 3) / a0 - 1)
        rows.append(row)
        log(f"lattice {m['formula']}: oc20 {row['oc20_pct']:+.2f}%  omat {row['omat_pct']:+.2f}%")
    results["lattice"] = rows
    seconds["lattice"] = round(time.time() - t_section)
    t_section = time.time()

# ---------------------------------------------------------------- strain
if "strain" in wanted:
    rows = []
    for sym in ("Pt", "Cu"):
        m = next(m for m in materials if m["formula"] == sym)
        for pct in (-2, -1, 0, 1, 2):
            bulk = w.bulk_from(m)
            bulk.set_cell(bulk.cell.array * (1 + pct / 100), scale_atoms=True)
            slab = w.build_slab(bulk, w.densest_plane(bulk))
            b = best(binding(slab, w.choose_sites(w.surface_sites(slab), "O", 3)))
            rows.append({"material": sym, "strain_pct": pct, "E": b["E"]})
            log(f"strain {sym} {pct:+d}%: {b['E']:.3f}")
    results["strain"] = rows
    seconds["strain"] = round(time.time() - t_section)
    t_section = time.time()

# ---------------------------------------------------------------- slab size
if "slab_size" in wanted:
    rows = []
    for sym in ("Pt", "Ru"):
        m = next(m for m in materials if m["formula"] == sym)
        bulk = w.bulk_from(m)
        hkl = w.densest_plane(bulk)
        for thickness, width in ((7, 7), (10, 7), (13, 7), (7, 10), (10, 10)):
            slab = w.build_slab(bulk, hkl, thickness=thickness, width=width, max_atoms=400)
            b = best(binding(slab, w.choose_sites(w.surface_sites(slab), "O", 2)))
            rows.append({"material": sym, "thickness_A": thickness, "width_A": width, "n_atoms": len(slab),
                         "layers": int(len(slab) / sum(top_layer(slab).values())), "E": b["E"], "site": b["site"]})
            log(f"slab {sym} {thickness}x{width} ({len(slab)} atoms): {b['E']:.3f} on {b['site']}")
    results["slab_size"] = rows
    seconds["slab_size"] = round(time.time() - t_section)
    t_section = time.time()

# ---------------------------------------------------------------- sites and tolerance
if "sites" in wanted:
    site_rows, tol_rows = [], []
    for m in materials:
        bulk = w.bulk_from(m)
        slab = w.build_slab(bulk, w.densest_plane(bulk))
        sites = w.surface_sites(slab)
        every = w.choose_sites(sites, "O", len(sites))
        rows = binding(slab, every)
        standard = rows[:STANDARD["sites"]]
        b_all, b_std = best(rows), best(standard)
        site_rows.append({"material": m["formula"], "n_sites": len(rows), "E_standard": b_std["E"],
                          "site_standard": b_std["site"], "E_all": b_all["E"], "site_all": b_all["site"],
                          "sites": [{k: r[k] for k in ("site", "E", "anomaly", "converged")} for r in rows]})
        tight = b_std["atoms"].copy()
        tight.set_constraint(b_std["atoms"].constraints)
        e_tight, conv = relax(tight, fmax=0.01, steps=1000)
        clean = b_std["slab"].copy()
        clean.set_constraint(b_std["slab"].constraints)
        e_clean, _ = relax(clean, fmax=0.01, steps=1000)
        tol_rows.append({"material": m["formula"], "E_standard": b_std["E"], "E_tight": e_tight - e_clean - E_O,
                         "converged": conv})
        log(f"sites {m['formula']}: standard {b_std['E']:.3f} ({b_std['site']}), all {len(rows)} sites "
            f"{b_all['E']:.3f} ({b_all['site']}); tight {tol_rows[-1]['E_tight']:.3f}")
    results["sites"] = site_rows
    results["tolerance"] = tol_rows
    seconds["sites"] = round(time.time() - t_section)
    t_section = time.time()

# ---------------------------------------------------------------- segregation
if "segregation" in wanted:
    rows = []
    for m in materials:
        if m.get("source") != "Materials Project":
            continue
        bulk = w.bulk_from(m)
        hkl = w.densest_plane(bulk)
        slab = w.build_slab(bulk, hkl)
        z = slab.positions[:, 2]
        top = np.where(z > z.max() - 0.6)[0]
        variants = [("as cut from the bulk", slab)]
        for el in sorted(set(slab.get_chemical_symbols())):
            skin = slab.copy()
            symbols = skin.get_chemical_symbols()
            for i in top:
                symbols[i] = el
            skin.set_chemical_symbols(symbols)
            variants.append((f"top layer all {el}", skin))
        for label, s_ in variants:
            b = best(binding(s_, w.choose_sites(w.surface_sites(s_), "O", 3)))
            rows.append({"material": m["formula"], "face": list(hkl), "surface": label, "top_layer": top_layer(s_),
                         "E": b["E"], "site": b["site"]})
            log(f"segregation {m['formula']} {label}: {b['E']:.3f} on {b['site']}")
    results["segregation"] = rows
    results.pop("alloy_faces", None)
    seconds["segregation"] = round(time.time() - t_section)

results["section_seconds"] = seconds
results["seconds"] = sum(seconds.values())
(HERE / "results").mkdir(exist_ok=True)
OUT.write_text(json.dumps(results, indent=1))
log("wrote results/atomistic.json")
