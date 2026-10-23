"""How close does UMA come to DFT for molecules on oxide surfaces?

Run with the Python that has fairchem, after extract_oc22.py has written cache/oc22_pairs.json:

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/oxide_check/oxide_check.py single
    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/oxide_check/oxide_check.py relax
    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/oxide_check/oxide_check.py nh3
    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 experiments/oxide_check/oxide_check.py anatase101

The reference is OC22's DFT (PBE+U, spin-polarized; Tran et al. 2023): adsorbates (O, H, OH, H2O, HO2, CO, N, C,
O2) on oxide slabs from its two validation splits, in-domain and out-of-domain materials. Only systems with one
molecule count, as the app places one; for each, the binding without gas references is
dE = E(molecule on slab) - E(slab), and a model's error is its dE minus DFT's.

  single  every model and task scores DFT's relaxed structures as they are (a stratified sample: up to SAMPLE pairs
          per adsorbate and split)
  relax   the app's way: the two modes the app offers (1.1 catalysis and materials) relax each
          structure from DFT's geometry (BFGS after the app's 0.03 A nudge, to 0.05 eV/A), on a smaller sample
  nh3     a test outside OC22: NH3 binding, built and relaxed exactly as the app's worker does (Materials Project's
          crystals in data/, the worker's slab, sites, nudge, relaxation, gas reference and clean-surface reference,
          in every model and task). On anatase TiO2 the measurement is 1.23 +/- 0.1 eV to desorb one NH3 from the
          five-fold Ti of the (101) face at zero coverage (TPD), with optB88-vdW DFT at 1.20 eV (Koust et al.
          2018). The app's own face for anatase is (001); (101) is (1,0,0) in this cell, chosen as Advanced would.
          V2O5, hot SCR's catalyst to beat, runs on the app's face to show how far the modes spread there.
  anatase101  the measured surface itself: anatase (101) cut between its O-Ti-O layers, so two-fold O and five-fold
          Ti end it as on the real crystal (the worker's cut leaves a dangling O there), 4 and 5 layers thick,
          10.3 x 7.6 A (one NH3 per four surface Ti), the bottom half fixed; NH3 on a five-fold Ti, then the
          worker's nudge, relaxation and references.

UMA-S 1.2.1's oxide mode (oc22) was trained on OC22; whether its validation systems were held out is unstated, so
this test may run inside its training data. Writes results/oxide.json.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import make_supercell, surface
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list

HERE = Path(__file__).resolve().parent
CKPT = Path.home() / "uma" / "checkpoints"
MODELS = [("uma-s-1p1", "oc20"), ("uma-s-1p1", "omat"), ("uma-s-1p2p1", "oc22"), ("uma-s-1p2p1", "oc20"),
          ("uma-s-1p2p1", "omat")]
RELAX_MODELS = [("uma-s-1p1", "oc20"), ("uma-s-1p1", "omat")]
SAMPLE, RELAX_SAMPLE = 40, 3  # pairs per adsorbate and split; pairs per adsorbate
FMAX, STEPS = 0.05, 250
NH3_SURFACES = [("anatase", "app face", None, {}), ("anatase", "(101)", (1, 0, 0), {}),
                ("anatase", "(101), thicker", (1, 0, 0), {"thickness": 10.0}), ("v2o5", "app face", None, {})]


def worker():
    """The app's UMA worker, for its nudge and relaxation."""
    path = HERE.parents[1] / "src" / "catalyst_engine" / "design" / "uma_worker.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("uma_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atoms_of(rec: dict) -> Atoms:
    a = Atoms(numbers=rec["numbers"], positions=rec["pos"], cell=rec["cell"], pbc=True)
    if rec.get("fixed"):
        a.set_constraint(FixAtoms(indices=[i for i, f in enumerate(rec["fixed"]) if f]))
    return a


def anatase_101(bulk: Atoms, layers: int) -> Atoms:
    """Anatase (101) cut through the widest gap between atomic planes, the space between two O-Ti-O layers, with
    `layers` of them, 10.3 x 7.6 A across, 10 A of vacuum each side and the bottom half fixed. `bulk` is the
    Materials Project primitive cell, whose first two vectors are the conventional a and b."""
    conv = make_supercell(bulk, [[1, 0, 0], [0, 1, 0], [1, 1, 2]])
    big = surface(conv, (1, 0, 1), layers + 2, vacuum=None, periodic=True)
    big.wrap()
    z, height = big.positions[:, 2], big.cell[2, 2]
    planes = np.unique(np.round(z, 2))
    gaps = np.diff(planes)
    cut = planes[np.argmax(gaps)] + gaps.max() / 2
    slab = big[((z - cut) % height) < layers * height / (layers + 2)]
    slab.positions[:, 2] = (slab.positions[:, 2] - cut) % height + 10.0
    slab.cell[2] = [0, 0, slab.positions[:, 2].max() + 10.0]
    slab.pbc = True
    slab = slab.repeat((1, 2, 1))
    zs = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(mask=zs < zs.min() + 0.5 * (zs.max() - zs.min())))
    return slab


def bonds_of(atoms: Atoms) -> np.ndarray:
    """Neighbors within 2.4 A of each atom: 5 for the surface Ti of anatase (101), 2 for its top O."""
    return np.bincount(neighbor_list("i", atoms, 2.4), minlength=len(atoms))


def one_molecule(p: dict) -> bool:
    """One adsorbate on the slab, and the adsorbate-on-slab system holds the slab's atoms plus that molecule's."""
    if p["nads"] != 1:
        return False
    extra = collections.Counter(p["adslab"]["numbers"])
    extra.subtract(collections.Counter(p["slab"]["numbers"]))
    want = collections.Counter(Atoms(p["adsorbate"]).numbers)
    return {k: v for k, v in extra.items() if v} == dict(want)


def sample(pairs: list[dict], per: int, keys: tuple[str, ...], seed: int) -> list[dict]:
    groups = collections.defaultdict(list)
    for p in pairs:
        groups[tuple(p[k] for k in keys)].append(p)
    rng = random.Random(seed)
    out = []
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda p: p["sid"])
        out += rng.sample(g, min(per, len(g)))
    return out


def main() -> None:
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    torch.set_num_threads(int(os.environ.get("CE_THREADS", max(1, min(8, (os.cpu_count() or 4) - 2)))))
    pairs = [p for p in json.loads((HERE / "cache" / "oc22_pairs.json").read_text()) if one_molecule(p)]
    out_path = HERE / "results" / "oxide.json"
    out_path.parent.mkdir(exist_ok=True)
    t0 = time.time()

    def save(section, key, rows):
        """Merge one model's rows into results/oxide.json, re-read first so modes run one after another keep each
        other's sections."""
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        results.setdefault(section, {})[key] = rows
        results["generated"] = time.strftime("%Y-%m-%d")
        out_path.write_text(json.dumps(results, indent=1))

    def log(msg):
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    units = {}

    def calc(model, task):
        if model not in units:
            units[model] = pretrained_mlip.load_predict_unit(str(CKPT / f"{model}.pt"), device="cpu")
        return FAIRChemCalculator(units[model], task_name=task)

    if mode == "anatase101":
        w = worker()
        bulk = w.bulk_from(json.loads((HERE / "data" / "anatase.json").read_text()))
        for model, task in MODELS:
            c = calc(model, task)
            mol = w.GAS["NH3"].copy()
            mol.cell = [15.0, 15.1, 15.2]
            mol.center()
            mol.pbc = True
            ref = w.relax_with(mol, c, FMAX, STEPS)[0]
            rows = []
            for layers in (4, 5):
                slab = anatase_101(bulk, layers)
                start = bonds_of(slab)
                e_slab = w.relax_with(w.nudge(slab), c, FMAX, STEPS)[0]
                ti = [i for i, el in enumerate(slab.get_chemical_symbols()) if el == "Ti"]
                i = max(ti, key=lambda i: slab.positions[i, 2])
                site = {"kind": "top", "label": "top Ti", "xy": slab.positions[i, :2].tolist(),
                        "z": float(slab.positions[i, 2])}
                adslab = w.place(slab, "NH3", site)
                e, n, conv = w.relax_with(adslab, c, FMAX, STEPS)

                def energy(a):
                    a.calc = c
                    return float(a.get_potential_energy())

                e_ref, drop = w.clean_reference(adslab, len(slab), e_slab, energy,
                                                lambda a: w.relax_with(a, c, FMAX, STEPS)[0])
                after = bonds_of(slab)
                n_ti = int(np.argmin(adslab.get_distances(len(slab), ti, mic=True)))
                rows.append({"layers": layers, "n_atoms": len(slab), "formula": slab.get_chemical_formula(),
                             "site_bonds_built": int(start[i]), "site_bonds_relaxed": int(after[i]),
                             "N_Ti_A": round(float(adslab.get_distance(len(slab), ti[n_ti], mic=True)), 3),
                             "E_bind": e - e_ref - ref, "slab_drop_eV": round(drop, 3), "converged": conv,
                             "steps": n, "anomaly": w.anomaly(adslab, len(slab), "NH3")})
                log(f"{model}/{task} anatase (101) {layers} layers: {rows[-1]['E_bind']:+.3f} eV, "
                    f"N-Ti {rows[-1]['N_Ti_A']} A {rows[-1]['anomaly']}")
            save("anatase101", f"{model}/{task}", rows)
    elif mode == "nh3":
        w = worker()
        for model, task in MODELS:
            c = calc(model, task)

            def relax(atoms):
                return w.relax_with(atoms, c, FMAX, STEPS)

            def energy(atoms):
                atoms.calc = c
                return float(atoms.get_potential_energy())

            mol = w.GAS["NH3"].copy()
            mol.cell = [15.0, 15.1, 15.2]
            mol.center()
            mol.pbc = True
            ref = relax(mol)[0]
            rows = []
            for material, face, hkl, size in NH3_SURFACES:
                bulk = w.bulk_from(json.loads((HERE / "data" / f"{material}.json").read_text()))
                hkl = tuple(hkl or w.densest_plane(bulk))
                slab = w.build_slab(bulk, hkl, **size)
                e_slab = relax(w.nudge(slab))[0]
                sites = [(site, "app") for site in
                         w.choose_sites(w.surface_sites(slab), "NH3", w.SETTINGS["standard"]["sites"])]
                if not any(site["label"] == "top Ti" for site, _ in sites) and "Ti" in slab.get_chemical_symbols():
                    # the measured site, the five-fold Ti, can sit deeper than the worker's 1.2 A site depth
                    i = max((i for i, el in enumerate(slab.get_chemical_symbols()) if el == "Ti"),
                            key=lambda i: slab.positions[i, 2])
                    sites.append(({"kind": "top", "label": "top Ti", "xy": slab.positions[i, :2].tolist(),
                                   "z": float(slab.positions[i, 2])}, "added"))
                for site, source in sites:
                    under = int(np.argmin(np.linalg.norm(slab.positions - [*site["xy"], site["z"]], axis=1)))
                    others = [j for j in range(len(slab)) if j != under]
                    bonds = int((slab.get_distances(under, others, mic=True) < 2.4).sum())  # O around a metal top site
                    adslab = w.place(slab, "NH3", site)
                    e, n, conv = relax(adslab)
                    e_ref, drop = w.clean_reference(adslab, len(slab), e_slab, energy, lambda a: relax(a)[0])
                    rows.append({"material": material, "face": face, "hkl": list(hkl), "n_atoms": len(slab),
                                 "formula": slab.get_chemical_formula(), "cell_A": slab.cell.lengths()[:2].round(2).tolist(),
                                 "site": site["label"], "site_from": source, "site_bonds": bonds, "E_bind": e - e_ref - ref, "slab_drop_eV": round(drop, 3),
                                 "converged": conv, "steps": n, "anomaly": w.anomaly(adslab, len(slab), "NH3")})
                    log(f"{model}/{task} {material} {face} {site['label']}: {rows[-1]['E_bind']:+.3f} eV {rows[-1]['anomaly']}")
            save("nh3", f"{model}/{task}", rows)
    elif mode == "single":
        chosen = sample(pairs, SAMPLE, ("adsorbate", "split"), seed=0)
        log(f"{len(chosen)} pairs for single points")
        for model, task in MODELS:
            c = calc(model, task)
            slab_e: dict[int, float] = {}
            rows = []
            for p in chosen:
                s = p["slab"]["sid"]
                if s not in slab_e:
                    a = atoms_of(p["slab"])
                    a.calc = c
                    slab_e[s] = float(a.get_potential_energy())
                a = atoms_of(p["adslab"])
                a.calc = c
                e = float(a.get_potential_energy())
                rows.append({"sid": p["sid"], "split": p["split"], "adsorbate": p["adsorbate"], "bulk": p["bulk"],
                             "bulk_id": p["bulk_id"], "dE_dft": p["adslab"]["energy"] - p["slab"]["energy"],
                             "dE": e - slab_e[s], "E_slab_dft": p["slab"]["energy"], "E_slab": slab_e[s]})
            save("single", f"{model}/{task}", rows)
            err = np.array([r["dE"] - r["dE_dft"] for r in rows])
            log(f"{model}/{task}: {len(rows)} pairs, median error {np.median(err):+.3f} eV, "
                f"MAE {np.abs(err).mean():.3f} eV")
    else:
        w = worker()
        chosen = sample(pairs, RELAX_SAMPLE, ("adsorbate",), seed=1)
        log(f"{len(chosen)} pairs for relaxations")
        for model, task in RELAX_MODELS:
            c = calc(model, task)
            slab_e: dict[int, tuple[float, float]] = {}  # slab sid: (energy at DFT's geometry, relaxed)

            def at_dft(rec):
                a = atoms_of(rec)
                a.calc = c
                return float(a.get_potential_energy())

            rows = []
            for p in chosen:
                s = p["slab"]["sid"]
                if s not in slab_e:
                    slab_e[s] = (at_dft(p["slab"]), w.relax_with(w.nudge(atoms_of(p["slab"])), c, FMAX, STEPS)[0])
                e_single = at_dft(p["adslab"])
                a = w.nudge(atoms_of(p["adslab"]))
                start = a.positions.copy()
                e, n, conv = w.relax_with(a, c, FMAX, STEPS)
                rows.append({"sid": p["sid"], "split": p["split"], "adsorbate": p["adsorbate"], "bulk": p["bulk"],
                             "bulk_id": p["bulk_id"], "dE_dft": p["adslab"]["energy"] - p["slab"]["energy"],
                             "dE": e - slab_e[s][1], "dE_single": e_single - slab_e[s][0], "converged": conv,
                             "steps": n, "moved_A": w.largest_move(start, a)})
            save("relax", f"{model}/{task}", rows)
            err = np.array([r["dE"] - r["dE_dft"] for r in rows])
            log(f"{model}/{task} relaxed: {len(rows)} pairs, median error {np.median(err):+.3f} eV, "
                f"MAE {np.abs(err).mean():.3f} eV, converged {sum(r['converged'] for r in rows)}")


if __name__ == "__main__":
    main()
