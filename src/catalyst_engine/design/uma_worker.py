"""Atomistic screening worker: binding energies of adsorbates on candidate surfaces with Meta's UMA model.

The app launches this file as a script with the Python that has fairchem (CE_UMA_PYTHON), one screen folder per
run:

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 uma_worker.py data/screens/<id>

It reads <id>/job.json, writes progress to status.json, one JSON line per relaxation to results.jsonl and each
relaxed structure to structures/*.cif. It imports only numpy and ASE at module level (fairchem and torch load in
main), so the app and the tests can use its geometry functions.

For each material: take the bulk crystal, cut its close-packed face (the most tightly packed outer atomic layer
among Miller indices up to 1, preferring a layer with the bulk's composition), pick the termination whose
composition matches the bulk, relax the
slab, place each adsorbate on the surface's top, bridge and hollow sites, relax, and keep the binding energy

    E_bind = E(slab + adsorbate) - E(slab) - E_ref(adsorbate)

with gas references from the same model: H = 1/2 H2, O = H2O - H2, N = 1/2 N2, C = CO - O, OH = O + H, and the
molecules themselves for CO, NO and NH3. Negative is bound; more negative binds more strongly.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.build import surface
from ase.constraints import FixAtoms
from ase.io import write

# Binding atom first, at the origin; molecules point away from the surface (+z).
_NH3_H = [(0.94 * np.cos(t), 0.94 * np.sin(t), 0.38) for t in np.radians([0, 120, 240])]
ADSORBATE_ATOMS = {
    "H": Atoms("H"),
    "C": Atoms("C"),
    "N": Atoms("N"),
    "O": Atoms("O"),
    "OH": Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.98)]),
    "CO": Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.15)]),
    "NO": Atoms("NO", positions=[(0, 0, 0), (0, 0, 1.17)]),
    "NH3": Atoms("NH3", positions=[(0, 0, 0), *_NH3_H]),
}
# Height of the binding atom above the local top-layer height, by site kind (Angstrom)
HEIGHTS = {
    "H": {"top": 1.6, "bridge": 1.2, "hollow": 1.0},
    "C": {"top": 1.8, "bridge": 1.4, "hollow": 1.1},
    "N": {"top": 1.8, "bridge": 1.4, "hollow": 1.1},
    "O": {"top": 1.9, "bridge": 1.5, "hollow": 1.2},
    "OH": {"top": 2.0, "bridge": 1.6, "hollow": 1.4},
    "CO": {"top": 1.9, "bridge": 1.5, "hollow": 1.3},
    "NO": {"top": 1.8, "bridge": 1.5, "hollow": 1.3},
    "NH3": {"top": 2.1, "bridge": 2.1, "hollow": 2.1},
}
# Site kinds to try first, per adsorbate (the quick setting keeps the first two sites)
PREFERENCE = {
    "H": ["hollow", "top", "bridge"],
    "C": ["hollow", "bridge", "top"],
    "N": ["hollow", "bridge", "top"],
    "O": ["hollow", "bridge", "top"],
    "OH": ["bridge", "top", "hollow"],
    "CO": ["top", "hollow", "bridge"],
    "NO": ["hollow", "top", "bridge"],
    "NH3": ["top"],
}
GAS = {
    "H2": Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)]),
    "N2": Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.10)]),
    "H2O": Atoms("OH2", positions=[(0, 0, 0), (0.76, 0, 0.59), (-0.76, 0, 0.59)]),
    "CO": Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.13)]),
    "NO": Atoms("NO", positions=[(0, 0, 0), (0, 0, 1.15)]),
    "NH3": Atoms("NH3", positions=[(0, 0, 0), *_NH3_H]),
}
# Each adsorbate's reference as a combination of gas molecules
REFERENCE = {
    "H": {"H2": 0.5},
    "O": {"H2O": 1.0, "H2": -1.0},
    "N": {"N2": 0.5},
    "C": {"CO": 1.0, "H2O": -1.0, "H2": 1.0},
    "OH": {"H2O": 1.0, "H2": -0.5},
    "CO": {"CO": 1.0},
    "NO": {"NO": 1.0},
    "NH3": {"NH3": 1.0},
}
NONMETALS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "Te"}
SETTINGS = {  # sites per adsorbate, force tolerance (eV/A), step cap
    "quick": {"sites": 2, "fmax": 0.10, "steps": 150},
    "standard": {"sites": 3, "fmax": 0.05, "steps": 250},
    "thorough": {"sites": 5, "fmax": 0.03, "steps": 400},
}


def bulk_from(material: dict) -> Atoms:
    """A bulk crystal from a job entry: a Materials Project structure or an element's reference crystal."""
    if material.get("ase_bulk"):
        return ase_bulk(material["ase_bulk"])
    return Atoms(material["species"], scaled_positions=material["frac"], cell=material["lattice"], pbc=True)


def _composition(symbols: list[str]) -> dict[str, float]:
    u, n = np.unique(symbols, return_counts=True)
    return dict(zip(u, n / n.sum()))


def _mismatch(comp: dict[str, float], target: dict[str, float]) -> float:
    return sum(abs(comp.get(el, 0.0) - f) for el, f in target.items()) + sum(comp[el] for el in comp if el not in target)


def plane_layers(bulk: Atoms, hkl, tol: float = 0.5) -> tuple[float, float]:
    """For one repeat of the plane `hkl`: atoms per square Angstrom in its fullest atomic layer (atoms within `tol`
    of each other in height form one layer), and how far that layer's composition sits from the bulk's (0 = same)."""
    s = surface(bulk, hkl, 1, vacuum=None, periodic=True)
    period = s.cell.array[2, 2]
    order = np.argsort(np.mod(s.positions[:, 2], period))
    z = np.mod(s.positions[:, 2], period)[order]
    symbols = np.array(s.get_chemical_symbols())[order]
    gaps = np.diff(np.append(z, z[0] + period))  # the gap above each atom, wrapping around the repeat
    cuts = np.where(gaps > tol)[0]
    if len(cuts) == 0:
        layers = [symbols]
    else:
        layers = [np.roll(symbols, -(a + 1))[: b - a] for a, b in zip(cuts, np.append(cuts[1:], cuts[0] + len(z)))]
    fullest = max(layers, key=len)
    area = np.linalg.norm(np.cross(s.cell.array[0], s.cell.array[1]))
    return len(fullest) / area, _mismatch(_composition(list(fullest)), _composition(bulk.get_chemical_symbols()))


def densest_plane(bulk: Atoms) -> tuple[int, int, int]:
    """Miller indices (up to 1, in the cell's own basis) of the face whose outer atomic layer is most tightly packed:
    the close-packed face of a metal or ordered alloy. Among faces within 3% of the densest, the one whose layer
    matches the bulk composition wins, then the widest interplanar spacing, then the simplest indices."""
    recip = np.linalg.inv(bulk.cell.array).T  # rows are reciprocal vectors without the 2 pi
    rows = []
    for hkl in itertools.product((-1, 0, 1), repeat=3):
        if hkl == (0, 0, 0) or next(h for h in hkl if h) < 0:
            continue
        density, mismatch = plane_layers(bulk, hkl)
        spacing = 1.0 / np.linalg.norm(np.dot(hkl, recip))
        rows.append((hkl, density, mismatch, spacing))
    top = max(r[1] for r in rows)
    close = [r for r in rows if r[1] >= 0.97 * top]
    return min(close, key=lambda r: (round(r[2], 2), -round(r[3], 2), sum(map(abs, r[0]))))[0]


def build_slab(bulk: Atoms, hkl, thickness: float = 7.0, width: float = 7.0, max_atoms: int = 90) -> Atoms:
    """A slab of the plane `hkl`: the termination whose composition best matches the bulk, at least `thickness`
    thick and `width` wide, the bottom half fixed, 10 A of vacuum on each side."""
    d = 1.0 / np.linalg.norm(np.dot(hkl, np.linalg.inv(bulk.cell.array).T))
    layers = max(4, int(np.ceil((thickness + 6.0) / d)) + 2)
    big = surface(bulk, hkl, layers, vacuum=None, periodic=True)
    big.wrap()
    z = big.positions[:, 2]
    target = _composition(bulk.get_chemical_symbols())
    best, best_score = None, np.inf
    for z0 in np.unique(np.round(z, 1)):
        if z0 + thickness > z.max() + 0.05:
            break
        window = big[(z >= z0 - 0.05) & (z <= z0 + thickness + 0.05)]
        if len(window) == 0:
            continue
        score = _mismatch(_composition(window.get_chemical_symbols()), target)
        if score < best_score - 1e-9:
            best, best_score = window, score
    slab = best if best is not None else big
    a, b = slab.cell.array[0], slab.cell.array[1]
    # perpendicular widths of the in-plane cell, so skewed cells still reach `width` in every direction
    area = np.linalg.norm(np.cross(a, b))
    wa, wb = area / np.linalg.norm(b), area / np.linalg.norm(a)
    ra, rb = max(1, int(np.ceil(width / wa))), max(1, int(np.ceil(width / wb)))
    while ra * rb * len(slab) > max_atoms and (ra > 1 or rb > 1):
        if ra >= rb:
            ra -= 1
        else:
            rb -= 1
    slab = slab.repeat((ra, rb, 1))
    # surface() lays the plane in xy; a vertical third vector keeps 20 A of vacuum straight above the face
    extent = np.ptp(slab.positions[:, 2])
    slab.set_cell([slab.cell.array[0], slab.cell.array[1], [0.0, 0.0, extent + 20.0]], scale_atoms=False)
    slab.center(axis=2)
    slab.wrap()
    slab.pbc = True
    zs = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(mask=zs < zs.min() + 0.5 * (zs.max() - zs.min())))
    return slab


def _neighbors_2d(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Copies of the 2D points shifted by every in-plane lattice vector combination in -1..1."""
    shifts = [i * cell[0, :2] + j * cell[1, :2] for i in (-1, 0, 1) for j in (-1, 0, 1)]
    return np.concatenate([points + s for s in shifts])


def surface_sites(slab: Atoms, depth: float = 1.2) -> list[dict]:
    """Top, bridge and hollow sites on the upper face: [{kind, label, xy, z}], most distinct first."""
    z = slab.positions[:, 2]
    top = np.where(z > z.max() - depth)[0]
    symbols = np.array(slab.get_chemical_symbols())
    sites: list[dict] = []
    # top: the highest atom of each element on the face, metals before nonmetals
    for el in sorted(set(symbols[top]), key=lambda e: (e in NONMETALS, e)):
        i = max((i for i in top if symbols[i] == el), key=lambda i: z[i])
        sites.append({"kind": "top", "label": f"top {el}", "xy": slab.positions[i, :2].tolist(), "z": float(z[i])})
    xy = slab.positions[top, :2]
    cell = slab.cell.array
    images = _neighbors_2d(xy, cell)
    zz = np.tile(z[top], 9)
    # bridge: midpoint to the nearest neighbor of the first top atom
    if len(top) >= 2 or len(images) > 1:
        d = np.linalg.norm(images - xy[0], axis=1)
        d[d < 1e-6] = np.inf
        j = int(np.argmin(d))
        mid = (xy[0] + images[j]) / 2
        sites.append({"kind": "bridge", "label": "bridge", "xy": mid.tolist(), "z": float(max(z[top[0]], zz[j]))})
    # hollows: centroids of triangles of the first atom and two near neighbors that are near each other; on a
    # close-packed face the first two distinct ones are the fcc and hcp hollows
    d = np.linalg.norm(images - xy[0], axis=1)
    order = [k for k in np.argsort(d) if d[k] > 1e-6][:6]
    found: list[np.ndarray] = []
    for j, k in itertools.combinations(order, 2):
        if np.linalg.norm(images[j] - images[k]) < 1.4 * min(d[j], d[k]):
            cen = (xy[0] + images[j] + images[k]) / 3
            if all(np.linalg.norm(cen - f) > 0.5 for f in found) and len(found) < 2:
                found.append(cen)
                sites.append({"kind": "hollow", "label": f"hollow {len(found)}", "xy": cen.tolist(),
                              "z": float(max(z[top[0]], zz[j], zz[k]))})
    return sites


def choose_sites(sites: list[dict], adsorbate: str, n: int) -> list[dict]:
    """The `n` sites to try for an adsorbate, in its preferred order of site kinds."""
    rank = {k: i for i, k in enumerate(PREFERENCE[adsorbate])}
    usable = [s for s in sites if s["kind"] in rank]
    usable.sort(key=lambda s: rank[s["kind"]])
    return usable[:n]


def place(slab: Atoms, adsorbate: str, site: dict) -> Atoms:
    ads = ADSORBATE_ATOMS[adsorbate].copy()
    h = HEIGHTS[adsorbate][site["kind"]]
    ads.translate([site["xy"][0], site["xy"][1], site["z"] + h])
    out = slab.copy()
    out.set_constraint(slab.constraints)
    out.extend(ads)
    return out


def anomaly(adslab: Atoms, n_slab: int, adsorbate: str) -> str:
    """'desorbed' if the adsorbate drifted off the surface, 'dissociated' if its bonds broke, else ''."""
    ads = adslab[n_slab:]
    d = adslab.get_distances(n_slab, list(range(n_slab)), mic=True)
    if d.min() > 3.0:
        return "desorbed"
    ref = ADSORBATE_ATOMS[adsorbate]
    if len(ref) > 1:
        d0 = ref.get_all_distances()
        d1 = ads.get_all_distances(mic=True)
        bonded = (d0 > 0) & (d0 < 1.3)
        if np.any(d1[bonded] > 1.5 * d0[bonded]):
            return "dissociated"
    return ""


def _save_status(folder: Path, **fields) -> None:
    path = folder / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields, updated=time.time())
    tmp = folder / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    os.replace(tmp, path)


def main(folder: Path) -> None:
    job = json.loads((folder / "job.json").read_text())
    setting = SETTINGS[job.get("thoroughness", "standard")]
    adsorbates = job["adsorbates"]
    materials = job["materials"]
    per_material = 1 + setting["sites"] * len(adsorbates)  # an estimate until each surface's sites are known
    total = per_material * len(materials)
    _save_status(folder, state="loading", done=0, total=total, message="Loading the UMA model", pid=os.getpid(),
                 started=time.time())

    import torch
    from ase.optimize import LBFGS
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 2)))
    calc = FAIRChemCalculator(pretrained_mlip.load_predict_unit(job["checkpoint"], device="cpu"), task_name=job["task"])

    def relax(atoms: Atoms) -> tuple[float, int, bool]:
        atoms.calc = calc
        opt = LBFGS(atoms, logfile=None)
        converged = bool(opt.run(fmax=setting["fmax"], steps=setting["steps"]))
        return float(atoms.get_potential_energy()), opt.get_number_of_steps(), converged

    _save_status(folder, state="running", message="Gas-phase references")
    gas: dict[str, float] = {}
    for name in sorted({g for a in adsorbates for g in REFERENCE[a]}):
        mol = GAS[name].copy()
        mol.cell = [15.0, 15.1, 15.2]
        mol.center()
        mol.pbc = True
        gas[name] = relax(mol)[0]
    refs = {a: sum(w * gas[g] for g, w in REFERENCE[a].items()) for a in adsorbates}

    (folder / "structures").mkdir(exist_ok=True)
    out = (folder / "results.jsonl").open("a")
    done = 0
    for n, m in enumerate(materials):
        start = done
        label = m["label"]
        tag = m["id"].replace(":", "_").replace("/", "_")
        try:
            _save_status(folder, message=f"{label}: building and relaxing the surface", done=done)
            t0 = time.time()
            bulk = bulk_from(m)
            hkl = tuple(m.get("facet") or densest_plane(bulk))
            slab = build_slab(bulk, hkl)
            e_slab, steps, conv = relax(slab)
            write(folder / "structures" / f"{tag}__slab.cif", slab)
            out.write(json.dumps({"material": m["id"], "adsorbate": "*", "site": "clean", "facet": list(hkl),
                                  "n_atoms": len(slab), "energy": e_slab, "steps": steps, "converged": conv,
                                  "seconds": round(time.time() - t0, 1), "structure": f"structures/{tag}__slab.cif"}) + "\n")
            out.flush()
            done += 1
            sites = surface_sites(slab)
            total += sum(len(choose_sites(sites, a, setting["sites"])) for a in adsorbates) + 1 - per_material
            _save_status(folder, total=total)
            for a in adsorbates:
                for site in choose_sites(sites, a, setting["sites"]):
                    _save_status(folder, message=f"{label}: {a} on the {site['label']} site", done=done)
                    t0 = time.time()
                    adslab = place(slab, a, site)
                    e, steps, conv = relax(adslab)
                    name = f"{tag}__{a}__{site['label'].replace(' ', '-')}.cif"
                    write(folder / "structures" / name, adslab)
                    out.write(json.dumps({"material": m["id"], "adsorbate": a, "site": site["label"],
                                          "E_bind": e - e_slab - refs[a], "steps": steps, "converged": conv,
                                          "anomaly": anomaly(adslab, len(slab), a), "n_ads": len(adslab) - len(slab),
                                          "seconds": round(time.time() - t0, 1), "structure": f"structures/{name}"}) + "\n")
                    out.flush()
                    done += 1
        except Exception as exc:  # one material failing leaves the rest of the screen running
            out.write(json.dumps({"material": m["id"], "adsorbate": "*", "site": "error", "error": str(exc)}) + "\n")
            out.flush()
            traceback.print_exc()
            done = max(done, start + 1)
    out.close()
    _save_status(folder, state="finished", done=done, total=done, message="Finished", finished=time.time())


if __name__ == "__main__":
    root = Path(sys.argv[1])
    try:
        main(root)
    except Exception as exc:
        traceback.print_exc()
        _save_status(root, state="failed", message=f"{type(exc).__name__}: {exc}")
        raise
