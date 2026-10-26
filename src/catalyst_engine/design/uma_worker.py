"""Atomistic screening worker: binding energies of adsorbates on candidate surfaces with Meta's UMA model.

The app launches this file as a script with the Python that has fairchem (CE_UMA_PYTHON), one screen folder per
run:

    TORCHDYNAMO_DISABLE=1 /opt/miniconda3/bin/python3 uma_worker.py data/screens/<id>

It reads <id>/job.json, writes progress to status.json, one JSON line per relaxation to results.jsonl and each
relaxed structure to structures/*.cif. It imports only numpy and ASE at module level (fairchem and torch load in
main), so the app and the tests can use its geometry functions.

For each material: take the bulk crystal, cut its close-packed face (the most tightly packed outer atomic layer
among Miller indices up to 1, preferring a layer with the bulk's composition), pick the termination whose
composition matches the bulk, nudge
its free atoms a few hundredths of an Angstrom and relax it with BFGS, so it settles in a true minimum rather than
balancing on a symmetric point, place each adsorbate on the surface's top, bridge and hollow sites, relax, and keep the binding
energy

    E_bind = E(slab + adsorbate) - E(slab) - E_ref(adsorbate)

E(slab) is the relaxed clean surface, or, when the surface under the adsorbate settled lower on its own, that surface
relaxed with the adsorbate lifted off; the binding energy then counts only the bond.

A zeolite entry (kind "framework", built by zeolites.py) skips the slab: each starting arrangement of its active
site relaxes whole in its fixed cell, the most stable is E(slab), and each adsorbate binds to the site atom (the
metal ion, or the proton) along the most open direction.

Gas references come from the same model: H = 1/2 H2, O = H2O - H2, N = 1/2 N2, C = CO - O, OH = O + H,
OOH = 2 H2O - 3/2 H2, and the molecules themselves for CO, NO and NH3. The electrochemical intermediates take the
same gases: COOH and the formate OCHO = CO + H2O - 1/2 H2, NHx = 1/2 N2 + x/2 H2, N2Hx = N2 + x/2 H2,
NOH = NO + 1/2 H2, and the carbon intermediates of CO2 reduction and methanol oxidation CHxO = CO + x/2 H2 (formyl CHO
and COH, formaldehyde CH2O and CHOH, methoxy CH3O and hydroxymethyl CH2OH). Negative is bound; more negative binds
more strongly. H, O, OH and OOH against H2O and H2 are the computational hydrogen electrode's references
(design/electro.py).

Formate (OCHO), diazene (HNNH), HNNH2 and formaldehyde (CH2O) bind through two atoms side by side; on a bridge site
they lie along the bridge, one binding atom toward each metal atom (ALIGNED).
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
from ase.geometry import find_mic
from ase.io import write
from ase.optimize import BFGS

try:
    from catalyst_engine.design import zeolites
except ImportError:  # run as a script by the UMA Python, which holds numpy, ASE and fairchem only
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zeolites

# Binding atom first, at the origin; molecules point away from the surface (+z).
_NH3_H = [(0.94 * np.cos(t), 0.94 * np.sin(t), 0.38) for t in np.radians([0, 120, 240])]
ADSORBATE_ATOMS = {
    "H": Atoms("H"),
    "C": Atoms("C"),
    "N": Atoms("N"),
    "O": Atoms("O"),
    "OH": Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.98)]),
    # hydroperoxyl: the bound O at the origin, the O-O bond (1.40 A) leaning out, the H (0.98 A) beyond it
    "OOH": Atoms("OOH", positions=[(0, 0, 0), (0.95, 0, 1.03), (1.86, 0, 1.40)]),
    "CO": Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.15)]),
    "NO": Atoms("NO", positions=[(0, 0, 0), (0, 0, 1.17)]),
    "NH3": Atoms("NH3", positions=[(0, 0, 0), *_NH3_H]),
    # carboxyl: bound through C, the C=O (1.26 A) and C-OH (1.35 A) bonds leaning out to either side
    "COOH": Atoms("COOH", positions=[(0, 0, 0), (1.05, 0, 0.70), (-1.15, 0, 0.70), (-1.15, 0, 1.68)]),
    # formate, bound through both O (C-O 1.28 A); the origin sits midway between them, the C-H points up
    "OCHO": Atoms("OOCH", positions=[(-1.10, 0, 0), (1.10, 0, 0), (0, 0, 0.65), (0, 0, 1.75)]),
    "NH": Atoms("NH", positions=[(0, 0, 0), (0, 0, 1.03)]),
    "NH2": Atoms("NHH", positions=[(0, 0, 0), (0.82, 0, 0.62), (-0.82, 0, 0.62)]),
    # N2H end-on: the N-N bond (1.20 A) tilted, the H (1.04 A) on the outer N
    "NNH": Atoms("NNH", positions=[(0, 0, 0), (0.35, 0, 1.15), (1.35, 0, 1.45)]),
    "NNH2": Atoms("NNHH", positions=[(0, 0, 0), (0, 0, 1.28), (0.87, 0, 1.80), (-0.87, 0, 1.80)]),
    # diazene and N2H3 bound through both N (N-N 1.30 and 1.44 A); the origin sits midway between them
    "HNNH": Atoms("NNHH", positions=[(-0.65, 0, 0), (0.65, 0, 0), (-1.20, 0, 0.85), (1.20, 0, 0.85)]),
    "HNNH2": Atoms("NNHHH", positions=[(-0.72, 0, 0), (0.72, 0, 0), (-1.30, 0, 0.84), (1.10, 0.82, 0.55),
                                         (1.10, -0.82, 0.55)]),
    "NOH": Atoms("NOH", positions=[(0, 0, 0), (0, 0, 1.36), (0.92, 0, 1.71)]),
    # formyl, bound through C: the C=O (1.22 A) and C-H (1.10 A) bonds leaning out to either side at 119 degrees
    "CHO": Atoms("CHO", positions=[(0, 0, 0), (-0.95, 0, 0.55), (1.05, 0, 0.62)]),
    # hydroxymethylidyne, bound through C, the C-O (1.32 A) upright and the H (0.99 A) on O
    "COH": Atoms("COH", positions=[(0, 0, 0), (0, 0, 1.32), (0.92, 0, 1.70)]),
    # formaldehyde bound through both C and O (C-O 1.36 A); the origin sits midway between them, the CH2 tilted up
    "CH2O": Atoms("COHH", positions=[(-0.68, 0, 0), (0.68, 0, 0), (-1.18, 0.85, 0.45), (-1.18, -0.85, 0.45)]),
    # hydroxymethylene, bound through C: C-O 1.35 A and C-H 1.10 A leaning out, the H on O (0.98 A) pointing up
    "CHOH": Atoms("CHOH", positions=[(0, 0, 0), (-0.95, 0, 0.55), (1.10, 0, 0.78), (0.81, 0, 1.72)]),
    # methoxy, bound through O, the C-O (1.43 A) upright and its three H (1.10 A) at the tetrahedral angle
    "CH3O": Atoms("OCHHH", positions=[(0, 0, 0), (0, 0, 1.43),
                                       *[(1.04 * np.cos(t), 1.04 * np.sin(t), 1.80) for t in np.radians([0, 120, 240])]]),
    # hydroxymethyl, bound through C at the tetrahedral angle to its O (1.38 A) and two H (1.10 A), the H on O upright
    "CH2OH": Atoms("COHHH", positions=[(0, 0, 0), (1.30, 0, 0.46), (-0.52, 0.90, 0.37), (-0.52, -0.90, 0.37),
                                        (1.30, 0, 1.44)]),
}
# Adsorbates bound through two atoms side by side: on a bridge site their x axis lies along the bridge
ALIGNED = {"OCHO", "HNNH", "HNNH2", "CH2O"}
# Height of the binding atom above the local top-layer height, by site kind (Angstrom)
HEIGHTS = {
    "H": {"top": 1.6, "bridge": 1.2, "hollow": 1.0},
    "C": {"top": 1.8, "bridge": 1.4, "hollow": 1.1},
    "N": {"top": 1.8, "bridge": 1.4, "hollow": 1.1},
    "O": {"top": 1.9, "bridge": 1.5, "hollow": 1.2},
    "OH": {"top": 2.0, "bridge": 1.6, "hollow": 1.4},
    "OOH": {"top": 2.0, "bridge": 1.7, "hollow": 1.5},
    "CO": {"top": 1.9, "bridge": 1.5, "hollow": 1.3},
    "NO": {"top": 1.8, "bridge": 1.5, "hollow": 1.3},
    "NH3": {"top": 2.1, "bridge": 2.1, "hollow": 2.1},
    "COOH": {"top": 1.95, "bridge": 1.55, "hollow": 1.35},
    "OCHO": {"top": 2.0, "bridge": 2.0, "hollow": 1.9},
    "NH": {"top": 1.8, "bridge": 1.4, "hollow": 1.1},
    "NH2": {"top": 1.9, "bridge": 1.5, "hollow": 1.3},
    "NNH": {"top": 1.9, "bridge": 1.5, "hollow": 1.3},
    "NNH2": {"top": 1.9, "bridge": 1.5, "hollow": 1.2},
    "HNNH": {"top": 1.9, "bridge": 1.9, "hollow": 1.7},
    "HNNH2": {"top": 2.0, "bridge": 2.0, "hollow": 1.8},
    "NOH": {"top": 1.8, "bridge": 1.4, "hollow": 1.1},
    "CHO": {"top": 1.95, "bridge": 1.55, "hollow": 1.35},
    "COH": {"top": 1.8, "bridge": 1.4, "hollow": 1.15},
    "CH2O": {"top": 1.9, "bridge": 1.9, "hollow": 1.7},
    "CHOH": {"top": 1.95, "bridge": 1.5, "hollow": 1.3},
    "CH3O": {"top": 2.0, "bridge": 1.6, "hollow": 1.35},
    "CH2OH": {"top": 2.05, "bridge": 1.6, "hollow": 1.4},
}
# Site kinds to try first, per adsorbate (the quick setting keeps the first two sites)
PREFERENCE = {
    "H": ["hollow", "top", "bridge"],
    "C": ["hollow", "bridge", "top"],
    "N": ["hollow", "bridge", "top"],
    "O": ["hollow", "bridge", "top"],
    "OH": ["bridge", "top", "hollow"],
    "OOH": ["top", "bridge", "hollow"],
    "CO": ["top", "hollow", "bridge"],
    "NO": ["hollow", "top", "bridge"],
    "NH3": ["top"],
    "COOH": ["top", "bridge", "hollow"],
    "OCHO": ["bridge", "top", "hollow"],
    "NH": ["hollow", "bridge", "top"],
    "NH2": ["bridge", "hollow", "top"],
    "NNH": ["top", "bridge", "hollow"],
    "NNH2": ["top", "hollow", "bridge"],
    "HNNH": ["bridge", "top", "hollow"],
    "HNNH2": ["bridge", "top", "hollow"],
    "NOH": ["hollow", "bridge", "top"],
    "CHO": ["top", "bridge", "hollow"],
    "COH": ["hollow", "bridge", "top"],
    "CH2O": ["bridge", "top", "hollow"],
    "CHOH": ["bridge", "top", "hollow"],
    "CH3O": ["hollow", "bridge", "top"],
    "CH2OH": ["top", "bridge", "hollow"],
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
    "OOH": {"H2O": 2.0, "H2": -1.5},
    "CO": {"CO": 1.0},
    "NO": {"NO": 1.0},
    "NH3": {"NH3": 1.0},
    "COOH": {"CO": 1.0, "H2O": 1.0, "H2": -0.5},
    "OCHO": {"CO": 1.0, "H2O": 1.0, "H2": -0.5},
    "NH": {"N2": 0.5, "H2": 0.5},
    "NH2": {"N2": 0.5, "H2": 1.0},
    "NNH": {"N2": 1.0, "H2": 0.5},
    "NNH2": {"N2": 1.0, "H2": 1.0},
    "HNNH": {"N2": 1.0, "H2": 1.0},
    "HNNH2": {"N2": 1.0, "H2": 1.5},
    "NOH": {"NO": 1.0, "H2": 0.5},
    "CHO": {"CO": 1.0, "H2": 0.5},
    "COH": {"CO": 1.0, "H2": 0.5},
    "CH2O": {"CO": 1.0, "H2": 1.0},
    "CHOH": {"CO": 1.0, "H2": 1.0},
    "CH3O": {"CO": 1.0, "H2": 1.5},
    "CH2OH": {"CO": 1.0, "H2": 1.5},
}
NONMETALS = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "Te"}
SETTINGS = {  # sites per adsorbate, force tolerance (eV/A), step cap
    "quick": {"sites": 2, "fmax": 0.10, "steps": 150},
    "standard": {"sites": 3, "fmax": 0.05, "steps": 250},
    "thorough": {"sites": 5, "fmax": 0.03, "steps": 400},
}


def bulk_from(material: dict) -> Atoms:
    """A bulk crystal from a job entry: a Materials Project structure or an element's reference crystal."""
    if material.get("ase_bulk"):  # the cubic cell when a face is chosen in cubic indices
        return ase_bulk(material["ase_bulk"], cubic=bool(material.get("cubic")))
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


def relax_with(atoms: Atoms, calc, fmax: float, steps: int) -> tuple[float, int, bool]:
    """Relax with BFGS: energy, steps taken, converged. BFGS steps downhill along every direction that curves down,
    so it leaves a balance point once the nudge tips it; LBFGS takes the same steps elsewhere but can walk back."""
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return float(atoms.get_potential_energy()), opt.get_number_of_steps(), converged


def nudge(slab: Atoms, stdev: float = 0.03, seed: int = 0) -> Atoms:
    """Shift the free atoms by about `stdev` Angstrom, the same way every run, so a relaxation can leave a symmetric
    balance point (zero force, higher energy) for the minimum beside it. The fixed bottom half stays put."""
    slab.rattle(stdev=stdev, seed=seed)
    return slab


def largest_move(before: np.ndarray, after: Atoms) -> float:
    """How far the atom that moved most travelled from `before`, in Angstrom, across the periodic boundaries."""
    return float(find_mic(after.positions - before, after.cell, after.pbc)[1].max())


def clean_reference(adslab: Atoms, n_slab: int, e_slab: float, energy, relax, tol: float = 0.05) -> tuple[float, float]:
    """The clean-surface energy to set a binding energy against, and how far it lies below the relaxed clean slab.
    Usually the relaxed clean slab itself; when the surface under the adsorbate holds less energy than that (by more
    than `tol` eV), the surface settled lower on its own, and its relaxation with the adsorbate lifted off is the
    reference. `energy` and `relax` take Atoms and return eV."""
    bare = adslab[:n_slab]
    if energy(bare) >= e_slab - tol:
        return e_slab, 0.0
    e_bare = relax(bare)
    return min(e_slab, e_bare), max(0.0, e_slab - e_bare)


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
        axis = (images[j] - xy[0]) / np.linalg.norm(images[j] - xy[0])
        sites.append({"kind": "bridge", "label": "bridge", "xy": mid.tolist(), "z": float(max(z[top[0]], zz[j])),
                      "axis": axis.tolist()})
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
    if adsorbate in ALIGNED and "axis" in site:  # one binding atom toward each atom of the bridge
        ads.rotate(float(np.degrees(np.arctan2(site["axis"][1], site["axis"][0]))), "z", center=(0, 0, 0))
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
        bonded = (d0 > 0) & (d0 < 1.5)  # bonds up to the O-O of OOH (1.40 A); the H-H of NH3 (1.63 A) stays out
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


def _framework(m: dict, tag: str, label: str, adsorbates: list[str], refs: dict, relax, energy, out, folder: Path,
               done: int, total: int) -> tuple[int, int]:
    """One zeolite site: relax each starting arrangement, then each adsorbate on the site atom of each. Writes
    the same rows as a surface; returns the updated (done, total) counts."""
    site = m["site"]
    total += len(m["variants"]) * (1 + len(adsorbates))
    _save_status(folder, total=total)
    relaxed = []
    for k, v in enumerate(m["variants"]):
        _save_status(folder, message=f"{label}: relaxing the site ({k + 1} of {len(m['variants'])})", done=done)
        t0 = time.time()
        s = nudge(Atoms(v["species"], scaled_positions=v["frac"], cell=m["lattice"], pbc=True))
        e, steps, conv = relax(s)
        write(folder / "structures" / f"{tag}__site{k}.cif", s)
        relaxed.append({"energy": e, "atoms": s, "steps": steps, "converged": conv, "seconds": time.time() - t0,
                        "structure": f"structures/{tag}__site{k}.cif"})
        done += 1
    best = min(relaxed, key=lambda r: r["energy"])
    out.write(json.dumps({"material": m["id"], "adsorbate": "*", "site": "clean", "kind": "framework",
                          "n_atoms": len(best["atoms"]), "energy": best["energy"], "steps": best["steps"],
                          "converged": best["converged"], "seconds": round(sum(r["seconds"] for r in relaxed), 1),
                          "structure": best["structure"]}) + "\n")
    out.flush()
    for a in adsorbates:
        for k, r in enumerate(relaxed):
            s = r["atoms"]
            name = s[site].symbol + (f" {k + 1}" if len(relaxed) > 1 else "")
            _save_status(folder, message=f"{label}: {a} on the {name} site", done=done)
            t0 = time.time()
            reach = 1.6 if s[site].symbol == "H" else HEIGHTS[a]["top"]
            adslab = zeolites.place_at(s, site, ADSORBATE_ATOMS[a], reach)
            e, steps, conv = relax(adslab)
            e_ref, drop = clean_reference(adslab, len(s), best["energy"], energy, lambda x: relax(x)[0])
            path = f"{tag}__{a}__{name.replace(' ', '-')}.cif"
            write(folder / "structures" / path, adslab)
            out.write(json.dumps({"material": m["id"], "adsorbate": a, "site": name, "E_bind": e - e_ref - refs[a],
                                  "slab_drop_eV": round(drop, 3), "steps": steps, "converged": conv,
                                  "anomaly": anomaly(adslab, len(s), a), "n_ads": len(adslab) - len(s),
                                  "seconds": round(time.time() - t0, 1), "structure": f"structures/{path}"}) + "\n")
            out.flush()
            done += 1
    return done, total


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
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 2)))
    unit = pretrained_mlip.load_predict_unit(os.path.expanduser(job["checkpoint"]), device="cpu")
    calc = FAIRChemCalculator(unit, task_name=job["task"])

    def relax(atoms: Atoms) -> tuple[float, int, bool]:
        return relax_with(atoms, calc, setting["fmax"], setting["steps"])

    def energy(atoms: Atoms) -> float:
        atoms.calc = calc
        return float(atoms.get_potential_energy())

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
            if m.get("kind") == "framework":
                done, total = _framework(m, tag, label, adsorbates, refs, relax, energy, out, folder, done,
                                         total - per_material)
                continue
            _save_status(folder, message=f"{label}: building and relaxing the surface", done=done)
            t0 = time.time()
            bulk = bulk_from(m)
            hkl = tuple(m.get("facet") or densest_plane(bulk))
            slab = build_slab(bulk, hkl, **job.get("slab", {}))
            built = slab.positions.copy()
            e_slab, steps, conv = relax(nudge(slab))
            write(folder / "structures" / f"{tag}__slab.cif", slab)
            out.write(json.dumps({"material": m["id"], "adsorbate": "*", "site": "clean", "facet": list(hkl),
                                  "n_atoms": len(slab), "energy": e_slab, "steps": steps, "converged": conv,
                                  "moved_A": round(largest_move(built, slab), 3),
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
                    e_ref, drop = clean_reference(adslab, len(slab), e_slab, energy, lambda s: relax(s)[0])
                    name = f"{tag}__{a}__{site['label'].replace(' ', '-')}.cif"
                    write(folder / "structures" / name, adslab)
                    out.write(json.dumps({"material": m["id"], "adsorbate": a, "site": site["label"],
                                          "E_bind": e - e_ref - refs[a], "slab_drop_eV": round(drop, 3),
                                          "steps": steps, "converged": conv,
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
