"""Zeolite active sites built on a framework crystal: aluminium in place of silicon, and the cation it holds.

A zeolite is a silica framework of corner-sharing SiO4 tetrahedra (T atoms) around pores. Each Al in place of a Si
leaves one negative charge on the framework, which a cation balances; that cation is where molecules bind:

  H      one Al, a proton on one of the four O around it (the Bronsted acid site); one variant per O
  M+     one Al in a six-membered ring, a metal ion (Cu+, Ag+) over the ring
  M2+    two Al across a six-membered ring (third neighbours), a metal ion (Cu2+, Fe2+) in the ring
  MOH+   one Al in an eight-membered ring, a metal hydroxide ([CuOH]+, [FeOH]+) in the ring

Rings are cycles of T atoms joined through O that close in real space and have no shortcut across them. The pore
is the point of the cell farthest from every atom. Imports only numpy and ASE, like uma_worker.py, so the UMA
worker's Python can use it too.
"""

from __future__ import annotations

import itertools

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

FORMS = {
    "H": "Acid form (H+, one Al)",
    "M+": "Metal ion M+ (one Al, six-membered ring)",
    "M2+": "Metal ion M2+ (two Al across a six-membered ring)",
    "MOH+": "Metal hydroxide [M-OH]+ (one Al, eight-membered ring)",
}
BOND_CUTOFF = 1.9  # T-O bonds in silicates run 1.55-1.75 A
# Materials Project frameworks checked against the IZA structure database (cell, ring sizes, density)
KNOWN = {"mp-aaaabamv": "Chabazite (CHA)"}


def code(material_id: str) -> str:
    """The IZA three-letter code of a known framework, e.g. CHA; empty for others."""
    name = KNOWN.get(material_id, "")
    return name[name.find("(") + 1:name.find(")")] if "(" in name else ""


def is_framework(atoms: Atoms) -> bool:
    """A silica framework: Si and O only, every Si with four O."""
    symbols = set(atoms.get_chemical_symbols())
    if symbols != {"Si", "O"}:
        return False
    i, j = neighbor_list("ij", atoms, BOND_CUTOFF)
    counts = np.bincount(i[np.array([atoms[b].symbol == "O" for b in j], dtype=bool)], minlength=len(atoms))
    return all(counts[k] == 4 for k, a in enumerate(atoms) if a.symbol == "Si")


def t_graph(z: Atoms) -> dict[int, list[tuple[int, int, np.ndarray]]]:
    """T atom -> [(neighbouring T, bridging O, lattice offset of that T)] through T-O-T bridges."""
    i, j, shift = neighbor_list("ijS", z, BOND_CUTOFF)
    sym = np.array(z.get_chemical_symbols())
    t_of_o: dict[int, list[tuple[int, np.ndarray]]] = {}
    for a, b, s in zip(i, j, shift):
        if sym[a] == "O" and sym[b] != "O":
            t_of_o.setdefault(int(a), []).append((int(b), s))
    graph: dict[int, list] = {}
    for o, ts in t_of_o.items():
        for (t1, s1), (t2, s2) in itertools.permutations(ts, 2):
            graph.setdefault(t1, []).append((t2, o, s2 - s1))
    return graph


def rings(z: Atoms, size: int) -> list[list[tuple[int, np.ndarray]]]:
    """Rings of `size` T atoms: [(T index, lattice offset)] in ring order, one entry per distinct ring."""
    graph = t_graph(z)
    found, seen = [], set()
    for start in graph:
        stack = [[(start, np.zeros(3, int))]]
        while stack:
            path = stack.pop()
            t, off = path[-1]
            for nb, _, s in graph[t]:
                o2 = off + s
                if nb == start and not o2.any() and len(path) == size:
                    keys = {frozenset((a, tuple(b - c)) for a, b in path) for _, c in path}
                    if not keys & seen:
                        seen |= keys
                        found.append(list(path))
                elif len(path) < size and all(not (nb == a and (o2 == b).all()) for a, b in path):
                    stack.append(path + [(nb, o2)])
    out = []
    for ring in found:  # a shortcut joins two members that are not next to each other: two smaller rings, not one
        pts = ring_xyz(z, ring)
        n = len(ring)
        if not any(abs(k - m) not in (1, n - 1) and np.linalg.norm(pts[k] - pts[m]) < 3.4
                   for k in range(n) for m in range(k + 1, n)):
            out.append(ring)
    return out


def ring_xyz(z: Atoms, ring) -> np.ndarray:
    return np.array([z.positions[t] + o @ z.cell.array for t, o in ring])


_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def clearances(z: Atoms, points: np.ndarray, skip: int | None = None) -> np.ndarray:
    """Distance from each point to the nearest atom of `z` (leaving out atom `skip`), across periodic boundaries."""
    cell = z.cell.array
    fp = np.atleast_2d(points) @ np.linalg.inv(cell)
    best = np.full(len(fp), np.inf)
    for k, p in enumerate(z.get_scaled_positions(wrap=False)):
        if k == skip:
            continue
        d = fp - p
        d -= np.round(d)
        best = np.minimum(best, np.linalg.norm((d[:, None, :] + _SHIFTS[None]) @ cell, axis=2).min(axis=1))
    return best


def clearance(z: Atoms, point: np.ndarray, skip: int | None = None) -> float:
    return float(clearances(z, point, skip)[0])


def windows(z: Atoms, size: int) -> list:
    """Rings of `size` whose centre stands clear of the framework (open windows), most open first."""
    found = rings(z, size)
    room = [clearance(z, ring_xyz(z, r).mean(axis=0)) for r in found]
    return [r for _, r in sorted(zip(room, found), key=lambda x: -x[0])]


def pore_center(z: Atoms, n: int = 24) -> np.ndarray:
    """The point of the cell farthest from every atom: the middle of the largest cage or channel."""
    pts = ((np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n) @ z.cell.array
    return pts[int(np.argmax(clearances(z, pts)))]


def nearest_image(z: Atoms, point: np.ndarray, near: np.ndarray) -> np.ndarray:
    """The periodic image of `point` closest to `near`."""
    frac = np.linalg.solve(z.cell.array.T, point - near)
    return near + (frac - np.round(frac)) @ z.cell.array


def open_direction(z: Atoms, atom: int, reach: float, prefer: np.ndarray | None = None) -> np.ndarray:
    """The unit vector from `atom` along which a point `reach` away sits farthest from every other atom; among
    directions within 0.1 A of the best clearance, the one closest to `prefer`."""
    golden = np.pi * (3 - 5 ** 0.5)
    k = np.arange(400)
    zz = 1 - 2 * (k + 0.5) / 400
    dirs = np.stack([np.sqrt(1 - zz ** 2) * np.cos(golden * k), np.sqrt(1 - zz ** 2) * np.sin(golden * k), zz], 1)
    room = clearances(z, z.positions[atom] + reach * dirs, skip=atom)
    ok = dirs[room >= room.max() - 0.1]
    if prefer is None:
        return ok[0]
    return ok[np.argmax(ok @ (prefer / np.linalg.norm(prefer)))]


def _o_of(z: Atoms, t: int) -> list[int]:
    i, j = neighbor_list("ij", z, BOND_CUTOFF)
    return [int(b) for a, b in zip(i, j) if a == t and z[b].symbol == "O"]


def _proton(z: Atoms, o: int) -> np.ndarray:
    """An H 0.97 A from framework O, pointing away from the two T atoms it bridges."""
    i, j, d = neighbor_list("ijD", z, BOND_CUTOFF)
    away = -sum(v / np.linalg.norm(v) for a, b, v in zip(i, j, d) if a == o and z[b].symbol != "O")
    return z.positions[o] + 0.97 * away / np.linalg.norm(away)


def _ring_frame(z: Atoms, ring, pore: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ring atoms' positions, the ring centre, and the ring's normal pointing toward the pore."""
    pts = ring_xyz(z, ring)
    centre = pts.mean(axis=0)
    normal = np.linalg.svd(pts - centre)[2][2]
    toward = nearest_image(z, pore, centre) - centre
    return pts, centre, (normal if normal @ toward > 0 else -normal)


def build(z: Atoms, form: str, metal: str = "Cu") -> tuple[list[Atoms], int]:
    """Starting structures for one active site on framework `z` (each relaxes; the most stable counts) and the
    index of the atom molecules bind to: the proton for the acid form, the metal otherwise."""
    pore = pore_center(z)
    if form == "H":
        t0 = int(min(i for i, a in enumerate(z) if a.symbol == "Si"))
        variants = []
        for o in _o_of(z, t0):
            s = z.copy()
            s[t0].symbol = "Al"
            s += Atoms("H", positions=[_proton(z, o)])
            variants.append(s)
        return variants, len(z)
    if form in ("M+", "M2+"):
        six = windows(z, 6)
        if not six:
            raise ValueError("This framework has no six-membered ring to hold the metal ion.")
        ring = six[0]
        pts, centre, normal = _ring_frame(z, ring, pore)
        s = z.copy()
        s[ring[0][0]].symbol = "Al"
        if form == "M+":
            lean = pts[0] - centre
            spot = centre + 0.8 * lean / np.linalg.norm(lean) + 0.6 * normal
        else:
            s[ring[3][0]].symbol = "Al"
            spot = centre + 0.3 * normal
        s += Atoms(metal, positions=[spot])
        return [s], len(z)
    if form == "MOH+":
        eight = windows(z, 8)
        if not eight:
            raise ValueError("This framework has no eight-membered ring to hold the metal hydroxide.")
        ring = eight[0]
        pts, centre, normal = _ring_frame(z, ring, pore)
        s = z.copy()
        s[ring[0][0]].symbol = "Al"
        to_al = (pts[0] - centre) / np.linalg.norm(pts[0] - centre)
        metal_at = centre + (np.linalg.norm(pts[0] - centre) - 2.9) * to_al + 0.5 * normal
        o_at = metal_at + 1.80 * normal
        h_dir = (normal + 0.5 * to_al) / np.linalg.norm(normal + 0.5 * to_al)
        s += Atoms(f"{metal}OH", positions=[metal_at, o_at, o_at + 0.97 * h_dir])
        return [s], len(z)
    raise ValueError(f"Unknown zeolite form {form!r}; known: {sorted(FORMS)}")


def place_at(s: Atoms, site: int, molecule: Atoms, reach: float) -> Atoms:
    """`s` with `molecule` (its binding atom first, at the origin, the rest along +z) bound to atom `site`: the
    binding atom `reach` away, straight out along the O-H bond for a proton, otherwise along the most open direction
    (toward the pore when several are open), the molecule pointing away from the site."""
    if s[site].symbol == "H":
        o = int(np.argmin(s.get_distances(site, [i for i in range(len(s)) if i != site], mic=True)))
        o = o if o < site else o + 1
        u = s.get_distance(o, site, mic=True, vector=True)
        u /= np.linalg.norm(u)
    else:
        toward = nearest_image(s, pore_center(s), s.positions[site]) - s.positions[site]
        u = open_direction(s, site, reach, toward)
    m = molecule.copy()
    if len(m) > 1:
        m.rotate((0, 0, 1), u)
    m.translate(s.positions[site] + reach * u)
    return s + m


def si_al(atoms: Atoms) -> float:
    symbols = atoms.get_chemical_symbols()
    return symbols.count("Si") / max(1, symbols.count("Al"))
