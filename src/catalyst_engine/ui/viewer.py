"""Interactive 3D view of a crystal or a surface with molecules on it, drawn by 3Dmol.js in the browser."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st
from ase import Atoms

THREE_DMOL = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"


def _repeats(atoms: Atoms, surface: bool, target: float = 9.0, max_atoms: int = 400) -> tuple[int, int, int]:
    lengths = atoms.cell.lengths()
    reps = [max(1, int(np.ceil(target / max(l, 1e-6)))) for l in lengths]
    if surface:
        reps[2] = 1
        reps[0], reps[1] = min(reps[0], 2), min(reps[1], 2)
    while np.prod(reps) * len(atoms) > max_atoms and max(reps) > 1:
        reps[int(np.argmax(reps))] -= 1
    return tuple(reps)


def show(atoms: Atoms, n_highlight: int = 0, height: int = 380, surface: bool = False, spin: bool = False) -> None:
    """Draw `atoms` repeated a few times so the crystal or surface reads as a material, with the cell outlined.
    The last `n_highlight` atoms (the adsorbed molecule) draw larger in every copy."""
    n = len(atoms)
    reps = _repeats(atoms, surface)
    big = atoms.repeat(reps)
    marked = [k * n + i for k in range(int(np.prod(reps))) for i in range(n - n_highlight, n)] if n_highlight else []
    xyz = f"{len(big)}\n\n" + "\n".join(f"{s} {x:.4f} {y:.4f} {z:.4f}"
                                         for s, (x, y, z) in zip(big.get_chemical_symbols(), big.positions))
    a, b, c = atoms.cell.array
    corners = {(i, j, k): (i * a + j * b + k * c).tolist() for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    edges = [(p, q) for p in corners for q in corners if sum(abs(x - y) for x, y in zip(p, q)) == 1 and p < q]
    if surface:  # the vacuum box adds little; outline only the surface cell's footprint at the slab's top
        z_top = float(atoms.positions[:, 2].max())
        foot = {(i, j): (i * a + j * b).tolist() for i in (0, 1) for j in (0, 1)}
        lines = [([*foot[p][:2], z_top], [*foot[q][:2], z_top])
                 for p, q in [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))]]
    else:
        lines = [(corners[p], corners[q]) for p, q in edges]
    surface_style = {"sphere": {"scale": 0.42, "colorscheme": "Jmol"}}
    crystal_style = {"sphere": {"scale": 0.3, "colorscheme": "Jmol"}, "stick": {"radius": 0.12, "colorscheme": "Jmol"}}
    base_style = surface_style if surface else crystal_style
    html = f"""
<div id="mol" style="width:100%;height:{height}px;position:relative;border-radius:8px;"></div>
<script src="{THREE_DMOL}"></script>
<script>
  const viewer = $3Dmol.createViewer(document.getElementById("mol"), {{backgroundColor: "white", backgroundAlpha: 0}});
  viewer.addModel({json.dumps(xyz)}, "xyz");
  // Jmol colors, the same as the legend under the view; metals as spheres, the molecule larger with its bonds
  viewer.setStyle({{}}, {json.dumps(base_style)});
  const marked = {json.dumps(marked)};
  if (marked.length) viewer.setStyle({{index: marked}}, {{sphere: {{scale: 0.5, colorscheme: "Jmol"}},
                                                        stick: {{radius: 0.18, colorscheme: "Jmol"}}}});
  for (const [p, q] of {json.dumps(lines)}) {{
    viewer.addLine({{start: {{x: p[0], y: p[1], z: p[2]}}, end: {{x: q[0], y: q[1], z: q[2]}}, color: "#94a3b8"}});
  }}
  viewer.zoomTo();
  {"viewer.rotate(-65, 'x'); viewer.zoom(1.15);" if surface else "viewer.rotate(20, 'y'); viewer.rotate(-15, 'x');"}
  {"viewer.spin('y', 0.4);" if spin else ""}
  viewer.render();
</script>"""
    st.iframe(html, height=height + 10)  # the page builds this HTML itself from computed structures


def element_colors(symbols: list[str]) -> str:
    """A one-line legend of the elements in view, with their 3Dmol (Jmol) colors."""
    jmol = {"H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "Pt": "#D0D0E0", "Pd": "#006985",
            "Rh": "#0A7D8C", "Ru": "#248F8F", "Ir": "#175487", "Au": "#FFD123", "Ag": "#C0C0C0", "Cu": "#C88033",
            "Ni": "#50D050", "Co": "#F090A0", "Fe": "#E06633", "Mo": "#54B5B5", "W": "#2194D6", "V": "#A6A6AB",
            "Ti": "#BFC2C7", "Mn": "#9C7AC7", "Ce": "#FFFFC7", "Zn": "#7D80B0", "Al": "#BFA6A6", "Si": "#F0C8A0"}
    return " ".join(f'<span style="color:{jmol.get(s, "#888")};font-size:1.1em">●</span> {s}'
                    for s in dict.fromkeys(symbols))
