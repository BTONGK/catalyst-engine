"""Materials explorer: search the Materials Project, look at a crystal in 3D with its properties, and add it to
the design list."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ase.data import atomic_numbers, reference_states

from catalyst_engine.design import elements, materials
from catalyst_engine.ui import common, viewer

st.title("Materials")
st.markdown("Look up any crystal before you screen it: its structure in 3D, how stable it is, whether it conducts, "
            "and how easy its elements are to source. Add the ones you like to your design list.")

CRYSTALS = {"fcc": "face-centered cubic", "bcc": "body-centered cubic", "hcp": "hexagonal close-packed",
            "bct": "body-centered tetragonal"}
cands: dict[str, dict] = st.session_state.setdefault("design_candidates", {})
status = common.engine_status()


def add_button(entry: dict, key: str) -> None:
    if entry["id"] in cands:
        st.button("On your design list", icon=":material/check:", disabled=True, key=key)
    elif st.button("Add to my design list", icon=":material/add:", type="primary", key=key):
        cands[entry["id"]] = entry
        st.rerun()


def show_crystal(atoms, title: str, rows: list[tuple[str, str]], key: str) -> None:
    view, facts = st.columns([3, 2], gap="large")
    with view:
        viewer.show(atoms, height=420, spin=st.session_state.get(f"mat_spin_{key}", False))
        st.markdown(viewer.element_colors(atoms.get_chemical_symbols()), unsafe_allow_html=True)
        st.caption("Drag to turn, scroll to zoom. The gray box is one repeat unit of the crystal.")
    with facts:
        st.markdown(f"### {title}")
        st.markdown("\n".join(f"- **{k}:** {v}" for k, v in rows))
        st.toggle("Spin", key=f"mat_spin_{key}")
        st.download_button("Download the structure (CIF)", materials.cif_text(atoms), file_name=f"{title}.cif",
                           icon=":material/download:", key=f"mat_cif_{key}")


search_tab, metals_tab = st.tabs(["Search the Materials Project", "Pure metals"])

with search_tab:
    if not status["mp_key"]:
        st.info("Searching needs a free Materials Project API key. Sign in at materialsproject.org, copy the key from "
                "your dashboard, and save it in ~/.config/.pmgrc.yaml as a line reading PMG_MAPI_KEY: your-key. "
                "The Pure metals tab works right away.", icon=":material/key:")
    else:
        c = st.columns([3, 1, 1])
        els = c[0].multiselect("Elements to combine", sorted(elements.CRUST_MG_PER_KG), default=["Cu", "Pd"],
                               format_func=lambda e: f"{e} · {elements.name(e)}", key="mat_els")
        mix = c[1].selectbox("Mix up to", [1, 2, 3], index=1, format_func=lambda n: f"{n} element{'s' * (n > 1)}",
                             key="mat_mix")
        kind = c[2].selectbox("Kind", ["Metals and alloys", "Oxides", "Everything"], key="mat_kind")
        c = st.columns(2)
        max_hull = c[0].slider("Stability: energy above the most stable form (eV/atom)", 0.0, 0.20, 0.05, 0.01,
                               key="mat_hull", help="0 is the most stable form. Up to about 0.05 can often be made.")
        max_atoms = c[1].slider("Largest crystal cell (atoms)", 2, 60, 24, key="mat_atoms",
                                help="Smaller cells compute faster in a screen.")
        rows: list[dict] = []
        if els:
            try:
                rows = common.mp_search(tuple(sorted(els)), mix, kind == "Oxides", max_hull, max_atoms)
            except materials.MaterialsProjectError as exc:
                st.error(str(exc))
            if kind == "Metals and alloys":
                rows = [r for r in rows if "O" not in r["elements"]]
            elif kind == "Oxides":
                rows = [r for r in rows if "O" in r["elements"]]
        if rows:
            table = pd.DataFrame([{
                "Material": r["formula"], "ID": r["material_id"], "Stability (eV/atom)": round(r["e_hull"], 3),
                "Conducts": "metal" if r["is_metal"] else f"gap {r['band_gap']:.1f} eV" if r["band_gap"] is not None else "",
                "Atoms in cell": r["nsites"], "Crystal": f"{r['crystal_system']} {r['space_group']}",
                "Supply": elements.supply_label(r["elements"]),
                "Made in a lab": "yes" if r["theoretical"] is False else "predicted"} for r in rows])
            st.caption(f"{len(rows)} materials, most stable first. Click a row to see it in 3D.")
            sel = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                               selection_mode="single-row", key="mat_table", height=280)
            row = rows[sel.selection.rows[0]] if sel.selection.rows else rows[0]
            atoms = materials.atoms_of(common.mp_structure(row["material_id"]))
            facts = [("Materials Project ID", row["material_id"]),
                     ("Crystal", f"{row['crystal_system']}, space group {row['space_group']}"),
                     ("Repeat unit", materials.lattice_summary(atoms)),
                     ("Atoms in the repeat unit", str(len(atoms))),
                     ("Stability", f"{row['e_hull']:.3f} eV/atom above the most stable form"
                      + (" (the most stable form)" if row["e_hull"] < 1e-4 else "")),
                     ("Formation energy", f"{row['formation_energy']:.2f} eV/atom" if row["formation_energy"] is not None else "n/a"),
                     ("Electronic", "metal (conducts)" if row["is_metal"] else f"band gap {row['band_gap']:.2f} eV"
                      if row["band_gap"] is not None else "n/a"),
                     ("Density", f"{row['density']:.2f} g/cm³" if row["density"] else "n/a"),
                     ("Supply", f"{elements.supply_label(row['elements'])}; scarcest element "
                      f"{elements.name(elements.scarcest(row['elements'])[0])}")]
            show_crystal(atoms, row["formula"], facts, "mp")
            add_button({"id": row["material_id"], "label": f"{row['formula']} ({row['material_id']})",
                        "formula": row["formula"], "e_hull": row["e_hull"], "elements": row["elements"],
                        "source": "Materials Project", "nsites": row["nsites"]}, key=f"mat_add_{row['material_id']}")
        elif els:
            st.caption("Try more elements, a looser stability limit or a larger cell to widen the search.")

with metals_tab:
    symbol = st.selectbox("Metal", elements.METALS, format_func=lambda e: f"{e} · {elements.name(e)}", key="mat_metal")
    entry = materials.element_entry(symbol)
    atoms = materials.atoms_of(entry)
    show_crystal(atoms, symbol, [("Name", elements.name(symbol)),
                                 ("Crystal", CRYSTALS.get(reference_states[atomic_numbers[symbol]]["symmetry"], "")),
                                 ("Repeat unit", materials.lattice_summary(atoms)),
                                 ("Supply", elements.supply_label([symbol])),
                                 ("Source", "Reference crystal at room temperature")], "metal")
    add_button(entry, key=f"mat_add_{entry['id']}")

st.divider()
st.markdown(f"**Your design list:** {len(cands)} material{'s' * (len(cands) != 1)}"
            + (f": {', '.join(m['formula'] for m in cands.values())}" if cands else ""))
common.link("views/design.py", "Go to Design a catalyst to run them", ":material/arrow_forward:")
