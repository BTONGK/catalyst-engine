"""Pieces of the electrocatalysis pages that mirror the thermal Design and Results pages: the candidate list, the
3D structures tab and the table of every site tried."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from ase.io import read

from catalyst_engine.design import elements, materials
from catalyst_engine.design.electro import ADSORBATES
from catalyst_engine.ui import common, viewer

METALS = elements.METALS


def candidates_section(*, pid: str, reference_formula: str, reference_name: str, suggested: list[str],
                       cands: dict[str, dict], advanced: bool, status: dict, prefix: str = "design") -> None:
    """Step 3 of a design: pure metals, the Materials Project search, and the list with its faces (Advanced).
    `cands` is the session's candidate dict for this kind of screen; widget keys start with `prefix`."""
    with st.container(border=True):
        st.markdown("### 3. Which materials should the app try?")
        st.caption(f"{reference_name or reference_formula} is always included as the catalyst to beat. "
                   "Add pure metals, search the Materials Project's 150,000+ crystals, or both.")
        metals_tab, mp_tab = st.tabs(["Pure metals", "Search the Materials Project"])
        with metals_tab:
            picks = [e for e in suggested if e in METALS and e != reference_formula][:6]
            picked = st.multiselect("Metals", METALS, default=picks, key=f"{prefix}_metals_{pid}",
                                    format_func=lambda e: f"{e} · {elements.name(e)}")
            if st.button("Add these metals", icon=":material/add:", key=f"{prefix}_add_metals"):
                for e in picked:
                    entry = materials.element_entry(e)
                    cands[entry["id"]] = entry
                st.rerun()
        with mp_tab:
            if not status["mp_key"]:
                st.info("Add your Materials Project API key (free at materialsproject.org) to ~/.config/.pmgrc.yaml as "
                        "PMG_MAPI_KEY to search its crystals.", icon=":material/key:")
            else:
                c = st.columns([3, 1, 1])
                els = c[0].multiselect("Elements to combine", sorted(elements.CRUST_MG_PER_KG),
                                       default=[e for e in suggested if e in elements.CRUST_MG_PER_KG][:4],
                                       key=f"{prefix}_mp_els_{pid}")
                mix = c[1].selectbox("Mix up to", [1, 2, 3], index=1, format_func=lambda n: f"{n} element{'s' * (n > 1)}",
                                     key=f"{prefix}_mp_mix")
                kind = c[2].selectbox("Kind", ["Metals and alloys", "Oxides"], key=f"{prefix}_mp_kind")
                c = st.columns(2)
                max_hull = c[0].slider("Stability: energy above the most stable form (eV/atom)", 0.0, 0.10, 0.02, 0.01,
                                       key=f"{prefix}_mp_hull", help="0 means the most stable form; up to 0.05 can often be made.")
                max_atoms = c[1].slider("Largest crystal cell (atoms)", 4, 40, 16, key=f"{prefix}_mp_atoms",
                                        help="Smaller cells compute faster.")
                if els:
                    try:
                        rows = common.mp_search(tuple(sorted(els)), mix, kind == "Oxides", max_hull, max_atoms)
                    except materials.MaterialsProjectError as exc:
                        st.error(str(exc))
                        rows = []
                    if kind == "Metals and alloys":
                        rows = [r for r in rows if "O" not in r["elements"]]
                    if rows:
                        table = pd.DataFrame([{"Material": r["formula"], "ID": r["material_id"],
                                               "Stability (eV/atom)": round(r["e_hull"], 3),
                                               "Atoms in cell": r["nsites"], "Crystal": f"{r['crystal_system']} {r['space_group']}",
                                               "Supply": elements.supply_label(r["elements"])} for r in rows])
                        sel = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                                           selection_mode="multi-row", key=f"{prefix}_mp_table_{pid}")
                        chosen = [rows[i] for i in sel.selection.rows]
                        a, b = st.columns([1, 2])
                        with a:
                            st.caption(f"{len(rows)} materials found. Tick rows to add them; the first ticked row shows in 3D.")
                            if st.button(f"Add {len(chosen)} selected", icon=":material/add:", disabled=not chosen,
                                         key=f"{prefix}_add_mp"):
                                for r in chosen:
                                    cands[r["material_id"]] = {"id": r["material_id"], "label": f"{r['formula']} ({r['material_id']})",
                                                               "formula": r["formula"], "e_hull": r["e_hull"],
                                                               "elements": r["elements"], "source": "Materials Project",
                                                               "nsites": r["nsites"]}
                                st.rerun()
                        with b:
                            if chosen:
                                atoms = materials.atoms_of(common.mp_structure(chosen[0]["material_id"]))
                                viewer.show(atoms, height=260)
                                st.caption(f"{chosen[0]['formula']}: {materials.lattice_summary(atoms)}")
                    else:
                        st.caption("Pick elements to see the stable materials made of them.")
        st.markdown(f"**Your list** ({len(cands)} material{'s' * (len(cands) != 1)} plus the catalyst to beat)")
        if cands:
            listing = pd.DataFrame([{"Remove": False, "Material": m["label"], "Source": m["source"],
                                     "Supply": elements.supply_label(m["elements"]),
                                     **({"Face": common.face_text(m)} if advanced else {}), "_id": i} for i, m in cands.items()])
            edited = st.data_editor(listing, hide_index=True, width="stretch", key=f"{prefix}_list_{advanced}",
                                    column_config={"_id": None, "Remove": st.column_config.CheckboxColumn(width="small"),
                                                   "Face": st.column_config.TextColumn(
                                                       help="auto: the close-packed face. Or three Miller indices such "
                                                            "as 1 1 0: in the cubic cell for a cubic metal, otherwise in "
                                                            "the crystal's own cell.")},
                                    disabled=["Material", "Source", "Supply"])
            records = edited.to_dict("records")
            gone = [r["_id"] for r in records if r["Remove"]]
            if gone:
                for i in gone:
                    cands.pop(i, None)
                st.rerun()
            if advanced:
                changed = False
                for r in records:
                    m = cands.get(r["_id"])
                    if m is None or m.get("kind") == "framework" or r["Face"] == common.face_text(m):
                        continue
                    try:
                        hkl = common.parse_face(r["Face"])
                    except ValueError:
                        st.warning(f"{m['label']}: write the face as three whole numbers, such as 1 1 1, or auto.")
                        continue
                    if hkl is None:
                        m.pop("facet", None)
                        m.pop("cubic", None)
                    else:
                        m["facet"] = hkl
                        if common.cubic_metal(m):
                            m["cubic"] = True
                    changed = True
                if changed:
                    st.rerun()
        else:
            st.caption("Empty so far. Add pure metals or Materials Project crystals above.")


def reference_entries(reference_formula: str, reference_name: str, cands: dict[str, dict], ref_choice: str) -> list[dict]:
    """The job entries of a screen: the catalyst to beat first (the reaction's own, or the candidate chosen in
    Advanced), then every candidate with its crystal structure."""
    if reference_formula in METALS:
        ref = materials.element_entry(reference_formula)
    else:
        ref = materials.job_entry(materials.most_stable(reference_formula))
    ref_label = reference_name or reference_formula

    def full(m: dict) -> dict:
        return m if m.get("ase_bulk") or m.get("kind") == "framework" else {**m, **common.mp_structure(m["id"])}

    if ref_choice in cands:
        chosen = cands[ref_choice]
        entries = [{**full(chosen), "is_reference": True, "ref_name": chosen["formula"],
                    "label": f"{chosen['label']} (catalyst to beat)"}, {**ref, "label": ref_label}]
    else:
        entries = [{**ref, "is_reference": True, "label": f"{ref_label} (catalyst to beat)"}]
    for i, m in cands.items():
        if i in {e["id"] for e in entries}:
            continue
        entries.append(full(m))
    return entries


def structures_tab(folder: Path, preds: list, chosen_face: set[str], key: str) -> None:
    """The 3D view of each finished material: its clean surface and every molecule on every site tried."""
    done = [p for p in preds if p.slab is not None]
    if not done:
        st.caption("Structures appear as each material finishes.")
        return
    a, b = st.columns([1, 2], gap="large")
    with a:
        pick = st.selectbox("Material", [p.material for p in done],
                            format_func={p.material: p.label for p in done}.get, key=f"res_3d_{key}")
        p = next(q for q in done if q.material == pick)
        zeolite = p.slab.get("kind") == "framework"
        bare = "Empty site" if zeolite else "Clean surface"
        views = {bare: (p.slab["structure"], 0)}
        best_view = bare
        for ads, bnd in p.binding.items():
            for site in bnd["sites"]:
                tag = f"{ads} on the {site['site']} site ({site['E']:+.2f} eV)" + (f", {site['note']}" if site["note"] else "")
                views[tag] = (site["structure"], site["n_ads"])
                if best_view == bare and site["structure"] == bnd["structure"]:
                    best_view = tag
        view = st.radio("Show", list(views), index=list(views).index(best_view), key=f"res_view_{key}_{pick}")
        if zeolite:
            st.caption(f"Zeolite site: the framework's whole cell, {p.slab['n_atoms']} atoms, every atom free to "
                       "move. Larger spheres are the molecule. The view shows eight copies of the cell, so each "
                       "molecule shows up in every copy. Drag to rotate, scroll to zoom.")
        else:
            facet = tuple(p.slab.get("facet"))
            face = (f"the face you chose, Miller indices {facet}" if p.material in chosen_face
                    else f"the close-packed face, Miller indices {facet}")
            st.caption(f"Surface: {face} in the crystal's own cell, {p.slab['n_atoms']} atoms, bottom half "
                       "held fixed. Larger spheres are the molecule. The view repeats the computed patch 2 × 2, "
                       "so each molecule shows up four times. Drag to rotate, scroll to zoom.")
        for ads, bnd in p.binding.items():
            if bnd["E"] is not None:
                st.markdown(f"**{ADSORBATES[ads].capitalize()} ({ads})**: binds at {bnd['E']:+.2f} eV on the "
                            f"{bnd['site']} site, best of {bnd['tried']} tried.")
    with b:
        path, n_ads = views[view]
        atoms = read(folder / path)
        viewer.show(atoms, n_highlight=n_ads, height=440, surface=not zeolite, span=18.0 if zeolite else 9.0)
        st.html(f"<div style='font-size:0.85rem'>{viewer.element_colors(atoms.get_chemical_symbols())}</div>")


def site_rows(preds: list) -> list[dict]:
    rows = []
    for p in preds:
        for ads, bnd in p.binding.items():
            for s in bnd["sites"]:
                rows.append({"Material": p.label, "Molecule": ads, "Site": s["site"], "Binding (eV)": round(s["E"], 3),
                             "Settled": "yes" if s["converged"] else "stopped at the step limit",
                             "Note": s["note"] or "kept"})
    return rows


def sites_and_downloads(preds: list, ranking_rows: list[dict], choice: str) -> None:
    """The table of every site tried, with the two CSV downloads."""
    rows = site_rows(preds)
    if not rows:
        return
    st.markdown("**Every site tried**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Desorbed: the molecule drifted off the surface. Dissociated: it broke apart. Both are left out "
               "of the lowest binding energy. The surface beneath settled lower: the surface relaxed further "
               "under the molecule, so its binding energy is set against that surface with the molecule lifted "
               "off, and counts only the bond.")
    c = st.columns(2)
    c[0].download_button("Download the ranking (CSV)", pd.DataFrame(ranking_rows).to_csv(index=False),
                         file_name=f"{choice}-ranking.csv", icon=":material/download:")
    c[1].download_button("Download every site (CSV)", pd.DataFrame(rows).to_csv(index=False),
                         file_name=f"{choice}-sites.csv", icon=":material/download:")
