"""Design a catalyst: pick a reaction, set the conditions, choose candidate materials, start a screen."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.design import elements, materials, screening
from catalyst_engine.design.spec import ADSORBATES
from catalyst_engine.ui import common, viewer

METALS = elements.METALS
LEVELS = {"Quick": "quick", "Standard": "standard", "Thorough": "thorough"}

st.title("Design a catalyst")
st.markdown("Pick a reaction, set the conditions, choose materials to try, and the app predicts which ones "
            "work best. Each step below takes a minute; the computer does the rest.")

status = common.engine_status()
if not status["uma"]["ready"]:
    st.warning(f"The atomistic model (UMA) is set up on another computer. Point CE_UMA_PYTHON at a Python with "
               f"fairchem installed and CE_UMA_CHECKPOINT at the model file. Looked for {status['uma']['python']} "
               f"and {status['uma']['checkpoint']}.", icon=":material/memory:")

ids = common.screening_pack_ids()
names = {pid: common.pack(pid).spec.name for pid in ids}
cands: dict[str, dict] = st.session_state.setdefault("design_candidates", {})

# ---------------------------------------------------------------- 1. reaction
with st.container(border=True):
    st.markdown("### 1. What should the catalyst do?")
    by_cat: dict[str, list[str]] = {}
    for pid in ids:
        by_cat.setdefault(common.pack(pid).spec.category, []).append(pid)
    rank = {c: i for i, c in enumerate(["Clean air", "Hydrogen and energy", "Chemicals and fuels"])}
    order = [pid for cat in sorted(by_cat, key=lambda c: (rank.get(c, 9), c)) for pid in sorted(by_cat[cat], key=names.get)]
    wanted = st.session_state.pop("design_pack_wanted", None)  # set by the Reactions page
    if wanted in order:
        st.session_state["design_pack"] = wanted
    elif st.session_state.get("design_pack") not in order:  # widget state clears when you visit another page
        last = st.session_state.get("design_pack_last")
        st.session_state["design_pack"] = last if last in order else order[0]
    pid = st.selectbox("Reaction", order, format_func=lambda p: f"{names[p]}  ·  {common.pack(p).spec.category}",
                       key="design_pack")
    st.session_state["design_pack_last"] = pid
    pack = common.pack(pid)
    spec = pack.spec.screening
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"**{pack.spec.summary or pack.spec.name}**")
        for pw_id, pw in pack.spec.pathways.items():
            tag = "wanted" if pw.role == "desired" else "side reaction"
            st.markdown(f"<span class='ce-eq'>{pw.equation}</span> &nbsp; :gray[{tag}]", unsafe_allow_html=True)
    with right:
        held = " and ".join(f"{ADSORBATES[a]} ({a})" for a in spec.descriptor)
        st.markdown(f"**Catalyst to beat:** {spec.reference.name or spec.reference.formula}")
        st.markdown(f"**Ranks materials by:** how strongly their surface holds {held}")
        with st.expander("Why this works"):
            st.write(" ".join(spec.basis.split()))
    common.link("views/reactions.py", "Want another reaction? Add your own on the Reactions page", ":material/add_circle:")

# ---------------------------------------------------------------- 2. conditions
step = screening.design_step(pack)
temps_c = [t - 273.15 for t in step.kelvin()]
with st.container(border=True):
    st.markdown("### 2. Under what conditions?")
    st.caption("These start at the reaction's standard test. Change them to match your process.")
    c = st.columns(3)
    t_lo, t_hi = c[0].slider("Temperature range (°C)", 0, 1000, (int(round(min(temps_c))), int(round(max(temps_c)))),
                             step=10, key=f"design_trange_{pid}")
    pressure = c[1].number_input("Pressure (bar)", 0.1, 300.0, round(step.pressure_kPa / 100.0, 2), key=f"design_p_{pid}")
    ghsv = c[2].number_input("Gas flow per catalyst volume (space velocity, 1/h)", 100, 2_000_000,
                             int(step.ghsv_per_h), step=1000, key=f"design_ghsv_{pid}")
    t_design = st.slider("Compare catalysts at (°C)", t_lo, t_hi, int(np.clip(np.median(temps_c), t_lo, t_hi)),
                         step=5, key=f"design_tcmp_{pid}",
                         help="The temperature for the headline numbers in the results.")
    st.markdown("**Gas going in**")
    feed_df = st.data_editor(
        pd.DataFrame([{"Gas": s, "Amount": str(v)} for s, v in step.feed.items()]), num_rows="dynamic",
        hide_index=True, key=f"design_feed_{pid}",
        column_config={"Gas": st.column_config.SelectboxColumn(options=list(pack.spec.species), required=True),
                       "Amount": st.column_config.TextColumn(help="Like 10 %, 500 ppm, or balance for the rest")})
    feed = {r["Gas"]: r["Amount"] for r in feed_df.to_dict("records") if r.get("Gas") and r.get("Amount")}
    conditions = {"temperatures_C": [float(t) for t in np.linspace(t_lo, t_hi, 9)], "pressure_bar": float(pressure),
                  "ghsv_per_h": float(ghsv), "feed": feed}
    try:
        grid = screening.step_for({**conditions, "temperatures_C": list(np.linspace(t_lo, t_hi, 25))})
        grid.feed_fractions(pack.network.species, pack.spec.water)
        feed_ok = True
    except ValueError as exc:
        st.error(f"Check the gas list: {exc}")
        feed_ok = False
    if feed_ok:
        metric = pack.spec.metrics[spec.metric]
        curves = screening.metric_curves(pack, grid, [pack.network.parameters(), screening.fast_params(pack)], spec.metric)
        names_ = [f"{spec.reference.name or spec.reference.formula} (catalyst to beat)", "Best possible (equilibrium)"]
        df = pd.DataFrame({"Temperature (°C)": np.tile(np.linspace(t_lo, t_hi, 25), 2), "Value": curves.ravel(),
                           "Curve": np.repeat(names_, 25)})
        chart = alt.Chart(df).mark_line(strokeWidth=3).encode(
            x=alt.X("Temperature (°C):Q", scale=alt.Scale(domain=[t_lo, t_hi], nice=False)),
            y=alt.Y("Value:Q", title=metric.kind.capitalize()),
            color=alt.Color("Curve:N", scale=alt.Scale(domain=names_, range=[common.REFERENCE_GRAY, common.ACCENT]),
                            legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("Curve:N", scale=alt.Scale(domain=names_, range=[[1, 0], [6, 4]]), legend=None),
            tooltip=["Curve:N", alt.Tooltip("Temperature (°C):Q", format=".0f"), alt.Tooltip("Value:Q", format=".2f")])
        st.altair_chart(chart.properties(height=300), width="stretch")
        st.caption(f"{metric.kind.capitalize()}: {metric.description}" if metric.description else "")
        st.caption("Before any screening: how the catalyst to beat performs under these conditions, and the most any "
                   "catalyst could reach. A better catalyst moves the gray curve toward the green one.")
        common.range_notes(pack, metric.kind, metric.species, t_lo, t_hi, float(np.nanmin(curves)))

# ---------------------------------------------------------------- 3. candidates
reference_formula = spec.reference.formula
with st.container(border=True):
    st.markdown("### 3. Which materials should the app try?")
    st.caption(f"{spec.reference.name or reference_formula} is always included as the catalyst to beat. "
               "Add pure metals, search the Materials Project's 150,000+ crystals, or both.")
    metals_tab, mp_tab = st.tabs(["Pure metals", "Search the Materials Project"])
    with metals_tab:
        suggested = [e for e in spec.suggested_elements if e in METALS and e != reference_formula][:6]
        picked = st.multiselect("Metals", METALS, default=suggested, key=f"design_metals_{pid}",
                                format_func=lambda e: f"{e} · {elements.name(e)}")
        if st.button("Add these metals", icon=":material/add:", key="design_add_metals"):
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
                                   default=[e for e in spec.suggested_elements if e in elements.CRUST_MG_PER_KG][:4],
                                   key=f"design_mp_els_{pid}")
            mix = c[1].selectbox("Mix up to", [1, 2, 3], index=1, format_func=lambda n: f"{n} element{'s' * (n > 1)}",
                                 key="design_mp_mix")
            kind = c[2].selectbox("Kind", ["Metals and alloys", "Oxides"], key="design_mp_kind")
            c = st.columns(2)
            max_hull = c[0].slider("Stability: energy above the most stable form (eV/atom)", 0.0, 0.10, 0.02, 0.01,
                                   key="design_mp_hull", help="0 means the most stable form; up to 0.05 can often be made.")
            max_atoms = c[1].slider("Largest crystal cell (atoms)", 4, 40, 16, key="design_mp_atoms",
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
                                       selection_mode="multi-row", key=f"design_mp_table_{pid}")
                    chosen = [rows[i] for i in sel.selection.rows]
                    a, b = st.columns([1, 2])
                    with a:
                        st.caption(f"{len(rows)} materials found. Tick rows to add them; the first ticked row shows in 3D.")
                        if st.button(f"Add {len(chosen)} selected", icon=":material/add:", disabled=not chosen,
                                     key="design_add_mp"):
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
                                 "Supply": elements.supply_label(m["elements"]), "_id": i} for i, m in cands.items()])
        edited = st.data_editor(listing, hide_index=True, width="stretch", key="design_list",
                                column_config={"_id": None, "Remove": st.column_config.CheckboxColumn(width="small")},
                                disabled=["Material", "Source", "Supply"])
        gone = [r["_id"] for r in edited.to_dict("records") if r["Remove"]]
        if gone:
            for i in gone:
                cands.pop(i, None)
            st.rerun()
    else:
        st.caption("Empty so far. Add pure metals or Materials Project crystals above.")

# ---------------------------------------------------------------- 4. run
with st.container(border=True):
    st.markdown("### 4. Run the screen")
    c = st.columns(2)
    level = c[0].segmented_control("How careful", list(LEVELS), default="Standard", key="design_level",
                                   help="Quick tries 2 spots per molecule on each surface, Standard 3, Thorough 5 "
                                        "with tighter settling.")
    name = c[1].text_input("Name this screen", f"{pack.spec.name} screen", key=f"design_name_{pid}")
    with st.expander("Model settings"):
        task = st.radio("UMA mode", ["oc20", "omat"], horizontal=True, key="design_task",
                        format_func={"oc20": "Catalysis (trained on surfaces with molecules)",
                                     "omat": "Materials (trained on bulk crystals)"}.get,
                        help="Catalysis mode is the default; the Model accuracy page compares the two.")
    level = level or "Standard"
    ref_elements = [reference_formula] if reference_formula in METALS else ["O"]  # a compound reference costs more
    secs = screening.seconds_estimate([{"elements": ref_elements}, *cands.values()], len(spec.descriptor),
                                      LEVELS[level])
    n_mat, minutes = len(cands) + 1, max(1, round(secs / 60))
    st.markdown(f"**{n_mat} material{'s' * (n_mat > 1)}** (the catalyst to beat included), {len(spec.descriptor)} "
                f"molecule type{'s' * (len(spec.descriptor) > 1)} each. Estimated time on this computer: "
                f"**about {minutes} minute{'s' * (minutes > 1)}**. It runs in the background; you can leave this page.")
    ready = status["uma"]["ready"] and feed_ok and bool(cands)
    if st.button("Start the screen", type="primary", icon=":material/play_arrow:", disabled=not ready,
                 key="design_start"):
        with st.spinner("Collecting crystal structures..."):
            try:
                if reference_formula in METALS:
                    ref = materials.element_entry(reference_formula)
                else:
                    ref = materials.job_entry(materials.most_stable(reference_formula))
                ref = {**ref, "is_reference": True, "label": f"{spec.reference.name or reference_formula} (catalyst to beat)"}
                entries = [ref]
                for i, m in cands.items():
                    if i == ref["id"]:
                        continue
                    entries.append(m if m.get("ase_bulk") else {**m, **common.mp_structure(i)})
                folder = screening.create_screen(pack, name, conditions, entries, LEVELS[level], task, float(t_design))
                screening.launch(folder)
            except (materials.MaterialsProjectError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["results_screen"] = folder.name
                st.switch_page("views/results.py")
    if not cands:
        st.caption("Add at least one material in step 3 to start.")
