"""Reactions: the library by category, a form that adds a new reaction, and each reaction's chemistry in detail."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.chem.formula import element_counts, molar_mass
from catalyst_engine.chem.network import parse_equation
from catalyst_engine.design import builder, elements
from catalyst_engine.design.spec import ADSORBATES
from catalyst_engine.packs import PACKS_DIR
from catalyst_engine.ui import common

st.title("Reactions")
st.markdown("Every reaction the app can design catalysts for. Pick one to start a design, add your own with a short "
            "form, or open one to see its chemistry in detail.")

library_tab, add_tab, inside_tab = st.tabs(["Library", "Add a reaction", "Inside a reaction"])

# ---------------------------------------------------------------- library
with library_tab:
    ids = common.pack_ids()
    by_cat: dict[str, list] = {}
    for pid in ids:
        by_cat.setdefault(common.pack(pid).spec.category, []).append(common.pack(pid))
    for cat in sorted(by_cat, key=common.category_order):
        st.markdown(f"#### {cat}")
        packs = sorted(by_cat[cat], key=lambda p: p.spec.name)
        for start in range(0, len(packs), 2):
            for col, p in zip(st.columns(2, gap="medium"), packs[start:start + 2]):
                with col.container(border=True, height="stretch"):
                    built = not (PACKS_DIR / p.id).exists()
                    st.markdown(f"**{p.spec.name}**" + ("  :violet-badge[Built in the app]" if built else ""))
                    st.write(p.spec.summary or " ".join(p.spec.description.split()[:30]))
                    wanted = [pw for pw in p.spec.pathways.values() if pw.role == "desired"]
                    st.markdown("<br>".join(f"<span class='ce-eq'>{pw.equation}</span>" for pw in wanted),
                                unsafe_allow_html=True)
                    scr = p.spec.screening
                    if scr is not None and not scr.on_design_page:
                        st.caption(f"Catalyst to beat: {scr.reference.name or scr.reference.formula} · runs in the "
                                   "Catalyst playground; it joins the Design page once its ranking check passes "
                                   "(Manual, Methodology tab, limits)")
                        a, b = st.columns([3, 1])
                    elif scr is not None:
                        st.caption(f"Catalyst to beat: {scr.reference.name or scr.reference.formula} · ranks materials "
                                   f"by their grip on {', '.join(ADSORBATES[a] for a in scr.descriptor)}")
                        a, b = st.columns([3, 1])
                        if a.button("Design a catalyst for this", key=f"lib_design_{p.id}", icon=":material/arrow_forward:"):
                            st.session_state["design_pack_wanted"] = p.id
                            st.switch_page("views/design.py")
                        if built:
                            with b.popover("Remove", icon=":material/delete:"):
                                st.write(f"Remove {p.spec.name} from the library? Its file goes too.")
                                if st.button("Yes, remove it", key=f"lib_rm_{p.id}", type="primary"):
                                    builder.delete_reaction(p.id)
                                    st.rerun()

# ---------------------------------------------------------------- add a reaction
with add_tab:
    st.markdown("Fill in the four parts below and press **Add this reaction**. The app checks the chemistry, sets the "
                "speed of the catalyst to beat from the temperature you give, and adds the reaction to every page.")

    with st.container(border=True):
        st.markdown("##### 1. The reaction")
        c = st.columns([2, 1])
        name = c[0].text_input("Name", placeholder="Dry reforming of methane", key="new_name")
        category = c[1].selectbox("Group", common.CATEGORIES, index=1, key="new_cat")
        summary = st.text_input("One sentence on what it does", placeholder="Turns methane and CO2 into syngas.",
                                key="new_summary")
        wanted = st.text_input("Wanted reaction", placeholder="CH4 + CO2 <=> 2 CO + 2 H2", key="new_wanted")
        st.caption("Write gases as formulas joined by +. Use -> for a reaction that runs one way, and <=> for one that "
                   "can run backward (its equilibrium then limits the result). Examples: 2 CO + O2 -> 2 CO2, "
                   "N2 + 3 H2 <=> 2 NH3.")
        side_text = st.text_area("Side reactions to watch (optional, one per line)", placeholder="CO2 + H2 <=> CO + H2O",
                                 height=80, key="new_side")
        side = [line.strip() for line in side_text.splitlines() if line.strip()]
        parsed = None
        if wanted.strip():
            try:
                for eq in [wanted, *side]:
                    stoich, _ = parse_equation(eq)
                    bal: dict[str, float] = {}
                    for sp, nu in stoich.items():
                        for el, n in element_counts(sp).items():
                            bal[el] = bal.get(el, 0.0) + nu * n
                    off = {el: round(v, 3) for el, v in bal.items() if abs(v) > 1e-9}
                    if off:
                        raise ValueError(f"{eq} leaves atoms over: {off}")
                parsed = builder.species_in([wanted, *side])
                st.success(f"Balanced. Gases: {', '.join(parsed)}", icon=":material/check:")
            except ValueError as exc:
                st.error(f"Check the equation: {exc}")

    if parsed:
        go_in, come_out = builder.reactants(wanted), builder.products(wanted)
        with st.container(border=True):
            st.markdown("##### 2. What counts as a good result")
            kind = st.radio("Rank catalysts by", ["Conversion", "Yield"], horizontal=True, key="new_kind",
                            captions=["the share of a gas going in that reacts",
                                      "how much of a product comes out per unit of a gas going in"])
            c = st.columns(2)
            if kind == "Conversion":
                gas = c[0].selectbox("Gas going in", go_in, key="new_conv")
                metric = {"kind": "conversion", "species": [gas]}
            else:
                prod = c[0].selectbox("Product", come_out, key="new_prod")
                base = c[1].selectbox("Per unit of", go_in, key="new_base")
                metric = {"kind": "yield", "species": [prod], "denominator": [base]}

        with st.container(border=True):
            st.markdown("##### 3. How to rank materials")
            st.caption("Catalysts work best when they hold one key atom or molecule just right: firmly enough to "
                       "react, lightly enough to let go. Pick it here; the app computes how strongly each material holds it.")
            c = st.columns(2)
            descriptor = c[0].multiselect("Key atom or molecule", list(ADSORBATES), default=builder.suggest_descriptor(wanted),
                                          format_func=lambda a: f"{a} · {ADSORBATES[a]}", key=f"new_desc_{wanted}",
                                          help="Oxidations: O. Nitrogen chemistry: N. Adding hydrogen to carbon: C and O. "
                                               "Two picks average their binding energies.")
            suggested = c[1].multiselect("Metals to suggest first", elements.METALS, default=["Pt", "Pd", "Ni", "Cu"],
                                         key="new_suggested")
            c = st.columns(3)
            ref_formula = c[0].text_input("Catalyst to beat (formula)", "Pt", key="new_ref",
                                          help="A pure metal such as Pt or Ni, or a compound such as V2O5. Compounds come "
                                               "from the Materials Project.")
            ref_name = c[1].text_input("Its plain name", "Platinum", key="new_ref_name")
            ea = c[2].slider("Activation energy (kJ/mol)", 30, 200, 80, 5, key="new_ea",
                             help="How quickly the reaction speeds up with heat. 60 to 100 is typical.")
            basis = st.text_area("Why this key molecule (optional; a paper or a sentence of reasoning)", height=68,
                                 key="new_basis")

        with st.container(border=True):
            st.markdown("##### 4. Standard conditions")
            st.caption("The starting point on the Design page; each design can change them.")
            seed = pd.DataFrame([{"Gas": g, "Amount": "10 %"} for g in go_in] + [{"Gas": "N2", "Amount": "balance"}])
            feed_df = st.data_editor(seed, num_rows="dynamic", hide_index=True, key=f"new_feed_{wanted}",
                                     column_config={"Amount": st.column_config.TextColumn(
                                         help="Like 10 %, 500 ppm, or balance for the rest")})
            feed = {str(r["Gas"]).strip(): str(r["Amount"]).strip() for r in feed_df.to_dict("records")
                    if r.get("Gas") and r.get("Amount")}
            c = st.columns(3)
            t_lo, t_hi = c[0].slider("Temperature range (°C)", 0, 1000, (150, 500), step=10, key="new_trange")
            pressure = c[1].number_input("Pressure (bar)", 0.1, 300.0, 1.0, key="new_p")
            ghsv = c[2].number_input("Space velocity (1/h)", 100, 2_000_000, 20000, step=1000, key="new_ghsv")
            half = st.slider("Where does the catalyst to beat convert half of what it could? (°C)", t_lo, t_hi,
                             int((t_lo + t_hi) / 2), step=5, key="new_half",
                             help="From a paper, a datasheet or experience. The app sets the reaction speed so the "
                                  "catalyst to beat matches it.")

        if st.button("Add this reaction", type="primary", icon=":material/add_circle:", key="new_go",
                     disabled=not (name.strip() and descriptor and ref_formula.strip())):
            with st.spinner("Checking the chemistry and setting the speed..."):
                try:
                    pid = builder.create_reaction(
                        name=name, category=category, summary=summary or name, wanted=wanted, side=side,
                        metric=metric, descriptor=descriptor, reference_formula=ref_formula, reference_name=ref_name,
                        suggested=suggested, feed=feed, pressure_bar=pressure, ghsv_per_h=ghsv,
                        temperatures_C=[float(t) for t in np.linspace(t_lo, t_hi, 9)], half_C=float(half),
                        Ea_kJ=float(ea), basis=basis)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["design_pack_wanted"] = pid
                    st.success(f"{name} is in the library.", icon=":material/check_circle:")
                    common.link("views/design.py", f"Design a catalyst for {name}", ":material/arrow_forward:")
        st.caption("New reactions are saved in data/reactions inside the catalyst-engine folder, as a pack.yaml file "
                   "you can edit later.")

# ---------------------------------------------------------------- one reaction in detail
with inside_tab:
    p = common.pack_picker("reactions_pack")
    net = p.network
    spec = p.spec

    with st.container(border=True):
        st.markdown(f"### {spec.name}")
        st.write(spec.description.strip())
        st.markdown(
            f":gray-badge[{p.id}] :gray-badge[version {spec.version}] "
            + (":orange-badge[Illustrative parameters]" if "illustrative" in spec.status else f":blue-badge[{spec.status}]")
        )

    pathways_tab, rates_tab, metrics_tab, protocols_tab = st.tabs(
        ["Pathways", "Rates vs temperature", "Species and metrics", "Test protocols"]
    )

    with pathways_tab:
        st.caption("Desired pathways make the product you want. Side pathways waste reagent or make something you must avoid.")
        ids = list(spec.pathways)
        for start in range(0, len(ids), 2):
            cols = st.columns(2)
            for col, pid in zip(cols, ids[start:start + 2]):
                pw = spec.pathways[pid]
                with col.container(border=True, height="stretch"):
                    a, b = st.columns([3, 1], vertical_alignment="center")
                    a.markdown(f"**{pid}**")
                    with b:
                        st.badge("Desired" if pw.role == "desired" else "Side", color="green" if pw.role == "desired" else "orange")
                    st.latex(common.equation_latex(pw.equation))
                    st.latex(common.rate_latex(pw))
                    t_ref_c = net.t_ref - 273.15
                    st.markdown(
                        f"k at {t_ref_c:,.0f} °C = **{common.fmt(np.exp(pw.rate.ln_k_ref))}** (SI units) · "
                        f"activation energy **{pw.rate.Ea_kJ:g} kJ/mol**"
                    )
                    if pw.note:
                        st.caption(pw.note)

    with rates_tab:
        temps = [t for proto in p.protocols.values() for step in proto.steps for t in step.kelvin()]
        lo_c, hi_c = min(temps) - 273.15 - 30, max(temps) - 273.15 + 30
        t_c = np.linspace(lo_c, hi_c, 80)
        ln_k = net.constants(net.defaults, t_c + 273.15).ln_k
        df = pd.concat(
            [pd.DataFrame({"T_C": t_c, "k": np.exp(ln_k[:, j]), "pathway": pw.id,
                           "role": spec.pathways[pw.id].role}) for j, pw in enumerate(net.pathways)]
        )
        desired = iter(["#0f766e", "#14b8a6", "#0369a1", "#38bdf8"])
        side = iter(["#ea580c", "#dc2626", "#b45309", "#f59e0b", "#be185d", "#a16207"])
        palette = {pw.id: next(desired if spec.pathways[pw.id].role == "desired" else side, "#64748b") for pw in net.pathways}
        lines = (
            alt.Chart(df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("T_C:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
                y=alt.Y("k:Q", title="Rate constant k (log scale)", scale=alt.Scale(type="log")),
                color=alt.Color("pathway:N", scale=alt.Scale(domain=list(palette), range=list(palette.values())),
                                legend=alt.Legend(orient="bottom", title=None)),
                strokeDash=alt.StrokeDash("role:N", scale=alt.Scale(domain=["desired", "side"], range=[[1, 0], [6, 4]]),
                                          legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["pathway:N", alt.Tooltip("T_C:Q", title="°C", format=".0f"), alt.Tooltip("k:Q", format=".3g")],
            )
        )
        ref = alt.Chart(pd.DataFrame({"T_C": [net.t_ref - 273.15]})).mark_rule(color=common.REFERENCE_GRAY, strokeDash=[2, 2]).encode(x="T_C:Q")
        st.altair_chart((lines + ref).properties(height=380), width="stretch")
        st.markdown(
            """
    **How to read it.** Each line is one pathway's intrinsic rate constant at the pack's parameters. A steeper
    line has a higher activation energy, so it gains more with every degree. Where a dashed side pathway climbs
    toward a solid desired pathway, selectivity falls and the catalyst wastes reagent. The gray rule marks the
    reference temperature where catalysts are compared. A better catalyst moves the desired line up, the side
    lines down, or flattens the side lines.
    """
        )
        st.caption("Coverage and inhibition terms scale these rates further at real gas compositions; the playground includes them.")

    with metrics_tab:
        st.markdown("##### Species")
        rows = []
        for s, sp in spec.species.items():
            involved = [pw.id for j, pw in enumerate(net.pathways) if net.nu[j, net.index(s)] != 0]
            rows.append({"Species": s, "Formula": sp.formula, "Molar mass (g/mol)": round(molar_mass(sp.formula), 2),
                         "Pathways": ", ".join(involved) or "carrier gas"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        st.markdown("##### Metrics")
        st.caption("What every test reports, computed the same way for every catalyst.")
        rows = [{"Metric": mid, "What it measures": m.description or "", "Unit": m.label_unit(),
                 "Target": common.target_text(m.target)} for mid, m in spec.metrics.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"What it measures": st.column_config.TextColumn(width="large")})

    with protocols_tab:
        for proto in p.protocols.values():
            with st.container(border=True):
                st.markdown(f"#### {proto.id}")
                st.markdown(f":gray-badge[version {proto.version}] :blue-badge[{proto.status}]")
                st.write(proto.description.strip())
                an = proto.analyzer
                st.markdown(
                    f"**Analyzer:** {an.name} reads {', '.join(an.measured)}. Reading noise: "
                    f"{an.sigma_abs_ppm:g} ppm plus {an.sigma_rel:.0%} of the reading."
                )
                for q in proto.open_questions:
                    st.warning(q, icon=":material/help:")
                rows = [
                    {
                        "Step": step.id,
                        "Purpose": " ".join(step.purpose.split()),
                        "Feed": " · ".join(f"{s} {v}" for s, v in step.feed.items()) + f" ({step.feed_basis} basis)",
                        "Space velocity (1/h)": f"{step.ghsv_per_h:,.0f}",
                        "Temperatures (°C)": ", ".join(f"{t - 273.15:,.0f}" for t in step.kelvin()),
                    }
                    for step in proto.steps
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             column_config={"Purpose": st.column_config.TextColumn(width="large"),
                                            "Feed": st.column_config.TextColumn(width="medium")})
