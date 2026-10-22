"""Each reaction pack in plain language: pathways, how fast each runs with temperature, test plans."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.chem.formula import molar_mass
from catalyst_engine.ui import common
from catalyst_engine.units import f_to_k, k_to_f

st.title("Reactions")
st.caption("Each reaction is a folder under reactions/. Add a folder and it appears here.")

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
                t_ref_f = k_to_f(net.t_ref)
                st.markdown(
                    f"k at {t_ref_f:,.0f} °F = **{common.fmt(np.exp(pw.rate.ln_k_ref))}** (SI units) · "
                    f"activation energy **{pw.rate.Ea_kJ:g} kJ/mol**"
                )
                if pw.note:
                    st.caption(pw.note)

with rates_tab:
    temps = [t for proto in p.protocols.values() for step in proto.steps for t in step.kelvin()]
    lo_f, hi_f = k_to_f(min(temps)) - 50, k_to_f(max(temps)) + 50
    t_f = np.linspace(lo_f, hi_f, 80)
    ln_k = net.constants(net.defaults, f_to_k(t_f)).ln_k
    df = pd.concat(
        [pd.DataFrame({"T_F": t_f, "k": np.exp(ln_k[:, j]), "pathway": pw.id,
                       "role": spec.pathways[pw.id].role}) for j, pw in enumerate(net.pathways)]
    )
    desired = iter(["#0f766e", "#14b8a6", "#0369a1", "#38bdf8"])
    side = iter(["#ea580c", "#dc2626", "#b45309", "#f59e0b", "#be185d", "#a16207"])
    palette = {pw.id: next(desired if spec.pathways[pw.id].role == "desired" else side, "#64748b") for pw in net.pathways}
    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("T_F:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
            y=alt.Y("k:Q", title="Rate constant k (log scale)", scale=alt.Scale(type="log")),
            color=alt.Color("pathway:N", scale=alt.Scale(domain=list(palette), range=list(palette.values())),
                            legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("role:N", scale=alt.Scale(domain=["desired", "side"], range=[[1, 0], [6, 4]]),
                                      legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["pathway:N", alt.Tooltip("T_F:Q", title="°F", format=".0f"), alt.Tooltip("k:Q", format=".3g")],
        )
    )
    ref = alt.Chart(pd.DataFrame({"T_F": [k_to_f(net.t_ref)]})).mark_rule(color=common.REFERENCE_GRAY, strokeDash=[2, 2]).encode(x="T_F:Q")
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
                    "Temperatures (°F)": ", ".join(f"{k_to_f(t):,.0f}" for t in step.kelvin()),
                }
                for step in proto.steps
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         column_config={"Purpose": st.column_config.TextColumn(width="large"),
                                        "Feed": st.column_config.TextColumn(width="medium")})
