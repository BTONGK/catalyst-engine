"""Free-energy playground: set each intermediate's free energy by hand and watch the limiting potential, the
overpotential, the free-energy diagram and the place on the volcano follow. The electrochemical counterpart of the
thermal app's Catalyst playground."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.design import electro
from catalyst_engine.ui import electro_ui

st.title("Free-energy playground")
st.markdown("Move each intermediate's free energy and see what it does to the potential a reaction needs. The same "
            "arithmetic runs behind every screen; here the numbers are yours.")

ids = electro.list_electro()
pid = st.selectbox("Reaction", ids, format_func=lambda i: electro_ui.spec(i).name, key="epg_pack")
s = electro_ui.spec(pid)
ideal = s.ideal()
st.markdown(f"<span class='ce-eq'>{s.equation}</span> &nbsp; :gray[{s.direction}, {s.electrons} electrons, "
            f"equilibrium at {s.equilibrium_V:.2f} V vs RHE]", unsafe_allow_html=True)
if not s.on_design_page:
    st.caption(f"{s.name} runs here on free energies you set. The Design page adds it once UMA's binding energies "
               "on its catalysts pass their accuracy check (Manual, Methodology tab, limits).")

left, right = st.columns([2, 3], gap="large")
with left:
    st.markdown("**Free energies at 0 V vs RHE (eV)**")
    if st.button("Set the ideal catalyst", icon=":material/star:", key="epg_ideal"):
        for a in s.adsorbates:
            st.session_state[f"epg_g_{pid}_{a}"] = round(ideal[a], 2)
    g = {}
    for a in s.adsorbates:
        label = next(i.label for i in s.intermediates if i.name == a)
        g[a] = st.slider(label, -2.0, 6.0, float(st.session_state.get(f"epg_g_{pid}_{a}", round(ideal[a], 2))), 0.01,
                         key=f"epg_g_{pid}_{a}", help=f"The ideal catalyst holds it at {ideal[a]:.2f} eV.")
    u_l, k = s.limiting_potential(g)
    eta = s.overpotential(g)
    m = st.columns(2)
    m[0].metric("Limiting potential", f"{u_l:.2f} V vs RHE", border=True)
    m[1].metric("Overpotential", f"{eta:.2f} V", border=True)
    st.markdown(f"Limiting step: **{s.step_names()[k]}** ({s.steps(g)[k]:+.2f} eV at 0 V). "
                + ("Every step is level at the equilibrium potential: this is the ideal catalyst." if eta < 0.005 else
                   f"Lowering that step by {eta:.2f} eV, with the others kept level, would reach the ideal."))
    if s.activation:
        a = s.activation
        b = s.barrier(g)
        route = s.limiting_route(g)
        st.markdown(f"{a.label} barrier: **{b:+.2f} eV**, from the {a.from_}* binding these free energies imply. "
                    + (f"It sits at or below zero, so the route “{a.route}” is open"
                       + (" and reaches as far as the other." if route == electro.EITHER else
                          " and sets the limiting potential.") if b <= 0 else
                       f"It stands above zero, so the route “{s.route}” sets the limiting potential."))
    u_at = st.slider("Speed at (V vs RHE)", float(round(s.equilibrium_V - 1.0, 2)), float(round(s.equilibrium_V + 1.0, 2)),
                     float(s.compare_V), 0.01, key=f"epg_u_{pid}")
    up = s.hurdle(g, u_at)
    factor = float(np.exp(-up / electro.K_T))
    st.markdown(f"At {u_at:.2f} V the largest hurdle{' on the faster route' if s.activation else ''} is "
                f"**{up:.2f} eV**: the surface runs at "
                + ("**its full speed**, every step downhill." if up < 0.005 else
                   f"**{electro_ui.fmt_ratio(factor).replace(' slower', '')} below** its full speed.")
                + (f" Faster route there: {s.route_at(g, u_at)}." if s.activation else ""))

with right:
    st.markdown("**Free-energy diagram**")
    potentials = {"0 V": 0.0, f"equilibrium, {s.equilibrium_V:.2f} V": s.equilibrium_V, f"limiting, {u_l:.2f} V": u_l}
    electro_ui.diagram_chart(s, g, potentials)
    if s.volcano_steps:
        st.markdown("**On the volcano**")
        x = float(sum(w * g[a] for a, w in s.descriptor.items()))
        pts = pd.DataFrame([{"x": x, "Material": "yours", "U": u_l, "lo": x - s.sigma_descriptor,
                             "hi": x + s.sigma_descriptor, "Kind": "Candidate"}])
        pts["lo"] = min(pts["lo"].min(), -0.5)
        pts["hi"] = max(pts["hi"].max(), 2.0)
        electro_ui.volcano_chart(s, pts, "higher" if s.direction == "reduction" else "lower")
        st.caption(f"The dot is your surface by its {electro_ui.descriptor_text(s)}; the gray line is where the scaling "
                   "relations would put its limiting potential. "
                   + ("Above" if s.direction == "reduction" else "Below") + " the line, your surface beats the scaling.")
