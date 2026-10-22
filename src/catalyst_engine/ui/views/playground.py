"""Change a catalyst's pathway rates and watch the whole test protocol respond against its targets."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common

MULTIPLIERS = sorted({float(f"{v:.2g}") for v in np.geomspace(0.1, 10, 41)})
EA_SHIFTS = (-40, 40)

st.title("Catalyst playground")
st.caption(
    "Each slider stands in for a formulation change. Every move reruns the full test protocol "
    "through the reactor model, so the charts show what the bench would read."
)

top = st.columns([2, 2, 3])
with top[0]:
    p = common.pack_picker("pg_pack")
net = p.network
proto_id = top[1].selectbox("Test protocol", list(p.protocols), key=f"pg_proto_{p.id}")
proto = p.protocol(proto_id)


def _key(pid: str, what: str) -> str:
    return f"pg:{p.id}:{pid}:{what}"


def load(overrides: dict[str, float]) -> None:
    """Set the sliders from a parameter set (rounded to the nearest slider stop)."""
    for pw in net.pathways:
        d_ln = overrides.get(f"{pw.id}.ln_k_ref", net.defaults[f"{pw.id}.ln_k_ref"]) - net.defaults[f"{pw.id}.ln_k_ref"]
        st.session_state[_key(pw.id, "mult")] = min(MULTIPLIERS, key=lambda m: abs(np.log(m) - d_ln))
        d_ea = overrides.get(f"{pw.id}.Ea_kJ", net.defaults[f"{pw.id}.Ea_kJ"]) - net.defaults[f"{pw.id}.Ea_kJ"]
        st.session_state[_key(pw.id, "ea")] = int(np.clip(5 * round(d_ea / 5), *EA_SHIFTS))


presets = {"Pack defaults (the reference)": {}} | common.catalyst_presets(p.id)
with top[2]:
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    preset = a.selectbox("Start from a named catalyst", list(presets), key=f"pg_preset_{p.id}")
    b.button("Load", on_click=load, args=(presets[preset],), width="stretch")

controls, charts = st.columns([1, 2.6], gap="large")

params: dict[str, float] = {}
with controls:
    st.markdown("##### Your catalyst")
    order = sorted(net.pathways, key=lambda pw: p.spec.pathways[pw.id].role != "desired")
    for pw in order:
        role = p.spec.pathways[pw.id].role
        with st.container(border=True):
            st.markdown(f"**{pw.id}** " + (":green-badge[Desired]" if role == "desired" else ":orange-badge[Side]"))
            st.caption(p.spec.pathways[pw.id].equation)
            mult = st.select_slider("Activity", MULTIPLIERS, value=1.0, key=_key(pw.id, "mult"),
                                    format_func=lambda v: f"{v:g}×",
                                    help="Rate multiplier at the reference temperature.")
            shift = st.slider("Activation energy shift (kJ/mol)", *EA_SHIFTS, value=0, step=5, key=_key(pw.id, "ea"),
                              help="Higher makes the rate climb faster with temperature.")
            if mult != 1.0:
                params[f"{pw.id}.ln_k_ref"] = net.defaults[f"{pw.id}.ln_k_ref"] + float(np.log(mult))
            if shift:
                lo, hi = net.bounds(f"{pw.id}.Ea_kJ")
                params[f"{pw.id}.Ea_kJ"] = float(np.clip(net.defaults[f"{pw.id}.Ea_kJ"] + shift, lo, hi))
    st.button("Reset to reference", on_click=load, args=({},), icon=":material/restart_alt:", width="stretch")

ref = common.run_protocol(p, proto.id, {})
yours = common.run_protocol(p, proto.id, params)

with charts:
    step_ids = [s.id for s in proto.steps]
    step_id = st.segmented_control("Protocol step", step_ids, default=step_ids[0], required=True,
                                   key=f"pg_step_{p.id}_{proto.id}")
    step = proto.step(step_id)
    st.caption(f"{' '.join(step.purpose.split())} · space velocity {step.ghsv_per_h:,.0f} 1/h")
    r, y = ref[ref["step"] == step_id], yours[yours["step"] == step_id]
    temps = y["T_F"].to_numpy()

    shown = [mid for mid in p.spec.metrics if np.isfinite(r[mid]).any() or np.isfinite(y[mid]).any()]
    targeted = [mid for mid in shown if p.spec.metrics[mid].target is not None]

    def meets(frame: pd.DataFrame, mid: str) -> np.ndarray:
        t = p.spec.metrics[mid].target
        v = frame[mid].to_numpy()
        with np.errstate(invalid="ignore"):
            return v >= t.min if t.min is not None else v <= t.max

    if targeted:
        all_y = np.logical_and.reduce([meets(y, m) for m in targeted])
        all_r = np.logical_and.reduce([meets(r, m) for m in targeted])
        n_y, n_r = int(all_y.sum()), int(all_r.sum())
        k1, k2 = st.columns([1, 2], vertical_alignment="center")
        k1.metric("Meets every target", f"{n_y} of {len(temps)} temps",
                  delta=f"{n_y - n_r:+d} vs reference" if n_y != n_r else None, border=True)
        badges = []
        for mid in targeted:
            ny = int(meets(y, mid).sum())
            color = "green" if ny == len(temps) else "orange" if ny else "red"
            badges.append(f":{color}-badge[{mid} {common.target_text(p.spec.metrics[mid].target)}: {ny} of {len(temps)}]")
        k2.markdown("**Per target, your catalyst**  \n" + " ".join(badges))
        if n_y:
            st.success(f"Your catalyst meets every target at {', '.join(f'{t:,.0f}' for t in temps[all_y])} °F.",
                       icon=":material/check_circle:")
        else:
            st.info("Your catalyst misses at least one target at every temperature in this step.", icon=":material/info:")

    grid = st.columns(2)
    for i, mid in enumerate(shown):
        m = p.spec.metrics[mid]
        long = pd.concat([
            pd.DataFrame({"T_F": r["T_F"], "value": r[mid], "series": "Reference"}),
            pd.DataFrame({"T_F": y["T_F"], "value": y[mid], "series": "Your catalyst"}),
        ])
        with grid[i % 2]:
            st.markdown(f"**{common.metric_title(p, mid)}**")
            st.altair_chart(common.metric_chart(long, m.label_unit(), m.target), width="stretch")

    gases = pd.concat([
        pd.DataFrame({"T_F": y["T_F"], "value": y[f"out_{s}_ppm"], "species": s}) for s in proto.analyzer.measured
    ])
    gas_chart = (
        alt.Chart(gases)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("T_F:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
            y=alt.Y("value:Q", title="ppm wet"),
            color=alt.Color("species:N", legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["species:N", alt.Tooltip("T_F:Q", title="°F"), alt.Tooltip("value:Q", title="ppm", format=".3g")],
        )
        .properties(height=220)
    )
    with grid[len(shown) % 2]:
        st.markdown(f"**What the {proto.analyzer.name} reads with your catalyst**")
        st.altair_chart(gas_chart, width="stretch")

    with st.expander("Parameters behind your catalyst"):
        rows = []
        for name, default in net.defaults.items():
            if name.endswith(("ln_k_ref", "Ea_kJ")):
                rows.append({"Parameter": name, "Reference": round(default, 3),
                             "Your catalyst": round(params.get(name, default), 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("ln k is the natural log of the rate constant at the reference temperature; Ea is in kJ/mol.")
