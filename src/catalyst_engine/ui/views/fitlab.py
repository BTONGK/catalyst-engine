"""Fit rate constants to bench readings and project catalyst life from aging data."""

import csv
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.fit.decay import fit_decay
from catalyst_engine.fit.kinetics import fit_kinetics, observations_from_table, simulated_readings
from catalyst_engine.ui import common
from catalyst_engine.units import k_to_f

QUANTITY = {
    "ln_k_ref": "ln k at reference temperature",
    "Ea_kJ": "Activation energy (kJ/mol)",
    "ln_K_ref": "Adsorption strength, ln K",
    "dH_kJ": "Adsorption heat (kJ/mol)",
}

SHORT = {"ln_k_ref": "ln k", "Ea_kJ": "Ea", "ln_K_ref": "ln K", "dH_kJ": "ΔH"}


def short(name: str) -> str:
    parts = name.split(".")
    return " ".join([parts[0]] + ([parts[2]] if len(parts) == 4 else []) + [SHORT[parts[-1]]])


st.title("Fit lab")
kin_tab, aging_tab = st.tabs(["Kinetics from bench readings", "Aging and catalyst life"])

with kin_tab:
    st.caption(
        "The fitter turns analyzer readings into rate constants with error bars. It also reports which "
        "parameters the test plan confounds, the way a DOE reports aliased factors."
    )
    c = st.columns([2, 2, 3])
    with c[0]:
        p = common.pack_picker("fit_pack")
    proto = p.protocol(c[1].selectbox("Test protocol", list(p.protocols), key=f"fit_proto_{p.id}"))
    with c[2]:
        source = st.segmented_control("Readings", ["Demo catalyst", "Upload a CSV"], default="Demo catalyst",
                                      required=True, key="fit_source")

    truth = None
    if source == "Demo catalyst":
        seed = st.session_state.setdefault("fit_seed", 1)
        truth = {k: v + (0.4 if k.endswith("ln_k_ref") else -8.0)
                 for k, v in p.network.defaults.items() if k in p.network.fitted}
        rows = simulated_readings(p, proto, truth, seed=seed)
        a, b = st.columns([4, 1], vertical_alignment="center")
        a.info(
            "The demo hides a catalyst about 50% more active than the pack defaults, runs the protocol through the "
            f"reactor, and adds {proto.analyzer.name} reading noise. The fitter then has to find the hidden values.",
            icon=":material/science:",
        )
        if b.button("New noise draw", icon=":material/casino:", width="stretch"):
            st.session_state["fit_seed"] = seed + 1
            st.rerun()
    else:
        template = pd.DataFrame(
            [{"step": s.id, "T_F": round(k_to_f(t)), **{f"out_{sp}_ppm": "" for sp in proto.analyzer.measured}}
             for s in proto.steps for t in s.kelvin()]
        )
        a, b = st.columns([3, 1], vertical_alignment="bottom")
        upload = a.file_uploader("Bench readings (CSV)", type="csv")
        b.download_button("Template for this protocol", template.to_csv(index=False), f"{proto.id}-template.csv",
                          icon=":material/download:", width="stretch")
        st.caption("Columns: step, T_F (or T_C or T_K), out_<species>_ppm for each species read. "
                   "Optional in_<species>_ppm columns record the measured inlet.")
        rows = list(csv.DictReader(io.StringIO(upload.getvalue().decode()))) if upload else []

    step_ids = [s.id for s in proto.steps]
    c = st.columns([3, 1, 1], vertical_alignment="bottom")
    steps = c[0].multiselect("Steps to include", step_ids, default=step_ids, key=f"fit_steps_{proto.id}",
                             help="Remove a step to see which parameters it was pinning down.")
    starts = c[1].select_slider("Fit restarts", [1, 2, 3, 4, 6], value=2, help="More restarts guard against local minima; each adds a few seconds.")
    go = c[2].button("Fit", type="primary", icon=":material/play_arrow:", width="stretch", disabled=not (rows and steps))

    signature = (p.id, proto.id, source, st.session_state.get("fit_seed"), tuple(steps), starts,
                 None if source == "Demo catalyst" else hash(str(rows)))
    if go:
        used = proto.model_copy(update={"steps": [s for s in proto.steps if s.id in steps]})
        try:
            with st.spinner("Fitting... a few seconds per restart"):
                obs = observations_from_table(p, used, [r for r in rows if str(r["step"]) in steps])
                st.session_state["fit_result"] = (signature, fit_kinetics(p, used, obs, n_starts=starts), truth)
        except (ValueError, KeyError) as exc:
            st.error(f"These readings give an error: {exc}")

    stored = st.session_state.get("fit_result")
    if stored and stored[1].pack == p.id:
        sig, res, truth_used = stored
        if sig != signature:
            st.caption(":orange[Inputs changed since this fit. Press Fit to update it.]")
        aliased = set(res.poorly_determined)
        k = st.columns(4)
        k[0].metric("Fit quality (reduced χ²)", f"{res.chi2_reduced:.2f}", border=True,
                    help="Near 1 means the model matches the readings to within the analyzer's noise.")
        k[1].metric("Readings used", res.n_points, border=True)
        k[2].metric("Parameters fitted", len(res.fitted), border=True)
        k[3].metric("Measured well", f"{len(res.fitted) - len(aliased)} of {len(res.fitted)}", border=True)
        if aliased:
            st.warning(
                f"This test plan confounds {', '.join(sorted(aliased))}: different combinations of them explain the "
                "readings equally well. Add a step that moves them independently, the way a DOE breaks an alias.",
                icon=":material/warning:",
            )
        else:
            st.success("Every parameter is pinned down by this test plan.", icon=":material/check_circle:")

        table = []
        for name in res.fitted:
            parts = name.split(".")
            quantity = QUANTITY[parts[-1]] + (f", {parts[2]}" if len(parts) == 4 else "")
            row = {"Pathway": parts[0], "Quantity": quantity, "Fitted": res.params[name], "± 1σ": res.std[name],
                   "Status": "Aliased in this test plan" if name in aliased else "Measured well"}
            if truth_used and name in truth_used:
                row["Hidden truth"] = truth_used[name]
                row["Miss (σ)"] = abs(res.params[name] - truth_used[name]) / max(res.std[name], 1e-12)
            table.append(row)
        st.dataframe(
            pd.DataFrame(table), hide_index=True,
            column_config={
                "Fitted": st.column_config.NumberColumn(format="%.3f"),
                "± 1σ": st.column_config.NumberColumn(format="%.3f"),
                "Hidden truth": st.column_config.NumberColumn(format="%.3f"),
                "Miss (σ)": st.column_config.NumberColumn(format="%.1f", help="Under 2 is typical for an honest error bar."),
            },
        )

        left, right = st.columns(2, gap="large")
        resid = pd.DataFrame(res.residuals)
        resid["T_F"] = resid["T_K"].map(k_to_f).round(0)
        top = float(max(resid["measured_ppm"].max(), resid["model_ppm"].max()) * 1.05)
        parity = (
            alt.Chart(pd.DataFrame({"x": [0, top]})).mark_line(color=common.REFERENCE_GRAY, strokeDash=[4, 4])
            .encode(x="x:Q", y="x:Q")
            + alt.Chart(resid).mark_circle(size=60, opacity=0.8)
            .encode(
                x=alt.X("measured_ppm:Q", title="Measured (ppm wet)", scale=alt.Scale(type="symlog", constant=1)),
                y=alt.Y("model_ppm:Q", title="Model (ppm wet)", scale=alt.Scale(type="symlog", constant=1)),
                color=alt.Color("species:N", legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["step:N", "T_F:Q", "species:N", alt.Tooltip("measured_ppm:Q", format=".2f"),
                         alt.Tooltip("model_ppm:Q", format=".2f")],
            )
        ).properties(height=340)
        left.markdown("**Model vs readings** (points on the dashed line match exactly)")
        left.altair_chart(parity, width="stretch")

        n = len(res.fitted)
        labels = [short(name) for name in res.fitted]
        corr = pd.DataFrame([{"a": labels[i], "b": labels[j], "r": float(np.nan_to_num(res.correlation[i, j]))}
                             for i in range(n) for j in range(n)])
        heat = (
            alt.Chart(corr).mark_rect()
            .encode(
                x=alt.X("a:N", title=None, sort=labels, axis=alt.Axis(labelAngle=-50, labelOverlap=False, labelLimit=160)),
                y=alt.Y("b:N", title=None, sort=labels, axis=alt.Axis(labelLimit=160)),
                color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                                legend=alt.Legend(title="r")),
                tooltip=["a:N", "b:N", alt.Tooltip("r:Q", format=".2f")],
            )
        ).properties(height=340)
        right.markdown("**Parameter correlation** (dark squares off the diagonal mark confounding)")
        right.altair_chart(heat, width="stretch")
    elif rows:
        st.info("Press Fit to estimate this catalyst's rate constants.", icon=":material/play_arrow:")

with aging_tab:
    st.caption(
        "Activity is a pathway's rate constant after aging divided by its fresh value. The fit picks the decay "
        "form that best matches the points and projects life with an uncertainty band."
    )
    if "aging_data" not in st.session_state:
        t = np.array([0, 24, 72, 168, 336, 504, 750, 1000.0])
        a = 0.55 + 0.45 / (1 + 2e-3 * 0.45 * t) + np.random.default_rng(3).normal(0, 0.008, len(t))
        st.session_state["aging_data"] = pd.DataFrame({"hours": t, "activity": np.round(np.clip(a, 0, 1.05), 3)})

    left, right = st.columns([1, 2.4], gap="large")
    with left:
        upload = st.file_uploader("Aging data (CSV: hours, activity, optional sigma)", type="csv", key="aging_upload")
        if upload is not None:
            st.session_state["aging_data"] = pd.read_csv(upload)
        st.caption("Demo points from a simulated hydrothermal aging run. Edit any cell to try your own.")
        data = st.data_editor(st.session_state["aging_data"], num_rows="dynamic", hide_index=True, key="aging_editor",
                              column_config={"hours": st.column_config.NumberColumn("Hours", min_value=0.0),
                                             "activity": st.column_config.NumberColumn("Activity", min_value=0.0, format="%.3f")})
        horizon = st.number_input("Project to (hours)", value=24000.0, min_value=100.0, step=1000.0, format="%.0f",
                                  help="24,000 hours is about three years of continuous operation.")
        eol = st.slider("End of life at activity", 0.5, 0.95, 0.70, step=0.01)
        log_x = st.toggle("Log time axis", value=True)

    with right:
        data = data.dropna(subset=["hours", "activity"])
        if len(data) < 3:
            st.info("Enter at least three aging points to fit.", icon=":material/info:")
        else:
            sigma = data["sigma"].to_numpy() if "sigma" in data and data["sigma"].notna().all() else None
            fit = fit_decay(data["hours"].to_numpy(), data["activity"].to_numpy(), sigma)
            lo, hi = fit.interval(horizon)
            life = fit.hours_to(eol)
            reach = fit.extrapolation_ratio(horizon)
            k = st.columns(2) + st.columns(2)
            k[0].metric("Best-fit form", "first-order" if fit.model == "exp" else "second-order",
                        help="first-order: steady loss of sites. second-order: sintering-like, slowing as it goes.", border=True)
            k[1].metric("Levels off at", f"{fit.a_eq:.2f}", border=True)
            k[2].metric(f"Activity at {horizon:,.0f} h", f"{fit.activity(horizon):.2f}",
                        help=f"90% interval {lo:.2f} to {hi:.2f}", border=True)
            k[3].metric(f"Hours to {eol:.2f}", f"levels off above {eol:.2f}" if np.isinf(life) else f"{life:,.0f}",
                        border=True, help="Read this with the band: its lower edge shows how soon end of life could come.")
            if reach > 3:
                st.warning(
                    f"This projection reaches {reach:,.0f}× past the longest aging point ({fit.t_max_h:,.0f} h). "
                    "Treat it as a planning number and confirm with longer aging.", icon=":material/warning:")

            t_grid = np.unique(np.concatenate([[0.0], np.geomspace(1.0, horizon, 90)]))
            band = pd.DataFrame([(t, float(fit.activity(t)), *fit.interval(t)) for t in t_grid],
                                columns=["hours", "fit", "lo", "hi"])
            ticks = [t for t in (0, 10, 100, 1000, 10000, 100000) if t <= horizon] + [horizon]
            x_scale = alt.Scale(type="symlog", constant=10) if log_x else alt.Scale()
            x = alt.X("hours:Q", title="Aging time (hours)", scale=x_scale,
                      axis=alt.Axis(values=ticks, format=",.0f") if log_x else alt.Axis(format=",.0f"))
            chart = (
                alt.Chart(band).mark_area(opacity=0.18, color=common.ACCENT).encode(x=x, y="lo:Q", y2="hi:Q")
                + alt.Chart(band).mark_line(color=common.ACCENT, strokeWidth=2.5)
                .encode(x=x, y=alt.Y("fit:Q", title="Activity (fresh = 1)", scale=alt.Scale(zero=False)))
                + alt.Chart(data).mark_circle(size=70, color="#0f172a").encode(x=x, y="activity:Q",
                                                                               tooltip=["hours:Q", "activity:Q"])
                + alt.Chart(pd.DataFrame({"hours": [fit.t_max_h]})).mark_rule(color=common.REFERENCE_GRAY, strokeDash=[4, 4]).encode(x=x)
                + alt.Chart(pd.DataFrame({"y": [eol]})).mark_rule(color=common.TARGET_RED, strokeDash=[3, 3]).encode(y="y:Q")
            ).properties(height=380)
            st.markdown("**Fit with 90% band** · gray line: longest test · red line: end of life")
            st.altair_chart(chart, width="stretch")
