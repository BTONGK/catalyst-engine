"""Published bench measurements reproduced with the engine's reactor. Text comes from experiments/<name>/report.yaml
(kind: literature); numbers from the results/summary.json its replicate.py writes."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common

HELD_OUT = "#d97706"

reports = common.experiment_reports("literature")
st.title("Literature check")
if not reports:
    st.info("Literature checks appear here once an experiment folder holds a report.yaml with kind: literature.")
    st.stop()
name = st.selectbox("Check", list(reports), format_func=lambda k: reports[k]["title"]) if len(reports) > 1 else next(iter(reports))
rep = reports[name]
data = rep["data"]

st.markdown(f"##### {rep['question']}")
st.caption(f"{rep['model']} · run {rep['run_date']} · text in experiments/{name}/report.yaml")
if data is None:
    st.info(f"Results appear here after experiments/{name}/replicate.py writes {rep['summary']}.")
    st.stop()

cats = data["catalysts"]
hold = data["holdout_from_C"]
t_max = max(p["T_C"] for c in cats.values() for pan in c["full"]["panels"] for p in pan["points"])
FITS = {"full": "Fit to every point", "holdout": f"Fit below {hold:.0f} °C, predict above"}
src = data["source"]
source_link = f"[{' '.join(src['cite'].split())}]({src['url']})"


def run_chart(panel: dict, holdout: bool) -> alt.LayerChart:
    """Model curve with its 90% band, and the measured points (held-out points in orange)."""
    curve = pd.DataFrame(panel["curve"])
    pts = pd.DataFrame(panel["points"])
    pts["Points"] = np.where(pts.used_in_fit, "Used in the fit", "Held out, predicted")
    x_scale = alt.Scale(domain=[curve.T_C.min(), curve.T_C.max()])
    x = alt.X("T_C:Q", title="Temperature (°C)", scale=x_scale)
    lo, hi = min(curve.lo.min(), pts.measured.min()), max(curve.hi.max(), pts.measured.max())
    y_scale = alt.Scale(domain=[min(0.0, lo), hi * 1.04], nice=True)
    layers = []
    if holdout:
        shade = pd.DataFrame({"x": [hold - 25.0], "x2": [curve.T_C.max()]})
        layers.append(alt.Chart(shade).mark_rect(color=HELD_OUT, opacity=0.07).encode(
            x=alt.X("x:Q", scale=x_scale, title="Temperature (°C)"), x2="x2:Q"))
    layers += [
        alt.Chart(curve).mark_area(color=common.ACCENT, opacity=0.16).encode(
            x=x, y=alt.Y("lo:Q", title=panel["unit"], scale=y_scale), y2="hi:Q"),
        alt.Chart(curve).mark_line(color=common.ACCENT, strokeWidth=2.5).encode(x=x, y="value:Q"),
        alt.Chart(pts).mark_point(filled=True, size=80, opacity=1).encode(
            x="T_C:Q", y="measured:Q",
            color=alt.Color("Points:N", scale=alt.Scale(domain=["Used in the fit", "Held out, predicted"][: 1 + holdout],
                                                         range=["#0f172a", HELD_OUT][: 1 + holdout]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("T_C:Q", title="°C"), alt.Tooltip("measured:Q", title="Measured", format=".3g"),
                     alt.Tooltip("model:Q", title="Model", format=".3g"), alt.Tooltip("lo:Q", title="Band low", format=".3g"),
                     alt.Tooltip("hi:Q", title="Band high", format=".3g")],
        ),
    ]
    return alt.layer(*layers).properties(height=300)


def gap_text(err: dict) -> str:
    return f"{err['mean_abs']:.0f} ppm" if "ppm" in err["unit"] else f"{err['mean_abs']:.3f}"


tabs = st.tabs(["Summary", *[c["name"] for c in cats.values()], "Gate screen", "Method and sources"])

with tabs[0]:
    st.success(" ".join(rep["verdict"].split()), icon=":material/verified:")
    head = rep.get("headline", {})
    run, label = head.get("run"), head.get("label", "")
    st.caption(f"Mean gap in {label}, model against measured: fitted to every point, and predicted for "
               f"{hold:.0f}–{t_max:.0f} °C from a fit to the points below {hold:.0f} °C.")
    cols = st.columns(2 * len(cats))
    for i, c in enumerate(cats.values()):
        fit_err = c["full"]["errors_fit"].get(run)
        pred = c["holdout"]["errors_predicted"]
        n_in, n_all = sum(e["in_band"] for e in pred.values()), sum(e["n"] for e in pred.values())
        if fit_err:
            cols[2 * i].metric(f"{c['name']}, fitted", f"{fit_err['mean_abs']:.3f}", border=True,
                               help=f"Mean gap in {label}, model against measured, fitted to every point")
        if run in pred:
            cols[2 * i + 1].metric(f"{c['name']}, predicted", f"{pred[run]['mean_abs']:.3f}",
                                   delta=f"{n_in} of {n_all} points in the 90% band", delta_color="off", delta_arrow="off",
                                   border=True, help=f"Mean gap in {label} at the {hold:.0f}–{t_max:.0f} °C points, "
                                                     f"predicted from a fit to the points below {hold:.0f} °C")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What the check showed")
        st.markdown("\n".join(f"- {f}" for f in rep["findings"]))
    with right:
        st.subheader("What it means for the engine and the bench")
        st.markdown("\n".join(f"{i}. {f}" for i, f in enumerate(rep["implications"], start=1)))
        st.markdown(f"**Next test.** {' '.join(rep['next_test'].split())}")
    with st.expander("Limits of this check"):
        st.markdown("\n".join(f"- {x}" for x in rep["limits"]))

for tab, (cat, c) in zip(tabs[1:], cats.items()):
    with tab:
        st.caption(f"{c['form'][:1].upper()}{c['form'][1:]}. {' '.join(src['conditions'].split())}")
        which = st.segmented_control("Fit", list(FITS), format_func=FITS.get, default="full", required=True, key=f"lit-{cat}")
        fit = c[which]
        if which == "holdout":
            st.caption(f"Fitted on the points below {hold:.0f} °C; the shaded range shows the prediction for the rest. "
                       "The teal band is the model's 90% range from the fit's own uncertainty.")
        else:
            st.caption("Fitted on every point. The teal band is the model's 90% range from the fit's own uncertainty.")
        for col, panel in zip(st.columns(len(fit["panels"]), gap="large"), fit["panels"]):
            with col:
                st.markdown(f"**{panel['title']}**")
                st.altair_chart(run_chart(panel, which == "holdout"), width="stretch")
                err = fit["errors_fit"].get(panel["id"])
                pred = fit["errors_predicted"].get(panel["id"])
                parts = [f"mean gap over the fitted points {gap_text(err)} (largest {err['max_abs']:.3g})"] if err else []
                if pred:
                    parts.append(f"at the held-out points {gap_text(pred)}, {pred['in_band']} of {pred['n']} inside the band")
                st.caption("Model against measured: " + "; ".join(parts) + ".")

        st.markdown(f"**Fitted parameters** (rate constants at the pack's reference temperature, {data['t_ref_C']:.0f} °C "
                    f"= {data['t_ref_F']:,.0f} °F, per m³ of monolith)")
        rows = []
        for pw, pw_name in data["pathways"].items():
            ln_sd = fit["std"][f"{pw}.ln_k_ref"]
            rows.append({
                "Pathway": pw_name,
                "k (1/s)": f"{fit['k_ref_per_s'][pw]:,.0f} (×/÷ {np.exp(ln_sd):.2f})",
                "Activation energy, this fit (kJ/mol)": f"{fit['params'][f'{pw}.Ea_kJ']:.0f} ± {fit['std'][f'{pw}.Ea_kJ']:.0f}",
                "Thesis model (kJ/mol)": f"{c['published_Ea_kJ'][pw]:g}",
                "Pack default (kJ/mol)": f"{data['defaults']['params'][f'{pw}.Ea_kJ']:g}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Ratio of the two rate constants at {data['t_ref_F']:,.0f} °F: {fit['k_ratio_scr_to_nh3_ox']:.1f} "
                   f"(pack default {data['defaults']['k_ratio_scr_to_nh3_ox']:.1f}). Reduced chi-square "
                   f"{fit['chi2_reduced']:.1f} over {fit['n_points']} points. Thesis values come from a model that "
                   "resolves washcoat diffusion, so they sit above the apparent values an ideal reactor gives.")
        with st.expander("Nitrogen balance in the ammonia-only run"):
            st.caption("N2 made equals half the ammonia used when ammonia burns to N2 alone.")
            st.dataframe(pd.DataFrame(c["n2_check"]).rename(columns={
                "T_C": "°C", "n2_ppm": "N2 measured (ppm)", "half_nh3_used_ppm": "Half the NH3 used (ppm)"}),
                hide_index=True, width="stretch")

with tabs[-2]:
    ys = data["yardstick"]
    st.markdown(f"**The published catalysts on the bench yardstick's {ys['step']} step**: "
                f"`{ys['protocol']}` at {ys['ghsv_per_h']:,.0f} 1/h, in the ideal reactor")
    st.caption("Solid lines use every published point, dashed lines only the points below "
               f"{hold:.0f} °C. Gray dashes are the pack's illustrative defaults.")
    band_fit = st.segmented_control("90% band for", list(FITS), format_func=FITS.get, default="full", required=True,
                                    key="lit-gate-band")
    rows = []
    for c in cats.values():
        for which, style in (("full", "solid"), ("holdout", "dashed")):
            for r in c[which]["yardstick"]:
                rows.append({"T_F": r["T_F"], "value": r["value"], "lo": r.get("lo"), "hi": r.get("hi"),
                             "Catalyst": c["name"], "fit": FITS[which], "dash": style})
    for r in data["defaults"]["yardstick"]:
        rows.append({"T_F": r["T_F"], "value": r["value"], "lo": None, "hi": None, "Catalyst": "Pack defaults",
                     "fit": "Illustrative", "dash": "dashed"})
    df = pd.DataFrame(rows)
    names = [c["name"] for c in cats.values()] + ["Pack defaults"]
    palette = [common.ACCENT, "#7c3aed", "#d97706", "#2563eb"][: len(cats)] + [common.REFERENCE_GRAY]
    color = alt.Color("Catalyst:N", scale=alt.Scale(domain=names, range=palette),
                      legend=alt.Legend(orient="bottom", title=None, symbolType="circle", symbolOpacity=1))
    temps = sorted(df.T_F.unique().tolist())
    x = alt.X("T_F:Q", title="Temperature (°F)", scale=alt.Scale(zero=False), axis=alt.Axis(values=temps, format=",d"))
    banded = df[df.fit == FITS[band_fit]]
    band = alt.Chart(banded).mark_area(opacity=0.16).encode(x=x, y="lo:Q", y2="hi:Q", color=color)
    lines = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.3).encode(
        x=x, y=alt.Y("value:Q", title="NOx conversion", scale=alt.Scale(zero=False, nice=True)), color=color,
        strokeDash=alt.StrokeDash("dash:N", scale=alt.Scale(domain=["solid", "dashed"], range=[[1, 0], [6, 4]]), legend=None),
        detail="fit:N",
        tooltip=["Catalyst:N", "fit:N", alt.Tooltip("T_F:Q", title="°F"), alt.Tooltip("value:Q", format=".3f"),
                 alt.Tooltip("lo:Q", format=".3f", title="Band low"), alt.Tooltip("hi:Q", format=".3f", title="Band high")])
    tdf = pd.DataFrame({"y": [ys["target"]], "label": [f"{ys['target_label']}: {ys['target']:g}"]})
    rule = alt.Chart(tdf).mark_rule(color=common.TARGET_RED, strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q")
    text = alt.Chart(tdf).mark_text(align="right", dx=-4, dy=-7, color=common.TARGET_RED, fontSize=11).encode(
        y="y:Q", text="label:N", x=alt.value("width"))
    st.altair_chart(alt.layer(band, lines, rule, text).properties(height=380), width="stretch")
    table = df.pivot_table(index=["Catalyst", "fit"], columns="T_F", values="value").reset_index()
    table.columns = [f"{c} °F" if isinstance(c, (int, float, np.integer)) else c.title() for c in table.columns]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c.endswith("°F")})

with tabs[-1]:
    st.subheader("Source")
    st.markdown(source_link)
    st.markdown("\n".join(f"- {f['label']} (PDF page {f['page']}); feed "
                          + ", ".join(f"{s} {v}" for s, v in f["feed"].items()) for f in data["figures"]))
    st.subheader("Method")
    st.write(" ".join(rep["method"].split()))
    st.markdown(f"**Reactor.** {data['reactor'].capitalize()}, {src['ghsv_per_h']:,.0f} 1/h. Fitted: "
                + ", ".join(f"`{p}`" for p in data["fitted"]) + ". Switched off: "
                + ", ".join(f"`{p}`" for p in data["switched_off"]) + ".")
    st.markdown(f"**Error model.** {' '.join(rep['error_model'].split())}")
    st.subheader("Limits")
    st.markdown("\n".join(f"- {x}" for x in rep["limits"]))
