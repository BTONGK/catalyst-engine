"""Published bench measurements reproduced with the engine's reactor. Text comes from experiments/<name>/report.yaml
(kind: literature); numbers from the results/summary.json its replicate.py writes."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common

HELD_OUT = "#d97706"
MAIN = "Pack's ammonia coverage"  # the main fit's model choice
COVER = "Ammonia coverage fitted"  # the choice whose profile maps the coverages the data allow
CHOICE_COLORS = [common.ACCENT, "#7c3aed", "#d97706", "#2563eb", "#be185d", "#65a30d"]
FED_COLORS = ["#0f766e", "#2563eb", "#7c3aed", "#be185d", "#d97706"]
RATIO_COLORS = ["#94a3b8", "#2563eb", "#0f766e"]
COV_FLOOR = 1e-4  # the coverage chart's left edge; a range open at the low end runs to it

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
sources = data["sources"]
cov = data.get("coverage")
t_ref = f"{data['t_ref_F']:,.0f} °F"
CHOICES = [MAIN, "Held-out fit", *sorted({a["label"] for c in cats.values() for a in c.get("alternatives", [])})]
CHOICE_COLOR = dict(zip(CHOICES, CHOICE_COLORS * 2))


def choice_scale(labels: list[str]) -> alt.Scale:
    """The same color for a model choice on every chart."""
    labels = [x for x in CHOICES if x in labels]
    return alt.Scale(domain=labels, range=[CHOICE_COLOR[x] for x in labels])


def short_cite(sid: str) -> str:
    """metkar2012 -> Metkar 2012; us7695703 -> US 7,695,703."""
    if sid.startswith("us") and sid[2:].isdigit():
        return f"US {int(sid[2:]):,}"
    return f"{sid[:-4].title()} {sid[-4:]}"


def cite_link(sid: str) -> str:
    return f"[{' '.join(sources[sid]['cite'].split())}]({sources[sid]['url']})"


def fit_labels(c: dict) -> dict[str, str]:
    return {"full": "Fit to every point", "holdout": c["holdout_test"]["label"]}


def held_where(hold: dict) -> str:
    """'from 500 °C' or the test's own short description."""
    return f"from {hold['from_C']:.0f} °C" if "from_C" in hold else hold.get("short", "held out")


def is_scr(panel: dict) -> bool:
    return panel.get("kind") == "scr" or panel["id"].startswith("scr.")


def scr_gap(fit: dict) -> dict | None:
    """The fitted gap over every SCR run that reads the first SCR run's quantity, weighted by readings."""
    pans = [p for p in fit["panels"] if is_scr(p)]
    if not pans:
        return None
    errs = [fit["errors_fit"][p["id"]] for p in pans if p["unit"] == pans[0]["unit"] and p["id"] in fit["errors_fit"]]
    n = sum(e["n"] for e in errs)
    return {"unit": pans[0]["unit"], "n": n, "mean_abs": sum(e["mean_abs"] * e["n"] for e in errs) / n} if n else None


def gap_text(err: dict) -> str:
    if "ppm" in err["unit"]:
        return f"{err['mean_abs']:.0f} ppm" if err["mean_abs"] >= 10 else f"{err['mean_abs']:.2g} ppm"
    return f"{err['mean_abs']:.3f}"


def wide_range(panel: dict) -> bool:
    """A concentration panel whose readings span more than twentyfold reads better on a log scale."""
    vals = [p["measured"] for p in panel["points"] if p["measured"] > 0]
    return "ppm" in panel["unit"] and len(vals) > 1 and max(vals) > 20 * min(vals)


def gap_with_unit(err: dict) -> str:
    """'12 ppm NH3 out' or '0.020 NO conversion'."""
    return f"{gap_text(err)} {err['unit'].replace(' (ppm)', '')}"


def at_ref(yardstick: list[dict]) -> dict:
    return next(r for r in yardstick if r["T_F"] == round(data["t_ref_F"]))


def ref_text(yardstick: list[dict]) -> str:
    r = at_ref(yardstick)
    return f"{r['value']:.2f}" + (f" ({r['lo']:.2f}–{r['hi']:.2f})" if "lo" in r else "")


def pct(x: float) -> str:
    """0.616 -> '62%', 0.032 -> '3.2%', 0.0013 -> '0.13%'."""
    v = 100.0 * x
    return f"{v:.0f}%" if v >= 10 else f"{v:.2g}%"


def ppm_list(vals: list[float]) -> str:
    """[60, 70, 82] -> '60–82'."""
    f = lambda v: f"{v:,.0f}" if v >= 10 else f"{v:g}"  # noqa: E731
    return f(vals[0]) if len(vals) == 1 else f"{f(vals[0])}–{f(vals[-1])}"


def cov_bounds(cp: dict) -> tuple[float, float]:
    """The 90% range of the coverage, an open end carried to 0 or 1."""
    return 0.0 if cp["lo_open"] else cp["lo"], 1.0 if cp["hi_open"] else cp["hi"]


def cov_text(cp: dict) -> str:
    lo, hi = cov_bounds(cp)
    return f"{pct(cp['best'])} ({pct(lo)}–{pct(hi)})"


def cover_choice(c: dict) -> dict | None:
    return next((a for a in c.get("alternatives", []) if a["label"] == COVER), None)


def choices(c: dict) -> list[dict]:
    """The main fit and each alternative, as one list of model choices."""
    main = {"label": MAIN, "chi2_reduced": c["full"]["chi2_reduced"], "n_params": c["n_params"],
            "poorly_determined": c["full"]["poorly_determined"], "params": c["full"]["params"], "std": c["full"]["std"],
            "errors_fit": c["full"]["errors_fit"], "yardstick": c["full"]["yardstick"], "note": "", "film": False,
            "coverage_ref": c["full"].get("coverage_ref"), "operating_map": c["full"].get("operating_map"),
            "curves": {p["id"]: p["curve"]["value"] for p in c["full"]["panels"]}}
    return [main, *c.get("alternatives", [])]


def run_chart(panel: dict, holdout: bool, hold_from: float | None, overlays: list[tuple[str, list]],
              log: bool = False) -> alt.LayerChart:
    """Model curve with its 90% band, the measured points (held-out points in orange), and any other model
    choices' curves as dashed lines."""
    curve = pd.DataFrame(panel["curve"])
    pts = pd.DataFrame(panel["points"])
    pts["Points"] = np.where(pts.used_in_fit, "Used in the fit", "Held out, predicted")
    held = holdout and not pts.used_in_fit.all()
    x_scale = alt.Scale(domain=[curve.T_C.min(), curve.T_C.max()])
    x = alt.X("T_C:Q", title="Temperature (°C)", scale=x_scale)
    lo, hi = min(curve.lo.min(), pts.measured.min()), max(curve.hi.max(), pts.measured.max())
    if log:  # from a tenth of the smallest positive reading; the curve and band rest on that floor below it
        floor = 0.1 * pts.measured[pts.measured > 0].min()
        y_scale = alt.Scale(type="log", domain=[floor, hi * 1.5])
        curve[["value", "lo", "hi"]] = curve[["value", "lo", "hi"]].clip(lower=floor)
        pts = pts[pts.measured > 0]
    elif "ppm" in panel["unit"]:
        y_scale = alt.Scale(domain=[min(0.0, lo), hi * 1.04], nice=True)
    else:  # a conversion or share: from a tenth below the lowest value, so close readings stay apart
        y_scale = alt.Scale(domain=[max(min(0.0, lo), np.floor(lo * 10) / 10 - 0.1), max(hi * 1.04, 1.0) if hi > 0.9 else hi * 1.04],
                            nice=True)
    layers = []
    if held and hold_from is not None:
        shade = pd.DataFrame({"x": [hold_from], "x2": [curve.T_C.max()]})
        layers.append(alt.Chart(shade).mark_rect(color=HELD_OUT, opacity=0.07).encode(
            x=alt.X("x:Q", scale=x_scale, title="Temperature (°C)"), x2="x2:Q"))
    layers += [
        alt.Chart(curve).mark_area(color=common.ACCENT, opacity=0.16).encode(
            x=x, y=alt.Y("lo:Q", title=panel["unit"], scale=y_scale), y2="hi:Q"),
        alt.Chart(curve).mark_line(color=common.ACCENT, strokeWidth=2.5).encode(x=x, y="value:Q"),
    ]
    if overlays:
        other = pd.concat([pd.DataFrame({"T_C": curve.T_C, "value": vals, "Model choice": label})
                           for label, vals in overlays])
        other["value"] = other["value"].clip(lower=y_scale.domain[0], upper=y_scale.domain[1])
        layers.append(alt.Chart(other).mark_line(strokeWidth=2, strokeDash=[6, 4]).encode(
            x=x, y="value:Q",
            color=alt.Color("Model choice:N", scale=choice_scale([label for label, _ in overlays]),
                            legend=alt.Legend(orient="bottom", title=None, labelLimit=320))))
    kinds = ["Used in the fit", "Held out, predicted"][: 1 + held]
    layers.append(alt.Chart(pts).mark_point(filled=True, size=70, opacity=1).encode(
        x="T_C:Q", y="measured:Q",
        color=alt.Color("Points:N", scale=alt.Scale(domain=kinds, range=["#0f172a", HELD_OUT][: len(kinds)]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("T_C:Q", title="°C"), alt.Tooltip("measured:Q", title="Measured", format=".3g"),
                 alt.Tooltip("model:Q", title="Model", format=".3g"), alt.Tooltip("lo:Q", title="Band low", format=".3g"),
                 alt.Tooltip("hi:Q", title="Band high", format=".3g")],
    ))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=280)


def coverage_chart() -> alt.LayerChart:
    """Each catalyst's 90% range of ammonia coverage at the yardstick's feed and the reference temperature, colored
    by the NO its source fed, against the pack's value."""
    rows = []
    for c in cats.values():
        cp = c.get("coverage_profile")
        if cp:
            lo, hi = cov_bounds(cp)
            no = c["fed_ppm"]["NO"]
            rows.append({"Catalyst": c["name"], "best": max(cp["best"], COV_FLOOR), "lo": max(lo, COV_FLOOR), "hi": hi,
                         "Fed": f"{ppm_list(no)} ppm NO", "order": max(no), "Best": pct(cp["best"]),
                         "90% range": f"{pct(lo)}–{pct(hi)}", "Source": short_cite(c["source"])})
    df = pd.DataFrame(rows)
    fed = list(df.sort_values("order").Fed.drop_duplicates())
    x = alt.X("lo:Q", title=f"NH3 coverage on the SCR site at {cov['ppm']:g} ppm NH3 and {t_ref}",
              scale=alt.Scale(type="log", domain=[COV_FLOOR, 1]),
              axis=alt.Axis(values=[1e-4, 1e-3, 1e-2, 0.1, 1], labelExpr="format(datum.value * 100, '~g') + '%'"))
    y = alt.Y("Catalyst:N", sort=list(df.Catalyst), title=None, axis=alt.Axis(labelLimit=200))
    color = alt.Color("Fed:N", scale=alt.Scale(domain=fed, range=FED_COLORS[: len(fed)]),
                      legend=alt.Legend(orient="bottom", title="Source feed"))
    tip = ["Catalyst:N", "Source:N", "Fed:N", "Best:N", "90% range:N"]
    ref = pd.DataFrame({"x": [cov["pack"]], "label": [f"Pack: {pct(cov['pack'])}"]})
    return alt.layer(
        alt.Chart(df).mark_rule(strokeWidth=7, opacity=0.45).encode(x=x, x2="hi:Q", y=y, color=color, tooltip=tip),
        alt.Chart(df).mark_point(filled=True, size=110, opacity=1).encode(x="best:Q", y=y, color=color, tooltip=tip),
        alt.Chart(ref).mark_rule(color=common.REFERENCE_GRAY, strokeDash=[4, 3], strokeWidth=2).encode(x="x:Q"),
        alt.Chart(ref).mark_text(align="right", dx=-5, dy=8, color="#5f6368", fontSize=11).encode(
            x="x:Q", y=alt.value(0), text="label:N"),
    ).properties(height=40 * len(df) + 120)  # the height holds the axis and legend too


def profile_charts(cp: dict) -> tuple[alt.LayerChart, alt.LayerChart]:
    """Chi-square rise and Gate 1 placement with the coverage held at each value."""
    prof = pd.DataFrame({"theta": cp["theta"], "dchi2": cp["dchi2"], "gate1": cp["gate1"]})
    prof["Coverage"] = prof.theta.map(pct)
    x = alt.X("theta:Q", title=f"NH3 coverage held at {cov['ppm']:g} ppm, {t_ref}",
              scale=alt.Scale(type="log", domain=[min(COV_FLOOR, prof.theta.min()), 1]),
              axis=alt.Axis(values=[1e-4, 1e-3, 1e-2, 0.1, 1], labelExpr="format(datum.value * 100, '~g') + '%'"))
    top = min(max(prof.dchi2.max(), 5.0), 30.0)
    tip = [alt.Tooltip("Coverage:N"), alt.Tooltip("dchi2:Q", title="Chi-square rise", format=".2f"),
           alt.Tooltip("gate1:Q", title="Gate 1", format=".3f")]
    rule = pd.DataFrame({"y": [data["coverage"]["dchi2_90"]], "label": ["90% limit"]})
    pack = pd.DataFrame({"x": [cov["pack"]]})
    left = alt.layer(
        alt.Chart(prof).mark_line(point=True, color=common.ACCENT, clip=True).encode(
            x=x, y=alt.Y("dchi2:Q", title="Chi-square rise from the best fit", scale=alt.Scale(domain=[0, top])),
            tooltip=tip),
        alt.Chart(rule).mark_rule(color=common.TARGET_RED, strokeDash=[3, 3]).encode(y="y:Q"),
        alt.Chart(rule).mark_text(align="left", dx=4, dy=-7, color=common.TARGET_RED, fontSize=11).encode(
            y="y:Q", x=alt.value(0), text="label:N"),
        alt.Chart(pack).mark_rule(color=common.REFERENCE_GRAY, strokeDash=[4, 3]).encode(x="x:Q"),
    ).properties(height=240)
    target = data["yardstick"]["target"]
    right = alt.layer(
        alt.Chart(prof).mark_line(point=True, color="#2563eb").encode(
            x=x, y=alt.Y("gate1:Q", title=f"Gate 1 placement at {t_ref}",
                         scale=alt.Scale(domain=[min(0.0, prof.gate1.min()), 1])), tooltip=tip),
        alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(color=common.TARGET_RED, strokeDash=[3, 3]).encode(y="y:Q"),
        alt.Chart(pack).mark_rule(color=common.REFERENCE_GRAY, strokeDash=[4, 3]).encode(x="x:Q"),
    ).properties(height=240)
    return left, right


tabs = st.tabs(["Summary", *[c["name"] for c in cats.values()], "Gate screen", "Method and sources"])

with tabs[0]:
    st.success(" ".join(rep["verdict"].split()), icon=":material/verified:")
    rows = []
    for c in cats.values():
        err = scr_gap(c["full"])
        pred = c["holdout"]["errors_predicted"]
        n_in, n_all = sum(e["in_band"] for e in pred.values()), sum(e["n"] for e in pred.values())
        cover, cp = cover_choice(c), c.get("coverage_profile")
        fitted_gate = ""
        if cover:
            fitted_gate = f"{at_ref(cover['yardstick'])['value']:.2f}"
            if cp:
                fitted_gate += f" ({cp['gate1_range'][0]:.2f}–{cp['gate1_range'][1]:.2f})"
        rows.append({
            "Catalyst": c["name"],
            "Source": short_cite(c["source"]),
            "Fed NO / NH3 (ppm)": f"{ppm_list(c['fed_ppm']['NO'])} / {ppm_list(c['fed_ppm']['NH3'])}" if "fed_ppm" in c else "",
            "Range (°C)": f"{c['range_C'][0]:.0f}–{c['range_C'][1]:.0f}",
            "Readings": c["full"]["n_points"],
            "SCR run, mean gap": gap_with_unit(err) if err else "",
            "Held-out points in the 90% band": f"{n_in} of {n_all} ({held_where(c['holdout_test'])})",
            f"NH3 coverage at {cov['ppm']:g} ppm, {t_ref} (90% range)" if cov else "NH3 coverage": cov_text(cp) if cp else "",
            f"Gate 1 at {t_ref}, pack coverage": ref_text(c["full"]["yardstick"]),
            f"Gate 1 at {t_ref}, coverage fitted": fitted_gate,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Mean gap: model against measured NO, fitted to every point. Held-out points: each catalyst's test fits "
               "part of its readings (its tab names which), then predicts the rest. NH3 coverage: the best fit and the "
               "range of coverages whose fit stays within the 90% limit, with the coverage held at each value in turn. "
               f"Gate 1 placement: NOx conversion on the bench yardstick at {t_ref} (target "
               f"{data['yardstick']['target']:g}): under the pack's coverage with the fit's 90% band, and with the "
               "coverage fitted, across the coverages the data allow.")
    if cov and any(c.get("coverage_profile") for c in cats.values()):
        st.markdown(f"**The ammonia coverage each source allows** at the yardstick's {cov['ppm']:g} ppm NH3 and {t_ref}: "
                    "point at the best fit, bar over the 90% range, dashes at the pack's illustrative value")
        st.altair_chart(coverage_chart(), width="stretch")
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
        src = sources[c["source"]]
        st.caption(f"{c['form'][:1].upper()}{c['form'][1:]}. {' '.join(src['conditions'].split())} "
                   f"Source: {short_cite(c['source'])}.")
        fits = fit_labels(c)
        which = st.segmented_control("Fit", list(fits), format_func=fits.get, default="full", required=True, key=f"lit-{cat}")
        fit = c[which]
        hold = c["holdout_test"]
        if which == "holdout":
            st.caption(f"{hold['label']}: the orange points are held out and predicted from the fit to the rest. "
                       "The teal band is the model's 90% range from the fit's own uncertainty.")
        else:
            st.caption("Fitted on every point. The teal band is the model's 90% range from the fit's own uncertainty.")
        alts = c.get("alternatives", [])
        log = st.toggle("Log scale for concentrations", value=any(wide_range(p) for p in fit["panels"]),
                        key=f"lit-log-{cat}") if any("ppm" in p["unit"] for p in fit["panels"]) else False
        shown = st.multiselect("Overlay other model choices", [a["label"] for a in alts], key=f"lit-overlay-{cat}",
                               placeholder="Model choices, fitted to every point") if alts else []
        panels = fit["panels"]
        n_steps = len({p["id"].split(".")[0] for p in panels})
        per_row = len(panels) if len(panels) <= 4 else min(n_steps, 4)
        for start in range(0, len(panels), per_row):
            for col, panel in zip(st.columns(per_row, gap="large"), panels[start: start + per_row]):
                with col:
                    st.markdown(f"**{panel['title']}**")
                    st.caption(panel["figure"])
                    overlays = [(a["label"], a["curves"][panel["id"]]) for a in alts if a["label"] in shown]
                    st.altair_chart(run_chart(panel, which == "holdout", hold.get("from_C"), overlays,
                                              log and wide_range(panel)), width="stretch")
                    err = fit["errors_fit"].get(panel["id"])
                    pred = fit["errors_predicted"].get(panel["id"])
                    parts = [f"mean gap over the fitted points {gap_text(err)} (largest {err['max_abs']:.3g})"] if err else []
                    if pred:
                        parts.append(f"at the held-out points {gap_text(pred)}, {pred['in_band']} of {pred['n']} inside the band")
                    st.caption("Model against measured: " + "; ".join(parts) + ".")

        st.markdown(f"**Fitted parameters** (rate constants at the pack's reference temperature, {data['t_ref_C']:.0f} °C "
                    f"= {t_ref}, per m³ of catalyst as tested)")
        rows = []
        for pw in c["pathways"]:
            ln_sd = fit["std"][f"{pw}.ln_k_ref"]
            ea = f"{pw}.Ea_kJ"
            row = {
                "Pathway": data["pathways"][pw],
                "k (1/s)": f"{fit['k_ref_per_s'][pw]:,.3g} (×/÷ {np.exp(ln_sd):.2g})",
                "Activation energy, this fit (kJ/mol)": (f"{fit['params'][ea]:.0f} ± {fit['std'][ea]:.0f}" if ea in fit["params"]
                                                         else f"{data['defaults']['params'][ea]:g}, held at the pack's value"),
            }
            if c["published_Ea_kJ"]:
                published = c["published_Ea_kJ"].get(pw, "")
                row[f"Published, {c['published_model']} (kJ/mol)"] = f"{published:g}" if isinstance(published, (int, float)) else published
            row |= {"Pack default (kJ/mol)": f"{data['defaults']['params'][ea]:g}",
                    "Spread": "wide" if any(n.startswith(f"{pw}.") for n in fit["poorly_determined"]) else "tight"}
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        off = [data["pathways"][k.split(".")[0]] for k in c["switched_off"]]
        st.caption(f"Reduced chi-square {fit['chi2_reduced']:.1f} over {fit['n_points']} readings. Spread marks a "
                   "parameter whose uncertainty runs past the fitter's limits (×/÷ 2 on k, ±30 kJ/mol on the "
                   "activation energy). Switched off for this catalyst: " + ", ".join(off) + ".")

        st.markdown(f"**Model choices**, each fitted to every point, with the Gate 1 placement at {t_ref} they give")
        crow = []
        for ch in choices(c):
            gaps = "; ".join(gap_with_unit(e) for e in ch["errors_fit"].values())
            crow.append({"Model choice": ch["label"], "Fitted numbers": ch["n_params"],
                         "Reduced chi-square": ch["chi2_reduced"],
                         f"NH3 coverage at {cov['ppm']:g} ppm" if cov else "NH3 coverage":
                             pct(ch["coverage_ref"]) if ch.get("coverage_ref") is not None else "",
                         f"Gate 1 at {t_ref}": ref_text(ch["yardstick"]),
                         "Mean gaps": gaps, "Wide spread": ", ".join(ch["poorly_determined"])})
        st.dataframe(pd.DataFrame(crow), hide_index=True, width="stretch",
                     column_config={"Reduced chi-square": st.column_config.NumberColumn(format="%.2f")})
        notes = [f"{a['label']}: {a['note']}" for a in alts if a.get("note")]
        if any(a.get("film") for a in alts):
            notes.append("Film included: the fit adds the gas-to-wall film of the source's monolith channels, so its "
                         "rates are intrinsic and its Gate 1 placement uses them in the ideal reactor.")
        if any(a.get("washcoat") for a in alts):
            notes.append("Washcoat diffusion included: each pathway's rate also carries the effectiveness factor of a "
                         "washcoat slab of the source's effective thickness and effective diffusivity; its Gate 1 placement "
                         "likewise uses the intrinsic rates in the ideal reactor.")
        if notes:
            st.caption(" ".join(notes))
        cp = c.get("coverage_profile")
        if cp and cov:
            with st.expander(f"Ammonia coverage the data allow: {cov_text(cp)}"):
                left, right = st.columns(2, gap="large")
                prof_left, prof_right = profile_charts(cp)
                left.altair_chart(prof_left, width="stretch")
                right.altair_chart(prof_right, width="stretch")
                st.caption(f"Each point refits every reading ({cp['choice'].lower()}, the other numbers free) with the "
                           "coverage held at that value. Coverages under the red line, a chi-square within 2.71 of the "
                           f"best{' scaled by the reduced chi-square of ' + format(cp['scaled_by'], '.2f') if cp['scaled_by'] > 1 else ''}, "
                           f"form the 90% range. Right: the Gate 1 placement at {t_ref} each of those fits gives. Gray "
                           "dashes mark the pack's coverage.")
        with st.expander("Fitted values under each model choice"):
            prow = []
            for ch in choices(c):
                for k, v in ch["params"].items():
                    prow.append({"Model choice": ch["label"], "Parameter": k, "Value": v, "± (1 sd)": ch["std"].get(k)})
                for k, v in ch.get("fixed", {}).items():
                    prow.append({"Model choice": ch["label"], "Parameter": k, "Value": v, "± (1 sd)": None})
            st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch",
                         column_config={"Value": st.column_config.NumberColumn(format="%.3f"),
                                        "± (1 sd)": st.column_config.NumberColumn(format="%.3f")})
        for check in c["checks"]:
            with st.expander(check["title"]):
                st.caption(check["caption"])
                st.dataframe(pd.DataFrame(check["rows"]), hide_index=True, width="stretch")

with tabs[-2]:
    ys = data["yardstick"]
    st.markdown(f"**The published catalysts on the bench yardstick's {ys['step']} step**: "
                f"`{ys['protocol']}` at {ys['ghsv_per_h']:,.0f} 1/h, in the ideal reactor")
    picked = st.multiselect("Model choices", CHOICES, default=[MAIN, COVER], key="lit-gate-choices")
    band_on = st.toggle("90% bands", value=True, key="lit-gate-bands")
    st.caption("Each chart is one catalyst's rate constants run through the yardstick's Gate 1 step. "
               f"{MAIN} keeps the pack's ammonia coverage on the SCR site; the held-out fit uses the same choice on "
               "the points its test keeps. Gray dashes are the pack's illustrative defaults.")
    rows = []
    for c in cats.values():
        sets = [(ch["label"], ch["yardstick"]) for ch in choices(c)] + [("Held-out fit", c["holdout"]["yardstick"])]
        for label, ysk in sets:
            if label in picked:
                rows += [{"T_F": r["T_F"], "value": r["value"], "lo": r.get("lo"), "hi": r.get("hi"),
                          "Catalyst": c["name"], "Model choice": label} for r in ysk]
    df = pd.DataFrame(rows, columns=["T_F", "value", "lo", "hi", "Catalyst", "Model choice"])
    dfl = pd.DataFrame([{"T_F": r["T_F"], "value": r["value"]} for r in data["defaults"]["yardstick"]])
    temps = sorted(dfl.T_F.unique().tolist())
    x = alt.X("T_F:Q", title="Temperature (°F)", scale=alt.Scale(domain=[min(temps), max(temps)]),
              axis=alt.Axis(values=temps, format=",d"))
    color = alt.Color("Model choice:N", scale=choice_scale(picked),
                      legend=alt.Legend(orient="bottom", title=None, symbolType="circle", symbolOpacity=1,
                                        labelLimit=320, columns=2))
    tdf = pd.DataFrame({"y": [ys["target"]], "label": [f"{ys['target_label']}: {ys['target']:g}"]})
    charts = []
    for c in cats.values():
        mine = df[df.Catalyst == c["name"]]
        low = min([0.0, *mine.value.dropna(), *(mine.lo.dropna() if band_on else [])])  # NH3 burned to NO can go below 0
        y = alt.Y("value:Q", title="NOx conversion", scale=alt.Scale(domain=[np.floor(low * 10) / 10, 1]))
        layers = [alt.Chart(dfl).mark_line(color=common.REFERENCE_GRAY, strokeDash=[4, 3], strokeWidth=1.5).encode(x=x, y=y)]
        if band_on and len(mine):
            layers.append(alt.Chart(mine.dropna(subset=["lo"])).mark_area(opacity=0.14).encode(x=x, y="lo:Q", y2="hi:Q",
                                                                                                color=color))
        layers += [
            alt.Chart(mine).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2.3).encode(
                x=x, y=y, color=color,
                tooltip=["Catalyst:N", "Model choice:N", alt.Tooltip("T_F:Q", title="°F"),
                         alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f", title="Band low"),
                         alt.Tooltip("hi:Q", format=".3f", title="Band high")]),
            alt.Chart(tdf).mark_rule(color=common.TARGET_RED, strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q"),
            alt.Chart(tdf).mark_text(align="right", dx=-4, dy=-7, color=common.TARGET_RED, fontSize=11).encode(
                y="y:Q", text="label:N", x=alt.value("width")),
        ]
        charts.append(alt.layer(*layers).properties(title=c["name"], height=280))
    for i in range(0, len(charts), 2):
        for col, chart in zip(st.columns(2, gap="large"), charts[i: i + 2]):
            col.altair_chart(chart, width="stretch")
    if len(df):
        table = df.pivot_table(index=["Catalyst", "Model choice"], columns="T_F", values="value").reset_index()
        table.columns = [f"{c} °F" if isinstance(c, (int, float, np.integer)) else c for c in table.columns]
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns if c.endswith("°F")})

    if all(c["full"].get("operating_map") for c in cats.values()):
        st.subheader(f"More ammonia, more catalyst: the design step at {t_ref}")
        st.caption(f"NOx conversion on the design step at {t_ref} with NH3 fed at 1.0, 1.2 and 1.4 times the NO, and "
                   f"with up to six times the catalyst (space velocity down from {ys['ghsv_per_h']:,.0f} to 5,000 1/h). "
                   "Where the lines flatten, ammonia burned to N2 sets the ceiling; more ammonia lifts it and leaves "
                   "more at the outlet. Hover a point for the NH3 left (wet ppm; ×1.11 gives ppmvd at 15% O2).")
        map_choice = st.segmented_control("Model choice", [COVER, MAIN], default=COVER, required=True, key="lit-map-choice")
        mrows = []
        for c in cats.values():
            ch = next((x for x in choices(c) if x["label"] == map_choice), None)
            for r in (ch or {}).get("operating_map") or []:
                mrows.append({"Catalyst": c["name"], "NH3/NO": f"{r['ratio']:.1f}", "volume": ys["ghsv_per_h"] / r["ghsv_per_h"],
                              "GHSV": r["ghsv_per_h"], "value": r["value"], "slip": r["nh3_out_ppm"]})
        mdf = pd.DataFrame(mrows)
        ratios = sorted(mdf["NH3/NO"].unique())
        xm = alt.X("volume:Q", title=f"Catalyst volume (× the design step's at {ys['ghsv_per_h']:,.0f} 1/h)",
                   scale=alt.Scale(domain=[1, mdf.volume.max()]))
        cm = alt.Color("NH3/NO:N", scale=alt.Scale(domain=ratios, range=RATIO_COLORS[: len(ratios)]),
                       legend=alt.Legend(orient="bottom", title="NH3/NO fed"))
        mcharts = []
        for c in cats.values():
            mine = mdf[mdf.Catalyst == c["name"]]
            if not len(mine):
                continue
            ym = alt.Y("value:Q", title="NOx conversion", scale=alt.Scale(domain=[np.floor(min(0.0, mine.value.min()) * 10) / 10, 1]))
            mcharts.append(alt.layer(
                alt.Chart(mine).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2.3).encode(
                    x=xm, y=ym, color=cm,
                    tooltip=["Catalyst:N", "NH3/NO:N", alt.Tooltip("GHSV:Q", title="1/h", format=","),
                             alt.Tooltip("value:Q", title="NOx conversion", format=".3f"),
                             alt.Tooltip("slip:Q", title="NH3 left (ppm, wet)", format=".1f")]),
                alt.Chart(tdf).mark_rule(color=common.TARGET_RED, strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q"),
            ).properties(title=c["name"], height=240))
        for i in range(0, len(mcharts), 2):
            for col, chart in zip(st.columns(2, gap="large"), mcharts[i: i + 2]):
                col.altair_chart(chart, width="stretch")
        mdf["cell"] = mdf.apply(lambda r: f"{r.value:.2f} · {r.slip:.1f} ppm", axis=1)
        mtab = mdf.pivot_table(index=["Catalyst", "NH3/NO"], columns="GHSV", values="cell", aggfunc="first", sort=False)
        mtab = mtab[sorted(mtab.columns, reverse=True)].reset_index()
        mtab.columns = [f"{c:,} 1/h" if isinstance(c, (int, np.integer)) else c for c in mtab.columns]
        st.dataframe(mtab, hide_index=True, width="stretch")
        st.caption("Each cell: NOx conversion · NH3 left at the outlet (wet ppm).")

with tabs[-1]:
    st.subheader("Sources")
    for sid in dict.fromkeys(c["source"] for c in cats.values()):
        src = sources[sid]
        st.markdown(f"**{short_cite(sid)}.** {cite_link(sid)}")
        st.markdown("\n".join(f"- {f['label']} ({f.get('where', 'PDF page ' + str(f.get('page')))}); feed "
                              + ", ".join(f"{s} {v}" for s, v in f["feed"].items())
                              for f in data["figures"] if f["source"] == sid))
        st.caption(f"{' '.join(src['conditions'].split())} Modeled at {src['ghsv_per_h']:,.0f} 1/h (or a table's own "
                   f"space velocity) and {src['pressure_kPa']:g} kPa.")
    st.subheader("Method")
    st.write(" ".join(rep["method"].split()))
    st.markdown(f"**Reactor.** {data['reactor'][:1].upper()}{data['reactor'][1:]}.")
    st.markdown("\n".join(f"- {c['name']}: fitted " + ", ".join(f"`{p}`" for p in c["fitted"]) + "; switched off "
                          + ", ".join(f"`{p}`" for p in c["switched_off"]) for c in cats.values()))
    st.markdown(f"**Error model.** {' '.join(rep['error_model'].split())}")
    st.subheader("Limits")
    st.markdown("\n".join(f"- {x}" for x in rep["limits"]))
