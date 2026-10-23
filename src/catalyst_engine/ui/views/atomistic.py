"""How the atomistic model's binding energies compare with known chemistry. Text comes from
experiments/<name>/report.yaml; numbers from the results/summary.json its analyze.py writes."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common

MODE_COLORS = {"oc20": "#d97706", "omat": common.ACCENT}

reports = common.experiment_reports()
st.title("Model accuracy")
st.caption("How closely the atomistic model (UMA) matches published binding energies: the check behind every ranking on the Results page.")
if not reports:
    st.info("Model checks appear here once an experiment folder holds a report.yaml.")
    st.stop()
name = st.selectbox("Experiment", list(reports), format_func=lambda k: reports[k]["title"]) if len(reports) > 1 else next(iter(reports))
rep = reports[name]
data = rep["data"]
modes = {m["id"]: m for m in rep["modes"]}
mode_name = {k: m["name"] for k, m in modes.items()}
sources = rep.get("sources", {})

st.markdown(f"##### {rep['question']}")
st.caption(f"{rep['model']} · run {rep['run_date']} · text in experiments/{name}/report.yaml")
if data is None:
    st.info(f"Results appear here after experiments/{name}/analyze.py writes {rep['summary']}.")
    st.stop()

metals, oxides = data["metals"], data["oxides"]
cases = {c["label"]: c for c in oxides["cases"]}


def strength(e: float | None) -> float | None:
    """Binding strength in eV: positive means the molecule holds on."""
    return None if e is None else -e


def link(key: str) -> str:
    s = sources.get(key)
    return f"[{s['cite']}]({s['url']})" if s else ""


summary, metal_tab, oxide_tab, checks_tab, method_tab = st.tabs(
    ["Summary", "Metals", "Oxides", "Reference checks", "Method and sources"])

with summary:
    st.success(" ".join(rep["verdict"].split()), icon=":material/verified:")
    n_scaling = sum(abs(f["slope"] - f["expected"]) <= 0.05 for f in metals["scaling"])
    n_order = sum(o["agrees"] for o in metals["orderings"])
    head = rep.get("headline")
    n_runs = metals["relaxations"] + oxides["relaxations"]
    n_conv = metals["converged"] + oxides["converged"]
    cols = st.columns(4)
    cols[0].metric("Scaling rules matched", f"{n_scaling} of {len(metals['scaling'])}", border=True,
                   help="UMA's slope sits within 0.05 of bond-counting theory")
    cols[1].metric("Metal orderings matched", f"{n_order} of {len(metals['orderings'])}", border=True,
                   help="Pairs of metals where every d-band picture agrees which binds N or O harder")
    if head and head["case"] in cases:
        c = cases[head["case"]]
        cols[2].metric(head.get("label", head["case"]), f"{strength(c[head['mode']]):.2f} eV", border=True,
                       delta=head.get("compare"), delta_color="off", delta_arrow="off",
                       help=f"{mode_name[head['mode']]}. Known: {c['known']}")
    cols[3].metric("Relaxations converged", f"{n_conv} of {n_runs}", border=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What the test showed")
        st.markdown("\n".join(f"- {f}" for f in rep["findings"]))
    with right:
        st.subheader("What it means for the atomistic layer")
        st.markdown("\n".join(f"{i}. {f}" for i, f in enumerate(rep["implications"], start=1)))
        st.markdown(f"**Next test.** {' '.join(rep['next_test'].split())}")

    st.subheader("The two modes")
    for col, m in zip(st.columns(len(modes)), modes.values()):
        with col.container(border=True):
            dot = f"<span style='color:{MODE_COLORS.get(m['id'], common.ACCENT)}'>●</span>"
            st.markdown(f"{dot} **{m['name']}** ({m['id']})", unsafe_allow_html=True)
            st.caption(m["about"])
    with st.expander("Limits of this run"):
        st.markdown("\n".join(f"- {x}" for x in rep["limits"]))
        st.caption(" ".join(rep["speed"].split()))

with metal_tab:
    st.caption(f"Catalysis mode on each metal's close-packed surface. Binding energy relative to N2, H2O and H2 gas, in eV: "
               f"more negative binds harder. {len(metals['metals'])} metals, {len(metals['adsorbates'])} fragments.")
    df = pd.DataFrame(metals["binding"])
    order = df[df.adsorbate == "N"].sort_values("E").metal.tolist()
    base = alt.Chart(df).encode(
        x=alt.X("adsorbate:N", sort=metals["adsorbates"], title=None, axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=13)),
        y=alt.Y("metal:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13)),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("E:Q", scale=alt.Scale(scheme="blueorange", domainMid=0), legend=alt.Legend(title="eV")),
        tooltip=["metal", "adsorbate", alt.Tooltip("E:Q", format="+.2f", title="eV"), "site"],
    )
    labels = base.mark_text(fontSize=12).encode(
        text=alt.Text("E:Q", format="+.2f"),
        color=alt.condition("datum.E < -1.2 || datum.E > 2", alt.value("white"), alt.value("#0f172a")),
    )
    st.markdown("**Binding energy by metal and fragment** (metals sorted by nitrogen binding)")
    st.altair_chart((heat + labels).properties(height=340), width="stretch")

    st.markdown("**Scaling rules.** A fragment with hydrogens binds through the same atom, so its binding should track "
                "the bare atom's at a slope set by how many bonds the hydrogens take up.")
    for col, f in zip(st.columns(len(metals["scaling"])), metals["scaling"]):
        pts = pd.DataFrame(f["points"])
        xs = np.linspace(pts.x.min(), pts.x.max(), 2)
        xbar, ybar = pts.x.mean(), pts.y.mean()
        lines = pd.DataFrame({"x": np.concatenate([xs, xs]), "series": ["UMA fit"] * 2 + ["Theory slope"] * 2,
                              "y": np.concatenate([f["slope"] * xs + f["intercept"], f["expected"] * (xs - xbar) + ybar])})
        domain = ["UMA fit", "Theory slope"]
        line_chart = alt.Chart(lines).mark_line(strokeWidth=2).encode(
            x=alt.X("x:Q", title=f"{f['x']} binding (eV)", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title=f"{f['y']} binding (eV)", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=[common.ACCENT, common.REFERENCE_GRAY]),
                            legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [5, 4]]), legend=None),
        )
        dots = alt.Chart(pts).mark_circle(size=60, color=common.ACCENT).encode(
            x="x:Q", y="y:Q", tooltip=["metal", alt.Tooltip("x:Q", format="+.2f"), alt.Tooltip("y:Q", format="+.2f")])
        names = alt.Chart(pts).mark_text(dx=12, fontSize=11, color="#475569").encode(x="x:Q", y="y:Q", text="metal")
        with col:
            st.markdown(f"**{f['y']} vs {f['x']}**: slope {f['slope']:.2f}, theory {f['expected']:.2f}, R² {f['r2']:.2f}")
            st.altair_chart((line_chart + dots + names).properties(height=260), width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Periodic-table orderings** (stronger binder first)")
        st.dataframe(pd.DataFrame([{
            "Pair": f"{o['strong']} > {o['weak']}", "Atom": o["element"],
            "UMA (eV)": f"{o['e_strong']:+.2f} vs {o['e_weak']:+.2f}", "Matches": "✓" if o["agrees"] else "✗",
        } for o in metals["orderings"]]), hide_index=True, width="stretch", height=35 * (len(metals["orderings"]) + 1) + 3)
    with right:
        st.markdown("**Whole molecules on metals**: binding strength relative to the gas molecule (eV, higher holds harder)")
        mol = -pd.DataFrame(metals["molecules"]).set_index("metal").rename_axis("Metal")
        st.dataframe(mol, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in mol.columns})
        if metals["flagged"]:
            st.caption("Left out of the minimum: " + "; ".join(
                f"{r['metal']} {r['adsorbate']} from {r['site']} ({r['anomaly']})" for r in metals["flagged"]) + ".")

with oxide_tab:
    st.caption("Binding strength of each molecule relative to the free gas molecule, in eV: higher holds on harder. "
               "Gray bars mark the measured or published range where one exists.")
    rows, bands = [], []
    for c in oxides["cases"]:
        for mode in modes:
            if c.get(mode) is not None:
                rows.append({"case": c["label"], "mode": mode_name[mode], "strength": strength(c[mode])})
        if c.get("known_range"):
            lo, hi = c["known_range"]
            bands.append({"case": c["label"], "lo": -hi, "hi": -lo, "known": c["known"]})
    labels_order = [c["label"] for c in oxides["cases"]]
    domain = [mode_name[m] for m in modes]
    y = alt.Y("case:N", sort=labels_order, title=None, axis=alt.Axis(labelFontSize=13, labelLimit=260))
    band_chart = alt.Chart(pd.DataFrame(bands)).mark_bar(size=22, color=common.REFERENCE_GRAY, opacity=0.45).encode(
        x="lo:Q", x2="hi:Q", y=y, tooltip=[alt.Tooltip("known:N", title="Known")])
    dot_chart = alt.Chart(pd.DataFrame(rows)).mark_point(filled=True, size=160, opacity=1).encode(
        x=alt.X("strength:Q", title="Binding strength (eV)", scale=alt.Scale(domain=[0, 1.8])),
        y=y,
        color=alt.Color("mode:N", scale=alt.Scale(domain=domain, range=[MODE_COLORS[m] for m in modes]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["case", "mode", alt.Tooltip("strength:Q", format=".2f", title="eV")],
    )
    st.altair_chart((band_chart + dot_chart).properties(height=320), width="stretch")

    header = "| Case | " + " | ".join(mode_name[m] for m in modes) + " | Known | Closest contact after relaxing | Note |"
    lines = [header, "|---" * (len(modes) + 4) + "|"]
    for c in oxides["cases"]:
        vals = " | ".join(f"{strength(c[m]):.2f}" if c.get(m) is not None else "–" for m in modes)
        contact = " / ".join(c.get(f"contact_{m}") or "–" for m in modes)
        src = sources.get(c["source"])
        known = f"[{c['known']}]({src['url']})" if src else c["known"]
        lines.append(f"| {c['label']} | {vals} | {known} | {contact} | {c['note']} |")
    st.markdown("\n".join(lines))

    shift = oxides["monolayer_shift"]
    if shift:
        st.markdown("**Crowding.** A full layer of ammonia on titania holds each molecule more weakly than a lone molecule.")
        cols = st.columns(len(shift) + 1)
        for col, (mode, value) in zip(cols, shift.items()):
            col.metric(mode_name[mode], f"{value:.2f} eV weaker", border=True)
        cols[-1].metric("Measured", f"{oxides['monolayer_shift_known']:.2f} eV weaker", border=True)

with checks_tab:
    notes = rep.get("reference_notes", {})
    st.subheader("Gas molecules")
    st.write(" ".join(notes.get("gas", "").split()))
    gas = pd.DataFrame(data["gas"])
    if not gas.empty:
        gas["Gap"] = (gas.uma_eV - gas.reference_eV).round(3)
        gas["task"] = gas.task.map(mode_name)
        st.dataframe(gas.rename(columns={"task": "Mode", "quantity": "Quantity", "uma_eV": "UMA (eV)",
                                         "reference_eV": "Reference (eV)", "reference": "Reference from"}),
                     hide_index=True, width="stretch")
    st.subheader("Bulk lattice")
    st.write(" ".join(notes.get("lattice", "").split()))
    lat = pd.DataFrame(data["lattice"])
    if not lat.empty:
        lat["task"] = lat.task.map(mode_name)
        st.dataframe(lat.rename(columns={"crystal": "Crystal", "task": "Mode", "axis": "Axis", "experiment_A": "Experiment (Å)",
                                         "uma_A": "UMA (Å)", "change_pct": "Change (%)"}),
                     hide_index=True, width="stretch", height=35 * (len(lat) + 1) + 3)

with method_tab:
    for part in rep.get("method", []):
        st.markdown(f"**{part['heading']}.** {' '.join(part['text'].split())}")
    st.subheader("Sources")
    st.markdown("\n".join(f"- {link(k)}" for k in sources))
    st.subheader("Rerun it")
    st.caption(f"From the repository root, in the Python that has fairchem. Results land in experiments/{name}/results/.")
    st.code(rep["rerun"], language="bash")
