"""Lab data: catalysts, samples, runs with their raw files, campaign scores and the prediction log."""

import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.data.demo import build_demo, demo_packs
from catalyst_engine.data.ingest import (fit_rows, ingest_run, issues, parse_analyzer_text, derive_points,
                                         point_metrics, read_analyzer_file, run_points, settled)
from catalyst_engine.data.ledger import ANCHORS, ARMS, Ledger
from catalyst_engine.data.records import add_campaign, add_formulation, add_sample, direction, stored_protocol
from catalyst_engine.data.scoring import score_campaign
from catalyst_engine.data.store import Store, store_path
from catalyst_engine.ui import common

ARM_COLORS = {"model": common.ACCENT, "expert": "#0369a1", "random": "#d97706", "blinded": common.REFERENCE_GRAY}
STATE = {"hit": ("Hit", "green"), "miss": ("Miss", "gray"), "awaiting test": ("Awaiting test", "blue"),
         "awaiting reference": ("Awaiting reference", "blue"), "needs retest": ("Needs retest", "orange")}
SPECIES_COLORS = ["#0f766e", "#7c3aed", "#d97706", "#dc2626", "#0369a1", "#65a30d"]
RUN_STATUS = {"complete": ("Complete", "green"), "failed": ("Failed", "red"), "aborted": ("Aborted", "orange")}


def flag_text(flag: str) -> str:
    kind, _, rest = flag.partition(":")
    rest = rest.strip()
    return {
        "unsteady": f"{rest} still drifting",
        "temperature_unsteady": "temperature still moving",
        "temperature_off": "temperature away from setpoint",
        "short_hold": "hold too short to average",
        "off_protocol": "setpoint outside the protocol",
        "inlet_from_protocol": "inlet taken from the protocol feed",
        "below_lod": f"{rest} below detection limit",
        "negative": f"{rest} reads below zero",
        "inlet_mismatch": f"inlet differs from the protocol feed ({rest})",
        "missing": rest,
    }.get(kind, flag)


def chip(label: str, color: str) -> str:
    return f":{color}-badge[{label}]"


def components_text(components: list[dict]) -> str:
    parts = []
    for c in components:
        amount = c["amount"] if isinstance(c["amount"], str) else f"{c['amount']:g}"
        unit = f" {c['unit']}" if c.get("unit") else ""
        parts.append(f"{c['material']} ({c.get('role', '')}, {amount}{unit})" if amount != "balance"
                     else f"{c['material']} ({c.get('role', 'support')})")
    return " · ".join(parts)


st.title("Lab data")
st.caption(
    "Every catalyst, sample and bench run, with its raw analyzer file, plus the prediction log that records each "
    "pick before the bench sees it. Records are append-only: a correction is a new record that names the one it "
    "replaces, and failed runs stay on file."
)

lab_path, sim_path = store_path("lab"), store_path("simulated")
lab_exists = (lab_path / "store.db").exists()
lab_has_runs = lab_exists and Store(lab_path).counts()["runs"] > 0
choice = st.segmented_control("Store", ["Lab store", "Simulated demo"], required=True, key="lab_store",
                              default="Lab store" if lab_has_runs else "Simulated demo")

if choice == "Simulated demo":
    packs_with_demo = demo_packs()
    a, b = st.columns([5, 1], vertical_alignment="center")
    a.info(
        "Simulated demo: made-up catalysts and simulated analyzer logs that run every part of the platform, "
        "from picks to scores. These numbers stay here; claims and gate scores come from the lab store.",
        icon=":material/science:",
    )
    if b.button("Rebuild demo", icon=":material/refresh:", width="stretch",
                help="Rebuilds the simulated store from the pack's demo.yaml") and packs_with_demo:
        with st.spinner("Simulating runs"):
            build_demo(common.pack(packs_with_demo[0])).close()
    if not (sim_path / "store.db").exists():
        if packs_with_demo:
            with st.spinner("Building the demo store"):
                build_demo(common.pack(packs_with_demo[0])).close()
        else:
            st.info("Add a demo.yaml to a pack to build the demo store.")
            st.stop()
    store = Store(sim_path)
else:
    if not lab_exists:
        st.info(
            f"The lab store lives at `{lab_path}`, outside git. It starts with your first record: add the reference "
            "catalyst on the Add data tab, or create the empty store now.",
            icon=":material/inventory_2:",
        )
        if st.button("Create the lab store", icon=":material/add:", type="primary"):
            Store.open_or_create(lab_path, "lab").close()
            st.rerun()
        st.stop()
    store = Store(lab_path)

ledger = Ledger(store)
check = ledger.verify()
raw_rows = store.rows("raw_files")
raw_ok = sum(store.verify_raw(r["sha256"]) for r in raw_rows)
runs = store.current("runs")
formulations = {f["id"]: f for f in store.rows("formulations")}
samples = {s["id"]: s for s in store.rows("samples")}
campaigns = store.rows("campaigns", order="id")

m = st.columns(5)
m[0].metric("Catalysts", len(store.current("formulations")))
m[1].metric("Samples", len(store.current("samples")))
m[2].metric("Runs", len(runs), f"{sum(r['status'] == 'complete' for r in runs)} complete", delta_color="off",
            delta_arrow="off")
m[3].metric("Prediction log", f"{check.entries} entries", "Chain intact" if check.ok else "Check failed",
            delta_color="normal" if check.ok else "inverse", delta_arrow="off")
m[4].metric("Raw files", f"{raw_ok} of {len(raw_rows)}", "match their hashes", delta_color="off", delta_arrow="off")
if not check.ok:
    st.error(f"The prediction log fails its check: {check.problem}", icon=":material/gpp_bad:")

camp_tab, runs_tab, cat_tab, log_tab, add_tab = st.tabs(
    ["Campaigns", "Runs", "Catalysts and samples", "Prediction log", "Add data"])


def pack_of_protocol(digest: str):
    return common.pack(store.get("protocols", digest)["pack"])


# Campaigns ---------------------------------------------------------------------------------------------------------
with camp_tab:
    if not campaigns:
        st.info("A campaign fixes the protocol version, the rig, the reference catalyst and the hit rule before "
                "any pick. Open one on the Add data tab.", icon=":material/flag:")
    else:
        names = {c["id"]: f"{c['id']} · {c['name']}" for c in campaigns}
        cid = st.selectbox("Campaign", list(names), format_func=names.get, index=len(names) - 1, key="lab_campaign")
        camp = store.get("campaigns", cid)
        p = common.pack(camp["pack"])
        rule = camp["hit_rule"]
        sc = score_campaign(store, p, cid)
        better = direction(p, rule)
        temps = rule["temperatures_F"]
        metric_label = common.metric_title(p, rule["metric"])
        proto = stored_protocol(store, camp["protocol_hash"])

        with st.container(border=True):
            a, b = st.columns([4, 1], vertical_alignment="top")
            a.markdown(f"#### {camp['name']}")
            with b:
                st.badge("Blinded" if sc.blinded else "Arms revealed", color="orange" if sc.blinded else "green")
            t_text = " and ".join(f"{t:,.0f}" for t in temps)
            if rule["rule"] == "beats_reference":
                verb = "beats" if better == "higher" else "comes in under"
                a.markdown(f"**Hit rule:** {rule['metric']} {verb} the reference by {rule['margin']:g} or more at "
                           f"{t_text} °F on the {rule['step']} step.")
            else:
                a.markdown(f"**Hit rule:** {rule['metric']} meets the pack target "
                           f"({common.target_text(p.spec.metrics[rule['metric']].target)}) at {t_text} °F on the "
                           f"{rule['step']} step.")
            ref_name = formulations[camp["reference_formulation_id"]]["name"]
            st.markdown(
                f":gray-badge[Protocol {proto.id} v{proto.version} · {camp['protocol_hash'][:10]}] "
                f":gray-badge[Rig {camp['rig']}] :gray-badge[Reference {camp['reference_formulation_id']}: {ref_name}]"
            )
            if camp["notes"]:
                st.caption(camp["notes"])

        tested = [pk for pk in sc.picks if pk.state in ("hit", "miss")]
        cols = st.columns(max(len(sc.arms), 1) + 1)
        ref_text = ", ".join(f"{v:.3f}" if v is not None else "pending" for v in sc.reference.values())
        cols[0].metric(f"Reference at {' / '.join(f'{t:,.0f}' for t in temps)} °F", ref_text,
                       f"mean of {len(sc.reference_runs)} runs", delta_color="off", delta_arrow="off",
                       help=f"{rule['metric']} of the reference catalyst")
        if sc.blinded:
            cols[1].metric("Picks", len(sc.picks), f"{len(tested)} scored", delta_color="off", delta_arrow="off",
                           help="Arm hit rates appear once the arms are revealed.")
        for col, arm in zip(cols[1:], sc.arms):
            rate = f"{arm.hit_rate:.0%}" if arm.hit_rate is not None else "–"
            col.metric(f"{arm.arm.title()} picks", f"{arm.hits} of {arm.tested}", f"{rate} hit rate",
                       delta_color="off", delta_arrow="off")

        if sc.comparisons:
            text = "; ".join(f"model against {c['against']}, p = {c['p_value']:.2f}" for c in sc.comparisons)
            st.caption(f"One-sided Fisher exact test, {text}. Gate 4 plans about 76 picks per arm, enough to tell a "
                       "20% hit rate from 5% at 95% confidence.")

        st.markdown(f"##### {metric_label}")
        st.caption("Each dot is one candidate's counted run. Gray line: the reference; dashed line: the hit "
                   "threshold." if rule["rule"] == "beats_reference" else "Each dot is one candidate's counted run.")
        chart_rows = []
        for pk in sc.picks:
            for t, v in pk.values.items():
                if v is not None:
                    chart_rows.append({"name": pk.name, "arm": pk.arm or "blinded", "T": f"{t:,.0f} °F", "value": v,
                                       "state": STATE[pk.state][0]})
        if chart_rows:
            df = pd.DataFrame(chart_rows)
            charts = st.columns(len(temps))
            arms_present = [a for a in [*ARMS, "blinded"] if a in set(df["arm"])]
            for col, t in zip(charts, temps):
                sub = df[df["T"] == f"{t:,.0f} °F"]
                dots = alt.Chart(sub).mark_circle(size=140, opacity=0.9).encode(
                    x=alt.X("value:Q", title=rule["metric"], scale=alt.Scale(zero=False, nice=True)),
                    y=alt.Y("name:N", title=None, sort=list(sub.sort_values("value", ascending=False)["name"])),
                    color=alt.Color("arm:N", scale=alt.Scale(domain=arms_present,
                                                             range=[ARM_COLORS[a] for a in arms_present]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["name:N", "arm:N", "state:N", alt.Tooltip("value:Q", format=".3f")],
                )
                layers = [dots]
                ref = sc.reference.get(t)
                if ref is not None:
                    lines = pd.DataFrame({"x": [ref], "kind": ["reference"]})
                    layers.append(alt.Chart(lines).mark_rule(color=common.REFERENCE_GRAY, strokeWidth=2).encode(x="x:Q"))
                    if rule["rule"] == "beats_reference":
                        edge = ref + rule["margin"] if better == "higher" else ref - rule["margin"]
                        layers.append(alt.Chart(pd.DataFrame({"x": [edge]})).mark_rule(
                            color=common.ACCENT, strokeDash=[4, 3]).encode(x="x:Q"))
                with col:
                    st.markdown(f"**{t:,.0f} °F**")
                    st.altair_chart(alt.layer(*layers).properties(height=36 * len(sub) + 60), width="stretch")

        st.markdown("##### Picks")
        pick_rows = []
        for pk in sc.picks:
            row = {"Formulation": pk.formulation, "Name": pk.name, "Arm": pk.arm or "blinded",
                   "Result": STATE[pk.state][0], "Counted run": pk.run or ""}
            row.update({f"{t:,.0f} °F": pk.values.get(t) for t in temps})
            row["Other runs"] = ", ".join(
                [f"{r} (failed)" for r in pk.failed_runs] + [f"{r} (point still settling)" for r in pk.unsettled_runs]
                + [f"{r} (before the pick)" for r in pk.early_runs])
            pick_rows.append(row)
        if pick_rows:
            st.dataframe(pd.DataFrame(pick_rows), hide_index=True, height=35 * (len(pick_rows) + 1) + 3,
                         column_config={**{f"{t:,.0f} °F": st.column_config.NumberColumn(format="%.3f") for t in temps},
                                        "Other runs": st.column_config.TextColumn(width="medium")})
            st.caption("A pick counts once: its first complete run logged after the pick with every hit-rule point "
                       "settled. Failed runs, retests and runs logged before the pick stay listed.")

        if sc.predictions:
            st.markdown("##### Predictions against measurements")
            pdf = pd.DataFrame(sc.predictions)
            pdf["name"] = pdf["formulation"].map(lambda f: formulations[f]["name"])
            scored = pdf.dropna(subset=["measured"])
            a, b = st.columns([3, 2], gap="large")
            if not scored.empty:
                lo = float(min(scored["low"].min(), scored["measured"].min())) - 0.02
                hi = float(max(scored["high"].max(), scored["measured"].max())) + 0.02
                axis = alt.Scale(domain=[lo, hi], zero=False)
                diag = alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]})).mark_line(
                    color=common.REFERENCE_GRAY, strokeDash=[4, 3]).encode(
                    x=alt.X("x:Q", title="Predicted", scale=axis), y=alt.Y("y:Q", title="Measured", scale=axis))
                scored = scored.assign(inside=scored["inside"].map({True: "Inside", False: "Outside"}))
                inside_color = alt.Color("inside:N", scale=alt.Scale(domain=["Inside", "Outside"],
                                                                     range=[common.ACCENT, "#d97706"]),
                                         legend=alt.Legend(orient="bottom", title="Measured value vs interval"))
                bars = alt.Chart(scored).mark_rule(strokeWidth=2, opacity=0.6).encode(
                    x=alt.X("low:Q", scale=axis), x2="high:Q", y=alt.Y("measured:Q", scale=axis), color=inside_color)
                dots = alt.Chart(scored).mark_circle(size=70).encode(
                    x=alt.X("predicted:Q", scale=axis), y=alt.Y("measured:Q", scale=axis), color=inside_color,
                    tooltip=["name:N", alt.Tooltip("temperature_F:Q", title="°F"),
                             alt.Tooltip("predicted:Q", format=".3f"), alt.Tooltip("measured:Q", format=".3f")])
                a.altair_chart((diag + bars + dots).properties(height=340), width="stretch")
            with b:
                if sc.coverage is not None:
                    st.metric("Intervals that held the measurement", f"{sc.coverage:.0%}",
                              f"{int(scored['inside'].eq('Inside').sum())} of {len(scored)}", delta_color="off",
                              delta_arrow="off")
                if not scored.empty:
                    st.metric("Mean absolute error", f"{(scored['measured'] - scored['predicted']).abs().mean():.3f}")
                st.caption("Each bar spans a logged prediction interval, at the measured value. A calibrated model "
                           "holds the measurement inside its interval as often as it claims.")

# Runs --------------------------------------------------------------------------------------------------------------
with runs_tab:
    if not runs:
        st.info("Runs appear here once an analyzer log is ingested on the Add data tab.", icon=":material/timeline:")
    else:
        def run_label(r: dict) -> str:
            f = formulations[samples[r["sample_id"]]["formulation_id"]]
            return f"{r['id']} · {f['name']} · {r['status']}"

        labels = {r["id"]: run_label(r) for r in runs}
        rid = st.selectbox("Run", list(labels), format_func=labels.get, key="lab_run")
        run = store.get("runs", rid)
        p = pack_of_protocol(run["protocol_hash"])
        protocol, points, metrics = run_points(store, p, rid)
        derivation = store.latest_derivation(rid)
        sample = samples[run["sample_id"]]
        form = formulations[sample["formulation_id"]]

        with st.container(border=True):
            a, b = st.columns([4, 1])
            a.markdown(f"#### {rid}: {form['name']}")
            label, color = RUN_STATUS.get(run["status"], (run["status"], "gray"))
            with b:
                st.badge(label, color=color)
            raw = store.rows("raw_files", "sha256 = ?", (run["raw_sha256"],))[0]
            intact = store.verify_raw(run["raw_sha256"])
            st.markdown(
                f":gray-badge[Sample {sample['id']} · {sample['form']}] :gray-badge[{run['campaign_id'] or 'outside a campaign'}] "
                f":gray-badge[Rig {run['rig']}] :gray-badge[Operator {run['operator']}] "
                f":gray-badge[Started {run['started_at'][:16].replace('T', ' ')} UTC]"
            )
            st.markdown(
                f"Raw file **{raw['original_name']}**, SHA-256 `{run['raw_sha256'][:16]}…` "
                + (":green-badge[matches its hash]" if intact else ":red-badge[changed since it was stored]")
                + f" · protocol {protocol.id} v{protocol.version} · points by {derivation['method'] if derivation else ''}"
            )
            if run["notes"]:
                st.caption(run["notes"])
            run_flags = derivation["run_flags"] if derivation else []
            missing = [f.split(":", 1)[1].strip() for f in run_flags if f.startswith("missing")]
            planned = sum(len(s.kelvin()) for s in protocol.steps)
            if missing:
                st.warning(f"The log covers {planned - len(missing)} of {planned} protocol points. Still to run: "
                           + "; ".join(missing) + ".", icon=":material/report:")
            for f in run_flags:
                if not f.startswith("missing"):
                    st.warning(flag_text(f), icon=":material/report:")

        data = read_analyzer_file(store.raw_path(run["raw_sha256"]), protocol.analyzer.measured)
        series = pd.DataFrame(data.rows)
        series["minutes"] = series["time_s"] / 60.0
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        species = c2.multiselect("Species", data.species, default=data.species[:1] + data.species[2:3],
                                 key=f"lab_species_{rid}")
        c1.markdown("##### Analyzer log")
        long = series.melt(id_vars=["minutes", "mode", "step"], value_vars=species, var_name="species", value_name="ppm")
        windows = pd.DataFrame([{"start": pt["window_start_s"] / 60, "end": pt["window_end_s"] / 60,
                                 "label": f"{pt['step']} {pt['setpoint_F']:,.0f} °F",
                                 "state": "Settled" if settled(pt) else "Still settling"} for pt in points])
        bypass = series[series["mode"] == "bypass"].groupby("step")["minutes"].agg(["min", "max"]).reset_index()
        x = alt.X("minutes:Q", title="Time (min)")
        shade = alt.Chart(windows).mark_rect(opacity=0.18).encode(
            x=alt.X("start:Q"), x2="end:Q",
            color=alt.Color("state:N", scale=alt.Scale(domain=["Settled", "Still settling"],
                                                      range=[common.ACCENT, "#d97706"]),
                            legend=alt.Legend(orient="bottom", title="Averaging window")),
            tooltip=["label:N", "state:N"]) if not windows.empty else None
        bypass_shade = alt.Chart(bypass).mark_rect(opacity=0.12, color=common.REFERENCE_GRAY).encode(
            x="min:Q", x2="max:Q", tooltip=[alt.Tooltip("step:N", title="Bypass (inlet) for step")])
        lines = alt.Chart(long).mark_line(strokeWidth=1.4).encode(
            x=x, y=alt.Y("ppm:Q", title="Reading (ppm, wet)", axis=alt.Axis(minExtent=56)),
            color=alt.Color("species:N", scale=alt.Scale(domain=data.species,
                                                         range=SPECIES_COLORS[:len(data.species)]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["species:N", alt.Tooltip("minutes:Q", format=".1f"), alt.Tooltip("ppm:Q", format=".2f")])
        layers = [bypass_shade] + ([shade] if shade is not None else []) + [lines]
        st.altair_chart(alt.layer(*layers).resolve_scale(color="independent").properties(height=300), width="stretch")
        temp = alt.Chart(series).mark_line(color="#b45309", strokeWidth=1.4).encode(
            x=x, y=alt.Y("T_F:Q", title="Temperature (°F)", scale=alt.Scale(zero=False), axis=alt.Axis(minExtent=56)))
        st.altair_chart(temp.properties(height=150), width="stretch")
        st.caption("Gray bands: bypass, where the analyzer reads the inlet. Colored bands: the last two minutes of "
                   "each hold, averaged into one point.")

        st.markdown("##### Steady-state points")
        show_metrics = [mid for mid in p.spec.metrics][:3]
        prow = []
        for pt, mvals in zip(points, metrics):
            row = {"Step": pt["step"], "Setpoint (°F)": pt["setpoint_F"], "Mean T (°F)": round(pt["T_F"], 1),
                   "Readings": pt["n"]}
            row.update({f"{s} out (ppm)": pt["outlet"].get(s) for s in data.species})
            row.update({mid: mvals.get(mid) for mid in show_metrics})
            row["Checks"] = "; ".join(flag_text(f) for f in issues(pt)) or "settled"
            row["Below detection"] = ", ".join(f.split(":")[1].strip() for f in pt["flags"] if f.startswith("below_lod"))
            prow.append(row)
        if prow:
            num = {c: st.column_config.NumberColumn(format="%.2f") for c in prow[0] if c.endswith("(ppm)")}
            num.update({mid: st.column_config.NumberColumn(format="%.3f") for mid in show_metrics})
            st.dataframe(pd.DataFrame(prow), hide_index=True, height=35 * (len(prow) + 1) + 3, column_config=num)
        d1, d2 = st.columns(2)
        d1.download_button("Download the raw file", store.raw_path(run["raw_sha256"]).read_bytes(),
                           file_name=raw["original_name"], mime="text/csv",
                           icon=":material/download:", width="stretch")
        rows_for_fit = fit_rows(p, protocol, points)
        if rows_for_fit:
            d2.download_button("Download settled points for the Fit lab", pd.DataFrame(rows_for_fit).to_csv(index=False),
                               file_name=f"{rid}-points.csv", mime="text/csv", icon=":material/query_stats:",
                               width="stretch", help="Upload this file on the Fit lab page to fit rate constants.")

# Catalysts and samples ---------------------------------------------------------------------------------------------
with cat_tab:
    current = store.current("formulations")
    if not current:
        st.info("Catalysts appear here once added on the Add data tab.", icon=":material/science:")
    else:
        run_counts = pd.Series([samples[r["sample_id"]]["formulation_id"] for r in runs]).value_counts().to_dict()
        rows = [{"Formulation": f["id"], "Name": f["name"], "Role": f["role"], "Composition": components_text(f["components"]),
                 "Preparation": f["preparation"], "Samples": sum(s["formulation_id"] == f["id"] for s in samples.values()),
                 "Runs": run_counts.get(f["id"], 0), "Added": f["created_at"][:10], "By": f["created_by"]}
                for f in current]
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(35 * (len(rows) + 1) + 3, 600),
                     column_config={"Composition": st.column_config.TextColumn(width="large")})
        replaced = [f for f in store.rows("formulations") if f["id"] not in {c["id"] for c in current}]
        if replaced:
            with st.expander(f"Replaced records ({len(replaced)})"):
                st.dataframe(pd.DataFrame([{"Formulation": f["id"], "Name": f["name"]} for f in replaced]), hide_index=True)
        st.markdown("##### Samples")
        srows = [{"Sample": s["id"], "Formulation": s["formulation_id"], "Name": formulations[s["formulation_id"]]["name"],
                  "Form": s["form"], "Mass (g)": s["mass_g"], "Volume (mL)": s["volume_mL"], "Aged from": s["aged_from"] or "",
                  "Runs": sum(r["sample_id"] == s["id"] for r in runs)} for s in store.current("samples")]
        st.dataframe(pd.DataFrame(srows), hide_index=True, height=min(35 * (len(srows) + 1) + 3, 600))

# Prediction log ----------------------------------------------------------------------------------------------------
with log_tab:
    a, b = st.columns([3, 2], gap="large")
    with a:
        if check.ok:
            st.success(f"Chain intact: {check.entries} entries. Head hash `{check.head[:16]}…`",
                       icon=":material/verified:")
        st.caption("Each entry's hash covers the one before it, so an edit to any past entry breaks every hash "
                   "after it. Picks and predictions enter before the run; each ingested run enters too, so the "
                   "order of the chain shows the pick came first.")
    with b:
        if store.simulated:
            st.caption("Anchors record the lab log in the repository. The demo log stays local.")
        else:
            anchors = ANCHORS.read_text().strip().splitlines()[-1] if ANCHORS.exists() else ""
            if anchors.startswith("| 2"):
                st.caption(f"Last anchor: {anchors.split('|')[1].strip()}, entries {anchors.split('|')[2].strip()}")
            if st.button("Anchor the head hash", icon=":material/anchor:", disabled=not check.ok or not check.entries,
                         help="Appends the head hash to docs/prediction_anchors.md. Commit that file to timestamp "
                              "every entry up to now."):
                ledger.anchor()
                st.success("Anchored. Commit docs/prediction_anchors.md to timestamp it.")

    blinded = {c["id"] for c in campaigns if not ledger.unblinded(c["id"])}

    def summary(e: dict) -> str:
        pl = e["payload"]
        if e["kind"] == "pick":
            arm = "blinded" if pl["campaign"] in blinded else pl["arm"]
            return f"{pl['campaign']}: picked {pl['formulation']} ({arm}). {pl.get('rationale', '')}".strip()
        if e["kind"] == "prediction":
            interval = f" [{pl['low']:.3f}, {pl['high']:.3f}]" if pl.get("low") is not None else ""
            return (f"{pl['campaign']}: {pl['formulation']} {pl['metric']} at {pl['temperature_F']:,.0f} °F "
                    f"({pl['step']}) = {pl['value']:.3f}{interval}")
        if e["kind"] == "run":
            return f"Run {pl['run']} ({pl['status']}) of sample {pl['sample']}, raw {pl['raw_sha256'][:12]}"
        if e["kind"] == "unblind":
            return f"{pl['campaign']}: arms revealed"
        return pl.get("text", "")

    kinds = st.pills("Show", ["pick", "prediction", "run", "unblind", "note"], selection_mode="multi",
                     default=["pick", "prediction", "run", "unblind", "note"], key="lab_log_kinds")
    entries = [e for e in ledger.entries() if e["kind"] in (kinds or [])]
    st.dataframe(pd.DataFrame([{"#": e["seq"], "Time (UTC)": e["created_at"].replace("T", " ")[:19], "Kind": e["kind"],
                                "Entry": summary(e), "By": e["payload"].get("by", ""), "Hash": e["hash"][:12]}
                               for e in reversed(entries)]),
                 hide_index=True, height=min(35 * (len(entries) + 1) + 3, 640),
                 column_config={"Entry": st.column_config.TextColumn(width="large")})

# Add data ----------------------------------------------------------------------------------------------------------
with add_tab:
    if store.simulated:
        st.info("Adding records works in the lab store. Switch the store above to Lab store.", icon=":material/edit:")
    else:
        add_pack = common.pack_picker("lab_add_pack")
        cur_forms = store.current("formulations")
        refs = [f for f in cur_forms if f["role"] == "reference"]
        cands = [f for f in cur_forms if f["role"] == "candidate"]
        fname = {f["id"]: f"{f['id']} · {f['name']}" for f in cur_forms}
        what = st.segmented_control("Add", ["Catalyst", "Sample", "Campaign", "Pick", "Prediction", "Run"],
                                    default="Catalyst", required=True, key="lab_add_what")

        if what == "Catalyst":
            with st.form("lab_add_formulation", clear_on_submit=True):
                name = st.text_input("Name", placeholder="e.g. Reference hot-SCR catalyst")
                role = st.radio("Role", ["candidate", "reference"], horizontal=True)
                comp = st.data_editor(pd.DataFrame([{"material": "", "role": "active", "amount": "", "unit": "wt%"}]),
                                      num_rows="dynamic", hide_index=True, width="stretch")
                prep = st.text_area("Preparation", height=80)
                notes = st.text_input("Notes")
                if st.form_submit_button("Add catalyst", type="primary"):
                    comps = [{"material": r["material"].strip(), "role": r["role"],
                              "amount": float(r["amount"]) if str(r["amount"]).replace(".", "", 1).isdigit() else str(r["amount"]),
                              "unit": r["unit"]} for r in comp.to_dict("records") if str(r["material"]).strip()]
                    rec = add_formulation(store, name.strip(), comps, role, prep, notes)
                    st.success(f"Added {rec['id']}: {rec['name']}")

        elif what == "Sample":
            if not cur_forms:
                st.info("Add a catalyst first; each sample belongs to one.")
            else:
                with st.form("lab_add_sample", clear_on_submit=True):
                    fid = st.selectbox("Catalyst", list(fname), format_func=fname.get)
                    form = st.text_input("Form", "monolith core")
                    c = st.columns(2)
                    mass = c[0].number_input("Mass (g)", min_value=0.0, value=None, format="%.3f")
                    vol = c[1].number_input("Volume (mL)", min_value=0.0, value=None, format="%.2f")
                    notes = st.text_input("Notes")
                    if st.form_submit_button("Add sample", type="primary"):
                        rec = add_sample(store, fid, form, mass, vol, notes=notes)
                        st.success(f"Added {rec['id']} of {fid}")

        elif what == "Campaign":
            if not refs:
                st.info("A campaign compares candidates with a reference catalyst. Add the reference first.")
            else:
                proto_id = st.selectbox("Protocol", list(add_pack.protocols), key="lab_camp_proto")
                proto = add_pack.protocol(proto_id)
                with st.form("lab_add_campaign"):
                    name = st.text_input("Name", placeholder="Batch 1")
                    rig = st.text_input("Rig", "bench-1")
                    ref_id = st.selectbox("Reference", [f["id"] for f in refs], format_func=fname.get)
                    c = st.columns(3)
                    metric = c[0].selectbox("Metric", list(add_pack.spec.metrics))
                    step = c[1].selectbox("Step", [s.id for s in proto.steps])
                    rule_kind = c[2].selectbox("Rule", ["beats_reference", "meets_target"])
                    temps_text = st.text_input("Temperatures (°F, comma separated)", "1050, 1150")
                    margin = st.number_input("Margin past the reference", value=0.03, format="%.3f")
                    notes = st.text_input("Notes")
                    if st.form_submit_button("Open campaign", type="primary"):
                        temps_in = [float(t) for t in temps_text.replace(" ", "").split(",") if t]
                        rec = add_campaign(store, add_pack, name, proto_id, rig, ref_id,
                                           {"metric": metric, "step": step, "temperatures_F": temps_in,
                                            "rule": rule_kind, "margin": margin}, notes)
                        st.success(f"Opened {rec['id']} on protocol hash {rec['protocol_hash'][:10]}")

        elif what == "Pick":
            open_camps = [c for c in campaigns if not ledger.unblinded(c["id"])]
            if not open_camps or not cands:
                st.info("Picks need an open campaign and at least one candidate catalyst.")
            else:
                with st.form("lab_pick", clear_on_submit=True):
                    cid = st.selectbox("Campaign", [c["id"] for c in open_camps])
                    fid = st.selectbox("Candidate", [f["id"] for f in cands], format_func=fname.get)
                    arm = st.radio("Arm", list(ARMS), horizontal=True)
                    rationale = st.text_input("Rationale")
                    if st.form_submit_button("Log pick", type="primary"):
                        e = ledger.pick(cid, fid, arm, rationale)
                        st.success(f"Logged pick #{e['seq']}, hash {e['hash'][:12]}")

        elif what == "Prediction":
            if not campaigns or not cands:
                st.info("Predictions need a campaign and a candidate catalyst.")
            else:
                with st.form("lab_predict", clear_on_submit=True):
                    cid = st.selectbox("Campaign", [c["id"] for c in campaigns])
                    fid = st.selectbox("Candidate", [f["id"] for f in cands], format_func=fname.get)
                    c = st.columns(3)
                    metric = c[0].selectbox("Metric", list(add_pack.spec.metrics))
                    step = c[1].text_input("Step", "kinetic")
                    temp_f = c[2].number_input("Temperature (°F)", value=1050.0)
                    c = st.columns(3)
                    value = c[0].number_input("Predicted value", value=0.8, format="%.3f")
                    low = c[1].number_input("Interval low", value=0.76, format="%.3f")
                    high = c[2].number_input("Interval high", value=0.84, format="%.3f")
                    model = st.text_input("Model and version")
                    if st.form_submit_button("Log prediction", type="primary"):
                        e = ledger.predict(cid, fid, metric, step, temp_f, value, low, high, model)
                        st.success(f"Logged prediction #{e['seq']}, hash {e['hash'][:12]}")

        else:  # Run
            cur_samples = store.current("samples")
            if not cur_samples:
                st.info("Runs belong to a sample. Add the sample first.")
            else:
                up = st.file_uploader("Analyzer log (CSV)", type=["csv", "txt"], key="lab_upload")
                sname = {s["id"]: f"{s['id']} · {formulations[s['formulation_id']]['name']}" for s in cur_samples}
                c = st.columns(3)
                sid = c[0].selectbox("Sample", list(sname), format_func=sname.get)
                cid = c[1].selectbox("Campaign", ["(outside a campaign)"] + [x["id"] for x in campaigns])
                rig = c[2].text_input("Rig", campaigns[-1]["rig"] if campaigns else "bench-1")
                proto_id = st.selectbox("Protocol", list(add_pack.protocols), key="lab_run_proto")
                notes = st.text_input("Notes", key="lab_run_notes")
                if up is not None:
                    text = up.getvalue().decode()
                    proto = add_pack.protocol(proto_id)
                    try:
                        parsed = parse_analyzer_text(text, proto.analyzer.measured)
                        pts, rflags = derive_points(add_pack, proto, parsed)
                        vals = point_metrics(add_pack, proto, pts)
                        st.markdown(f"**Preview:** {len(parsed.rows)} readings, {len(pts)} points; "
                                    + ("all settled." if all(settled(x) for x in pts) else
                                       f"{sum(not settled(x) for x in pts)} still settling."))
                        for f in rflags:
                            st.warning(flag_text(f), icon=":material/report:")
                        st.dataframe(pd.DataFrame([{"Step": x["step"], "Setpoint (°F)": x["setpoint_F"],
                                                    **{k: round(v, 3) for k, v in mv.items() if np.isfinite(v)},
                                                    "Checks": "; ".join(flag_text(f) for f in issues(x)) or "settled"}
                                                   for x, mv in zip(pts, vals)]), hide_index=True)
                        if st.button("Save run", type="primary", icon=":material/save:"):
                            with tempfile.TemporaryDirectory() as tmp:
                                path = Path(tmp) / up.name
                                path.write_bytes(up.getvalue())
                                saved = ingest_run(store, add_pack, path, sample_id=sid, rig=rig, protocol_id=proto_id,
                                                   campaign_id=None if cid.startswith("(") else cid, notes=notes,
                                                   original_name=up.name)
                            st.success(f"Stored {saved['id']} ({saved['status']}), raw {saved['raw_sha256'][:12]}")
                    except ValueError as exc:
                        st.error(str(exc))
                with st.expander("File format"):
                    st.markdown(
                        "One row per analyzer reading, written by the rig logger. Header lines starting with `#` "
                        "can carry `rig`, `protocol`, `sample`, `operator` and `started_at`.\n\n"
                        "```\ntime_s,step,setpoint_F,mode,T_F,NO (ppm),NO2 (ppm),NH3 (ppm),N2O (ppm)\n"
                        "0,design,650,bypass,640.2,22.6,0.1,22.4,0.0\n```\n\n"
                        "`mode` is `bypass` while the gas skips the catalyst (the inlet reading) and `reactor` "
                        "otherwise. Each hold's last two minutes become one point."
                    )

store.close()
