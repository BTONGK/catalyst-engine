"""Results of catalyst screens: progress while one runs, then the ranking, the curves, the volcano and 3D views."""

from __future__ import annotations

import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from ase.io import read

from catalyst_engine.design import screening
from catalyst_engine.design.spec import ADSORBATES
from catalyst_engine.packs import list_packs
from catalyst_engine.ui import common, viewer

PALETTE = ["#0f766e", "#7c3aed", "#ea580c", "#2563eb", "#db2777", "#65a30d", "#0891b2", "#a16207"]

st.title("Results")

screens = screening.list_screens()
if not screens:
    st.info("Your screens appear here. Start one on the Design page and its results fill in as they come in.",
            icon=":material/science:")
    common.link("views/design.py", "Design a catalyst", ":material/add_circle:")
    st.stop()


def label_of(folder) -> str:
    job = screening.read_job(folder)
    req = job["request"]
    state = screening.read_status(folder).get("state", "")
    when = datetime.fromisoformat(req["created"]).strftime("%b %d, %H:%M")
    return f"{req['name']}  ·  {when}  ·  {state}"


names = [f.name for f in screens]
wanted = st.session_state.get("results_screen")
choice = st.selectbox("Screen", names, index=names.index(wanted) if wanted in names else 0,
                      format_func=lambda n: label_of(screening.screens_dir() / n), key="results_pick")
st.session_state["results_screen"] = choice
folder = screening.screens_dir() / choice
job = screening.read_job(folder)
req = job["request"]
if req["pack"] not in list_packs():
    st.error(f"This screen used the reaction '{req['pack_name']}'. Add it back on the Reactions page to see the results.")
    st.stop()
pack = common.pack(req["pack"])
spec = pack.spec.screening
metric = pack.spec.metrics[spec.metric]
ref_entry = next((m for m in req["materials"] if m.get("is_reference")), {})
ref_name = ref_entry.get("ref_name") or spec.reference.name or spec.reference.formula
chosen_face = {m["id"] for m in req["materials"] if m.get("facet")}
t_cmp = req["design_temperature_C"]


def fmt_ratio(r: float | None) -> str:
    if r is None or not np.isfinite(r):
        return ""
    if r >= 1000:
        return "Over 1,000× faster"
    if r <= 1e-3:
        return "Over 1,000× slower"
    if r >= 1.5:
        return f"{r:,.0f}× faster" if r >= 10 else f"{r:.1f}× faster"
    if r <= 1 / 1.5:
        return f"{1 / r:,.0f}× slower" if r <= 0.1 else f"{1 / r:.1f}× slower"
    return "About the same"


def fmt_range(lo: float, hi: float) -> str:
    """The likely range of a speed ratio; beyond 1,000× either way reads as 1,000×+."""
    def side(r: float) -> tuple[str, str]:
        x = r if r >= 1 else 1 / r
        text = "1,000×+" if x >= 1000 else f"{x:,.0f}×" if x >= 10 else f"{x:.1f}×"
        return text, "faster" if r >= 1 else "slower"
    (a, wa), (b, wb) = side(lo), side(hi)
    if a == b == "1,000×+":
        return f"Over 1,000× {wa}"
    if wa == wb == "slower":
        a, b = b, a  # the smaller slowdown first
    return f"{a} to {b} {wa}" if wa == wb else f"{a} {wa} to {b} {wb}"


def fmt_metric(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return ""
    return f"{100 * v:.0f}%" if metric.kind == "conversion" else f"{v:.3g}"


@st.cache_data(show_spinner=False)
def predictions(folder_name: str, stamp: float):
    f = screening.screens_dir() / folder_name
    return screening.predict(pack, screening.read_job(f), screening.read_results(f))


def progress_panel() -> str:
    status = screening.read_status(folder)
    state = status.get("state", "queued")
    if state in screening.RUNNING:
        done, total = status.get("done", 0), max(status.get("total", 0), 1)
        elapsed = time.time() - status.get("started", time.time())
        left = elapsed / done * (total - done) if done else None
        st.progress(min(done / total, 1.0), text=f"{status.get('message', 'Working')}  ·  {done} of {total} steps")
        st.caption(f"Running for {elapsed / 60:.0f} min" + (f", about {left / 60:.0f} min to go" if left else "")
                   + ". This page refreshes by itself; you can leave and come back.")
        if st.button("Stop this screen", icon=":material/stop_circle:"):
            screening.cancel(folder)
            st.rerun()
    elif state == "finished":
        took = status.get("finished", 0) - status.get("started", 0)
        st.success(f"Finished in {took / 60:.0f} min.", icon=":material/check_circle:")
    elif state == "cancelled":
        st.info("You stopped this screen. The results below cover what finished.", icon=":material/stop_circle:")
    else:
        st.error(f"The screen stopped: {status.get('message', '')}", icon=":material/error:")
        log = folder / "log.txt"
        if log.exists():
            with st.expander("Last lines of the log"):
                st.code("\n".join(log.read_text().splitlines()[-25:]))
    return state


def body() -> None:
    state = progress_panel()
    results_file = folder / "results.jsonl"
    stamp = results_file.stat().st_mtime if results_file.exists() else 0.0
    preds, curves = predictions(choice, stamp)
    ref = next((p for p in preds if p.is_reference), None)
    ranked = [p for p in preds if p.rank is not None]
    t_top = max(req["conditions"]["temperatures_C"])

    if ref is None or ref.descriptor is None:
        st.info(f"Predictions appear once {ref_name}, the catalyst to beat, is done. It runs first.",
                icon=":material/hourglass_top:")
    candidates = [p for p in ranked if not p.is_reference]
    if candidates:
        best = candidates[0]
        so_far = " so far" if state in screening.RUNNING else ""
        if ref.rank < best.rank:
            st.markdown(f"### {ref_name}, the catalyst to beat, leads{so_far}. Closest challenger: {best.formula}")
        else:
            st.markdown(f"### Best candidate{so_far}: {best.formula}")
        m = [c for _ in range(2) for c in st.columns(2)]  # 2 x 2, so long values stay readable on narrow screens
        m[0].metric(f"Predicted speed vs {ref_name}", fmt_ratio(best.rate_ratio), border=True,
                    help=f"Reaction rate at {t_cmp:.0f} °C relative to the catalyst to beat.")
        m[1].metric(f"{metric.kind.capitalize()} at {t_cmp:.0f} °C", fmt_metric(best.metric_at_design),
                    f"{ref_name}: {fmt_metric(ref.metric_at_design)}", delta_color="off", delta_arrow="off", border=True,
                    help=metric.description)
        m[2].metric("Light-off temperature", f"{best.t_half_C:.0f} °C" if best.t_half_C is not None else f"above {t_top:.0f} °C",
                    f"{ref_name}: {ref.t_half_C:.0f} °C" if ref.t_half_C is not None else None,
                    delta_color="off", delta_arrow="off", border=True,
                    help="The lowest temperature where it gets halfway to the best possible.")
        m[3].metric("Supply", best.supply, border=True)
        common.check_banner(spec.check, spec.check_note)
        values = [v for c in curves["materials"].values() for v in c["mid"]] + list(curves["best_possible"] or [])
        common.range_notes(pack, metric.kind, metric.species, min(req["conditions"]["temperatures_C"]), t_top,
                           float(np.nanmin(values)) if values else 0.0)
        spans = screening.leader_spans(curves, {p.material: ref_name if p.is_reference else p.formula for p in preds})
        if len(spans) > 1:
            st.info("The leader changes with temperature: " + "; ".join(
                f"{name} from {a:.0f} to {b:.0f} °C" for name, a, b in spans) + ". Heat moves the volcano's top "
                "toward materials that bind more strongly; the Volcano tab shows it at each temperature.",
                icon=":material/thermostat:")

    tabs = st.tabs(["Ranking", "Performance curves", "Volcano", "3D structures", "Details"])
    with tabs[0]:
        rows = []
        for p in preds:
            rows.append({
                "Rank": "" if p.rank is None else str(p.rank), "Material": f"{ref_name} (to beat)" if p.is_reference else p.label, "Speed vs " + ref_name: "Reference" if p.is_reference else fmt_ratio(p.rate_ratio),
                "Likely range": "" if p.is_reference or p.rate_range is None else fmt_range(*p.rate_range),
                "Speed ratio": p.rate_ratio,  # the exact number, for the CSV
                f"At {t_cmp:.0f} °C": fmt_metric(p.metric_at_design),
                "Light-off": f"{p.t_half_C:.0f} °C" if p.t_half_C is not None else ("" if p.rank is None else f"above {t_top:.0f} °C"),
                "Binding (eV)": None if p.descriptor is None else round(p.descriptor, 2),
                "Stability (eV/atom)": None if p.e_hull is None else round(p.e_hull, 3),
                "Supply": p.supply,
                "Status": "Done" if p.complete else ("Failed: " + p.error if p.error else "Waiting"),
            })
        table = pd.DataFrame(rows)
        if (table["Status"] == "Done").all():
            table = table.drop(columns="Status")
        st.dataframe(table, hide_index=True, width="stretch", height=36 * (len(table) + 1) + 4,
                     column_config={"Rank": st.column_config.TextColumn(width=50),
                                    "Binding (eV)": st.column_config.NumberColumn(width="small"),
                                    "Stability (eV/atom)": st.column_config.NumberColumn("Stability", width="small",
                                                                                         help="eV/atom above the most stable form"),
                                    "Material": st.column_config.TextColumn(width="medium"),
                                    "Likely range": st.column_config.TextColumn(width="medium"),
                                    "Speed ratio": None})
        st.caption(f"Speed: the predicted reaction rate at {t_cmp:.0f} °C relative to {ref_name}. Likely range: from "
                   f"the model's typical error in binding energy (±{spec.sigma_eV:.2f} eV) and from how far the "
                   "volcano's top moves with temperature. Binding: how strongly the "
                   "surface holds the molecules that decide this reaction (more negative is stronger). Stability: "
                   "0 is the most stable form of that composition.")
    with tabs[1]:
        if not curves["materials"]:
            st.caption("Curves appear once the catalyst to beat is done.")
        else:
            order = [p for p in preds if p.material in curves["materials"]]
            default = [p.material for p in order if not p.is_reference][:5]
            show = st.multiselect("Materials to show", [p.material for p in order], default=default,
                                  format_func={p.material: p.formula for p in order}.get, key=f"res_show_{choice}")
            t = curves["T_C"]
            frames = [pd.DataFrame({"T": t, "mid": curves["best_possible"], "lo": np.nan, "hi": np.nan,
                                    "Material": "Best possible"})]
            for p in order:
                if p.is_reference or p.material in show:
                    c = curves["materials"][p.material]
                    frames.append(pd.DataFrame({"T": t, "mid": c["mid"], "lo": c["lo"], "hi": c["hi"],
                                                "Material": f"{ref_name} (to beat)" if p.is_reference else p.formula}))
            df = pd.concat(frames)
            names_ = list(dict.fromkeys(df["Material"]))
            colors = [common.ACCENT if n == "Best possible" else common.REFERENCE_GRAY if n.endswith("(to beat)")
                      else PALETTE[(i % (len(PALETTE) - 1)) + 1] for i, n in enumerate(names_)]
            scale = alt.Scale(domain=names_, range=colors)
            color = alt.Color("Material:N", scale=scale, legend=alt.Legend(orient="bottom", title=None))
            x_axis = alt.X("T:Q", title="Temperature (°C)", scale=alt.Scale(domain=[float(t[0]), float(t[-1])], nice=False))
            band = alt.Chart(df.dropna(subset=["lo"])).mark_area(opacity=0.08).encode(
                x=x_axis, y="lo:Q", y2="hi:Q", color=alt.Color("Material:N", scale=scale, legend=None))
            line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
                x=x_axis, y=alt.Y("mid:Q", title=f"{metric.kind.capitalize()} ({metric.description or spec.metric})"
                                  if len(metric.description or "") < 40 else metric.kind.capitalize()),
                color=color,
                strokeDash=alt.condition(alt.datum.Material == "Best possible", alt.value([6, 4]), alt.value([1, 0])))
            rule = alt.Chart(pd.DataFrame({"T": [t_cmp]})).mark_rule(color="#94a3b8", strokeDash=[2, 3]).encode(x="T:Q")
            # separate color scales (same colors) so only the lines draw the legend
            st.altair_chart((band + line + rule).resolve_scale(color="independent").properties(height=380),
                            width="stretch")
            limit = ("on this honeycomb, where gas reaching the channel walls sets the limit"
                     if screening.is_monolith(req["conditions"].get("shape")) else "(equilibrium)")
            st.caption("Each line is the predicted performance in a reactor under your conditions; the shaded band is "
                       f"the likely range. The dashed green line is the most any catalyst could reach {limit}. "
                       "The dotted line marks the comparison temperature.")
    with tabs[2]:
        if ref is None or ref.descriptor is None:
            st.caption("The volcano appears once the catalyst to beat is done.")
        else:
            temps = sorted({round(t) for t in curves["T_C"]} | {round(t_cmp)})
            t_v = st.select_slider("Draw the volcano at", temps, value=round(t_cmp), format_func=lambda t: f"{t} °C",
                                   key=f"res_volcano_t_{choice}",
                                   help="The volcano's top moves with temperature; the dots move with it.")
            at_t = screening.step_for({**req["conditions"], "temperatures_C": [float(t_v)]})
            move = float(screening.peak_moves(pack, at_t, [t_v + 273.15])[0])
            kt = screening.K_B * (t_v + 273.15)
            pts = pd.DataFrame([{"Binding (eV)": p.descriptor, "Material": p.formula,
                                 "Speed": float(np.exp(-spec.barrier_shift(p.descriptor, ref.descriptor, move) / kt)),
                                 "lo": p.descriptor - spec.sigma_descriptor, "hi": p.descriptor + spec.sigma_descriptor,
                                 "Kind": "Catalyst to beat" if p.is_reference else "Candidate"}
                                for p in preds if p.rate_ratio is not None]).sort_values("Binding (eV)")
            top = ref.descriptor + spec.optimum_shift_eV + move
            x_lo, x_hi = min(pts["lo"].min(), top) - 0.25, max(pts["hi"].max(), top) + 0.25
            x, y = screening.volcano_line(pack, ref.descriptor, span=max(top - x_lo, x_hi - top), n=161,
                                          t_design_C=t_v, move=move)
            # decades shown: from just above the peak down to the slowest material, floored at a trillionth
            hi_dec = int(np.ceil(np.log10(y.max()))) + 1
            lo_dec = max(-12, int(np.floor(np.log10(max(1e-30, pts["Speed"].min())))) - 1)
            floor = 10.0 ** lo_dec
            pts["Shown"] = pts["Speed"].clip(lower=floor * 2)
            pts["Off the scale"] = pts["Speed"] < floor * 2
            step = 1 if hi_dec - lo_dec <= 7 else 3
            ticks = [10.0 ** d for d in range(hi_dec, lo_dec - 1, -1) if d % step == 0]
            names_ = {-3: "1/1,000", -6: "1/million", -9: "1/billion", -12: "1/trillion"}
            label = "".join(f"abs(log(datum.value)/LN10 - ({d})) < 0.01 ? '{(f'{10 ** d:,}×' if d >= 0 else names_.get(d, f'1/{10 ** -d:,}'))}' : "
                            for d in range(hi_dec, lo_dec - 1, -1)) + "''"
            x_enc = alt.X("Binding (eV):Q", scale=alt.Scale(domain=[x_lo, x_hi], nice=False),
                          title="Binding energy (eV), more negative = holds more strongly")
            y_enc = alt.Y("Speed:Q", title=f"Predicted speed vs {ref_name}",
                          scale=alt.Scale(type="log", domain=[floor, 10.0 ** hi_dec]),
                          axis=alt.Axis(values=ticks, labelExpr=label, grid=True))
            line = alt.Chart(pd.DataFrame({"Binding (eV)": x, "Speed": y})).mark_line(
                color="#94a3b8", strokeWidth=2, clip=True).encode(x=x_enc, y=y_enc)
            err = alt.Chart(pts).mark_rule(color="#cbd5e1", strokeWidth=3, clip=True).encode(x="lo:Q", x2="hi:Q", y="Shown:Q")
            dots = alt.Chart(pts).mark_point(size=120, opacity=0.95, filled=True, clip=True).encode(
                x="Binding (eV):Q", y=alt.Y("Shown:Q", scale=alt.Scale(type="log", domain=[floor, 10.0 ** hi_dec])),
                shape=alt.Shape("Off the scale:N", scale=alt.Scale(domain=[False, True], range=["circle", "triangle-down"]),
                                legend=None),
                color=alt.Color("Kind:N", scale=alt.Scale(domain=["Catalyst to beat", "Candidate"],
                                                          range=[common.REFERENCE_GRAY, common.ACCENT]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Material", alt.Tooltip("Binding (eV):Q", format=".2f"), alt.Tooltip("Speed:Q", format=".3g")])
            # labels beside each dot, alternating left and right from the fastest down on each side of the peak, so
            # close neighbors along a slope read apart; on the bottom edge they sit above
            pts["place"] = "above"
            for side in (pts["Binding (eV)"] <= top, pts["Binding (eV)"] > top):
                order = pts[side & ~pts["Off the scale"]].sort_values("Speed", ascending=False).index
                pts.loc[order, "place"] = np.where(np.arange(len(order)) % 2 == 0, "left", "right")
            offsets = {"above": {"dy": -14}, "left": {"dx": -9, "align": "right"}, "right": {"dx": 9, "align": "left"}}
            text = alt.layer(*[alt.Chart(pts[pts["place"] == place]).mark_text(
                fontSize=12, color="#475569", clip=True, **kw).encode(
                x="Binding (eV):Q", y="Shown:Q", text="Material:N") for place, kw in offsets.items()])
            corners = pd.DataFrame({"Binding (eV)": [x_lo + 0.03, x_hi - 0.03], "Speed": [10.0 ** (hi_dec - 0.4)] * 2,
                                    "label": ["← holds too strongly", "holds too weakly →"]})
            sides = alt.layer(*[alt.Chart(corners.iloc[[i]]).mark_text(align=a, fontSize=12, color="#64748b").encode(
                x="Binding (eV):Q", y="Speed:Q", text="label:N") for i, a in enumerate(("left", "right"))])
            st.altair_chart((line + err + dots + text + sides).properties(height=420), width="stretch")
            held = " + ".join(f"{w:g}×{a}" if w != 1 else a for a, w in spec.descriptor.items())
            anchor = pack.network.t_ref - 273.15
            where = (f"At {t_v} °C the top sits {abs(move):.2f} eV toward "
                     f"{'stronger' if move < 0 else 'weaker'} binding than at {anchor:.0f} °C, where the reaction sets it: "
                     f"{'heat' if move < 0 else 'cooling'} changes how readily the gas sticks. " if abs(move) >= 0.005
                     else f"The top sits where the reaction sets it at {anchor:.0f} °C; heat moves it toward stronger "
                          "binding, cooling toward weaker. ") if spec.peak_gas else ""
            st.caption(f"The volcano: materials that hold {held} too weakly (right) react slowly because the molecules "
                       f"leave before reacting; those that hold it too strongly (left) clog up. The best sit near the "
                       f"peak. {where}Bars show the binding energy's likely range"
                       + (", and triangles on the bottom edge mark materials slower still (hover for the number). "
                          if pts["Off the scale"].any() else ". ") + " ".join(spec.basis.split()))
    with tabs[3]:
        done = [p for p in preds if p.slab is not None]
        if not done:
            st.caption("Structures appear as each material finishes.")
        else:
            a, b = st.columns([1, 2], gap="large")
            with a:
                pick = st.selectbox("Material", [p.material for p in done],
                                    format_func={p.material: p.label for p in done}.get, key=f"res_3d_{choice}")
                p = next(q for q in done if q.material == pick)
                zeolite = p.slab.get("kind") == "framework"
                bare = "Empty site" if zeolite else "Clean surface"
                views = {bare: (p.slab["structure"], 0)}
                best_view = bare
                for ads, bnd in p.binding.items():
                    for site in bnd["sites"]:
                        tag = f"{ads} on the {site['site']} site ({site['E']:+.2f} eV)" + (f", {site['note']}" if site["note"] else "")
                        views[tag] = (site["structure"], site["n_ads"])
                        if best_view == bare and site["structure"] == bnd["structure"]:
                            best_view = tag
                view = st.radio("Show", list(views), index=list(views).index(best_view), key=f"res_view_{choice}_{pick}")
                if zeolite:
                    st.caption(f"Zeolite site: the framework's whole cell, {p.slab['n_atoms']} atoms, every atom free to "
                               "move. Larger spheres are the molecule. The view shows eight copies of the cell, so each "
                               "molecule shows up in every copy. Drag to rotate, scroll to zoom.")
                else:
                    facet = tuple(p.slab.get("facet"))
                    face = (f"the face you chose, Miller indices {facet}" if p.material in chosen_face
                            else f"the close-packed face, Miller indices {facet}")
                    st.caption(f"Surface: {face} in the crystal's own cell, {p.slab['n_atoms']} atoms, bottom half "
                               "held fixed. Larger spheres are the molecule. The view repeats the computed patch 2 × 2, "
                               "so each molecule shows up four times. Drag to rotate, scroll to zoom.")
                for ads, bnd in p.binding.items():
                    if bnd["E"] is not None:
                        st.markdown(f"**{ADSORBATES[ads].capitalize()} ({ads})**: binds at {bnd['E']:+.2f} eV on the "
                                    f"{bnd['site']} site, best of {bnd['tried']} tried.")
            with b:
                path, n_ads = views[view]
                atoms = read(folder / path)
                viewer.show(atoms, n_highlight=n_ads, height=440, surface=not zeolite, span=18.0 if zeolite else 9.0)
                st.html(f"<div style='font-size:0.85rem'>{viewer.element_colors(atoms.get_chemical_symbols())}</div>")
    with tabs[4]:
        site_rows = []
        for p in preds:
            for ads, bnd in p.binding.items():
                for s in bnd["sites"]:
                    site_rows.append({"Material": p.label, "Molecule": ads, "Site": s["site"], "Binding (eV)": round(s["E"], 3),
                                      "Settled": "yes" if s["converged"] else "stopped at the step limit",
                                      "Note": s["note"] or "kept"})
        if site_rows:
            st.markdown("**Every site tried**")
            st.dataframe(pd.DataFrame(site_rows), hide_index=True, width="stretch")
            st.caption("Desorbed: the molecule drifted off the surface. Dissociated: it broke apart. Both are left out "
                       "of the lowest binding energy. The surface beneath settled lower: the surface relaxed further "
                       "under the molecule, so its binding energy is set against that surface with the molecule lifted "
                       "off, and counts only the bond.")
            c = st.columns(2)
            c[0].download_button("Download the ranking (CSV)", pd.DataFrame(rows).to_csv(index=False),
                                 file_name=f"{choice}-ranking.csv", icon=":material/download:")
            c[1].download_button("Download every site (CSV)", pd.DataFrame(site_rows).to_csv(index=False),
                                 file_name=f"{choice}-sites.csv", icon=":material/download:")
        cond = req["conditions"]
        st.markdown("**Settings**")
        st.markdown(
            f"- Reaction: {pack.spec.name}\n"
            f"- Gas in: {', '.join(f'{k} {v}' for k, v in cond['feed'].items())}\n"
            f"- {min(cond['temperatures_C']):.0f}–{max(cond['temperatures_C']):.0f} °C, {cond['pressure_bar']:g} bar, "
            f"space velocity {cond['ghsv_per_h']:,.0f} 1/h\n"
            f"- Catalyst form: {common.form_text(cond.get('shape'))}\n"
            f"- Care level: {req['thoroughness']}; UMA mode: {req['task']}\n"
            f"- Folder: `{folder}`")
        with st.expander("How the predictions are made"):
            st.markdown(
                f"1. **Binding.** For each material, the app cuts its densest crystal face, places "
                f"{', '.join(ADSORBATES[a] for a in spec.descriptor)} on its top, bridge and hollow sites, and lets "
                "Meta's UMA model relax each arrangement. The lowest energy is the binding energy.\n"
                f"2. **Volcano.** {' '.join(spec.basis.split())} Each eV of binding away from the peak adds "
                f"{spec.slope_weak:g} eV (too weak) or {spec.slope_strong:g} eV (too strong) to the reaction's barrier.\n"
                f"3. **Reactor.** The barrier change sets each material's rate constant relative to {ref_name}, and the "
                "reactor model turns rates into performance under your conditions, with equilibrium where it applies.\n\n"
                f"Read the ranking as a shortlist: the best few candidates are the ones worth making and testing.")


if screening.read_status(folder).get("state") in screening.RUNNING:
    @st.fragment(run_every=5)
    def live() -> None:
        body()
        if screening.read_status(folder).get("state") not in screening.RUNNING:
            st.rerun()

    live()
else:
    body()
