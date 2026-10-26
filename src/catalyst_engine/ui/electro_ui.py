"""The electrocatalysis section's Design and Results pages: pick an electrochemical reaction and an operating
potential, screen materials, and read limiting potentials, overpotentials, free-energy diagrams and the volcano."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.design import electro, screening
from catalyst_engine.design.electro import ADSORBATES
from catalyst_engine.design.uma_worker import REFERENCE
from catalyst_engine.ui import common, screen_ui

LEVELS = {"Quick": "quick", "Standard": "standard", "Thorough": "thorough"}
TASKS = {"oc20": "Catalysis (trained on surfaces with molecules)", "omat": "Materials (trained on bulk crystals)"}
PALETTE = ["#0f766e", "#7c3aed", "#ea580c", "#2563eb", "#db2777", "#65a30d", "#0891b2", "#a16207"]


@st.cache_resource(show_spinner=False)
def _spec(spec_id: str, mtime: float) -> electro.ElectroSpec:
    return electro.load_electro(spec_id)


def spec(spec_id: str) -> electro.ElectroSpec:
    return _spec(spec_id, (electro.ELECTRO_DIR / f"{spec_id}.yaml").stat().st_mtime)


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
    def side(r: float) -> tuple[str, str]:
        x = r if r >= 1 else 1 / r
        text = "1,000×+" if x >= 1000 else f"{x:,.0f}×" if x >= 10 else f"{x:.1f}×"
        return text, "faster" if r >= 1 else "slower"
    (a, wa), (b, wb) = side(lo), side(hi)
    if a == b == "1,000×+":
        return f"Over 1,000× {wa}"
    if wa == wb == "slower":
        a, b = b, a
    return f"{a} to {b} {wa}" if wa == wb else f"{a} {wa} to {b} {wb}"


def descriptor_text(s: electro.ElectroSpec) -> str:
    parts = []
    for a, w in s.descriptor.items():
        parts.append(f"{'+' if w > 0 and parts else ('−' if w < 0 else '')}{'' if abs(w) == 1 else f'{abs(w):g}×'}{a}*")
    return " ".join(parts) + " free energy"


def screen_label(folder: Path) -> str:
    req = screening.read_job(folder)["request"]
    state = screening.read_status(folder).get("state", "")
    when = datetime.fromisoformat(req["created"]).strftime("%b %d, %H:%M")
    return f"{req['name']}  ·  {when}  ·  {state}"


# ---------------------------------------------------------------- design page

GAS_WORDS = {"H2O": "water", "H2": "hydrogen", "N2": "nitrogen", "CO": "carbon monoxide", "NO": "nitric oxide",
             "NH3": "ammonia"}


def joined(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def gases_text(s: electro.ElectroSpec) -> str:
    """The gas molecules the worker sets this reaction's binding energies against, in plain words."""
    used = {g for a in s.adsorbates for g in REFERENCE[a]}
    return joined([w for g, w in GAS_WORDS.items() if g in used])


def chemical_text(s: electro.ElectroSpec) -> str:
    """The steps of the path that move zero electrons, in one sentence; empty for a reaction with none."""
    if not s.chemical_steps:
        return ""
    names = s.step_names()
    one = len(s.chemical_steps) == 1
    return (f"{joined([f'{names[c.step]}' for c in s.chemical_steps])} {'moves' if one else 'move'} zero electrons, "
            f"so the potential leaves {'it' if one else 'them'} alone: {'its' if one else 'each one’s'} hurdle is "
            "its free energy, or its barrier where the file gives one, at every potential.")


def routes_text(s: electro.ElectroSpec) -> str:
    """The two routes of a reaction with a chemical step, in one sentence."""
    a = s.activation
    return (f"Two routes run side by side: “{s.route}” and “{a.route}”, where {a.label} makes {a.joins_at}* "
            f"straight away at a barrier that follows the {a.from_}* binding. The faster route carries the current.")


def design_page(advanced: bool, status: dict) -> None:
    ids = electro.list_electro()
    specs = {i: spec(i) for i in ids}
    cands: dict[str, dict] = st.session_state.setdefault("electro_candidates", {})

    with st.container(border=True):
        st.markdown("### 1. What should the electrode do?")
        wanted = st.session_state.pop("electro_pack_wanted", None)
        if wanted in ids:
            st.session_state["electro_pack"] = wanted
        elif st.session_state.get("electro_pack") not in ids:
            st.session_state["electro_pack"] = ids[0]
        pid = st.selectbox("Reaction", ids, format_func=lambda i: f"{specs[i].name}  ·  {specs[i].category}  ·  "
                           f"{common.check_short(specs[i].check)}", key="electro_pack",
                           help="Every electrochemical reaction file. Each one's check, against published DFT or "
                                "measurements, shows beside it: checked, missed, or awaiting published data.")
        s = specs[pid]
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(f"**{s.summary or s.name}**")
            st.markdown(f"<span class='ce-eq'>{s.equation}</span> &nbsp; :gray[{s.direction}, {s.electrons} electrons, "
                        f"equilibrium at {s.equilibrium_V:.2f} V vs RHE]", unsafe_allow_html=True)
            st.markdown("Intermediates along the path: " + " → ".join(i.label for i in s.intermediates))
            if s.activation:
                st.markdown(routes_text(s))
        with right:
            st.markdown(f"**Catalyst to beat:** {s.reference.name or s.reference.formula}")
            st.markdown(f"**Ranks materials by:** predicted speed at your potential, then overpotential. The "
                        f"{descriptor_text(s)} places each one on the volcano.")
            common.check_line(s.check, s.check_note)
            with st.expander("Why this works"):
                st.write(" ".join(s.basis.split()))

    with st.container(border=True):
        st.markdown("### 2. At what potential?")
        st.caption("Potentials are on the reversible hydrogen electrode (RHE) scale, which already accounts for "
                   "pH: the equilibrium and limiting potentials read the same at any pH. The temperature is 25 °C.")
        lo, hi = (s.equilibrium_V - 1.0, s.equilibrium_V + 0.3) if s.direction == "reduction" \
            else (s.equilibrium_V - 0.3, s.equilibrium_V + 1.0)
        u_cmp = st.slider("Compare catalysts at (V vs RHE)", float(round(lo, 2)), float(round(hi, 2)),
                          float(s.compare_V), 0.01, key=f"electro_u_{pid}",
                          help="The operating potential for the headline speed numbers. Each material's limiting "
                               "potential and overpotential read the same at any potential.")
        gap = (s.equilibrium_V - u_cmp) if s.direction == "reduction" else (u_cmp - s.equilibrium_V)
        st.markdown(f"That is **{abs(gap):.2f} V {'past' if gap >= 0 else 'short of'} the equilibrium potential** "
                    f"({s.equilibrium_V:.2f} V). A material whose limiting potential lies "
                    f"{'above' if s.direction == 'reduction' else 'below'} {u_cmp:.2f} V runs every step that "
                    f"moves an electron{' on its better route' if s.activation else ''} downhill there; the others pay "
                    "for their largest hurdle" + (", and a chemical step uphill holds back every material at every "
                                                  "potential" if s.chemical_steps else "") + ".")
        conditions = {"compare_V": float(u_cmp)}

    screen_ui.candidates_section(pid=pid, reference_formula=s.reference.formula,
                                 reference_name=s.reference.name, suggested=s.suggested_elements, cands=cands,
                                 advanced=advanced, status=status, prefix="electro")

    with st.container(border=True):
        st.markdown("### 4. Run the screen")
        c = st.columns(2)
        level = c[0].segmented_control("How careful", list(LEVELS), default="Standard", key="electro_level",
                                       help="Quick tries 2 spots per molecule on each surface, Standard 3, Thorough 5 "
                                            "with tighter settling.")
        name = c[1].text_input("Name this screen", f"{s.name} screen", key=f"electro_name_{pid}")
        ref_choice, slab = "", None
        if advanced:
            c = st.columns(2)
            task = c[0].selectbox("UMA mode", list(TASKS), index=list(TASKS).index(s.task), format_func=TASKS.get,
                                  key=f"electro_task_adv_{pid}",
                                  help="Catalysis mode suits metal surfaces, the surfaces the checks cover. On oxides the "
                                       "materials mode comes closest, 0.18–0.24 eV from oxide DFT, wider than "
                                       "the error bar (Manual, Methodology, limits).")
            ref_choice = c[1].selectbox(
                "Catalyst to beat", [""] + list(cands), key=f"electro_ref_{pid}",
                format_func=lambda k: f"{s.reference.name or s.reference.formula} (this reaction's)" if not k
                else cands[k]["label"],
                help="Every material is ranked against this one. Pick a material of the same family as your "
                     "candidates, such as an oxide when screening oxides.")
            c = st.columns(2)
            thickness = c[0].slider("Slab thickness (Å)", 7, 14, 7, key="electro_slab_t")
            width = c[1].slider("Slab width (Å)", 7, 12, 7, key="electro_slab_w")
            slab = {"thickness": float(thickness), "width": float(width)} if (thickness, width) != (7, 7) else None
        else:
            task = s.task
            st.caption(f"UMA mode: {TASKS[task]}. Advanced lets you change it.")
        level = level or "Standard"
        ref_elements = [s.reference.formula] if s.reference.formula in screen_ui.METALS else ["O"]
        secs = screening.seconds_estimate([{"elements": ref_elements}, *cands.values()], len(s.adsorbates), LEVELS[level])
        n_mat, minutes = len(cands) + 1, max(1, round(secs / 60))
        st.markdown(f"**{n_mat} material{'s' * (n_mat > 1)}** (the catalyst to beat included), {len(s.adsorbates)} "
                    f"intermediate{'s' * (len(s.adsorbates) > 1)} each. Estimated time on this computer: "
                    f"**about {minutes} minute{'s' * (minutes > 1)}**. It runs in the background; you can leave this page.")
        ready = status["uma"]["ready"] and bool(cands)
        if st.button("Start the screen", type="primary", icon=":material/play_arrow:", disabled=not ready,
                     key="electro_start"):
            with st.spinner("Collecting crystal structures..."):
                try:
                    entries = screen_ui.reference_entries(s.reference.formula, s.reference.name, cands, ref_choice)
                    folder = electro.create_screen(s, name, conditions, entries, LEVELS[level], task, slab=slab)
                    screening.launch(folder)
                except Exception as exc:  # a Materials Project problem or a bad entry; the page stays usable
                    st.error(str(exc))
                else:
                    st.session_state["electro_results_screen"] = folder.name
                    st.switch_page("views/electro_results.py")
        if not cands:
            st.caption("Add at least one material in step 3 to start.")


# ---------------------------------------------------------------- results page

@st.cache_data(show_spinner=False)
def _predictions(folder_name: str, stamp: float):
    f = electro.screens_dir() / folder_name
    job = screening.read_job(f)
    return electro.predict(spec(job["request"]["pack"]), job, screening.read_results(f))


def progress_panel(folder: Path) -> str:
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


def results_body(folder: Path, choice: str) -> None:
    job = screening.read_job(folder)
    req = job["request"]
    state = progress_panel(folder)
    s = spec(req["pack"])
    ref_entry = next((m for m in req["materials"] if m.get("is_reference")), {})
    ref_name = ref_entry.get("ref_name") or s.reference.name or s.reference.formula
    chosen_face = {m["id"] for m in req["materials"] if m.get("facet")}
    u_cmp = float(req["conditions"]["compare_V"])
    results_file = folder / "results.jsonl"
    stamp = results_file.stat().st_mtime if results_file.exists() else 0.0
    preds, curves = _predictions(choice, stamp)
    ref = next((p for p in preds if p.is_reference), None)
    ranked = [p for p in preds if p.rank is not None]
    better = "higher" if s.direction == "reduction" else "lower"

    if ref is None or not ref.complete:
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
        m = [c for _ in range(2) for c in st.columns(2)]
        m[0].metric("Overpotential", f"{best.overpotential_V:.2f} V", f"{ref_name}: {ref.overpotential_V:.2f} V",
                    delta_color="off", delta_arrow="off", border=True,
                    help="How far the limiting potential sits from the equilibrium potential. Smaller is better.")
        m[1].metric("Limiting potential", f"{best.limiting_V:.2f} V vs RHE", f"{ref_name}: {ref.limiting_V:.2f} V",
                    delta_color="off", delta_arrow="off", border=True,
                    help=f"Where every step that moves an electron{' on the better route' if s.activation else ''} "
                         "runs downhill; "
                         f"{better} is better for this reaction.")
        m[2].metric(f"Predicted speed vs {ref_name} at {u_cmp:.2f} V", fmt_ratio(best.rate_ratio), border=True,
                    help="From the largest hurdle at that potential: the largest step still uphill"
                         + (", or the chemical barrier, on each route, with the faster route counting"
                            if s.activation else "") + " (Sabatier analysis).")
        m[3].metric("Supply", best.supply, border=True)
        common.check_banner(s.check, s.check_note)
    missing = [p for p in preds if p.missing]
    if missing:
        st.info("On " + ", ".join(f"{p.formula} ({', '.join(p.missing)})" for p in missing) + " an intermediate "
                "broke apart or drifted off at every site tried, so that material's path stays open. "
                "A molecule that splits binds the surface too strongly to stay whole; one that drifts off binds "
                "too weakly to stay. The Details tab lists each site.", icon=":material/info:")

    tabs = st.tabs(["Ranking", "Activity vs potential", "Free-energy diagram", "Volcano", "3D structures", "Details"])
    with tabs[0]:
        rows = []
        for p in preds:
            rows.append({
                "Rank": "" if p.rank is None else str(p.rank),
                "Material": f"{ref_name} (to beat)" if p.is_reference else p.label,
                "Overpotential (V)": None if p.overpotential_V is None else round(p.overpotential_V, 2),
                "Limiting potential (V vs RHE)": None if p.limiting_V is None else round(p.limiting_V, 2),
                "Likely range (V)": "" if p.limiting_range is None else f"{p.limiting_range[0]:.2f} to {p.limiting_range[1]:.2f}",
                "Limiting step": "" if p.limiting_step is None else s.step_names()[p.limiting_step],
                **({f"{s.activation.label} barrier (eV)": None if p.barrier_eV is None else round(p.barrier_eV, 2),
                    f"Largest hurdle at {u_cmp:.2f} V (eV)": None if p.hurdle_eV is None else round(p.hurdle_eV, 2),
                    f"Faster route at {u_cmp:.2f} V": p.route}
                   if s.activation else {}),
                f"Speed vs {ref_name} at {u_cmp:.2f} V": "Reference" if p.is_reference else fmt_ratio(p.rate_ratio),
                "Speed range": "" if p.is_reference or p.rate_range is None else fmt_range(*p.rate_range),
                "Speed ratio": p.rate_ratio,
                **{f"{a}* (eV)": None if a not in p.free_energies else round(p.free_energies[a], 2) for a in s.adsorbates},
                "Stability (eV/atom)": None if p.e_hull is None else round(p.e_hull, 3),
                "Supply": p.supply,
                "Status": "Done" if p.complete else ("Failed: " + p.error if p.error else
                                                     ("Incomplete: " + ", ".join(p.missing) if p.missing else "Waiting")),
            })
        table = pd.DataFrame(rows)
        if (table["Status"] == "Done").all():
            table = table.drop(columns="Status")
        st.dataframe(table, hide_index=True, width="stretch", height=36 * (len(table) + 1) + 4,
                     column_config={"Rank": st.column_config.TextColumn(width=50), "Speed ratio": None,
                                    "Material": st.column_config.TextColumn(width="medium"),
                                    "Stability (eV/atom)": st.column_config.NumberColumn("Stability", width="small")})
        a = s.activation
        st.caption(f"Overpotential: how far the limiting potential, where every step that moves an electron on the "
                   f"{'better route' if a else 'path'} runs downhill, sits from {s.equilibrium_V:.2f} V. "
                   + (chemical_text(s) + " " if s.chemical_steps else "")
                   + (f"{routes_text(s)} On each route the hurdle at {u_cmp:.2f} V is its largest step still uphill, "
                      f"or its barrier when that is higher (Nørskov et al. 2004); the ranking sorts by the faster "
                      "route's hurdle, then by overpotential. The limiting potential counts the second route once "
                      "its barrier falls to zero. " if a else
                      f"The ranking sorts by the largest step still uphill at {u_cmp:.2f} V, then by overpotential. ")
                   + f"Likely ranges: the exact ends when every binding energy moves within ±{electro.SIGMA_EV:.2f} eV, "
                   "the model's typical error, all at once. Free energies: each intermediate at 0 V vs RHE, "
                   "against water and hydrogen gas; the ideal catalyst would hold them at "
                   + ", ".join(f"{a}* {v:.2f}" for a, v in s.ideal().items())
                   + " eV. Speed: from the hurdle at the comparison potential. Surface oxidation and "
                   "dissolution at the working potential sit outside this picture (Manual, Methodology tab).")
    with tabs[1]:
        if not curves["materials"]:
            st.caption("Curves appear once the catalyst to beat is done.")
        else:
            order = [p for p in preds if p.material in curves["materials"]]
            default = [p.material for p in order if not p.is_reference][:5]
            show = st.multiselect("Materials to show", [p.material for p in order], default=default,
                                  format_func={p.material: p.formula for p in order}.get, key=f"eres_show_{choice}")
            u = np.array(curves["U_V"])
            ref_up = np.array(curves["materials"][curves["reference"]]["uphill"])
            frames = [pd.DataFrame({"U": u, "ratio": np.exp((ref_up - np.array(curves["ideal"])) / electro.K_T),
                                    "Material": "Ideal catalyst"})]
            for p in order:
                if p.is_reference or p.material in show:
                    frames.append(pd.DataFrame({"U": u, "ratio": curves["materials"][p.material]["ratio"],
                                                "Material": f"{ref_name} (to beat)" if p.is_reference else p.formula}))
            df = pd.concat(frames)
            df["ratio"] = df["ratio"].clip(lower=1e-12, upper=1e12)
            names_ = list(dict.fromkeys(df["Material"]))
            colors = [common.ACCENT if n == "Ideal catalyst" else common.REFERENCE_GRAY if n.endswith("(to beat)")
                      else PALETTE[(i % (len(PALETTE) - 1)) + 1] for i, n in enumerate(names_)]
            line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
                x=alt.X("U:Q", title="Potential (V vs RHE)", scale=alt.Scale(domain=[float(u[0]), float(u[-1])], nice=False)),
                y=alt.Y("ratio:Q", title=f"Speed vs {ref_name}", scale=alt.Scale(type="log")),
                color=alt.Color("Material:N", scale=alt.Scale(domain=names_, range=colors),
                                legend=alt.Legend(orient="bottom", title=None)),
                strokeDash=alt.condition(alt.datum.Material == "Ideal catalyst", alt.value([6, 4]), alt.value([1, 0])),
                tooltip=["Material:N", alt.Tooltip("U:Q", format=".2f"), alt.Tooltip("ratio:Q", format=".3g")])
            rules = alt.Chart(pd.DataFrame({"U": [u_cmp, s.equilibrium_V], "what": ["comparison", "equilibrium"]})).mark_rule(
                color="#94a3b8", strokeDash=[2, 3]).encode(x="U:Q", tooltip="what:N")
            st.altair_chart((line + rules).properties(height=380), width="stretch")
            st.caption(f"Each line is a material's predicted speed relative to {ref_name} at the same potential: "
                       "the speed falls by about 50× for every 0.1 eV of the largest hurdle. Where every "
                       f"step of both {'faster routes' if s.activation else 'paths'} runs downhill the lines meet at 1"
                       + (f", and a line runs flat wherever its faster route waits on the {s.activation.label} "
                          "barrier, which holds at every potential" if s.activation else "")
                       + ". The dashed green line is the ideal catalyst, "
                       "whose steps all turn downhill at the equilibrium potential. Dotted lines mark the "
                       "comparison and equilibrium potentials.")
    with tabs[2]:
        done = [p for p in preds if p.complete]
        if not done:
            st.caption("The diagram appears once a material is done.")
        else:
            c = st.columns([1, 1])
            pick = c[0].selectbox("Material", [p.material for p in done],
                                  format_func={p.material: p.label for p in done}.get, key=f"eres_fed_{choice}")
            p = next(q for q in done if q.material == pick)
            options = {"0 V": 0.0, f"equilibrium, {s.equilibrium_V:.2f} V": s.equilibrium_V,
                       f"its limiting potential, {p.limiting_V:.2f} V": p.limiting_V,
                       f"the comparison potential, {u_cmp:.2f} V": u_cmp}
            at = c[1].multiselect("Draw at", list(options), default=list(options)[:3], key=f"eres_fed_u_{choice}_{pick}")
            diagram_chart(s, p.free_energies, {k: options[k] for k in options if k in at})
            step_text = "; ".join(f"{n}: {v:+.2f} eV" for n, v in zip(s.step_names(), p.steps))
            a = s.activation
            route_text = ""
            if a:
                route_text = (f" That is on {'either route' if p.limiting_route == electro.EITHER else f'the route “{p.limiting_route}”'}. "
                              f"The route “{a.route}” starts at {a.joins_at}*, in place of the steps before it, and "
                              f"pays the {a.label} barrier, {p.barrier_eV:+.2f} eV on {p.formula}; it counts for the "
                              "limiting potential once that barrier falls to zero.")
            st.caption(f"The free energy of each state along the path on {p.formula} at 0 V vs RHE: {step_text}. "
                       f"Lowering the potential tilts a reduction's path downhill (raising it tilts an oxidation's), "
                       f"{s.electrons} × 1 eV per volt across the whole path. The step that stays uphill longest, "
                       f"{s.step_names()[p.limiting_step]}, sets the limiting potential, {p.limiting_V:.2f} V."
                       + route_text)
    with tabs[3]:
        if ref is None or not ref.complete:
            st.caption("The volcano appears once the catalyst to beat is done.")
        else:
            pts = pd.DataFrame([{"x": p.descriptor, "Material": p.formula, "U": p.limiting_V,
                                 "lo": p.descriptor - s.sigma_descriptor, "hi": p.descriptor + s.sigma_descriptor,
                                 "Kind": "Catalyst to beat" if p.is_reference else "Candidate"}
                                for p in preds if p.complete])
            volcano_chart(s, pts, better)
            st.caption(f"Each dot is a material's own limiting potential against its {descriptor_text(s)}; bars show "
                       "the descriptor's likely range, and the dashed green line is the equilibrium potential, the "
                       "ceiling for a reduction and the floor for an oxidation. "
                       + ("The gray line is the volcano the scaling relations draw"
                          + (", with both routes" if s.volcano_barrier else "")
                          + ": the other intermediates follow the descriptor on most surfaces, so one number places "
                          "a material. Each dot uses that surface's own intermediates, so a dot "
                          f"{'above' if s.direction == 'reduction' else 'below'} the line beats those relations, "
                          "which is where new catalysts come from. "
                          if s.volcano_steps else "")
                       + " ".join(s.basis.split()))
    with tabs[4]:
        screen_ui.structures_tab(folder, preds, chosen_face, choice)
    with tabs[5]:
        screen_ui.sites_and_downloads(preds, rows, choice)
        st.markdown("**Settings**")
        st.markdown(
            f"- Reaction: {s.name} ({s.equation}), {s.direction}\n"
            f"- Comparison potential: {u_cmp:.2f} V vs RHE; equilibrium {s.equilibrium_V:.2f} V; 25 °C\n"
            f"- Care level: {req['thoroughness']}; UMA mode: {req['task']}\n"
            f"- Folder: `{folder}`")
        with st.expander("How the predictions are made"):
            uma = [f"{i.name}*" for i in s.intermediates if i.correction_from == "uma"]
            corr = "; ".join(f"{i.name}*: {i.correction_eV:+.2f} eV"
                             + (" from UMA" if i.correction_from == "uma" else "")
                             + (f", {i.water_eV:+.2f} eV from water" if i.water_eV else "")
                             + (f", {i.layer_eV:+.2f} eV from its neighbors in the layer" if i.layer_eV else "")
                             for i in s.intermediates)
            a = s.activation
            sign = "+" if s.direction == "reduction" else "−"
            st.markdown(
                f"1. **Binding.** For each material, the app cuts its densest crystal face, places "
                f"{', '.join(ADSORBATES[x] for x in s.adsorbates)} on its top, bridge and hollow sites, and lets "
                "Meta's UMA model relax each arrangement. The lowest energy is the binding energy, set against "
                f"{gases_text(s)} molecules in the gas.\n"
                f"2. **Free energies.** Each intermediate's free energy at 0 V vs RHE is its binding energy plus "
                f"corrections for zero-point motion, entropy and hydrogen bonds ({corr}), "
                + ("each from a published table" if not uma else
                   f"published for the rest and, for {joined(uma)}, from UMA's own vibrations on Pt(111), which miss "
                   "the published values by up to two tenths of an eV where both exist (Manual, Methodology tab, "
                   "limits)")
                + (", plus a shift from tabulated formation free energies that moves the gas reference to the "
                   "reaction's own zero" if any(i.shift_eV for i in s.intermediates) else "")
                + ". A proton and electron at 0 V count as half a hydrogen molecule (the computational hydrogen "
                "electrode, Nørskov et al. 2004).\n"
                "3. **Potential.** "
                + (f"Each step that moves an electron shifts by {sign}eU at a potential U. {chemical_text(s)} "
                   if s.chemical_steps else f"Every step moves one electron, so a potential U shifts each step by {sign}eU. ")
                + f"The limiting potential is the best potential where every step that moves an electron along "
                f"{'a route' if a else 'the path'} runs downhill; the overpotential is its distance from the "
                "equilibrium potential.\n"
                f"4. **Speed.** At the comparison potential the rate follows the largest hurdle h, exp(−h/kT) at "
                "25 °C (Sabatier analysis), so speeds are relative to the catalyst to beat. "
                + (f"{routes_text(s)} The {a.label} barrier is E_a = {a.slope:g} × ΔE({a.from_}*) "
                   f"{'−' if a.intercept < 0 else '+'} {abs(a.intercept):g} eV, less {a.bonus_eV:g} eV for its "
                   "looser transition state; on each route the hurdle is its largest step still uphill, or its "
                   "barrier when that is higher (Nørskov et al. 2004, Fig. 8).\n" if a else "\n")
                + f"5. **Likely ranges.** Every binding energy moves anywhere within ±{electro.SIGMA_EV:.2f} eV at "
                "once, and linear programming finds the exact highest and lowest limiting potential and hurdle"
                + (", each barrier moving with its binding energy"
                   if a or any(c.barrier for c in s.chemical_steps) else "") + ".\n\n"
                "Barriers of the proton transfers, the structure of the water layer and the surface's change of "
                "coverage with potential stay outside this picture (Manual, Methodology tab). Read the ranking as "
                "a shortlist.")


# ---------------------------------------------------------------- charts shared with the playground

def diagram_chart(s: electro.ElectroSpec, g: dict[str, float], potentials: dict[str, float]) -> None:
    """The free-energy staircase of one surface at one or more potentials."""
    segs, links = [], []
    for label, u in potentials.items():
        rows_ = electro.diagram(s, g, u)
        for r in rows_:
            segs.append({"x": r["step"], "x2": r["step"] + 0.8, "y": r["level"], "Potential": label, "State": r["state"]})
        for a, b in zip(rows_, rows_[1:]):
            links.append({"x": a["step"] + 0.8, "x2": b["step"], "y": a["level"], "y2": b["level"], "Potential": label})
    if not segs:
        st.caption("Pick at least one potential to draw.")
        return
    names_ = list(potentials)
    scale = alt.Scale(domain=names_, range=PALETTE[:len(names_)])
    states = [r["state"] for r in electro.diagram(s, g, 0.0)]
    x_axis = alt.X("x:Q", title="Along the reaction path", axis=alt.Axis(
        values=[k + 0.4 for k in range(len(states))],
        labelExpr="[" + ",".join(f"'{st_}'" for st_ in states) + "][round(datum.value - 0.4)]"))
    seg = alt.Chart(pd.DataFrame(segs)).mark_rule(strokeWidth=4).encode(
        x=x_axis, x2="x2:Q", y=alt.Y("y:Q", title="Free energy (eV)"),
        color=alt.Color("Potential:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["State:N", "Potential:N", alt.Tooltip("y:Q", format=".2f", title="eV")])
    link = alt.Chart(pd.DataFrame(links)).mark_rule(strokeDash=[3, 3], strokeWidth=1.5).encode(
        x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q", color=alt.Color("Potential:N", scale=scale, legend=None))
    st.altair_chart((seg + link).properties(height=360), width="stretch")


def volcano_chart(s: electro.ElectroSpec, pts: pd.DataFrame, better: str) -> None:
    """The limiting-potential volcano: the scaling line, the equilibrium potential and each material's own dot."""
    x_lo, x_hi = float(pts["lo"].min()) - 0.3, float(pts["hi"].max()) + 0.3
    line_df = None
    if s.volcano_steps:
        x, y = electro.volcano_line(s, x_lo, x_hi)
        line_df = pd.DataFrame({"x": x, "U": y})
    x_enc = alt.X("x:Q", scale=alt.Scale(domain=[x_lo, x_hi], nice=False), title=descriptor_text(s) + " (eV)")
    y_vals = list(pts["U"]) + [s.equilibrium_V] + (list(line_df["U"]) if line_df is not None else [])
    y_enc = alt.Y("U:Q", title=f"Limiting potential (V vs RHE), {better} is better",
                  scale=alt.Scale(domain=[min(y_vals) - 0.15, max(y_vals) + 0.15], nice=False))
    layers = []
    if line_df is not None:
        layers.append(alt.Chart(line_df).mark_line(color="#94a3b8", strokeWidth=2, clip=True).encode(x=x_enc, y=y_enc))
    layers.append(alt.Chart(pd.DataFrame({"U": [s.equilibrium_V]})).mark_rule(color=common.ACCENT, strokeDash=[6, 4]).encode(y="U:Q"))
    layers.append(alt.Chart(pts).mark_rule(color="#cbd5e1", strokeWidth=3, clip=True).encode(x="lo:Q", x2="hi:Q", y="U:Q"))
    layers.append(alt.Chart(pts).mark_point(size=120, opacity=0.95, filled=True, clip=True).encode(
        x=x_enc, y=y_enc,
        color=alt.Color("Kind:N", scale=alt.Scale(domain=["Catalyst to beat", "Candidate"],
                                                  range=[common.REFERENCE_GRAY, common.ACCENT]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Material", alt.Tooltip("x:Q", format=".2f", title="descriptor (eV)"),
                 alt.Tooltip("U:Q", format=".2f", title="limiting potential (V)")]))
    layers.append(alt.Chart(pts).mark_text(dy=-12, fontSize=12, color="#475569", clip=True).encode(
        x=x_enc, y=y_enc, text="Material:N"))
    st.altair_chart(alt.layer(*layers).properties(height=420), width="stretch")
