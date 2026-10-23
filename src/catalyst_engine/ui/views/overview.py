"""Where the project stands, read from docs/status.yaml and the packs on disk, plus the catalysis basics and the
user manual from docs/manual.yaml."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common

status = common.project_status()
packs = [common.pack(pid) for pid in common.pack_ids()]
phases = status["phases"]
phase_status = {ph["id"]: ph["status"] for ph in phases}

st.title("Catalyst engine")
st.markdown(f"##### {status['headline'].strip()}")
st.caption(f"Status as of {status['updated']}. Edit docs/status.yaml to update this page.")

cols = st.columns(4)
cols[0].metric("Reactions loaded", len(packs), border=True)
cols[1].metric("Reaction pathways modeled", sum(len(p.spec.pathways) for p in packs), border=True)
cols[2].metric("Build phases complete", f"{sum(ph['status'] == 'done' for ph in phases)} of {len(phases)}", border=True)
cols[3].metric("Decisions waiting on you", sum(d["status"] == "open" for d in status["decisions"]), border=True)

where, basics, manual, loop, code, plan = st.tabs(
    ["Where we are", "Catalysis basics", "Manual", "How the loop works", "Code map", "Full plan"])

with where:
    nxt, recent = st.columns([3, 2], gap="large")
    with nxt:
        st.subheader("Next steps")
        for i, s in enumerate(status.get("next_steps", []), start=1):
            with st.container(border=True):
                a, b = st.columns([5, 1], vertical_alignment="center")
                a.markdown(f"**{i}. {s['step']}**  \n:gray[{' '.join(s['why'].split())}]")
                b.badge(s["owner"], color="blue" if s["owner"] == "You" else "green")
    with recent:
        st.subheader("Recent work")
        st.markdown("\n".join(f"- :gray[{r['date']}] {' '.join(r['item'].split())}" for r in status.get("recent", [])))

    st.subheader("The proof sequence")
    st.caption("Each step uses the same engine. The claim grows from one reaction to many.")
    cols = st.columns(len(status["proof_sequence"]))
    for i, (col, item) in enumerate(zip(cols, status["proof_sequence"]), start=1):
        with col.container(border=True, height="stretch"):
            st.caption(f"Step {i}")
            st.markdown(f"**{item['step']}**")
            st.write(item["goal"])
            common.badge(item["status"])

    st.subheader("Build phases")
    done = sum(ph["status"] == "done" for ph in phases)
    st.progress(done / len(phases), text=f"{done} of {len(phases)} phases complete")
    for ph in phases:
        with st.container(border=True):
            st.markdown(f"{common.status_label(ph['status'])} **Phase {ph['id']} · {ph['name']}**  \n"
                        f":gray[{' '.join(ph['summary'].split())}]")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Findings so far")
        for f in status["findings"]:
            with st.expander(f["title"]):
                st.write(f["detail"].strip())
                st.caption(f"Source: {f['source']}")
    with right:
        st.subheader("Decisions")
        for d in sorted(status["decisions"], key=lambda d: d["status"] != "open"):
            with st.container(border=True):
                a, b = st.columns([5, 1], vertical_alignment="center")
                a.markdown(f"{d['item']}  \n:gray[Owner: {d['owner']}]")
                if d.get("detail"):
                    a.caption(" ".join(d["detail"].split()))
                with b:
                    common.badge(d["status"])

guide = common.manual()


def flat(text: str) -> str:
    return " ".join(text.split())


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {flat(t)}" for t in items)


def flow_chart(nodes: list[tuple[str, str]], edges: list[str], rankdir: str) -> str:
    lines = [f'digraph {{ rankdir={rankdir}; bgcolor="transparent"; nodesep=0.35; ranksep=0.35;',
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=12 margin="0.15,0.08" '
             'fillcolor="#ccfbf1" color="#0f766e" fontcolor="#0f172a"];',
             'edge [color="#64748b" fontname="Helvetica" fontsize=10 fontcolor="#64748b"];']
    lines += [f'{nid} [label="{label}"];' for nid, label in nodes] + edges + ["}"]
    return "\n".join(lines)


def sketch_labels(xcol: str, ycol: str, notes: list[tuple[float, float, str, str]]) -> alt.LayerChart:
    """Text labels placed in data units; one layer per label so each keeps its own alignment."""
    return alt.layer(*[alt.Chart(pd.DataFrame({xcol: [x], ycol: [y], "label": [text]})).mark_text(
        fontSize=12, color="#64748b", align=align, lineBreak="\n").encode(x=f"{xcol}:Q", y=f"{ycol}:Q", text="label:N")
        for x, y, text, align in notes])


def basics_figure(name: str) -> None:
    """Sketches for the Catalysis basics tab: shapes only, so they read the same for every reaction."""
    x = np.linspace(0, 1, 121)
    if name == "energy":
        base = 1.0 - 0.7 / (1 + np.exp(-(x - 0.5) / 0.06))
        paths = {"Reaction alone": base + 2.0 * np.exp(-((x - 0.5) / 0.12) ** 2),
                 "With a catalyst": base + 0.75 * np.exp(-((x - 0.34) / 0.08) ** 2) + 0.6 * np.exp(-((x - 0.66) / 0.08) ** 2)}
        df = pd.DataFrame([{"x": xi, "Energy": e, "Path": k} for k, ys in paths.items() for xi, e in zip(x, ys)])
        lines = alt.Chart(df).mark_line(strokeWidth=3).encode(
            x=alt.X("x:Q", title="Reaction progress →", axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("Energy:Q", title="Energy", axis=alt.Axis(labels=False, ticks=False, grid=False)),
            color=alt.Color("Path:N", scale=alt.Scale(range=["#94a3b8", "#0f766e"]), legend=alt.Legend(orient="top", title=None)))
        ends = sketch_labels("x", "Energy", [(0.0, 1.32, "Starting gases", "left"), (1.0, 0.12, "Products", "right")])
        st.altair_chart(alt.layer(lines, ends).properties(height=260), width="stretch")
    elif name == "window":
        t = x
        conv = 1 / (1 + np.exp(-(t - 0.3) / 0.045)) - 0.55 / (1 + np.exp(-(t - 0.88) / 0.05))
        df = pd.DataFrame({"t": t, "Conversion": conv})
        inside = df[df["Conversion"] >= 0.9]
        band = alt.Chart(pd.DataFrame({"a": [inside["t"].min()], "b": [inside["t"].max()]})).mark_rect(
            color="#0f766e", opacity=0.12).encode(x="a:Q", x2="b:Q")
        curve = alt.Chart(df).mark_line(strokeWidth=3, color="#0f766e").encode(
            x=alt.X("t:Q", title="Temperature →", axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("Conversion:Q", title="Conversion", scale=alt.Scale(domain=[0, 1.05]),
                    axis=alt.Axis(format="%", values=[0, 0.5, 1], grid=False)))
        notes = sketch_labels("t", "Conversion", [(0.32, 0.45, "← Light-off", "left"),
                                                  ((inside["t"].min() + inside["t"].max()) / 2, 1.03, "Working window", "center"),
                                                  (1.0, 0.3, "Side reactions\ntake over", "right")])
        st.altair_chart(alt.layer(band, curve, notes).properties(height=260), width="stretch")
        st.caption("A sketch: the shape is typical, the numbers come from each reaction.")
    elif name == "cycle":
        st.graphviz_chart(flow_chart(
            [("arrive", "Gas molecules arrive"), ("stick", "They stick to an active site"),
             ("react", "They react on the surface"), ("leave", "The products drift away")],
            ["arrive -> stick; stick -> react; react -> leave;",
             'leave -> stick [label=" the spot is free again" constraint=false style=dashed];'], "TB"))
    elif name == "rig":
        st.graphviz_chart(flow_chart(
            [("gas", "Gas bottles"), ("mix", "Flow controllers\nmix the feed"),
             ("tube", "Heated tube\nwith the catalyst"), ("fa", "Analyzer\nreads the gas"), ("csv", "CSV file\ninto the app")],
            ["gas -> mix -> tube -> fa -> csv;",
             'mix -> fa [label=" bypass: reads what goes in" constraint=false style=dashed];'], "TB"))


with basics:
    pre = guide["preface"]
    st.subheader(pre["title"])
    st.write(flat(pre["intro"]))
    for i, sec in enumerate(pre["sections"], start=1):
        with st.container(border=True):
            if sec.get("figure"):
                text_col, fig_col = st.columns([3, 2], gap="large", vertical_alignment="center")
            else:
                text_col, fig_col = st.container(), None
            with text_col:
                st.markdown(f"##### {i}. {sec['title']}")
                st.write(flat(sec["text"]))
                if sec.get("points"):
                    st.markdown(bullets(sec["points"]))
                if sec.get("packs"):
                    st.markdown(f"**Reactions in the app today ({len(packs)}):** "
                                + " · ".join(f":blue-badge[{p.spec.name}]" for p in packs))
            if fig_col is not None:
                with fig_col:
                    basics_figure(sec["figure"])
    st.caption("Text in docs/manual.yaml. Next: the Manual tab shows how to run an experiment in the app.")

with manual:
    st.subheader(guide["title"])
    st.write(flat(guide["intro"]))
    st.info("New to catalysis? Start with the Catalysis basics tab.", icon=":material/school:")

    st.markdown("#### What you need")
    st.markdown(bullets(guide["needs"]))

    st.markdown("#### Start here")
    for i, (col, item) in enumerate(zip(st.columns(len(guide["start"]), gap="medium"), guide["start"]), start=1):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{i}. {item['title']}**")
            st.write(flat(item["text"]))
            if item.get("code"):
                st.code(item["code"].strip(), language="bash")

    st.markdown("#### Run an experiment, step by step")
    st.caption(f"The same {len(guide['steps'])} steps for every reaction and every batch. "
               "Each card names where the step happens.")
    phase = None
    for i, s in enumerate(guide["steps"], start=1):
        if s.get("phase") and s["phase"] != phase:
            phase = s["phase"]
            st.markdown(f"**:gray[{phase.upper()}]**")
        with st.container(border=True):
            a, b = st.columns([3, 2], vertical_alignment="center")
            a.markdown(f"##### {i}. {s['title']}")
            b.badge(s["page"], icon=":material/location_on:", color="blue")
            st.markdown(bullets(s["do"]))
            if s.get("code"):
                st.code(s["code"].strip(), language="bash")
            st.markdown(f":green[**You get:**] {flat(s['get'])}")

    st.markdown("#### What each page is for")
    st.markdown("| Page | Use it for |\n|---|---|\n" + "\n".join(f"| **{r['page']}** | {r['use']} |" for r in guide["pages"]))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Good habits")
        st.markdown(bullets(guide["tips"]))
        st.markdown("#### When something looks off")
        for h in guide["help"]:
            with st.expander(h["q"]):
                st.write(h["a"])
    with right:
        st.markdown("#### Words used here")
        for g in guide["glossary"]:
            st.markdown(f"**:gray[{g['group'].upper()}]**")
            st.markdown("\n".join(f"- **{t['term']}**: {t['means']}" for t in g["terms"]))
    st.caption("Text in docs/manual.yaml.")

with loop:
    st.subheader("The discovery loop")
    st.caption("Green: built and tested. Blue: started. Orange: next. Gray: planned. Yellow: the physical lab.")
    fills = {"done": ("#d1fae5", "#059669"), "building": ("#dbeafe", "#2563eb"),
             "next": ("#ffedd5", "#ea580c"), "planned": ("#f1f5f9", "#94a3b8"), "lab": ("#fef9c3", "#ca8a04")}
    nodes = {
        "atom": ("Atomistic layer\nbinding energies from UMA", phase_status.get(5, "planned")),
        "model": ("Shared model\none model across reactions", phase_status.get(4, "planned")),
        "planner": ("Experiment planner\npicks the next formulations", phase_status.get(4, "planned")),
        "rig": ("Bench rig + FTIR\nmakes and tests catalysts", "lab"),
        "virtual": ("Virtual rig\nsimulated catalysts and FTIR files", phase_status.get(3, "planned")),
        "data": ("Data platform\nruns, raw files, prediction log", phase_status.get(2, "planned")),
        "fit": ("Kinetic and aging fits\nrates with error bars", phase_status.get(1, "done")),
        "packs": ("Reaction packs\neach reaction as a data file", phase_status.get(1, "done")),
        "kin": ("Kinetics engine\nany reaction network", phase_status.get(1, "done")),
        "value": ("System value\nsizing, cost, permits", phase_status.get(1, "done")),
        "agents": ("Agents\nrun the loop, write the report", phase_status.get(6, "planned")),
    }
    lines = ['digraph { rankdir=TB; bgcolor="transparent"; nodesep=0.6; ranksep=0.45;',
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=13 margin="0.2,0.1" width=2.6];',
             'edge [color="#64748b" fontname="Helvetica" fontsize=11 fontcolor="#475569"];']
    for nid, (label, stat) in nodes.items():
        fill, line = fills[stat]
        lines.append(f'{nid} [label="{label}" fillcolor="{fill}" color="{line}" fontcolor="#0f172a"];')
    lines += [
        "{rank=same; atom; packs;}", "{rank=same; model; kin;}", "{rank=same; planner; virtual;}",
        "{rank=same; data; value;}",
        'atom -> model; model -> planner [label=" predictions, logged first"];',
        'planner -> rig [label=" formulations"]; rig -> data [label=" FTIR files"];',
        'virtual -> data [style=dashed label=" stand-in until the lab opens"];',
        'data -> fit; fit -> model [label=" measured rates" constraint=false]; fit -> value [label=" catalyst rates"];',
        'packs -> kin; kin -> fit [style=dashed]; kin -> virtual;',
        'agents -> planner [style=dotted arrowhead=none]; agents -> data [style=dotted arrowhead=none];',
        "{rank=same; rig; agents;}",
        "}",
    ]
    st.graphviz_chart("\n".join(lines))
    st.markdown(
        """
**How to read it.** The model proposes catalysts and logs its predictions before any experiment runs.
The rig tests them; the fits turn raw FTIR readings into rate constants with error bars; those
measurements train the model for the next round. System value turns a catalyst's rates into what a
customer pays for: catalyst volume, fan power, ammonia and permit headroom.

**Why packs matter.** Every reaction lives in its own folder of data files. The kinetics engine, the
fits and the value tools read any pack, so a second reaction costs a folder of YAML. The test suite
checks that the engine code stays free of reaction-specific names.
"""
    )

with code:
    st.subheader("Where things live")
    st.caption("Open these when you want the detail behind a page in this app.")
    rows = [
        ("reactions/<pack>/pack.yaml", "One reaction: species, pathways, rate laws, metrics and targets", "Reactions page"),
        ("reactions/<pack>/protocols/", "Bench test plans: feeds, temperatures, space velocities, analyzer noise", "Reactions page"),
        ("reactions/<pack>/scenarios/", "Customer cases: an exhaust stream and competing designs", "Design scenarios page"),
        ("src/catalyst_engine/chem/", "Thermodynamics, the reaction network, the reactor solver", "Catalyst playground"),
        ("src/catalyst_engine/fit/", "Kinetic fits with uncertainty; aging fits and life projection", "Fit lab"),
        ("src/catalyst_engine/value/", "Tempering air, monolith sizing, permits, scenario costs", "Calculators, Design scenarios"),
        ("src/catalyst_engine/data/", "Lab store, analyzer ingest, campaign scoring, prediction log", "Lab data page"),
        ("reactions/<pack>/demo.yaml", "The made-up campaign behind the simulated demo store", "Lab data page"),
        ("data/lab/, data/simulated/", "The stores themselves, outside git", "Lab data page"),
        ("src/catalyst_engine/packs.py", "Loads and checks reaction packs", "Reactions page"),
        ("src/catalyst_engine/cli.py", "The `ce` command line", "Terminal"),
        ("src/catalyst_engine/ui/", "This app", "Here"),
        ("experiments/<name>/", "Model checks: scripts, results, and a report.yaml with the write-up", "Atomistic check page"),
        ("tests/", "The checks that run on every change", "Terminal: uv run pytest"),
        ("docs/PLAN.md", "The full plan", "Full plan tab"),
        ("docs/status.yaml", "Everything on the Where we are tab", "Where we are tab"),
        ("docs/manual.yaml", "Catalysis basics and the user manual: setup, the experiment steps, tips and words used", "Catalysis basics and Manual tabs"),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Path", "What it holds", "Shown in"]),
        hide_index=True,
    )

with plan:
    st.markdown((common.REPO / "docs" / "PLAN.md").read_text())
