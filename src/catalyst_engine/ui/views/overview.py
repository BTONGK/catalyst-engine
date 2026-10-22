"""Where the project stands, read from docs/status.yaml and the packs on disk."""

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

where, loop, code, plan = st.tabs(["Where we are", "How the loop works", "Code map", "Full plan"])

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
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Path", "What it holds", "Shown in"]),
        hide_index=True,
    )

with plan:
    st.markdown((common.REPO / "docs" / "PLAN.md").read_text())
