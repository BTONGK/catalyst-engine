"""The manual: how to use the app step by step, catalysis basics, the words used, and help. Text in
docs/manual.yaml; screenshots in docs/manual/."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common

guide = common.manual()
packs = [common.pack(pid) for pid in common.pack_ids()]
SHOTS = common.REPO / "docs" / "manual"


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
    elif name == "volcano":
        d = np.linspace(-1, 1, 121)
        act = np.exp(-np.abs(d) * 2.2)
        df = pd.DataFrame({"d": d, "Activity": act})
        curve = alt.Chart(df).mark_line(strokeWidth=3, color="#0f766e").encode(
            x=alt.X("d:Q", title="How strongly the surface holds the key molecule →",
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("Activity:Q", title="Activity", scale=alt.Scale(domain=[0, 1.15]),
                    axis=alt.Axis(labels=False, ticks=False, grid=False)))
        dots = pd.DataFrame({"d": [-0.75, -0.3, 0.05, 0.55], "Activity": np.exp(-np.abs([-0.75, -0.3, 0.05, 0.55]) * 2.2),
                             "m": ["A", "B", "C", "D"]})
        pts = alt.Chart(dots).mark_circle(size=140, color="#7c3aed").encode(x="d:Q", y="Activity:Q")
        names = alt.Chart(dots).mark_text(dy=-14, fontSize=12, color="#7c3aed").encode(x="d:Q", y="Activity:Q", text="m:N")
        notes = sketch_labels("d", "Activity", [(-0.98, 0.42, "Holds too weakly:\nmolecules drift off", "left"),
                                                (0.98, 0.42, "Holds too strongly:\nthe surface clogs", "right"),
                                                (0.0, 1.1, "Just right", "center")])
        st.altair_chart(alt.layer(curve, pts, names, notes).properties(height=260), width="stretch")
        st.caption("A sketch. Material C sits near the top; the app draws the real volcano for each reaction.")
    elif name == "screen":
        st.graphviz_chart(flow_chart(
            [("mp", "Crystal structure\n(Materials Project)"), ("cut", "Cut the surface,\nplace the molecules"),
             ("uma", "UMA computes\nhow strongly they stick"), ("volc", "Place on the volcano"),
             ("reactor", "Simulated reactor\nat your conditions"), ("rank", "Ranking, curves\nand 3D views")],
            ["mp -> cut -> uma -> volc -> reactor -> rank;"], "TB"))

st.title("Manual")
use_tab, basics_tab, words_tab, help_tab = st.tabs(["How to use the app", "Catalysis basics", "Words used", "Help"])

with use_tab:
    st.markdown(f"### {guide['title']}")
    st.write(flat(guide["intro"]))
    st.info("New to catalysts? Read the Catalysis basics tab first; it takes ten minutes.", icon=":material/school:")

    st.markdown("#### What you need")
    st.markdown(bullets(guide["needs"]))

    st.markdown("#### Start here")
    for i, (col, item) in enumerate(zip(st.columns(len(guide["start"]), gap="medium"), guide["start"]), start=1):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{i}. {item['title']}**")
            st.write(flat(item["text"]))
            if item.get("code"):
                st.code(item["code"].strip(), language="bash")

    st.markdown(f"#### Design a catalyst in {len(guide['steps'])} steps")
    st.caption("The same steps for every reaction. Each card names the page and the part of the page where the step "
               "happens.")
    phase = None
    for i, s in enumerate(guide["steps"], start=1):
        if s.get("phase") and s["phase"] != phase:
            phase = s["phase"]
            st.markdown(f"**:gray[{phase.upper()}]**")
        with st.container(border=True):
            shot = SHOTS / s["image"] if s.get("image") else None
            if shot is not None and shot.exists():
                text_col, img_col = st.columns([1, 1], gap="large")
            else:
                text_col, img_col = st.container(), None
            with text_col:
                a, b = st.columns([3, 2], vertical_alignment="center")
                a.markdown(f"##### {i}. {s['title']}")
                b.badge(s["page"], icon=":material/location_on:", color="blue")
                st.markdown(bullets(s["do"]))
                if s.get("code"):
                    st.code(s["code"].strip(), language="bash")
                st.markdown(f":green[**You get:**] {flat(s['get'])}")
            if img_col is not None:
                img_col.image(str(shot), width="stretch")

    st.markdown("#### What each page is for")
    st.markdown("| Page | Use it for |\n|---|---|\n" + "\n".join(f"| **{r['page']}** | {r['use']} |" for r in guide["pages"]))

    st.markdown("#### Good habits")
    st.markdown(bullets(guide["tips"]))

with basics_tab:
    pre = guide["preface"]
    st.markdown(f"### {pre['title']}")
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

with words_tab:
    cols = st.columns(len(guide["glossary"]), gap="large")
    for col, g in zip(cols, guide["glossary"]):
        with col:
            st.markdown(f"#### {g['group']}")
            st.markdown("\n".join(f"- **{t['term']}**: {t['means']}" for t in g["terms"]))

with help_tab:
    st.markdown("#### When something looks off")
    for h in guide["help"]:
        with st.expander(h["q"]):
            st.write(h["a"])
    st.caption("Text in docs/manual.yaml.")
