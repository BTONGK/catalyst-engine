"""The manual: how to use the app step by step, catalysis basics, the methodology, the words used, and help. Text in
docs/manual.yaml and experiments/design_method/report.yaml; screenshots in docs/manual/."""

import math

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

def method_flow(steps: list[dict]) -> str:
    """The method's steps as two columns read top to bottom: the atom-scale steps, then energy to performance."""
    split = next(i for i, s in enumerate(steps) if s.get("scale") == "performance")
    groups = [("atoms", "at the scale of atoms", range(split)), ("perf", "from energy to performance",
                                                                 range(split, len(steps)))]
    lines = []
    for gid, title, idx in groups:
        lines.append(f'subgraph cluster_{gid} {{ label="Steps {idx[0] + 1} to {idx[-1] + 1}: {title}"; '
                     'fontname="Helvetica"; fontsize=13; '
                     'fontcolor="#0f766e"; color="#cbd5e1"; style="rounded";')
        lines += [f's{i + 1} [label="{i + 1}. {steps[i]["title"]}" width=2.6];' for i in idx]
        lines.append(" -> ".join(f"s{i + 1}" for i in idx) + "; }")
    return flow_chart([], lines, "TB")


STATUS = {
    "checked": ("Checked", "green", ":material/verified:"),
    "gap": ("Checked, with a known gap", "orange", ":material/report:"),
    "published": ("Rests on published work", "blue", ":material/menu_book:"),
    "open": ("Open: the largest uncertainty", "red", ":material/help:"),
}


SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def two_figures(x: float) -> str:
    """Two significant figures in plain digits: 180, 2.5, 14,000."""
    return f"{float(f'{x:.2g}'):,g}"


def ratio_text(v: float) -> str:
    """A speed ratio a reader can picture: 0.14 stays, 0.00007 reads "1 in 14,000", 1e-17 reads "1 in 10¹⁷"."""
    if v >= 0.01:
        return f"{v:.2g}"
    n = 1 / v
    if n >= 1e15:
        return f"1 in 10{str(round(math.log10(n))).translate(SUPERSCRIPT)}"
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if n >= size:
            return f"1 in {two_figures(n / size)} {word}"
    return f"1 in {two_figures(n)}"


def method_table(spec: dict, tables: dict) -> None:
    """A table from the methodology summary: `spec` names the table and the columns to show, with their labels;
    columns listed under `ratios` read as "1 in N" and those under `percent` as percentages."""
    df = pd.DataFrame(tables[spec["table"]])
    cols = spec["columns"]
    for c in spec.get("ratios", []):
        df[c] = df[c].map(ratio_text)
    for c in spec.get("percent", []):
        df[c] = df[c].map(lambda v: f"{v:.1%}")
    df = df[list(cols)].rename(columns=cols)
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, list)).any():
            df[c] = df[c].map(lambda v: "(" + " ".join(str(x) for x in v) + ")" if isinstance(v, list) else v)
    wide = {cols[c]: st.column_config.TextColumn(width=112) for c in spec.get("ratios", [])}
    st.dataframe(df, hide_index=True, width="stretch", height="content", column_config=wide)


def method_chart(spec: dict, tables: dict) -> None:
    """A parity plot (two estimates of the same number) or lines against one variable."""
    df = pd.DataFrame(tables[spec["table"]])
    x = alt.X(f"{spec['x']}:Q", title=spec["x_title"], scale=alt.Scale(zero=False, nice=True))
    y = alt.Y(f"{spec['y']}:Q", title=spec["y_title"], scale=alt.Scale(zero=False, nice=True))
    if spec["kind"] == "parity":
        lo = float(min(df[spec["x"]].min(), df[spec["y"]].min())) - 0.2
        hi = float(max(df[spec["x"]].max(), df[spec["y"]].max())) + 0.2
        diag = alt.Chart(pd.DataFrame({spec["x"]: [lo, hi], spec["y"]: [lo, hi]})).mark_line(
            color="#94a3b8", strokeDash=[4, 4]).encode(x=x, y=y)
        pts = alt.Chart(df).mark_circle(size=90, color="#0f766e").encode(x=x, y=y, tooltip=list(df.columns))
        names = alt.Chart(df).mark_text(dx=10, align="left", fontSize=11, color="#475569").encode(
            x=x, y=y, text=f"{spec['label']}:N")
        chart = alt.layer(diag, pts, names)
    else:
        color = alt.Color(f"{spec['color']}:N", title=None, legend=alt.Legend(orient="bottom")) if spec.get("color") \
            else alt.value("#0f766e")
        chart = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(x=x, y=y, color=color, tooltip=list(df.columns))
    st.altair_chart(chart.properties(height=260), width="stretch")
    if spec.get("caption"):
        st.caption(spec["caption"])


def methodology(paper: dict) -> None:
    """The Methodology tab: a method paper whose numbers come from the experiment's summary."""
    values, tables = paper["data"]["values"], paper["data"]["tables"]

    def fill(text: str) -> str:
        return flat(text).format_map(values)

    st.markdown(f"### {paper['title']}")
    st.caption(fill(paper["byline"]))
    with st.container(border=True):
        st.markdown("**Abstract**")
        st.write(fill(paper["abstract"]))
    st.info(fill(paper["how_to_read"]), icon=":material/school:")
    st.markdown(" ".join(f":{color}-badge[{icon} {label}]" for label, color, icon in STATUS.values()))

    for n, sec in enumerate(paper["intro"], start=1):
        st.markdown(f"#### {n}. {sec['title']}")
        st.write(fill(sec["text"]))
        if sec.get("points"):
            st.markdown("\n".join(f"- {fill(p)}" for p in sec["points"]))
        if sec.get("flow"):
            st.graphviz_chart(method_flow(paper["steps"]))
            st.markdown("| Step | Takes in | Gives out | Tool |\n|---|---|---|---|\n" + "\n".join(
                f"| {i}. {s['title']} | {s['takes']} | {s['gives']} | {s['tool']} |"
                for i, s in enumerate(paper["steps"], start=1)))

    n_intro = len(paper["intro"])
    st.markdown(f"#### {n_intro + 1}. Methods and checks, step by step")
    st.caption("Each step: what the app does, the equation with the same idea in words, the assumptions it rests on "
               "(numbered A1, A2 … and collected in section " + str(n_intro + 4) + "), and how it was checked.")
    for i, s in enumerate(paper["steps"], start=1):
        label, color, icon = STATUS[s["status"]]
        with st.container(border=True):
            a, b = st.columns([3, 2], vertical_alignment="center")
            a.markdown(f"##### Step {i} · {s['title']}")
            b.badge(label, icon=icon, color=color)
            st.write(fill(s["does"]))
            for eq in s.get("equations", []):
                st.latex(eq["tex"])
                st.caption(f"In words: {fill(eq['words'])}")
            st.markdown("**Assumptions**")
            st.markdown("\n".join(f"- **{x['id']}. {fill(x['short'])}.** {fill(x['text'])}" for x in s["assumptions"]))
            st.markdown("**How it was checked**")
            for c in s["checks"]:
                st.markdown(f"- {fill(c['text'])}")
                if c.get("table") or c.get("chart"):
                    with st.expander(fill(c.get("see", "See the numbers"))):
                        if c.get("chart"):
                            method_chart(c["chart"], tables)
                        if c.get("table"):
                            method_table(c["table"], tables)
            if s.get("verdict"):
                st.markdown(f":{color}[**Verdict.**] {fill(s['verdict'])}")

    ex = paper["example"]
    st.markdown(f"#### {n_intro + 2}. {ex['title']}")
    st.write(fill(ex["text"]))
    method_table(ex["table"], tables)
    st.markdown("\n".join(f"- {fill(p)}" for p in ex["points"]))

    st.markdown(f"#### {n_intro + 3}. {paper['budget']['title']}")
    st.write(fill(paper["budget"]["text"]))
    st.markdown("| Source | Size (eV) | Where it comes from |\n|---|---|---|\n" + "\n".join(
        f"| {row['source']} | {fill(row['size'])} | {fill(row['basis'])} |" for row in paper["budget"]["rows"]))
    st.write(fill(paper["budget"]["after"]))

    st.markdown(f"#### {n_intro + 4}. Every assumption in one place")
    rows = [(x, i) for i, s in enumerate(paper["steps"], start=1) for x in s["assumptions"]]
    st.markdown("| | Assumption | Step | If it fails |\n|---|---|---|---|\n" + "\n".join(
        f"| **{x['id']}** | {fill(x['short'])} | {i} | {fill(x['effect'])} |" for x, i in rows))

    st.markdown(f"#### {n_intro + 5}. {paper['trust']['title']}")
    for group in paper["trust"]["groups"]:
        st.markdown(f"**{group['heading']}**")
        st.markdown("\n".join(f"- {fill(p)}" for p in group["points"]))

    lim = paper["limits"]
    st.markdown(f"#### {n_intro + 6}. {lim['title']}")
    st.write(fill(lim["text"]))
    st.markdown("| Left out | Why | How large | Meanwhile |\n|---|---|---|---|\n" + "\n".join(
        f"| **{row['part']}** | {fill(row['why'])} | {fill(row['size'])} | {fill(row['meanwhile'])} |"
        for row in lim["rows"]))
    for c in lim.get("checks", []):
        st.markdown(f"- {fill(c['text'])}")
        with st.expander(fill(c.get("see", "See the numbers"))):
            method_table(c["table"], tables)

    n = n_intro + 7
    if audit := paper.get("audit"):
        st.markdown(f"#### {n}. {audit['title']}")
        st.write(fill(audit["text"]))
        a, b = st.columns(2, gap="large")
        a.markdown("**Checked**")
        a.markdown("\n".join(f"- {fill(p)}" for p in audit["checked"]))
        b.markdown("**Changed in the latest review**")
        b.markdown("\n".join(f"- {fill(p)}" for p in audit["changed"]))
        n += 1

    st.markdown(f"#### {n}. Reproduce this page")
    st.write(fill(paper["reproduce"]["text"]))
    st.code(paper["reproduce"]["code"].strip(), language="bash")
    st.markdown("\n".join(f"- {fill(v)}" for v in paper["reproduce"]["versions"]))

    st.markdown("#### References")
    st.markdown("\n".join(f"{n}. {ref['cite']}" + (f" [link]({ref['url']})" if ref.get("url") else "")
                          for n, ref in enumerate(paper["references"], start=1)))
    st.caption("Text in experiments/design_method/report.yaml; numbers in its results/summary.json.")


st.title("Manual")
use_tab, basics_tab, method_tab, words_tab, help_tab = st.tabs(
    ["How to use the app", "Catalysis basics", "Methodology", "Words used", "Help"])

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

with method_tab:
    paper = next(iter(common.experiment_reports("methodology").values()))
    if paper["data"] is None:
        st.info("The methodology's numbers come from two scripts in experiments/design_method/; run them to fill this "
                "tab (the commands sit at the end of report.yaml).", icon=":material/science:")
    else:
        methodology(paper)

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
