"""The electrocatalysis manual: electrochemistry basics, how to use this section, the methodology paper rendered
from experiments/electro_check/report.yaml with its summary numbers, and the words used."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml

from catalyst_engine.ui import common

DOCS = common.REPO / "docs" / "electro"
STATUS = {"checked": ("Checked", "green", ":material/verified:"), "partly": ("Partly checked", "orange", ":material/rule:"),
          "untested": ("Untested", "gray", ":material/help:")}


@st.cache_data(show_spinner=False)
def _manual(mtime: float) -> dict:
    return yaml.safe_load((DOCS / "manual.yaml").read_text())


def manual() -> dict:
    return _manual((DOCS / "manual.yaml").stat().st_mtime)


def flat(text: str) -> str:
    return " ".join(text.split())


def method_table(spec: dict, tables: dict) -> None:
    df = pd.DataFrame(tables[spec["table"]])
    cols = spec["columns"]
    df = df[list(cols)].rename(columns=cols)
    st.dataframe(df, hide_index=True, width="stretch", height="content")


WHEN = {"load": ("Each load", "blue", ":material/upload_file:"),
        "screen": ("Each screen", "green", ":material/play_circle:")}


def loading_flow(stages: list[dict]) -> str:
    """The loading stages as two columns read top to bottom: each load, then each screen."""
    lines = ['digraph { rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.35;',
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=12 margin="0.15,0.08" '
             'fillcolor="#ccfbf1" color="#0f766e" fontcolor="#0f172a" width=2.6];',
             'edge [color="#64748b"];']
    for when, (label, _, _) in WHEN.items():
        idx = [i for i, s in enumerate(stages, start=1) if s["when"] == when]
        if not idx:
            continue
        lines.append(f'subgraph cluster_{when} {{ label="{label}"; labeljust="l"; fontname="Helvetica"; fontsize=13; '
                     'fontcolor="#0f766e"; color="#cbd5e1"; style="rounded";')
        lines += [f'g{i} [label="{i}. {stages[i - 1]["title"]}"];' for i in idx]
        lines.append(" -> ".join(f"g{i}" for i in idx) + "; }")
    order = range(1, len(stages) + 1)
    lines.append(" -> ".join(f"g{i}" for i in order) + " [style=invis];")
    lines += [f"g{a} -> g{b} [constraint=false];" for a, b in zip(order, list(order)[1:])
              if stages[a - 1]["when"] != stages[b - 1]["when"]]
    return "\n".join(lines + ["}"])


def loading_section(sec: dict, fill) -> None:
    """What the app does with a reaction file, stage by stage, with one reaction followed through."""
    st.write(fill(sec["text"]))
    st.graphviz_chart(loading_flow(sec["stages"]))
    for i, s in enumerate(sec["stages"], start=1):
        label, color, icon = WHEN[s["when"]]
        with st.container(border=True):
            a, b = st.columns([3, 2], vertical_alignment="center")
            a.markdown(f"##### Stage {i} · {s['title']}")
            b.badge(label, icon=icon, color=color)
            st.write(fill(s["does"]))
            for eq in s.get("equations", []):
                st.latex(eq["tex"])
                st.caption(f"In words: {fill(eq['words'])}")
            if s.get("checks"):
                st.markdown("**What must hold** (the app stops the load and names what it found)")
                st.markdown("\n".join(f"- {fill(c)}" for c in s["checks"]))
            st.markdown(f":blue[**In the example.**] {fill(s['example'])}")


def check_section(sec: dict, tables: dict, fill) -> None:
    """Each reaction against published calculations: the rule, the verdicts, the rows behind them, the corrections."""
    st.write(fill(sec["text"]))
    st.markdown("\n".join(f"- {fill(p)}" for p in sec["points"]))
    method_table(sec["table"], tables)
    for part in ("intermediates", "metals"):
        if sec.get(part):
            with st.expander(fill(sec[part].get("see", "See the numbers"))):
                method_table(sec[part], tables)
    st.write(fill(sec["result"]))
    if corr := sec.get("corrections"):
        st.write(fill(corr["text"]))
        with st.expander(fill(corr.get("see", "See the numbers"))):
            method_table(corr, tables)


st.title("Electrocatalysis manual")
doc = manual()
reports = common.experiment_reports("electro")
paper = next(iter(reports.values())) if reports else None

basics_tab, use_tab, method_tab, words_tab = st.tabs(["Electrochemistry basics", "How to use this section",
                                                      "Methodology", "Words and help"])

with basics_tab:
    pre = doc["preface"]
    st.markdown(f"### {pre['title']}")
    st.write(flat(pre["intro"]))
    for sec in pre["sections"]:
        st.markdown(f"#### {sec['title']}")
        for para in sec["paragraphs"]:
            st.write(flat(para))
        if sec.get("points"):
            st.markdown("\n".join(f"- {flat(p)}" for p in sec["points"]))

with use_tab:
    st.markdown(f"### {doc['title']}")
    st.write(flat(doc["intro"]))
    st.markdown("**You need**")
    st.markdown("\n".join(f"- {flat(n)}" for n in doc["needs"]))
    for n, step in enumerate(doc["steps"], start=1):
        with st.container(border=True):
            st.markdown(f"#### {n}. {step['title']}")
            st.caption(f"{step['phase']} · {step['page']}")
            a, b = st.columns([3, 2], gap="large") if step.get("image") else (st.container(), None)
            with a:
                st.markdown("\n".join(f"- {flat(d)}" for d in step["do"]))
                st.markdown(f"**You get:** {flat(step['get'])}")
            if b is not None and (DOCS / step["image"]).exists():
                b.image(str(DOCS / step["image"]))
    st.markdown("#### Pages in this section")
    st.markdown("| Page | Use it for |\n|---|---|\n" + "\n".join(f"| {p['page']} | {flat(p['use'])} |" for p in doc["pages"]))
    st.markdown("#### Tips")
    st.markdown("\n".join(f"- {flat(t)}" for t in doc["tips"]))

with method_tab:
    if paper is None or paper.get("data") is None:
        st.info("The methodology appears here once experiments/electro_check/ holds a report.yaml and its summary.")
    else:
        values, tables = paper["data"]["values"], paper["data"]["tables"]
        # section numbers in page order, so the text can point at them ({sec_limits}, {sec_reaction_check} ...)
        order = [k for k in ("loading",) if paper.get(k)] + ["method"] + \
            [k for k in ("reaction_check",) if paper.get(k)] + ["assumptions", "trust", "limits"] + \
            [k for k in ("audit",) if paper.get(k)] + ["reproduce"]
        num = {key: i for i, key in enumerate(order, start=1)}
        mapping = {**values, **{f"sec_{k}": v for k, v in num.items()}}

        def fill(text: str) -> str:
            return flat(text).format_map(mapping)

        st.markdown(f"### {paper['title']}")
        st.caption(fill(paper["byline"]))
        with st.container(border=True):
            st.markdown("**Abstract**")
            st.write(fill(paper["abstract"]))
        st.info(fill(paper["how_to_read"]), icon=":material/school:")
        st.markdown(" ".join(f":{color}-badge[{icon} {label}]" for label, color, icon in STATUS.values()))

        if paper.get("loading"):
            st.markdown(f"#### {num['loading']}. {paper['loading']['title']}")
            loading_section(paper["loading"], fill)

        st.markdown(f"#### {num['method']}. The method, step by step")
        for i, s in enumerate(paper["sections"], start=1):
            label, color, icon = STATUS[s["status"]]
            with st.container(border=True):
                a, b = st.columns([3, 2], vertical_alignment="center")
                a.markdown(f"##### Step {i} · {s['title']}")
                b.badge(label, icon=icon, color=color)
                st.write(fill(s["does"]))
                for eq in s.get("equations", []):
                    st.latex(eq["tex"])
                    st.caption(f"In words: {fill(eq['words'])}")
                if s.get("assumptions"):
                    st.markdown("**Assumptions**")
                    st.markdown("\n".join(f"- **{x['id']}. {fill(x['short'])}.** {fill(x['text'])}" for x in s["assumptions"]))
                st.markdown("**How it was checked**")
                for c in s["checks"]:
                    st.markdown(f"- {fill(c['text'])}")
                    if c.get("table"):
                        with st.expander(fill(c.get("see", "See the numbers"))):
                            method_table(c["table"], tables)
                if s.get("verdict"):
                    st.markdown(f":{color}[**Verdict.**] {fill(s['verdict'])}")

        if paper.get("reaction_check"):
            st.markdown(f"#### {num['reaction_check']}. {paper['reaction_check']['title']}")
            check_section(paper["reaction_check"], tables, fill)

        st.markdown(f"#### {num['assumptions']}. Every assumption in one place")
        rows = [(x, i) for i, s in enumerate(paper["sections"], start=1) for x in s.get("assumptions", [])]
        st.markdown("| | Assumption | Step | If it fails |\n|---|---|---|---|\n" + "\n".join(
            f"| **{x['id']}** | {fill(x['short'])} | {i} | {fill(x['effect'])} |" for x, i in rows))

        st.markdown(f"#### {num['trust']}. {paper['trust']['title']}")
        for group in paper["trust"]["groups"]:
            st.markdown(f"**{group['heading']}**")
            st.markdown("\n".join(f"- {fill(p)}" for p in group["points"]))

        lim = paper["limits"]
        st.markdown(f"#### {num['limits']}. {lim['title']}")
        st.write(fill(lim["text"]))
        st.markdown("| Left out | Why | How large | Meanwhile |\n|---|---|---|---|\n" + "\n".join(
            f"| **{row['part']}** | {fill(row['why'])} | {fill(row['size'])} | {fill(row['meanwhile'])} |"
            for row in lim["rows"]))

        if audit := paper.get("audit"):
            st.markdown(f"#### {num['audit']}. {audit['title']}")
            st.write(fill(audit["text"]))
            a, b = st.columns(2, gap="large")
            a.markdown("**Checked**")
            a.markdown("\n".join(f"- {fill(p)}" for p in audit["checked"]))
            b.markdown("**Changed in the latest review**")
            b.markdown("\n".join(f"- {fill(p)}" for p in audit["changed"]))

        st.markdown(f"#### {num['reproduce']}. Reproduce this page")
        st.write(fill(paper["reproduce"]["text"]))
        st.code(paper["reproduce"]["code"].strip(), language="bash")
        st.markdown("\n".join(f"- {fill(v)}" for v in paper["reproduce"]["versions"]))

        st.markdown("#### References")
        st.markdown("\n".join(f"{n}. {ref['cite']}" + (f" [link]({ref['url']})" if ref.get("url") else "")
                              for n, ref in enumerate(paper["references"], start=1)))
        st.caption("Text in experiments/electro_check/report.yaml; numbers in its results/summary.json"
                   + "".join(f" and {a.replace('../', 'experiments/')}" for a in paper.get("also", [])) + ".")

with words_tab:
    for group in doc["glossary"]:
        st.markdown(f"#### {group['group']}")
        st.markdown("\n".join(f"- **{t['term']}.** {flat(t['means'])}" for t in group["terms"]))
    st.markdown("#### When something looks off")
    for h in doc["help"]:
        with st.expander(h["q"]):
            st.write(flat(h["a"]))

