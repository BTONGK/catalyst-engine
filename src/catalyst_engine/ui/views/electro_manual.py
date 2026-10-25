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

        def fill(text: str) -> str:
            return flat(text).format_map(values)

        st.markdown(f"### {paper['title']}")
        st.caption(fill(paper["byline"]))
        with st.container(border=True):
            st.markdown("**Abstract**")
            st.write(fill(paper["abstract"]))
        st.info(fill(paper["how_to_read"]), icon=":material/school:")
        st.markdown(" ".join(f":{color}-badge[{icon} {label}]" for label, color, icon in STATUS.values()))

        st.markdown("#### 1. The method, step by step")
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
                        with st.expander(c.get("see", "See the numbers")):
                            method_table(c["table"], tables)
                if s.get("verdict"):
                    st.markdown(f":{color}[**Verdict.**] {fill(s['verdict'])}")

        st.markdown("#### 2. Every assumption in one place")
        rows = [(x, i) for i, s in enumerate(paper["sections"], start=1) for x in s.get("assumptions", [])]
        st.markdown("| | Assumption | Step | If it fails |\n|---|---|---|---|\n" + "\n".join(
            f"| **{x['id']}** | {fill(x['short'])} | {i} | {fill(x['effect'])} |" for x, i in rows))

        st.markdown(f"#### 3. {paper['trust']['title']}")
        for group in paper["trust"]["groups"]:
            st.markdown(f"**{group['heading']}**")
            st.markdown("\n".join(f"- {fill(p)}" for p in group["points"]))

        lim = paper["limits"]
        st.markdown(f"#### 4. {lim['title']}")
        st.write(fill(lim["text"]))
        st.markdown("| Left out | Why | How large | Meanwhile |\n|---|---|---|---|\n" + "\n".join(
            f"| **{row['part']}** | {fill(row['why'])} | {fill(row['size'])} | {fill(row['meanwhile'])} |"
            for row in lim["rows"]))

        st.markdown("#### 5. Reproduce this page")
        st.write(fill(paper["reproduce"]["text"]))
        st.code(paper["reproduce"]["code"].strip(), language="bash")
        st.markdown("\n".join(f"- {fill(v)}" for v in paper["reproduce"]["versions"]))

        st.markdown("#### References")
        st.markdown("\n".join(f"{n}. {ref['cite']}" + (f" [link]({ref['url']})" if ref.get("url") else "")
                              for n, ref in enumerate(paper["references"], start=1)))
        st.caption("Text in experiments/electro_check/report.yaml; numbers in its results/summary.json.")

with words_tab:
    for group in doc["glossary"]:
        st.markdown(f"#### {group['group']}")
        st.markdown("\n".join(f"- **{t['term']}.** {flat(t['means'])}" for t in group["terms"]))
    st.markdown("#### When something looks off")
    for h in doc["help"]:
        with st.expander(h["q"]):
            st.write(flat(h["a"]))

