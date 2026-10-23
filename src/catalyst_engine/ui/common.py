"""Shared pieces for the app: cached data, status badges, charts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from ase.data import atomic_numbers, reference_states

from catalyst_engine.packs import PACKS_DIR, Pack, list_packs, load_pack, pack_path, read_yaml
from catalyst_engine.units import k_to_f

REPO = Path(__file__).resolve().parents[3]
ACCENT = "#0f766e"
REFERENCE_GRAY = "#9aa0a6"
CATEGORIES = ["Clean air", "Hydrogen and energy", "Chemicals and fuels", "Other"]  # reaction groups, in display order
TARGET_RED = "#d93025"

STATUS = {
    "done": ("Done", "green"),
    "building": ("Building", "blue"),
    "next": ("Next", "orange"),
    "planned": ("Planned", "gray"),
    "open": ("Open", "orange"),
    "closed": ("Closed", "green"),
}


def style() -> None:
    st.html(
        """<style>
        .block-container {padding-top: 2.2rem; max-width: 1280px;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .ce-eq {font-family: ui-monospace, Menlo, monospace; font-size: 1.05rem;}
        .ce-muted {opacity: 0.7; font-size: 0.9rem;}
        </style>"""
    )


def badge(status: str) -> None:
    label, color = STATUS.get(status, (status.title(), "gray"))
    st.badge(label, color=color)


def status_label(status: str) -> str:
    label, color = STATUS.get(status, (status.title(), "gray"))
    return f":{color}-badge[{label}]"


def _mtime(path: Path) -> float:
    return max((p.stat().st_mtime for p in path.rglob("*.yaml")), default=0.0) if path.is_dir() else path.stat().st_mtime


# Cache keys include file modification times, so edits to YAML show up on the next rerun.
@st.cache_data(show_spinner=False)
def _status(mtime_key: float) -> dict:
    return yaml.safe_load((REPO / "docs" / "status.yaml").read_text())


def project_status() -> dict:
    return _status(_mtime(REPO / "docs" / "status.yaml"))


@st.cache_data(show_spinner=False)
def _manual(mtime_key: float) -> dict:
    return yaml.safe_load((REPO / "docs" / "manual.yaml").read_text())


def manual() -> dict:
    """The user manual shown on the Overview page."""
    return _manual(_mtime(REPO / "docs" / "manual.yaml"))


@st.cache_data(show_spinner=False)
def _report(folder: Path, mtime_key: float) -> dict:
    report = yaml.safe_load((folder / "report.yaml").read_text())
    summary = folder / report.get("summary", "results/summary.json")
    report["data"] = json.loads(summary.read_text()) if summary.exists() else None
    return report


def experiment_reports(kind: str = "atomistic") -> dict[str, dict]:
    """Experiments that publish a report.yaml of one `kind` (atomistic unless the report says otherwise), keyed by
    folder name; each carries its summary numbers as `data`."""
    out = {}
    for path in sorted((REPO / "experiments").glob("*/report.yaml")):
        stamps = [path.stat().st_mtime] + [p.stat().st_mtime for p in path.parent.glob("results/*.json")]
        report = _report(path.parent, max(stamps))
        if report.get("kind", "atomistic") == kind:
            out[path.parent.name] = report
    return out


@st.cache_resource(show_spinner=False)
def _pack(pack_id: str, mtime_key: float) -> Pack:
    return load_pack(pack_id)


def pack(pack_id: str) -> Pack:
    """Packs reload automatically when their YAML files change."""
    return _pack(pack_id, _mtime(pack_path(pack_id)))


def pack_ids() -> list[str]:
    return list_packs()


def pack_picker(key: str, label: str = "Reaction") -> Pack:
    """Starts on docs/status.yaml's focus_pack, the reaction the project is proving first."""
    ids = pack_ids()
    names = {pid: pack(pid).spec.name for pid in ids}
    focus = project_status().get("focus_pack")
    chosen = st.selectbox(label, ids, index=ids.index(focus) if focus in ids else 0, format_func=names.get, key=key)
    return pack(chosen)


@st.cache_data(show_spinner=False)
def simulate(pack_id: str, protocol_id: str, params: tuple[tuple[str, float], ...], mtime_key: float) -> pd.DataFrame:
    rows = _pack(pack_id, mtime_key).simulate(dict(params), protocol_id)
    df = pd.DataFrame(rows)
    df["T_F"] = df["T_K"].map(k_to_f).round(0)
    return df


def run_protocol(p: Pack, protocol_id: str, params: dict[str, float]) -> pd.DataFrame:
    return simulate(p.id, protocol_id, tuple(sorted(params.items())), _mtime(pack_path(p.id)))


def metric_chart(df: pd.DataFrame, unit: str, target=None, height: int = 220, log_y: bool = False) -> alt.LayerChart:
    """Lines per series (Reference dashed gray, Your catalyst in the accent) with an optional target line.
    `df` columns: T_F, value, series. Callers write the title above the chart, where long titles wrap."""
    df = df[np.isfinite(df["value"])]
    domain = ["Reference", "Your catalyst"]
    y_scale = alt.Scale(type="symlog", constant=0.1) if log_y else alt.Scale(zero=False, nice=True)
    lines = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5)
        .encode(
            x=alt.X("T_F:Q", title="Temperature (°F)", scale=alt.Scale(zero=False)),
            y=alt.Y("value:Q", title=unit, scale=y_scale),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=[REFERENCE_GRAY, ACCENT]),
                            legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[5, 4], [1, 0]]), legend=None),
            tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("T_F:Q", title="°F"),
                     alt.Tooltip("value:Q", title=unit, format=".3g")],
        )
    )
    layers = [lines]
    if target is not None:
        tdf = pd.DataFrame({"y": [target.value], "label": [f"{target.label or 'Target'}: {target.value:g}"]})
        layers.append(alt.Chart(tdf).mark_rule(color=TARGET_RED, strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q"))
        layers.append(
            alt.Chart(tdf).mark_text(align="right", dx=-4, dy=-7, color=TARGET_RED, fontSize=11)
            .encode(y="y:Q", text="label:N", x=alt.value("width"))
        )
    return alt.layer(*layers).properties(height=height)


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:.{digits}g}" if abs(v) < 1e4 else f"{v:,.0f}"


def metric_title(p: Pack, metric_id: str) -> str:
    """First sentence of a metric's description, for chart titles."""
    desc = (p.spec.metrics[metric_id].description or "").strip()
    return desc.split(". ")[0].rstrip(".") if desc else metric_id


def target_text(target) -> str:
    if target is None:
        return ""
    sign = "≥" if target.min is not None else "≤"
    return f"{sign} {target.value:g}" + (f" ({target.label})" if target.label else "")


def _tex_species(name: str) -> str:
    body = re.sub(r"([A-Za-z\)])(\d+)", r"\1_{\2}", name.replace("_", r"\_"))
    return rf"\mathrm{{{body}}}"


def equation_latex(equation: str) -> str:
    arrow = "<=>" if "<=>" in equation else "->"
    sides = []
    for side in equation.split(arrow):
        terms = []
        for term in side.split("+"):
            parts = term.split()
            coef, sp = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
            terms.append((coef + r"\,") * bool(coef) + _tex_species(sp))
        sides.append(" + ".join(terms))
    return sides[0] + (r" \rightleftharpoons " if arrow == "<=>" else r" \rightarrow ") + sides[1]


def rate_latex(pathway) -> str:
    """The rate law the engine evaluates for one pathway, written out."""
    rate = pathway.rate
    out = [r"r = k(T)"]
    for s, n in rate.orders.items():
        out.append(rf"C_{{{_tex_species(s)}}}" + ("" if n == 1 else f"^{{{n:g}}}"))
    for s in rate.coverage:
        kc = rf"K_{{{_tex_species(s)}}} C_{{{_tex_species(s)}}}"
        out.append(rf"\frac{{{kc}}}{{1 + {kc}}}")
    if rate.inhibition:
        denom = " + ".join(rf"K_{{{_tex_species(s)}}} C_{{{_tex_species(s)}}}" for s in rate.inhibition)
        m = rate.inhibition_exponent
        out.append(rf"\frac{{1}}{{(1 + {denom})" + ("" if m == 1 else f"^{{{m:g}}}") + "}")
    if "<=>" in pathway.equation:
        out.append(r"\left(1 - \frac{Q}{K_{eq}(T)}\right)")
    return r"\,".join(out)


def scenario_files() -> list[Path]:
    return sorted(PACKS_DIR.glob("*/scenarios/*.yaml"))


def catalyst_presets(pack_id: str) -> dict[str, dict[str, float]]:
    """Named catalysts declared in a pack's scenario files, usable as starting points."""
    presets = {}
    for path in sorted((pack_path(pack_id) / "scenarios").glob("*.yaml")):
        data = read_yaml(path)
        for name, cat in (data.get("catalysts") or {}).items():
            label = f"{name}: {cat.get('description', '')}".rstrip(": ")
            presets[label] = dict(cat.get("params") or {})
    return presets


# ---------------------------------------------------------------- catalyst design

def screening_pack_ids() -> list[str]:
    """Reactions that can rank materials (their pack declares a screening descriptor)."""
    return [pid for pid in pack_ids() if pack(pid).spec.screening is not None]


def link(page: str, label: str, icon: str | None = None) -> None:
    """A link to another page of the app; plain text when the page runs on its own (as in tests)."""
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.markdown(f"**{label}**")


@st.cache_data(show_spinner=False, ttl=60)
def engine_status() -> dict:
    from catalyst_engine.design import materials, screening

    return {"uma": screening.uma_setup(), "mp_key": bool(materials.api_key())}


@st.cache_data(show_spinner="Searching the Materials Project...", ttl=86400)
def mp_search(elements: tuple[str, ...], max_elements: int, add_oxygen: bool, max_e_hull: float,
              max_atoms: int) -> list[dict]:
    from catalyst_engine.design import materials

    return materials.search(list(elements), max_elements, add_oxygen, max_e_hull, max_atoms)


@st.cache_data(show_spinner=False, ttl=86400)
def mp_structure(material_id: str) -> dict:
    from catalyst_engine.design import materials

    return materials.structure(material_id)


def face_text(entry: dict) -> str:
    return " ".join(map(str, entry["facet"])) if entry.get("facet") else "auto"


def parse_face(text: str) -> list[int] | None:
    """Three Miller indices from text such as '1 1 0', '(110)' or '1,-1,0'; None for auto."""
    t = text.strip().lower().replace("(", " ").replace(")", " ").replace(",", " ")
    if t in ("", "auto"):
        return None
    parts = t.split()
    if len(parts) == 1 and len(parts[0]) == 3 and parts[0].isdigit():
        parts = list(parts[0])
    hkl = [int(x) for x in parts]
    if len(hkl) != 3 or not any(hkl):
        raise ValueError
    return hkl


def cubic_metal(entry: dict) -> bool:
    """A pure metal whose reference crystal is cubic: its face reads in the cubic cell, as textbooks write it."""
    sym = entry.get("ase_bulk")
    return bool(sym) and (reference_states[atomic_numbers[sym]] or {}).get("symmetry") in ("fcc", "bcc")


def form_text(shape: dict | None) -> str:
    """The catalyst's form in words: powder, or a coated honeycomb with its cells and wall."""
    if shape and shape.get("form") == "monolith":
        return f"coated honeycomb, {shape['cpsi']:.0f} cells per square inch, {shape['wall_mm']:g} mm wall and coat"
    return "powder"


def category_order(category: str) -> tuple[int, str]:
    """Sort key: the groups in CATEGORIES first, in that order, then any other group alphabetically."""
    return (CATEGORIES.index(category) if category in CATEGORIES else len(CATEGORIES), category)


def range_notes(pack: Pack, metric_kind: str, metric_species: list[str], t_lo: float, t_hi: float,
                lowest_value: float) -> None:
    """Notes under a design or its results: temperatures beyond the reaction's standard test, and a conversion
    that runs below zero."""
    from catalyst_engine.design import screening

    std = [t - 273.15 for t in screening.design_step(pack).kelvin()]
    if t_lo < min(std) - 25 or t_hi > max(std) + 25:
        st.info(f"This reaction's standard test runs from {min(std):.0f} to {max(std):.0f} °C, and its rates are set "
                "for that window. Results beyond it are an extrapolation, so treat them as rough.",
                icon=":material/thermostat:")
    if metric_kind == "conversion" and lowest_value < -0.005:  # half a point: below rounding noise in the solver
        gases = " and ".join(metric_species)
        leave, come = ("leave", "come") if len(metric_species) > 1 else ("leaves", "comes")
        st.warning(f"Conversion below zero means more {gases} {leave} the reactor than {come} in: at these "
                   f"temperatures the side reactions make {gases} faster than the main reaction removes it. A lower "
                   "temperature range compares the catalysts where they do useful work.", icon=":material/warning:")
