"""Catalyst engine app. Launch with `uv run ce ui`."""

import sys
from pathlib import Path

import streamlit as st

import catalyst_engine

# Streamlit reruns this script and the pages on every load but keeps imported modules. When any engine source file
# is newer than the loaded engine, drop the engine's modules and the app caches so this run imports fresh code.
newest_source = max(f.stat().st_mtime for f in Path(catalyst_engine.__file__).parent.rglob("*.py"))
if newest_source > getattr(catalyst_engine, "LOADED_AT", 0):
    for name in [n for n in sys.modules if n == "catalyst_engine" or n.startswith("catalyst_engine.")]:
        del sys.modules[name]
    st.cache_data.clear()
    st.cache_resource.clear()

from catalyst_engine.design import screening  # noqa: E402  (imported after the freshness check above)
from catalyst_engine.ui import common  # noqa: E402

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="Catalyst engine", page_icon=":material/science:", layout="wide")
common.style()
screening.add_examples()

nav = st.navigation(
    {
        "Start": [
            st.Page(VIEWS / "home.py", title="Home", icon=":material/home:", default=True),
            st.Page(VIEWS / "manual.py", title="Manual", icon=":material/menu_book:"),
        ],
        "Design": [
            st.Page(VIEWS / "design.py", title="Design a catalyst", icon=":material/add_circle:"),
            st.Page(VIEWS / "results.py", title="Results", icon=":material/leaderboard:"),
            st.Page(VIEWS / "materials.py", title="Materials", icon=":material/view_in_ar:"),
            st.Page(VIEWS / "reactions.py", title="Reactions", icon=":material/science:"),
        ],
        "Simulate": [
            st.Page(VIEWS / "playground.py", title="Catalyst playground", icon=":material/tune:"),
            st.Page(VIEWS / "scenarios.py", title="Design scenarios", icon=":material/factory:"),
            st.Page(VIEWS / "calculators.py", title="Calculators", icon=":material/calculate:"),
            st.Page(VIEWS / "fitlab.py", title="Fit lab", icon=":material/query_stats:"),
        ],
        "Check accuracy": [
            st.Page(VIEWS / "atomistic.py", title="Model accuracy", icon=":material/biotech:"),
            st.Page(VIEWS / "literature.py", title="Literature check", icon=":material/fact_check:"),
        ],
        "Lab (optional)": [
            st.Page(VIEWS / "labdata.py", title="Lab data", icon=":material/database:"),
        ],
        "Project": [
            st.Page(VIEWS / "overview.py", title="Project status", icon=":material/dashboard:"),
        ],
    },
    expanded=True,
)
with st.sidebar:
    engine = common.engine_status()
    st.caption(("UMA ready" if engine["uma"]["ready"] else "UMA needs setting up") + " · "
               + ("Materials Project connected" if engine["mp_key"] else "Materials Project key needed"))
nav.run()
