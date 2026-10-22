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

from catalyst_engine.ui import common  # noqa: E402  (imported after the freshness check above)

VIEWS = Path(__file__).parent / "views"

st.set_page_config(page_title="Catalyst engine", page_icon=":material/science:", layout="wide")
common.style()

nav = st.navigation(
    {
        "Project": [
            st.Page(VIEWS / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page(VIEWS / "labdata.py", title="Lab data", icon=":material/database:"),
            st.Page(VIEWS / "atomistic.py", title="Atomistic check", icon=":material/biotech:"),
        ],
        "Explore": [
            st.Page(VIEWS / "reactions.py", title="Reactions", icon=":material/science:"),
            st.Page(VIEWS / "playground.py", title="Catalyst playground", icon=":material/tune:"),
            st.Page(VIEWS / "scenarios.py", title="Design scenarios", icon=":material/factory:"),
        ],
        "Tools": [
            st.Page(VIEWS / "calculators.py", title="Calculators", icon=":material/calculate:"),
            st.Page(VIEWS / "fitlab.py", title="Fit lab", icon=":material/query_stats:"),
        ],
    },
    expanded=True,
)
with st.sidebar:
    st.caption("All catalyst parameters are illustrative until your review and the rig supply real ones.")
nav.run()
