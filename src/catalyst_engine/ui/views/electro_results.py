"""Results of electrochemical screens: progress while one runs, then the ranking by predicted speed at the comparison
potential (overpotential breaks ties), the activity against potential, the free-energy diagram, the volcano and the
3D views."""

from __future__ import annotations

import streamlit as st

from catalyst_engine.design import electro, screening
from catalyst_engine.ui import common, electro_ui

st.title("Electrocatalyst results")

screens = electro.list_screens()
if not screens:
    st.info("Your electrochemical screens appear here. Start one on the Design page and its results fill in as "
            "they come in.", icon=":material/bolt:")
    common.link("views/electro_design.py", "Design an electrocatalyst", ":material/add_circle:")
    st.stop()

names = [f.name for f in screens]
wanted = st.session_state.get("electro_results_screen")
choice = st.selectbox("Screen", names, index=names.index(wanted) if wanted in names else 0,
                      format_func=lambda n: electro_ui.screen_label(electro.screens_dir() / n), key="electro_results_pick")
st.session_state["electro_results_screen"] = choice
folder = electro.screens_dir() / choice
job = screening.read_job(folder)
if job["request"]["pack"] not in electro.list_electro():
    st.error(f"This screen used the reaction '{job['request']['pack_name']}', whose file is missing from "
             "reactions/electro/. Add it back to see the results.")
    st.stop()

if screening.read_status(folder).get("state") in screening.RUNNING:
    @st.fragment(run_every=5)
    def live() -> None:
        electro_ui.results_body(folder, choice)
        if screening.read_status(folder).get("state") not in screening.RUNNING:
            st.rerun()

    live()
else:
    electro_ui.results_body(folder, choice)
