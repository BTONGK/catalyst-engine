"""Design an electrocatalyst: pick an electrochemical reaction, set the potential, choose candidate materials,
start a screen."""

from __future__ import annotations

import streamlit as st

from catalyst_engine.ui import common, electro_ui

st.title("Design an electrocatalyst")
st.markdown("Pick an electrochemical reaction, set the potential, choose materials to try, and the app predicts "
            "which ones need the least voltage. Each step below takes a minute; the computer does the rest.")
advanced = st.toggle("Advanced", key="electro_advanced",
                     help="Adds a choice of crystal face for each material, the UMA mode, the slab size and the "
                          "catalyst to beat. The steps stay the same. The Manual's Methodology tab lists what stays "
                          "outside the model, and why.")

status = common.engine_status()
if not status["uma"]["ready"]:
    st.warning(f"The atomistic model (UMA) is set up on another computer. Point CE_UMA_PYTHON at a Python with "
               f"fairchem installed and CE_UMA_CHECKPOINT at the model file. Looked for {status['uma']['python']} "
               f"and {status['uma']['checkpoint']}.", icon=":material/memory:")

electro_ui.design_page(advanced, status)
