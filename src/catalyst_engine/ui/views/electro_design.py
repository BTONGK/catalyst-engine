"""Design an electrocatalyst: pick an electrochemical reaction, set the potential, choose candidate materials,
start a screen."""

from __future__ import annotations

import streamlit as st

from catalyst_engine.design import screening
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
    st.warning(f"The atomistic model (UMA) is waiting for its last install step. Once Meta approves your access "
               f"to UMA on Hugging Face, run the install command again: `{screening.INSTALL_COMMAND}`. "
               f"Developers can instead point CE_UMA_PYTHON at a Python with fairchem and CE_UMA_CHECKPOINT at "
               f"the model file. The app looked for {status['uma']['python']} and {status['uma']['checkpoint']}.",
               icon=":material/memory:")

electro_ui.design_page(advanced, status)
