"""Electrocatalysis home: what this section does, how it works, what it predicts, and the latest screens."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from catalyst_engine.design import electro, screening
from catalyst_engine.ui import common, electro_ui

st.title("Electrocatalyst engine")
st.markdown("##### Design electrocatalysts on a computer. Pick an electrochemical reaction, choose materials to try, "
            "and the app predicts the potential each one needs, with the free-energy diagram behind every number.")

with st.container(horizontal=True):
    common.link("views/electro_design.py", "Design an electrocatalyst", ":material/add_circle:")
    common.link("views/electro_manual.py", "Read the manual (5 minutes)", ":material/menu_book:")
    common.link("views/electro_playground.py", "Try the free-energy playground", ":material/tune:")

st.markdown("#### How it works")
steps = [
    (":material/bolt:", "1. Pick a reaction",
     "Choose what the electrode should do: evolve hydrogen, reduce oxygen in a fuel cell, or evolve oxygen in an "
     "electrolyzer. Set the potential to compare catalysts at, in volts on the hydrogen-electrode scale."),
    (":material/grid_view:", "2. Choose materials",
     "Tick pure metals, or search the Materials Project's 150,000+ crystals for alloys and oxides made of the "
     "elements you name."),
    (":material/memory:", "3. Let the computer work",
     "An AI model from Meta (UMA) places each intermediate of the reaction path on every material's surface and "
     "computes how strongly it binds. A screen of ten metals takes under an hour on this computer."),
    (":material/leaderboard:", "4. Read the results",
     "A ranking by overpotential against the catalyst to beat, each material's speed across potentials, its "
     "free-energy diagram, the volcano, and 3D structures you can turn with the mouse."),
]
for col, (icon, title, text) in zip(st.columns(len(steps), gap="medium"), steps):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{icon} {title}**")
        st.write(text)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("#### What the app predicts for each material")
    st.markdown(
        "- **The free energy of each intermediate** on its surface (eV), computed atom by atom and corrected to "
        "room temperature in water.\n"
        "- **Its limiting potential:** the voltage where the last step of the path turns downhill.\n"
        "- **Its overpotential:** how far that sits from the equilibrium potential; smaller is better.\n"
        "- **Which step limits it,** so you know what a better material has to change.\n"
        "- **Its speed against the catalyst to beat** at your potential, with a likely range.\n"
        "- **Its place on the volcano,** and whether it breaks the scaling relations that cap most surfaces.\n"
        "- **How easy its elements are to source,** and its 3D structure with every intermediate in place.")
    st.markdown("#### Where the numbers come from")
    st.markdown(
        "| Part | Source |\n|---|---|\n"
        "| Crystal structures | The Materials Project, plus reference crystals for pure metals |\n"
        "| Binding energies | UMA, Meta FAIR Chemistry's machine-learned model, trained on millions of DFT calculations; "
        "it runs on this computer |\n"
        "| Free energies | Published zero-point, entropy and water corrections (Nørskov et al. 2004, 2005) |\n"
        "| Potentials | The computational hydrogen electrode: a proton and electron at 0 V count as half a hydrogen "
        "molecule, and the potential tilts every step by one electron-volt per volt |\n"
        "| Speed | Sabatier analysis: the largest step still uphill at a potential sets the rate |")
    st.caption("The Manual's Methodology tab gives every equation and assumption, and the check against published "
               "tables behind each number.")

with right:
    status = common.engine_status()
    st.markdown("#### Ready to run")
    with st.container(border=True):
        uma = status["uma"]
        if uma["ready"]:
            st.markdown(":green[:material/check_circle:] **Atomistic model (UMA)** ready on this computer")
        else:
            st.markdown(":orange[:material/pending:] **Atomistic model (UMA)** needs setting up; the manual's "
                        "\"When something looks off\" section shows how")
        if status["mp_key"]:
            st.markdown(":green[:material/check_circle:] **Materials Project** connected")
        else:
            st.markdown(":orange[:material/key:] **Materials Project** needs a free API key to search crystals; "
                        "pure metals work right away")
        ids = electro.list_electro()
        st.markdown(f":green[:material/check_circle:] **{len(ids)} electrochemical reactions** ready to design for")

    st.markdown("#### Reactions ready now")
    specs = [electro_ui.spec(i) for i in ids]
    for s in specs:
        st.markdown(f":blue-badge[{s.name}] :gray[{s.equation} · equilibrium {s.equilibrium_V:.2f} V vs RHE · "
                    f"catalyst to beat {s.reference.name or s.reference.formula}]")
    st.caption("A new reaction is a file in reactions/electro/; the Manual's Methodology tab shows the form.")

st.markdown("#### Your latest electrochemical screens")
screens = electro.list_screens()[:6]
if screens:
    for folder in screens:
        req = screening.read_job(folder)["request"]
        state = screening.read_status(folder).get("state", "queued")
        badge = {"finished": ("Finished", "green"), "cancelled": ("Stopped", "gray"), "error": ("Error", "red"),
                 "stopped": ("Stopped", "gray")}.get(state, ("Running", "blue"))
        with st.container(border=True):
            a, b, c = st.columns([4, 1, 1], vertical_alignment="center")
            when = datetime.fromisoformat(req["created"]).strftime("%b %d, %H:%M")
            a.markdown(f"**{req['name']}**  \n:gray[{req['pack_name']} · {len(req['materials'])} materials · {when}]")
            b.badge(badge[0], color=badge[1])
            if c.button("Open", key=f"ehome_open_{folder.name}", icon=":material/arrow_forward:"):
                st.session_state["electro_results_screen"] = folder.name
                st.switch_page("views/electro_results.py")
else:
    st.caption("Your screens appear here once you start one on the Design page.")
