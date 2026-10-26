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
ids = electro.list_electro()
specs = [electro_ui.spec(i) for i in ids]
steps = [
    (":material/bolt:", "1. Pick a reaction",
     "Choose what the electrode should do, such as evolve hydrogen in an electrolyzer or reduce oxygen in a fuel "
     "cell. Set the potential to compare catalysts at, in volts on the hydrogen-electrode scale."),
    (":material/grid_view:", "2. Choose materials",
     "Tick pure metals, or search the Materials Project's 150,000+ crystals for alloys made of the elements "
     "you name."),
    (":material/memory:", "3. Let the computer work",
     "An AI model from Meta (UMA) places each intermediate of the reaction path on every material's surface and "
     "computes how strongly it binds. A screen of ten metals takes under an hour on this computer."),
    (":material/leaderboard:", "4. Read the results",
     "A ranking by predicted speed at your potential against the catalyst to beat, with each material's "
     "overpotential and likely range, its free-energy diagram, the volcano, and 3D structures you can turn with "
     "the mouse."),
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
        "- **Its limiting potential:** the voltage where every step that moves an electron runs downhill. Where a "
        "reaction has two routes, as oxygen reduction does, the better route counts.\n"
        "- **Its overpotential:** how far that sits from the equilibrium potential; smaller is better.\n"
        "- **Which step limits it, and by which route,** so you know what a better material has to change.\n"
        "- **Its speed against the catalyst to beat** at your potential, with the exact likely range when every "
        "binding energy moves within its error bar.\n"
        "- **Its place on the volcano,** and whether it breaks the scaling relations that cap most surfaces.\n"
        "- **How easy its elements are to source,** and its 3D structure with every intermediate in place.")
    st.markdown("#### Where the numbers come from")
    st.markdown(
        "| Part | Source |\n|---|---|\n"
        "| Crystal structures | The Materials Project, plus reference crystals for pure metals |\n"
        "| Binding energies | UMA, Meta FAIR Chemistry's machine-learned model, trained on millions of DFT calculations; "
        "it runs on this computer |\n"
        "| Free energies | Published zero-point, entropy and water corrections (Nørskov et al. 2004 and 2005, "
        "Rossmeisl et al. 2007, Skúlason et al. 2012, Peterson et al. 2010), UMA's own vibrations for the "
        "intermediates that await a published value, each marked on its results page, and tabulated formation free "
        "energies (the NBS tables) that put each reaction on its own scale |\n"
        "| Potentials | The computational hydrogen electrode: a proton and electron at 0 V count as half a hydrogen "
        "molecule, and the potential tilts every step that moves an electron by one electron-volt per volt; a "
        "chemical step, such as a molecule leaving the surface, stays put |\n"
        "| Speed | Sabatier analysis: on each route the largest hurdle at a potential, a step still uphill or a "
        "chemical barrier such as splitting O2, and the faster route sets the rate (Nørskov et al. 2004) |\n"
        "| Likely ranges | Every binding energy moved within UMA's typical error, ±0.15 eV, all at once; the exact "
        "highest and lowest values by linear programming |")
    st.caption("The Manual's Methodology tab gives every equation and assumption, the check against published "
               "tables behind each number, and what stays outside the method with its size.")

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
        st.markdown(f":green[:material/check_circle:] **{len(ids)} electrochemical reactions** ready to design for")
        checks = [s.check for s in specs]
        st.markdown(f":blue[:material/fact_check:] Checks against published DFT and measurements: "
                    f"**{checks.count('passed')}** passed, **{checks.count('missed')}** missed, "
                    f"**{checks.count('awaiting')}** await published data. The Design page shows each one's verdict.")

    st.markdown("#### Reactions ready now")
    for s in sorted(specs, key=lambda s: ["passed", "awaiting", "missed"].index(s.check)):
        st.markdown(f":{common.CHECK_LOOK[s.check][1]}-badge[{s.name}] :gray[{s.equation} · equilibrium "
                    f"{s.equilibrium_V:.2f} V vs RHE · catalyst to beat {s.reference.name or s.reference.formula}]")
    st.caption("Green: checked against published results. Blue: a part awaits published data. Orange: a part missed "
               "its check. A new reaction is a file in reactions/electro/; the Manual's Methodology tab shows the form.")

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
