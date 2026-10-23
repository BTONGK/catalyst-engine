"""Home: what the app does, the three steps, what it predicts, the engine's status and the latest screens."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from catalyst_engine.design import screening
from catalyst_engine.ui import common

st.title("Catalyst engine")
st.markdown("##### Design catalysts on a computer. Pick a reaction, choose materials to try, and the app predicts "
            "which ones work best, with a 3D view of every surface it computed.")

with st.container(horizontal=True):
    common.link("views/design.py", "Design a catalyst", ":material/add_circle:")
    common.link("views/manual.py", "Read the manual (5 minutes)", ":material/menu_book:")
    common.link("views/materials.py", "Explore materials", ":material/view_in_ar:")

st.markdown("#### How it works")
steps = [
    (":material/science:", "1. Pick a reaction",
     "Choose what the catalyst should do, such as cleaning exhaust or making ammonia or methanol, and set the "
     "temperature, pressure and gas mix of your process."),
    (":material/grid_view:", "2. Choose materials",
     "Tick pure metals, or search the Materials Project's 150,000+ crystals for alloys and oxides made of the "
     "elements you name."),
    (":material/memory:", "3. Let the computer work",
     "An AI model from Meta (UMA) places the key molecules on each material's surface and computes how strongly "
     "they stick. A screen of ten materials takes a few minutes to an hour on this computer."),
    (":material/leaderboard:", "4. Read the results",
     "A ranking against the catalyst to beat, predicted performance across your temperatures, the activity "
     "volcano, and 3D structures you can turn with the mouse."),
]
for col, (icon, title, text) in zip(st.columns(len(steps), gap="medium"), steps):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{icon} {title}**")
        st.write(text)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("#### What the app predicts for each material")
    st.markdown(
        "- **How strongly the key molecules stick** to its surface (binding energy, in eV), computed atom by atom.\n"
        "- **How fast it runs** compared with the catalyst to beat, read off the activity volcano.\n"
        "- **Its performance curve:** conversion or yield across your temperature range, in a reactor at your "
        "pressure, gas mix and flow.\n"
        "- **Its light-off temperature:** where it reaches half of the best possible result.\n"
        "- **A ranking** with a likely range for every number, so close calls show as close.\n"
        "- **How easy its elements are to source:** abundant, uncommon, scarce or precious.\n"
        "- **Its 3D structure:** the crystal, the surface, and each molecule in its best spot.")
    st.markdown("#### Where the numbers come from")
    st.markdown(
        "| Part | Source |\n|---|---|\n"
        "| Crystal structures | The Materials Project (materialsproject.org), plus reference crystals for pure metals |\n"
        "| Binding energies | UMA, Meta FAIR Chemistry's machine-learned model, trained on millions of quantum "
        "chemistry (DFT) calculations; it runs on this computer |\n"
        "| Activity | The Sabatier volcano: the best catalyst holds the key molecule just right, firmly enough to react and lightly enough to let go |\n"
        "| Performance curves | The app's reactor model, with equilibrium limits from Cantera's thermodynamic data |")
    st.caption("The Manual's Methodology tab gives every equation and assumption, and the check behind each step.")

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
        ids = common.screening_pack_ids()
        st.markdown(f":green[:material/check_circle:] **{len(ids)} reactions** ready to design for")

    st.markdown("#### Reactions ready now")
    by_cat: dict[str, list[str]] = {}
    for pid in ids:
        by_cat.setdefault(common.pack(pid).spec.category, []).append(common.pack(pid).spec.name)
    for cat in sorted(by_cat, key=common.category_order):
        st.markdown(f"**:gray[{cat.upper()}]**  \n" + " ".join(f":blue-badge[{n}]" for n in sorted(by_cat[cat])))
    common.link("views/reactions.py", "Add your own reaction", ":material/add:")

st.markdown("#### Your latest screens")
screens = screening.list_screens()[:6]
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
            if c.button("Open", key=f"home_open_{folder.name}", icon=":material/arrow_forward:"):
                st.session_state["results_screen"] = folder.name
                st.switch_page("views/results.py")
else:
    st.caption("Your screens appear here once you start one on the Design page.")
