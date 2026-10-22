"""Compare emission-control designs for one exhaust stream: catalyst volume, fan power, reagent, cost."""

import json
from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st

from catalyst_engine.packs import read_yaml
from catalyst_engine.ui import common
from catalyst_engine.units import parse_fraction
from catalyst_engine.value.scenario import ScenarioSpec, evaluate_scenario

st.title("Design scenarios")
st.caption(
    "One exhaust stream, several ways to clean it. The engine sizes the catalyst for each design "
    "and adds up what the customer pays for."
)

files = common.scenario_files()
if not files:
    st.info("Add a scenario file under reactions/<pack>/scenarios/ to compare designs here.")
    st.stop()

names = {f: read_yaml(f)["name"] for f in files}
path = st.selectbox("Scenario", files, format_func=names.get)
raw = read_yaml(path)
if raw.get("note"):
    st.info(" ".join(raw["note"].split()), icon=":material/info:")
st.caption(f"Source: {path.relative_to(common.REPO)}. Every number in it is an example until an integrator supplies real ones.")


@st.cache_data(show_spinner="Sizing each design...")
def run(spec_json: str) -> list[dict]:
    return [asdict(r) for r in evaluate_scenario(ScenarioSpec.model_validate_json(spec_json))]


pack = common.pack(raw["pack"])
key = f"sc:{path.name}"
with st.expander("Inputs", expanded=False, icon=":material/edit:"):
    with st.form(f"{key}:form", border=False):
        ex = raw["exhaust"]
        st.markdown("**Exhaust**")
        c = st.columns(4)
        flow = c[0].number_input("Flow (kg/s)", value=float(ex["flow_kg_s"]), min_value=1.0, step=5.0)
        t_exh = c[1].number_input("Temperature (°F)", value=float(ex["temperature_F"]), step=25.0)
        ambient = c[2].number_input("Ambient air (°F)", value=float(raw["ambient_F"]), step=5.0)
        pollutants = {}
        for i, (s, v) in enumerate(raw["pollutants"].items()):
            pollutants[s] = c[3].number_input(f"{s} at turbine exit (ppm wet)", value=parse_fraction(v) * 1e6,
                                              min_value=0.0, step=1.0, key=f"{key}:pol:{s}")
        ratio = None
        if raw.get("reagent"):
            rg = raw["reagent"]
            lo_hi = rg["ratio"] if isinstance(rg["ratio"], list) else [rg["ratio"], rg["ratio"]]
            ratio = st.slider(f"{rg['species']} dose range (mol per mol of {' + '.join(rg['to'])})", 0.5, 2.0,
                              (float(lo_hi[0]), float(lo_hi[1])), step=0.05,
                              help="Sizing picks the dose inside this range that needs the least catalyst.")

        st.markdown("**Targets**")
        targets = {}
        c = st.columns(max(len(raw["targets"]), 1))
        for col, (mid, bound) in zip(c, raw["targets"].items()):
            side = "min" if "min" in bound else "max"
            label = f"{mid} {'≥' if side == 'min' else '≤'}"
            targets[mid] = {side: col.number_input(label, value=float(bound[side]), key=f"{key}:t:{mid}",
                                                   help=pack.spec.metrics[mid].description if mid in pack.spec.metrics else None)}

        st.markdown("**Hardware and prices**")
        mo, fan, eco = raw["monolith"], raw["fan"], raw["economics"]
        c = st.columns(4)
        cpsi = c[0].number_input("Cell density (cpsi)", value=float(mo["cpsi"]), step=8.0)
        wall = c[1].number_input("Wall (mm)", value=float(mo["wall_mm"]), step=0.05, format="%.2f")
        face = c[2].number_input("Face velocity (m/s)", value=float(mo["face_velocity_m_s"]), step=0.5)
        dp = c[3].number_input("Fan pressure rise (kPa)", value=float(fan["pressure_rise_kPa"]), step=0.5)
        c = st.columns(4)
        eff = c[0].number_input("Fan efficiency", value=float(fan["efficiency"]), min_value=0.1, max_value=1.0, step=0.05)
        power = c[1].number_input("Electricity ($/MWh)", value=float(eco["electricity_usd_per_mwh"]), step=5.0)
        hours = c[2].number_input("Hours per year", value=float(eco["hours_per_year"]), step=500.0, max_value=8760.0)
        reagent_price = c[3].number_input("Reagent ($/tonne)", value=float(eco.get("reagent_usd_per_tonne", 0.0)), step=50.0)

        st.markdown("**Designs**")
        designs = st.data_editor(
            pd.DataFrame(raw["designs"]).reindex(columns=["name", "catalyst", "tempered_to_F"]),
            num_rows="dynamic", hide_index=True, key=f"{key}:designs",
            column_config={
                "name": st.column_config.TextColumn("Design", required=True),
                "catalyst": st.column_config.SelectboxColumn("Catalyst", options=list(raw["catalysts"]), required=True),
                "tempered_to_F": st.column_config.NumberColumn("Cool exhaust to (°F)", help="Leave blank to run at full exhaust temperature."),
            },
        )
        for name, cat in raw["catalysts"].items():
            st.caption(f"**{name}**: {cat.get('description', '')}")
        st.form_submit_button("Run scenario", type="primary", icon=":material/play_arrow:")

spec = dict(raw)
spec["exhaust"] = ex | {"flow_kg_s": flow, "temperature_F": t_exh}
spec["ambient_F"] = ambient
spec["pollutants"] = {s: f"{v} ppm" for s, v in pollutants.items()}
if ratio is not None:
    spec["reagent"] = raw["reagent"] | {"ratio": list(ratio)}
spec["targets"] = targets
spec["monolith"] = {"cpsi": cpsi, "wall_mm": wall, "face_velocity_m_s": face}
spec["fan"] = {"pressure_rise_kPa": dp, "efficiency": eff}
spec["economics"] = {"electricity_usd_per_mwh": power, "hours_per_year": hours, "reagent_usd_per_tonne": reagent_price}
spec["designs"] = [
    {k: v for k, v in row.items() if not (k == "tempered_to_F" and pd.isna(v))}
    for row in designs.dropna(subset=["name", "catalyst"]).to_dict("records")
]
try:
    results = run(json.dumps(spec, default=float))
except Exception as exc:  # show input problems on the page
    st.error(f"These inputs give an error: {exc}")
    st.stop()

feasible = [r for r in results if r["feasible"]]
least = min(feasible, key=lambda r: r["catalyst_volume_m3"])["name"] if feasible else None
cheapest = min(feasible, key=lambda r: r["fan_energy_usd_per_year"] + r["reagent_usd_per_year"])["name"] if feasible else None
reagent = raw.get("reagent", {}).get("species", "Reagent")

st.subheader("Results")
cols = st.columns(len(results))
for col, r in zip(cols, results):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{r['name']}**")
        st.caption(f"Catalyst at {r['catalyst_T_F']:,.0f} °F · {r['catalyst']}")
        badges = [":green-badge[Meets every target]" if r["feasible"] else ":red-badge[Out of reach at any catalyst volume]"]
        badges += [":violet-badge[Least catalyst]"] * (r["name"] == least)
        badges += [":blue-badge[Lowest operating cost]"] * (r["name"] == cheapest)
        st.markdown(" ".join(badges))
        if r["feasible"]:
            st.metric("Catalyst volume", f"{r['catalyst_volume_m3']:,.1f} m³")
            st.metric("Operating cost per year", f"${r['fan_energy_usd_per_year'] + r['reagent_usd_per_year']:,.0f}")
        st.metric("Tempering air", f"{r['tempering_mass_ratio']:.0%} of exhaust mass")
        st.metric("Fan power", f"{r['fan_power_kW']:,.0f} kW")
        if r["reagent_ratio"] is not None and r["feasible"]:
            st.metric(f"{reagent} dose ratio", f"{r['reagent_ratio']:.2f}")

if feasible:
    left, right = st.columns(2, gap="large")
    vol = pd.DataFrame({"design": [r["name"] for r in feasible], "m3": [r["catalyst_volume_m3"] for r in feasible]})
    design_axis = alt.X("design:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=220))
    left.markdown("**Catalyst to buy**")
    left.altair_chart(
        alt.Chart(vol).mark_bar(color=common.ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=56)
        .encode(x=design_axis, y=alt.Y("m3:Q", title="Catalyst volume (m³)"),
                tooltip=["design:N", alt.Tooltip("m3:Q", format=",.1f")])
        .properties(height=280),
        width="stretch",
    )
    cost = pd.DataFrame(
        [{"design": r["name"], "item": "Fan electricity", "usd": r["fan_energy_usd_per_year"]} for r in feasible]
        + [{"design": r["name"], "item": f"{reagent}", "usd": r["reagent_usd_per_year"]} for r in feasible]
    )
    right.markdown("**Operating cost per year**")
    right.altair_chart(
        alt.Chart(cost).mark_bar(size=56)
        .encode(x=design_axis, y=alt.Y("sum(usd):Q", title="$ per year", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("item:N", scale=alt.Scale(range=["#f59e0b", common.ACCENT]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["design:N", "item:N", alt.Tooltip("usd:Q", format="$,.0f")])
        .properties(height=280),
        width="stretch",
    )

with st.expander("Every number, side by side"):
    rows = {
        "Catalyst temperature (°F)": lambda r: f"{r['catalyst_T_F']:,.0f}",
        "Meets every target": lambda r: "yes" if r["feasible"] else "out of reach",
        "Tempering air (share of exhaust mass)": lambda r: f"{r['tempering_mass_ratio']:.1%}",
        f"{reagent} dose ratio": lambda r: f"{r['reagent_ratio']:.2f}" if r["reagent_ratio"] is not None else "–",
        "Space velocity (1/h)": lambda r: f"{r['ghsv_per_h']:,.0f}" if r["ghsv_per_h"] else "–",
        "Catalyst volume (m³)": lambda r: f"{r['catalyst_volume_m3']:,.1f}" if r["catalyst_volume_m3"] else "–",
        "Face area (m²)": lambda r: f"{r['face_area_m2']:,.0f}" if r["face_area_m2"] else "–",
        "Bed depth (m)": lambda r: f"{r['bed_depth_m']:.2f}" if r["bed_depth_m"] else "–",
        "Pressure drop (Pa)": lambda r: f"{r['pressure_drop_Pa']:,.0f}" if r["pressure_drop_Pa"] else "–",
        "Fan power (kW)": lambda r: f"{r['fan_power_kW']:,.0f}",
        "Fan electricity ($/yr)": lambda r: f"{r['fan_energy_usd_per_year']:,.0f}",
        f"{reagent} (tonnes/yr)": lambda r: f"{r['reagent_t_per_year']:,.1f}",
        f"{reagent} ($/yr)": lambda r: f"{r['reagent_usd_per_year']:,.0f}",
    }
    for mid in targets:
        rows[f"{mid} at the sized volume"] = lambda r, mid=mid: common.fmt(r["metrics"].get(mid))
    table = pd.DataFrame({r["name"]: [f(r) for f in rows.values()] for r in results}, index=list(rows))
    st.dataframe(table)

with st.expander("How the engine sizes each design"):
    st.markdown(
        """
1. **Tempering.** If a design cools the exhaust first, an enthalpy balance gives the ambient air it takes,
   using real heat capacities of the exhaust's water and CO2. That air dilutes the pollutants and needs a fan.
2. **Dose.** The reagent is dosed across the chosen range; for each dose the engine finds the highest space
   velocity (least catalyst) where the outlet still meets every target.
3. **Honeycomb physics.** Each channel limits how fast gas reaches the wall. The model includes this mass
   transfer, which caps what any catalyst can do at high temperature and high flow.
4. **Cost.** Catalyst volume comes from the flow and the space velocity; fan power from the tempering-air flow
   and pressure rise; reagent from the dose and the flow.
"""
    )
