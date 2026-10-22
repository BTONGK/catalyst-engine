"""Quick answers: tempering air, megawatts under a permit cap, concentration bases."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from catalyst_engine.ui import common
from catalyst_engine.units import Basis, convert, f_to_k
from catalyst_engine.value.permits import MOLAR_MASS, lb_per_mmbtu, max_mw_under_cap, tons_per_year
from catalyst_engine.value.tempering import tempering_air

EXAMPLE_EXHAUST = {"N2": 74.5, "O2": 12.5, "CO2": 3.9, "H2O": 8.3, "AR": 0.8}

st.title("Calculators")
temper_tab, permit_tab, basis_tab = st.tabs(["Tempering air", "Permit headroom", "Concentration basis"])


@st.cache_data(show_spinner=False)
def temper(exhaust: tuple[tuple[str, float], ...], t_exh_f: float, t_target_f: float, t_amb_f: float):
    r = tempering_air(dict(exhaust), f_to_k(t_exh_f), f_to_k(t_target_f), f_to_k(t_amb_f))
    return r.mass_ratio, r.mole_ratio, r.mixed_mole_fractions


with temper_tab:
    st.caption(
        "Conventional SCR catalysts work best below about 900 °F, so hot turbine exhaust gets cooled with "
        "ambient air first. This is the air a hot-running catalyst saves."
    )
    left, right = st.columns([1, 2], gap="large")
    with left:
        t_exh = st.slider("Exhaust temperature (°F)", 700, 1300, 1150, step=10)
        t_target = st.slider("Catalyst limit (°F)", 500, 1100, 900, step=10)
        t_amb = st.slider("Ambient air (°F)", 0, 120, 80, step=5)
        with st.expander("Exhaust composition (% by volume, wet)"):
            comp = {s: st.number_input(s, value=v, min_value=0.0, step=0.1, key=f"ex_{s}") for s, v in EXAMPLE_EXHAUST.items()}
        exhaust = tuple((s, v / 100) for s, v in comp.items())
    with right:
        if t_target >= t_exh:
            st.success("The exhaust is already at or below the catalyst limit: zero tempering air.", icon=":material/check_circle:")
        elif t_amb >= t_target:
            st.info("Pick ambient air cooler than the catalyst limit.", icon=":material/info:")
        else:
            mass, mole, mix = temper(exhaust, t_exh, t_target, t_amb)
            c = st.columns(3)
            c[0].metric("Air per kg of exhaust", f"{mass:.1%}", border=True)
            c[1].metric("Extra gas flow", f"{mole:.1%}", border=True, help="Added gas through the catalyst, by moles (so also by normal volume).")
            c[2].metric("O2 after mixing", f"{100 * mix.get('O2', 0):.1f}%", border=True)
            limits = np.arange(max(t_amb + 50, 500), t_exh, 10.0)
            curve = pd.DataFrame({"limit_F": limits, "air": [temper(exhaust, t_exh, float(t), t_amb)[0] for t in limits]})
            here = pd.DataFrame({"limit_F": [t_target], "air": [mass]})
            chart = (
                alt.Chart(curve).mark_line(color=common.ACCENT, strokeWidth=2.5)
                .encode(x=alt.X("limit_F:Q", title="Catalyst temperature limit (°F)"),
                        y=alt.Y("air:Q", title="Air per kg of exhaust", axis=alt.Axis(format="%")),
                        tooltip=[alt.Tooltip("limit_F:Q", title="°F"), alt.Tooltip("air:Q", format=".1%")])
                + alt.Chart(here).mark_point(size=120, filled=True, color=common.TARGET_RED).encode(x="limit_F:Q", y="air:Q")
            ).properties(title=f"Tempering air for {t_exh:,} °F exhaust: every degree of catalyst tolerance saves air", height=320)
            st.altair_chart(chart, width="stretch")

with permit_tab:
    st.caption(
        "Data-center turbine fleets often sit under an annual tons-per-year cap. A cleaner stack fits more megawatts "
        "under the same cap. Uses EPA Method 19 for natural gas."
    )
    left, right = st.columns([1, 2], gap="large")
    with left:
        species = st.selectbox("Pollutant", list(MOLAR_MASS), help="NOx is counted as NO2, as permits do.")
        ppm = st.number_input("Stack concentration (ppmvd @ 15% O2)", value=2.0, min_value=0.05, step=0.5)
        cap = st.number_input("Annual cap (short tons)", value=100.0, min_value=1.0, step=10.0, format="%.0f")
        heat_rate = st.number_input("Heat rate (Btu/kWh, HHV)", value=9500.0, step=100.0, format="%.0f")
        hours = st.number_input("Operating hours per year", value=8760.0, max_value=8760.0, step=100.0, format="%.0f")
    with right:
        rate = lb_per_mmbtu(ppm, species)
        mw = max_mw_under_cap(cap, rate, heat_rate, hours)
        c = st.columns(3)
        c[0].metric("Emission rate (lb/MMBtu)", f"{rate:.4f}", border=True)
        c[1].metric("Tons per year per MW", f"{tons_per_year(rate, 1.0, heat_rate, hours):.3f}", border=True)
        c[2].metric("Capacity under the cap", f"{mw:,.0f} MW", border=True)
        grid = np.round(np.linspace(0.5, 9.0, 35), 2)
        df = pd.DataFrame({"ppm": grid, "mw": [max_mw_under_cap(cap, lb_per_mmbtu(v, species), heat_rate, hours) for v in grid]})
        chart = (
            alt.Chart(df).mark_line(color=common.ACCENT, strokeWidth=2.5)
            .encode(x=alt.X("ppm:Q", title="ppmvd @ 15% O2"), y=alt.Y("mw:Q", title="MW under the cap"),
                    tooltip=[alt.Tooltip("ppm:Q", format=".2f"), alt.Tooltip("mw:Q", format=",.0f")])
            + alt.Chart(pd.DataFrame({"ppm": [ppm], "mw": [mw]})).mark_point(size=120, filled=True, color=common.TARGET_RED)
            .encode(x="ppm:Q", y="mw:Q")
        ).properties(title="Halving the stack concentration doubles the megawatts", height=320)
        st.altair_chart(chart, width="stretch")

with basis_tab:
    st.caption(
        "The same gas reads differently wet, dry, and corrected to a reference O2. Permits use ppmvd @ 15% O2; "
        "an FTIR on a hot line reads wet."
    )
    left, right = st.columns([1, 2], gap="large")
    bases = {"ppm wet": Basis(), "ppmvd (dry)": Basis(dry=True)}
    with left:
        value = st.number_input("Reading", value=2.0, min_value=0.0, step=0.5)
        o2_ref = st.number_input("Reference O2 for correction (%)", value=15.0, min_value=0.0, max_value=20.0, step=1.0)
        bases[f"ppmvd @ {o2_ref:g}% O2"] = Basis(dry=True, o2_ref_pct=o2_ref)
        src_label = st.selectbox("Reading is in", list(bases))
        h2o = st.slider("Water in the gas (% wet)", 0.0, 30.0, 10.0, step=0.5)
        o2 = st.slider("O2 in the gas (% wet)", 0.0, 20.0, 15.0, step=0.5)
    with right:
        try:
            rows = [{"Basis": label, "Value": float(convert(value, bases[src_label], b, h2o_frac=h2o / 100, o2_wet_frac=o2 / 100))}
                    for label, b in bases.items()]
            c = st.columns(len(rows))
            for col, row in zip(c, rows):
                col.metric(row["Basis"], f"{row['Value']:.3g}", border=True)
            dry_o2 = 100 * (o2 / 100) / (1 - h2o / 100)
            st.markdown(
                f"This gas holds **{dry_o2:.1f}% O2 on a dry basis**. The O2 correction scales a dry reading by "
                f"(20.9 − {o2_ref:g}) / (20.9 − {dry_o2:.1f}) = **{(20.9 - o2_ref) / (20.9 - dry_o2):.3f}**."
            )
        except ValueError as exc:
            st.info(f"{exc}. Lower the O2 or the water to correct this gas.", icon=":material/info:")
