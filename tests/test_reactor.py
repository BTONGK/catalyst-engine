import numpy as np
import pytest

from catalyst_engine.chem import thermo
from catalyst_engine.chem.formula import element_counts
from catalyst_engine.chem.network import Network, NetworkSpec
from catalyst_engine.chem.reactor import Conditions, outlet, outlet_rk4, solve
from catalyst_engine.packs import load_pack
from catalyst_engine.units import P_ATM, T_STD


def _first_order_network() -> Network:
    spec = NetworkSpec.model_validate({
        "reference_temperature_K": 800.0,
        "species": {s: {"formula": s} for s in ["N2O", "N2", "O2"]},
        "pathways": {"decomp": {"equation": "N2O -> N2 + 0.5 O2", "role": "desired",
                                "rate": {"key": "N2O", "orders": {"N2O": 1}, "ln_k_ref": np.log(20.0), "Ea_kJ": 100}}},
    })
    return Network.compile(spec)


def test_first_order_matches_analytic_conversion():
    net = _first_order_network()
    t, ghsv = 800.0, 30000.0
    feed = np.zeros(3)
    feed[net.index("N2O")], feed[net.index("N2")] = 100e-6, 1 - 100e-6
    out = outlet(net, {}, Conditions(t, P_ATM, ghsv, feed[None, :]))
    tau = 3600.0 / ghsv * T_STD / t  # actual residence time per unit catalyst volume
    expected = 1.0 - np.exp(-20.0 * tau)
    conversion = 1.0 - out[0, net.index("N2O")] / feed[net.index("N2O")]
    assert conversion == pytest.approx(expected, rel=2e-4)  # dilute: mole change shifts this < 1e-4


def test_mass_transfer_caps_a_fast_reaction():
    net = _first_order_network()
    t, ghsv, km_a = 800.0, 30000.0, 5.0
    feed = np.zeros(3)
    feed[net.index("N2O")], feed[net.index("N2")] = 100e-6, 1 - 100e-6
    cond = Conditions(t, P_ATM, ghsv, feed[None, :], km_a=np.array([[km_a]]))
    out = outlet(net, {"decomp.ln_k_ref": np.log(1e6)}, cond)
    tau = 3600.0 / ghsv * T_STD / t
    conversion = 1.0 - out[0, net.index("N2O")] / feed[net.index("N2O")]
    assert conversion == pytest.approx(1.0 - np.exp(-km_a * tau), rel=1e-3)


def test_washcoat_diffusion_scales_a_first_order_rate_by_the_slab_effectiveness():
    net = _first_order_network()
    t, ghsv, k, phi2_per_k = 800.0, 30000.0, 20.0, 0.2  # phi = 2, eta = tanh(2)/2
    feed = np.zeros(3)
    feed[net.index("N2O")], feed[net.index("N2")] = 100e-6, 1 - 100e-6
    cond = Conditions(t, P_ATM, ghsv, feed[None, :], wc_phi2=np.array([[phi2_per_k]]))
    out = outlet(net, {}, cond)
    tau = 3600.0 / ghsv * T_STD / t
    eta = np.tanh(2.0) / 2.0
    conversion = 1.0 - out[0, net.index("N2O")] / feed[net.index("N2O")]
    assert conversion == pytest.approx(1.0 - np.exp(-eta * k * tau), rel=2e-4)


@pytest.mark.parametrize("pack_id", ["scr_hot", "oxidation_engine"])
def test_adaptive_solver_agrees_with_rk4(pack_id):
    p = load_pack(pack_id)
    cond, _ = p.conditions(p.protocol())
    ref = outlet_rk4(p.network, {}, cond, 8192)
    out = outlet(p.network, {}, cond, rtol=1e-6, atol_ppm=1e-5)
    assert np.max(np.abs(out - ref)) * 1e6 < 1e-3  # ppm


def test_replaying_a_grid_reproduces_the_solution():
    p = load_pack("scr_hot")
    cond, _ = p.conditions(p.protocol())
    sol = solve(p.network, {}, cond)
    assert np.array_equal(outlet(p.network, {}, cond, grid=sol.grid), sol.outlet)


def test_stiff_low_space_velocity_converges():
    p = load_pack("scr_hot")
    cond, _ = p.conditions(p.protocol())
    cond.ghsv_per_h[:] = 50.0
    loose = outlet(p.network, {}, cond)
    tight = outlet(p.network, {}, cond, rtol=1e-7, atol_ppm=1e-6)
    assert np.max(np.abs(loose - tight)) * 1e6 < 0.05


@pytest.mark.parametrize("pack_id", ["scr_hot", "oxidation_engine"])
def test_elements_are_conserved(pack_id):
    # Outlets are mole fractions and moles change with reaction, but every element is conserved,
    # so the ratio of atoms of any element to oxygen atoms must match between feed and outlet.
    p = load_pack(pack_id)
    cond, _ = p.conditions(p.protocol())
    out = outlet(p.network, {}, cond, rtol=1e-7, atol_ppm=1e-6)
    atoms = {e: np.array([element_counts(p.spec.species[s].formula).get(e, 0) for s in p.network.species])
             for e in ("C", "H", "N")}
    o = np.array([element_counts(p.spec.species[s].formula).get("O", 0) for s in p.network.species])
    for e, counts in atoms.items():
        if counts.any():
            assert np.allclose(out @ counts / (out @ o), cond.feed @ counts / (cond.feed @ o), rtol=1e-6)


def test_reversible_pathway_reaches_equilibrium():
    p = load_pack("oxidation_engine")
    t = 900.0
    feed = p.protocol().steps[0].feed_fractions(p.network.species)
    cond = Conditions(t, P_ATM, 2000.0, feed[None, :])
    out = outlet(p.network, {"no_ox.ln_k_ref": 12.0}, cond, rtol=1e-7, atol_ppm=1e-6)[0]
    c_total = P_ATM / (8.314462618 * t)
    no, no2, o2 = (out[p.network.index(s)] * c_total for s in ("NO", "NO2", "O2"))
    k_c = thermo.equilibrium_constant_c({"NO": -1, "O2": -0.5, "NO2": 1}, t)
    assert no2 / (no * np.sqrt(o2)) == pytest.approx(k_c, rel=0.01)
