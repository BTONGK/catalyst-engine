import dataclasses

import numpy as np
import pytest

from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.fit.decay import accelerate, fit_decay
from catalyst_engine.fit.kinetics import fit_kinetics, observations_from_table
from catalyst_engine.packs import load_pack
from catalyst_engine.units import f_to_k


def _synthetic(pack, protocol, true, seed=1):
    rows = pack.simulate(true, protocol.id)
    rng = np.random.default_rng(seed)
    an = protocol.analyzer
    for r in rows:
        for s in an.measured:
            v = r[f"out_{s}_ppm"]
            r[f"out_{s}_ppm"] = max(0.0, v + rng.normal(0, np.hypot(an.sigma_abs_ppm, an.sigma_rel * v)))
    return rows


def _shifted_truth(pack):
    return {k: v + (0.4 if k.endswith("ln_k_ref") else -8.0) for k, v in pack.network.defaults.items()
            if k in pack.network.fitted}


@pytest.mark.parametrize("pack_id", ["scr_hot", "oxidation_engine"])
def test_fit_recovers_known_parameters(pack_id):
    p = load_pack(pack_id)
    proto = p.protocol()
    true = _shifted_truth(p)
    obs = observations_from_table(p, proto, _synthetic(p, proto, true))
    res = fit_kinetics(p, proto, obs, n_starts=2)
    assert not res.poorly_determined
    for name, value in true.items():
        assert abs(res.params[name] - value) < 4 * res.std[name], name


def test_fit_flags_what_a_protocol_cannot_separate():
    # Without the ammonia-only step, the three ammonia-oxidation pathways are not separable.
    p = load_pack("scr_hot")
    proto = p.protocol()
    trimmed = proto.model_copy(update={"steps": [s for s in proto.steps if s.id != "nh3_oxidation"]})
    rows = [r for r in _synthetic(p, proto, _shifted_truth(p)) if r["step"] != "nh3_oxidation"]
    obs = observations_from_table(p, trimmed, rows)
    res = fit_kinetics(p, trimmed, obs, n_starts=2)
    assert any(n.startswith("nh3_ox") for n in res.poorly_determined)
    assert not any(n.startswith("scr.") for n in res.poorly_determined)


def test_fit_with_film_mass_transfer_recovers_intrinsic_rates():
    # Readings from a reactor with a film resistance: fitted with the same film, the intrinsic rates come back;
    # fitted as an ideal reactor, the hot-end rates read low, so the apparent activation energy drops.
    p = load_pack("scr_hot")
    proto = p.protocol()
    true = _shifted_truth(p)
    obs = observations_from_table(p, proto, _synthetic(p, proto, true))
    km_a = np.full(len(p.network.pathways), 150.0)
    cond = Conditions(np.array([o.t_k for o in obs]), np.array([o.p_pa for o in obs]),
                      np.array([o.ghsv_per_h for o in obs]), np.array([o.feed for o in obs]), np.tile(km_a, (len(obs), 1)))
    ppm = outlet(p.network, p.network.parameters() | true, cond) * 1e6
    filmed = [dataclasses.replace(o, km_a=km_a, measured_ppm={s: float(ppm[i, p.network.index(s)]) for s in o.measured_ppm})
              for i, o in enumerate(obs)]
    res = fit_kinetics(p, proto, filmed, n_starts=2)
    for name, value in true.items():
        assert abs(res.params[name] - value) < 4 * res.std[name] + 1e-3, name
    ideal = fit_kinetics(p, proto, [dataclasses.replace(o, km_a=None) for o in filmed], fit=["scr.ln_k_ref", "scr.Ea_kJ"],
                         fixed={k: v for k, v in true.items() if not k.startswith("scr.")}, n_starts=2)
    assert ideal.params["scr.Ea_kJ"] < res.params["scr.Ea_kJ"]
    with pytest.raises(ValueError, match="every observation"):
        fit_kinetics(p, proto, filmed[:1] + obs[1:], n_starts=1)
    coated = [dataclasses.replace(filmed[0], wc_phi2=np.full(len(p.network.pathways), 1e-3))] + filmed[1:]
    with pytest.raises(ValueError, match="washcoat"):
        fit_kinetics(p, proto, coated, n_starts=1)


def test_observations_accept_measured_inlet():
    p = load_pack("scr_hot")
    proto = p.protocol()
    rows = [{"step": "design", "T_F": "1050", "in_NO_ppm": "24.0", "out_NO_ppm": "2.0"}]
    obs = observations_from_table(p, proto, rows)
    assert obs[0].t_k == pytest.approx(f_to_k(1050))
    assert obs[0].feed[p.network.index("NO")] == pytest.approx(24e-6)
    assert obs[0].measured_ppm == {"NO": 2.0}


def test_decay_fit_and_projection():
    t = np.array([0, 24, 50, 100, 250, 500, 1000.0])
    kd, a_eq = 2e-3, 0.80
    a = a_eq + (1 - a_eq) / (1 + kd * (1 - a_eq) * t)
    a = a + np.random.default_rng(0).normal(0, 0.005, len(t))
    fit = fit_decay(t, a, 0.005)
    assert fit.activity(2000.0) == pytest.approx(a_eq + (1 - a_eq) / (1 + kd * (1 - a_eq) * 2000), abs=0.03)
    lo, hi = fit.interval(2000.0)
    assert lo < fit.activity(2000.0) < hi
    assert fit.extrapolation_ratio(40000.0) == pytest.approx(40.0)
    assert np.isinf(fit.hours_to(0.5))  # it levels off above 0.5


def test_accelerate_is_arrhenius():
    assert accelerate(1.0, 0.0, 900.0, 1000.0) == pytest.approx(1.0)
    assert accelerate(1.0, 100.0, 900.0, 1000.0) > 1.0
