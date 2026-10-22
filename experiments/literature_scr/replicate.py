"""Reproduce published SCR measurements with the engine's ideal reactor.

    uv run python experiments/literature_scr/replicate.py

The digitized points in data/ (written by digitize.py) stand in for a lab's results. For each catalyst, the
scr_hot pack's own rate laws are fitted in the engine's isothermal plug-flow reactor, jointly to the ammonia
oxidation run (NH3 in O2, no NO) and the standard SCR run (NO + NH3), in two ways:

  full     every point, 100-600 C
  holdout  points up to 450 C only; the 500-600 C points are then predicted and compared, the hot-SCR range

Ammonia oxidation in these runs makes N2 only (the thesis's N balance closes on N2), so the pathways to NO and
N2O are switched off and the N2 readings serve as a check. Writes results/summary.json for the app's
Literature check page.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.fit.kinetics import fit_kinetics, observations_from_table
from catalyst_engine.packs import AnalyzerSpec, ProtocolSpec, StepSpec, load_pack, read_yaml
from catalyst_engine.units import f_to_k

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
PACK = "scr_hot"
HOLDOUT_FROM_C = 500.0
FIT = ["scr.ln_k_ref", "scr.Ea_kJ", "nh3_ox_n2.ln_k_ref", "nh3_ox_n2.Ea_kJ"]
OFF = {"nh3_ox_no.ln_k_ref": -30.0, "nh3_ox_n2o.ln_k_ref": -30.0}
# Reading error of the digitized points: analyzer scatter and plotting/digitizing, about 1% of a 500 ppm feed
# plus 3% of the reading. Points under 1 ppm are censored.
ANALYZER = AnalyzerSpec(name="digitized figure", measured=["NO", "NH3"], sigma_abs_ppm=5.0, sigma_rel=0.03,
                        lod_ppm={"NO": 1.0, "NH3": 1.0})
CURVE_C = np.arange(100.0, 651.0, 10.0)


def nominal(t_c: float) -> float:
    """The set point a digitized temperature belongs to: the runs step in 25 C or 50 C increments."""
    return float(round(t_c / 25.0) * 25.0)


def load_points() -> list[dict]:
    rows = []
    for path in sorted(DATA.glob("*.csv")):
        with path.open(newline="") as fh:
            rows += [r | {"T_C": float(r["T_C"]), "value": float(r["value"])} for r in csv.DictReader(fh)]
    return rows


def protocol_for(cat: str, spec: dict, points: list[dict]) -> ProtocolSpec:
    figs = {f["id"]: f for f in spec["figures"]}
    src = spec["sources"]["metkar2012"]
    steps = []
    for step_id, quantity in (("nh3_ox", "out_NH3_ppm"), ("scr", "no_conversion")):
        mine = [p for p in points if p["catalyst"] == cat and p["quantity"] == quantity]
        feed = figs[mine[0]["figure"]]["feed"]
        temps = sorted({nominal(p["T_C"]) for p in mine})
        steps.append(StepSpec(id=step_id, feed=feed, ghsv_per_h=src["ghsv_per_h"], temperatures_C=temps,
                              purpose=figs[mine[0]["figure"]]["label"]))
    return ProtocolSpec(id=f"metkar2012-{cat}", pack=PACK, version=1, status="literature", analyzer=ANALYZER,
                        steps=steps, description=src["conditions"].strip())


def fit_rows(pack, protocol: ProtocolSpec, cat: str, points: list[dict]) -> list[dict]:
    """Rows in the form fit_kinetics reads: NH3 outlet from the oxidation run, NO outlet from the SCR run."""
    no_in = protocol.step("scr").feed_fractions(pack.network.species, pack.spec.water)[pack.network.index("NO")] * 1e6
    rows = []
    for p in points:
        if p["catalyst"] != cat:
            continue
        if p["quantity"] == "out_NH3_ppm":
            rows.append({"step": "nh3_ox", "T_C": nominal(p["T_C"]), "out_NH3_ppm": max(p["value"], 0.0)})
        elif p["quantity"] == "no_conversion":
            rows.append({"step": "scr", "T_C": nominal(p["T_C"]), "out_NO_ppm": no_in * (1.0 - min(p["value"], 1.0))})
    return rows


PANELS = {
    "nh3_ox": {"title": "Ammonia oxidation run (NH3 + O2 only)", "unit": "NH3 out (ppm)", "species": "NH3"},
    "scr": {"title": "Standard SCR run (NO + NH3 + O2)", "unit": "NO conversion", "species": "NO"},
}
PATHWAYS = {"scr": "Standard SCR", "nh3_ox_n2": "NH3 oxidation to N2"}
YARDSTICK = {"protocol": "scr-yardstick-v1", "step": "design", "metric": "nox_conversion"}


def panel_value(step_id: str, net, feed: np.ndarray, out_ppm: np.ndarray) -> np.ndarray:
    """A run's figure quantity from outlet ppm: NH3 left for the oxidation run, NO conversion for the SCR run."""
    j = net.index(PANELS[step_id]["species"])
    return out_ppm[..., j] if step_id == "nh3_ox" else 1.0 - out_ppm[..., j] / (feed[j] * 1e6)


def draws(fit, n: int = 300, seed: int = 0) -> list[dict[str, float]]:
    """Parameter sets drawn from the fit's covariance, for the 90% band around each model curve."""
    std = np.array([fit.std[k] for k in FIT])
    cov = np.nan_to_num(fit.correlation) * np.outer(std, std)
    xs = np.random.default_rng(seed).multivariate_normal([fit.params[k] for k in FIT], cov, size=n)
    return [fit.params | dict(zip(FIT, x)) for x in xs]


def outlet_many(net, param_sets: list[dict[str, float]], cond: Conditions) -> np.ndarray:
    """Outlet ppm for several parameter sets in one reactor call: (sets, conditions, species)."""
    m, b = len(param_sets), cond.size
    tiled = Conditions(np.tile(cond.t_k, m), np.tile(cond.p_pa, m), np.tile(cond.ghsv_per_h, m), np.tile(cond.feed, (m, 1)))
    params = {k: np.repeat([ps[k] for ps in param_sets], b) for k in param_sets[0]}
    return outlet(net, params, tiled).reshape(m, b, -1) * 1e6


def panels(pack, protocol: ProtocolSpec, fit, rows: list[dict], fit_below_C: float) -> list[dict]:
    """One chart per run: the model curve with its 90% band, and each measured point beside the model there."""
    net = pack.network
    sets = [fit.params] + draws(fit)
    obs = observations_from_table(pack, protocol, rows)
    out = []
    for step in protocol.steps:
        y = step.feed_fractions(net.species, pack.spec.water)
        mine = [(o, r) for o, r in zip(obs, rows) if o.step == step.id]
        t_k = np.concatenate([CURVE_C + 273.15, [o.t_k for o, _ in mine]])
        cond = Conditions(t_k, step.pressure_kPa * 1000.0, step.ghsv_per_h, np.tile(y, (len(t_k), 1)))
        with np.errstate(over="ignore", invalid="ignore"):
            vals = panel_value(step.id, net, y, outlet_many(net, sets, cond))
        best, lo, hi = vals[0], np.nanpercentile(vals[1:], 5, axis=0), np.nanpercentile(vals[1:], 95, axis=0)
        n = len(CURVE_C)
        measured = panel_value(step.id, net, y, np.array([[r.get(f"out_{s}_ppm", 0.0) for s in net.species]
                                                         for _, r in mine]))
        digits = 2 if step.id == "nh3_ox" else 4
        best, lo, hi, measured = (np.round(a, digits) for a in (best, lo, hi, measured))
        out.append(PANELS[step.id] | {
            "id": step.id,
            "curve": {"T_C": CURVE_C.tolist(), "value": best[:n].tolist(), "lo": lo[:n].tolist(), "hi": hi[:n].tolist()},
            "points": [{"T_C": r["T_C"], "measured": float(m), "model": float(v), "lo": float(a), "hi": float(b),
                        "in_band": bool(a <= m <= b), "used_in_fit": r["T_C"] < fit_below_C}
                       for (_, r), m, v, a, b in zip(mine, measured, best[n:], lo[n:], hi[n:])],
        })
    return out


def yardstick(pack, fit_or_params) -> list[dict]:
    """The Gate 1 screen of the pack's own bench protocol, run on these parameters (with the fit's 90% band)."""
    protocol = pack.protocol(YARDSTICK["protocol"])
    cond, rows = pack.conditions(protocol)
    idx = np.array([i for i, r in enumerate(rows) if r.step == YARDSTICK["step"]])
    cond = Conditions(cond.t_k[idx], cond.p_pa[idx], cond.ghsv_per_h[idx], cond.feed[idx])
    sets = [fit_or_params] if isinstance(fit_or_params, dict) else [fit_or_params.params] + draws(fit_or_params)
    with np.errstate(over="ignore", invalid="ignore"):
        ppm = outlet_many(pack.network, sets, cond)
    vals = np.array([pack.metrics(cond.feed, o / 1e6)[YARDSTICK["metric"]] for o in ppm])
    out = []
    for j, t_k in enumerate(cond.t_k):
        row = {"T_F": round(t_k * 9 / 5 - 459.67), "value": round(float(vals[0, j]), 4)}
        if len(sets) > 1:
            row |= {"lo": round(float(np.nanpercentile(vals[1:, j], 5)), 4),
                    "hi": round(float(np.nanpercentile(vals[1:, j], 95)), 4)}
        out.append(row)
    return out


def describe(fit, pack) -> dict:
    t_ref = pack.network.t_ref
    k = {pw: float(np.exp(fit.params[f"{pw}.ln_k_ref"])) for pw in ("scr", "nh3_ox_n2")}
    return {
        "params": {n: round(fit.params[n], 4) for n in FIT},
        "std": {n: round(fit.std[n], 4) for n in FIT},
        "poorly_determined": fit.poorly_determined,
        "chi2_reduced": round(fit.chi2_reduced, 3),
        "n_points": fit.n_points,
        "k_ref_per_s": {pw: round(v, 3) for pw, v in k.items()},
        "k_ratio_scr_to_nh3_ox": round(k["scr"] / k["nh3_ox_n2"], 2),
        "t_ref_C": round(t_ref - 273.15, 1),
    }


def errors(panel_list: list[dict], used: bool) -> dict[str, dict]:
    """Model minus measured, per run, over the points that were (or were held out of) the fit."""
    out = {}
    for pan in panel_list:
        d = np.array([p["model"] - p["measured"] for p in pan["points"] if p["used_in_fit"] == used])
        if len(d):
            out[pan["id"]] = {"unit": pan["unit"], "n": int(len(d)), "mean_abs": round(float(np.mean(np.abs(d))), 4),
                              "max_abs": round(float(np.max(np.abs(d))), 4), "mean": round(float(np.mean(d)), 4),
                              "in_band": sum(p["in_band"] for p in pan["points"] if p["used_in_fit"] == used)}
    return out


def n2_check(points: list[dict], cat: str) -> list[dict]:
    """N2 made in the oxidation run against half the NH3 consumed: all N2 means the two agree."""
    nh3 = {nominal(p["T_C"]): p["value"] for p in points if p["catalyst"] == cat and p["quantity"] == "out_NH3_ppm"}
    n2 = {nominal(p["T_C"]): p["value"] for p in points if p["catalyst"] == cat and p["quantity"] == "out_N2_ppm"}
    feed = 500.0
    return [{"T_C": t, "n2_ppm": round(n2[t], 1), "half_nh3_used_ppm": round((feed - nh3[t]) / 2.0, 1)}
            for t in sorted(set(nh3) & set(n2))]


def main() -> None:
    spec = read_yaml(HERE / "sources.yaml")  # keeps the species key NO a string
    src = spec["sources"]["metkar2012"]
    pack = load_pack(PACK)
    points = load_points()
    defaults = pack.network.parameters()
    target = pack.spec.metrics[YARDSTICK["metric"]].target
    k_def = {pw: float(np.exp(defaults[f"{pw}.ln_k_ref"])) for pw in PATHWAYS}
    summary = {
        "run_date": date.today().isoformat(),
        "pack": PACK,
        "source": {k: src[k] for k in ("cite", "url", "conditions", "ghsv_per_h")},
        "figures": [{k: f[k] for k in ("id", "label", "page", "feed")} for f in spec["figures"]],
        "reactor": "isothermal plug flow, ideal (no mass-transfer limits)",
        "fitted": FIT,
        "switched_off": list(OFF),
        "pathways": PATHWAYS,
        "analyzer": ANALYZER.model_dump(),
        "holdout_from_C": HOLDOUT_FROM_C,
        "t_ref_C": round(pack.network.t_ref - 273.15, 1),
        "t_ref_F": round((pack.network.t_ref - 273.15) * 9 / 5 + 32, 1),
        "yardstick": YARDSTICK | {"target": target.value, "target_label": target.label,
                                  "ghsv_per_h": pack.protocol(YARDSTICK["protocol"]).step(YARDSTICK["step"]).ghsv_per_h},
        "defaults": {
            "params": {n: round(defaults[n], 4) for n in FIT},
            "k_ref_per_s": {pw: round(v, 3) for pw, v in k_def.items()},
            "k_ratio_scr_to_nh3_ox": round(k_def["scr"] / k_def["nh3_ox_n2"], 2),
            "yardstick": yardstick(pack, defaults),
        },
        "catalysts": {},
    }

    for cat, info in src["catalysts"].items():
        protocol = protocol_for(cat, spec, points)
        rows = fit_rows(pack, protocol, cat, points)
        result = {"name": info["name"], "form": info["form"], "published_Ea_kJ": src["published_Ea_kJ"][cat],
                  "n2_check": n2_check(points, cat)}
        for label, below in (("full", np.inf), ("holdout", HOLDOUT_FROM_C)):
            use = [r for r in rows if r["T_C"] < below]
            fit = fit_kinetics(pack, protocol, observations_from_table(pack, protocol, use), fit=FIT, fixed=OFF,
                               n_starts=8)
            entry = describe(fit, pack)
            entry["panels"] = panels(pack, protocol, fit, rows, below)
            entry["errors_fit"] = errors(entry["panels"], True)
            entry["errors_predicted"] = errors(entry["panels"], False)
            entry["yardstick"] = yardstick(pack, fit)
            result[label] = entry
            print(f"{cat} {label}: chi2 {entry['chi2_reduced']}, params {entry['params']}, "
                  f"poor {entry['poorly_determined']}")
            for kind in ("errors_fit", "errors_predicted"):
                for run, e in entry[kind].items():
                    print(f"   {kind} {run}: mean |d| {e['mean_abs']}, max {e['max_abs']}, n {e['n']}")
        summary["catalysts"][cat] = result

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
