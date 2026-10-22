"""Reproduce published SCR measurements with the engine's ideal reactor.

    uv run python experiments/literature_scr/replicate.py

The digitized points in data/ (written by digitize.py) stand in for a lab's results. Each dataset in sources.yaml
gathers one catalyst's runs, each run a bench step: an ammonia oxidation run (NH3 in O2, no NO) and a standard SCR
run (NO + NH3). The scr_hot pack's own rate laws are fitted to all of a catalyst's runs at once in the engine's
isothermal plug-flow reactor, in two ways:

  full     every point
  holdout  the dataset's held-out points (at or above a temperature, in the runs it names) are left out of the
           fit and then predicted, with a 90% band from the fit's own uncertainty

Pathways a dataset leaves out of its fit are switched off. Each alternative refits every point under another
model choice: more or fewer fitted parameters, parameters held at given values, or film mass transfer between the
gas and a monolith wall. Writes results/summary.json for the app's Literature check page.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from datetime import date
from pathlib import Path

import numpy as np

from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.fit.kinetics import fit_kinetics, observations_from_table
from catalyst_engine.packs import AnalyzerSpec, ProtocolSpec, StepSpec, load_pack, read_yaml
from catalyst_engine.value.monolith import Monolith

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
PACK = "scr_hot"
OFF = -30.0  # ln k of a switched-off pathway
LOD_PPM = 1.0  # digitized readings under 1 ppm count as below detection
STEPS = {"nh3_ox": "Ammonia oxidation run (NH3 + O2 only)", "scr": "Standard SCR run (NO + NH3 + O2)"}
# What each figure quantity is, as a function of the run's feed and outlet ppm.
QUANTITIES = {
    "out_NH3_ppm": {"unit": "NH3 out (ppm)", "digits": 2},
    "out_NO_ppm": {"unit": "NO out (ppm)", "digits": 1},
    "out_N2O_ppm": {"unit": "N2O out (ppm)", "digits": 2},
    "nox_selectivity": {"unit": "Share of burned NH3 that makes NOx", "digits": 4},
    "no_conversion": {"unit": "NO conversion", "digits": 4},
    "no_conversion_pct": {"unit": "NO conversion", "digits": 4},
}
PATHWAYS = {"scr": "Standard SCR", "nh3_ox_n2": "NH3 oxidation to N2", "nh3_ox_no": "NH3 oxidation to NO",
            "nh3_ox_n2o": "NH3 oxidation to N2O"}
YARDSTICK = {"protocol": "scr-yardstick-v1", "step": "design", "metric": "nox_conversion"}


def nominal(t_c: float, step: float = 25.0) -> float:
    """The set point a digitized temperature belongs to: most runs step in 25 C or 50 C increments; a figure that
    plots each point at its own measured temperature sets a finer step."""
    return float(round(round(t_c / step) * step, 1))


def load_points(spec: dict) -> list[dict]:
    step = {f["id"]: f.get("t_step_C", 25.0) for f in spec["figures"]}
    rows = []
    for path in sorted(DATA.glob("*.csv")):
        with path.open(newline="") as fh:
            rows += [r | {"T_C": nominal(float(r["T_C"]), step[r["figure"]]), "value": float(r["value"])}
                     for r in csv.DictReader(fh)]
    return rows


def run_points(points: list[dict], run: dict) -> list[dict]:
    """A run's digitized points, in the figure's quantity (percentages become fractions)."""
    out = [p for p in points if p["figure"] == run["figure"] and p["series"] == run["series"]]
    if not out:
        raise ValueError(f"no digitized points for {run}; run digitize.py")
    return [p | ({"value": p["value"] / 100.0, "quantity": "no_conversion"} if p["quantity"] == "no_conversion_pct" else {})
            for p in out]


def shown(quantity: str, value: float, digits: int) -> float:
    """A digitized reading as the fit uses it: concentrations and shares at or above zero, conversions at most 1."""
    return round(min(value, 1.0) if quantity == "no_conversion" else max(value, 0.0), digits)


def model_value(quantity: str, net, feed: np.ndarray, out_ppm: np.ndarray) -> np.ndarray:
    nh3, no = net.index("NH3"), net.index("NO")
    if quantity.startswith("out_"):
        return out_ppm[..., net.index(quantity[4:-4])]
    if quantity == "nox_selectivity":
        with np.errstate(divide="ignore", invalid="ignore"):
            return out_ppm[..., no] / (feed[nh3] * 1e6 - out_ppm[..., nh3])
    return 1.0 - out_ppm[..., no] / (feed[no] * 1e6)


class Dataset:
    def __init__(self, spec: dict, entry: dict, points: list[dict], pack):
        self.entry, self.pack, self.id = entry, pack, entry["id"]
        self.source = spec["sources"][entry["source"]]
        self.cat = entry["catalyst"]
        self.info = self.source["catalysts"][self.cat]
        figs = {f["id"]: f for f in spec["figures"]}
        self.runs = [r | {"points": run_points(points, r), "label": figs[r["figure"]]["label"],
                          "feed": figs[r["figure"]]["feed"]} for r in entry["runs"]]
        self.fit = [f"{pw}.{k}" for pw in entry["fit"] for k in ("ln_k_ref", "Ea_kJ")] + entry.get("fit_extra", [])
        self.off = {f"{pw}.ln_k_ref": OFF for pw in PATHWAYS if pw not in entry["fit"]}
        self.monolith = Monolith(**self.source["monolith"]) if "monolith" in self.source else None
        self.steps = {}
        for step_id in dict.fromkeys(r["step"] for r in self.runs):
            mine = [r for r in self.runs if r["step"] == step_id]
            temps = sorted({p["T_C"] for r in mine for p in r["points"]})
            self.steps[step_id] = StepSpec(id=step_id, feed=mine[0]["feed"], ghsv_per_h=self.source["ghsv_per_h"],
                                           pressure_kPa=self.source.get("pressure_kPa", 101.325),
                                           temperatures_C=temps, purpose=STEPS[step_id])
        self.rows = self._rows()
        err = entry["error"]
        measured = [s for s in pack.network.species if any(f"out_{s}_ppm" in r for r in self.rows)]
        self.analyzer = AnalyzerSpec(name="digitized figure", measured=measured, sigma_abs_ppm=err["sigma_abs_ppm"],
                                     sigma_rel=err["sigma_rel"], lod_ppm={s: err.get("lod_ppm", LOD_PPM) for s in measured})
        self.protocol = ProtocolSpec(id=self.id, pack=PACK, version=1, status="literature", analyzer=self.analyzer,
                                     steps=list(self.steps.values()), description=" ".join(self.source["conditions"].split()))
        hold = entry["holdout"]
        self.held_out = lambda step, t: step in hold["runs"] and t >= hold["from_C"]

    def feed(self, step_id: str) -> np.ndarray:
        return self.steps[step_id].feed_fractions(self.pack.network.species, self.pack.spec.water)

    def km_a(self, t_k: np.ndarray, feed: np.ndarray, p_pa: float) -> np.ndarray:
        """Film mass transfer from the gas to the monolith wall for each pathway's key species: (temperatures, pathways)."""
        net, species = self.pack.network, self.pack.spec.species
        gas = {species[s].thermo_id: float(y) for s, y in zip(net.species, feed) if y > 0}
        keys = [species[net.species[pw.key]].thermo_id for pw in net.pathways]
        return np.array([[self.monolith.km_a(k, float(t), p_pa, gas) for k in keys] for t in t_k])

    def observations(self, rows: list[dict], film: bool = False) -> list:
        obs = observations_from_table(self.pack, self.protocol, rows)
        if film:
            obs = [dataclasses.replace(o, km_a=self.km_a([o.t_k], o.feed, o.p_pa)[0]) for o in obs]
        return obs

    def _rows(self) -> list[dict]:
        """One fit row per (run step, temperature), carrying every outlet reading the figures give there."""
        net = self.pack.network
        rows: dict[tuple, dict] = {}
        for run in self.runs:
            y = self.feed(run["step"])
            for p in run["points"]:
                row = rows.setdefault((run["step"], p["T_C"]), {"step": run["step"], "T_C": p["T_C"]})
                q, v = p["quantity"], p["value"]
                if q.startswith("out_"):
                    if q in row:
                        raise ValueError(f"{self.id}: two {q} readings at {p['T_C']} C; give the figure a finer t_step_C")
                    row[q] = max(v, 0.0)
                elif q == "no_conversion":
                    row["out_NO_ppm"] = y[net.index("NO")] * 1e6 * (1.0 - min(v, 1.0))
                elif q == "nox_selectivity":
                    row["_nox_share"] = max(v, 0.0)
        for row in rows.values():  # NOx made = share x ammonia used, once the run's NH3 reading is in
            if "_nox_share" in row:
                used = self.feed(row["step"])[net.index("NH3")] * 1e6 - row["out_NH3_ppm"]
                row["out_NO_ppm"] = row.pop("_nox_share") * used
        return sorted(rows.values(), key=lambda r: (r["step"], r["T_C"]))


def draws(fit, names: list[str], n: int = 300, seed: int = 0) -> list[dict[str, float]]:
    """Parameter sets drawn from the fit's covariance, for the 90% band around each model curve."""
    std = np.array([fit.std[k] for k in names])
    cov = np.nan_to_num(fit.correlation) * np.outer(std, std)
    xs = np.random.default_rng(seed).multivariate_normal([fit.params[k] for k in names], cov, size=n)
    return [fit.params | dict(zip(names, x)) for x in xs]


def outlet_many(net, param_sets: list[dict[str, float]], cond: Conditions) -> np.ndarray:
    """Outlet ppm for several parameter sets in one reactor call: (sets, conditions, species)."""
    m, b = len(param_sets), cond.size
    tiled = Conditions(np.tile(cond.t_k, m), np.tile(cond.p_pa, m), np.tile(cond.ghsv_per_h, m), np.tile(cond.feed, (m, 1)),
                       None if cond.km_a is None else np.tile(cond.km_a, (m, 1)))
    params = {k: np.repeat([ps[k] for ps in param_sets], b) for k in param_sets[0]}
    return outlet(net, params, tiled).reshape(m, b, -1) * 1e6


def panels(ds: Dataset, fit, holdout: bool, names: list[str] | None = None, band: bool = True,
           film: bool = False) -> list[dict]:
    """One chart per run: the model curve with its 90% band, and each measured point beside the model there."""
    net = ds.pack.network
    sets = [fit.params] + (draws(fit, names or ds.fit) if band else [fit.params])
    temps = [p["T_C"] for r in ds.runs for p in r["points"]]
    curve_c = np.arange(np.floor(min(temps) / 50) * 50, max(temps) + 26, 10.0)
    out = []
    for run in ds.runs:
        step = ds.protocol.step(run["step"])
        y = ds.feed(run["step"])
        q = run["points"][0]["quantity"]
        t_c = np.concatenate([curve_c, [p["T_C"] for p in run["points"]]])
        p_pa = step.pressure_kPa * 1000.0
        cond = Conditions(t_c + 273.15, p_pa, step.ghsv_per_h, np.tile(y, (len(t_c), 1)),
                          ds.km_a(t_c + 273.15, y, p_pa) if film else None)
        with np.errstate(over="ignore", invalid="ignore"):
            vals = model_value(q, net, y, outlet_many(net, sets, cond))
        best, lo, hi = vals[0], np.nanpercentile(vals[1:], 5, axis=0), np.nanpercentile(vals[1:], 95, axis=0)
        n, digits = len(curve_c), QUANTITIES[q]["digits"]
        best, lo, hi = (np.round(a, digits) for a in (best, lo, hi))
        out.append({
            "id": f"{run['step']}.{q}",
            "title": STEPS[run["step"]],
            "figure": run["label"],
            "unit": QUANTITIES[q]["unit"],
            "curve": {"T_C": curve_c.tolist(), "value": best[:n].tolist(), "lo": lo[:n].tolist(), "hi": hi[:n].tolist()},
            "points": [{"T_C": p["T_C"], "measured": m, "model": float(v), "lo": float(a), "hi": float(b),
                        "in_band": bool(a <= m <= b), "used_in_fit": not (holdout and ds.held_out(run["step"], p["T_C"]))}
                       for p, m, v, a, b in zip(run["points"], (shown(q, p["value"], digits) for p in run["points"]),
                                                best[n:], lo[n:], hi[n:])],
        })
    return out


def yardstick(pack, params: dict[str, float], fit=None, names: list[str] | None = None) -> list[dict]:
    """The Gate 1 screen of the pack's own bench protocol, run on these parameters (with the fit's 90% band)."""
    protocol = pack.protocol(YARDSTICK["protocol"])
    cond, rows = pack.conditions(protocol)
    idx = np.array([i for i, r in enumerate(rows) if r.step == YARDSTICK["step"]])
    cond = Conditions(cond.t_k[idx], cond.p_pa[idx], cond.ghsv_per_h[idx], cond.feed[idx])
    sets = [params] + (draws(fit, names) if fit is not None else [])
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


def errors(panel_list: list[dict], used: bool) -> dict[str, dict]:
    """Model minus measured, per run, over the points that were (or were held out of) the fit."""
    out = {}
    for pan in panel_list:
        pts = [p for p in pan["points"] if p["used_in_fit"] == used]
        d = np.array([p["model"] - p["measured"] for p in pts])
        if len(d):
            out[pan["id"]] = {"unit": pan["unit"], "n": int(len(d)), "mean_abs": round(float(np.mean(np.abs(d))), 4),
                              "max_abs": round(float(np.max(np.abs(d))), 4), "mean": round(float(np.mean(d)), 4),
                              "in_band": sum(p["in_band"] for p in pts)}
    return out


def checks(ds: Dataset, points: list[dict]) -> list[dict]:
    """Consistency checks within the source's own figures."""
    out = []
    for check in ds.entry.get("checks", []):  # both read the ammonia-only run
        feed_nh3 = ds.feed("nh3_ox")[ds.pack.network.index("NH3")] * 1e6
        nh3 = {p["T_C"]: p["value"] for r in ds.runs if r["step"] == "nh3_ox" for p in r["points"]
               if p["quantity"] == "out_NH3_ppm"}
        if check == "n2_balance":
            fig = next(r["figure"] for r in ds.runs if r["step"] == "nh3_ox")
            n2 = {p["T_C"]: p["value"] for p in points if p["figure"] == fig and p["quantity"] == "out_N2_ppm"}
            out.append({"title": "Nitrogen balance in the ammonia-only run",
                        "caption": "N2 made equals half the ammonia used when ammonia burns to N2 alone.",
                        "rows": [{"°C": t, "N2 measured (ppm)": round(n2[t], 1),
                                  "Half the NH3 used (ppm)": round((feed_nh3 - nh3[t]) / 2.0, 1)}
                                 for t in sorted(set(nh3) & set(n2))]})
        elif check == "nox_two_figures":
            direct = {p["T_C"]: p["value"] for p in points if p["catalyst"] == ds.cat and p["quantity"] == "out_NOx_ppm"}
            share = {p["T_C"]: p["value"] for r in ds.runs for p in r["points"] if p["quantity"] == "nox_selectivity"}
            out.append({"title": "NOx made in the ammonia-only run, read two ways",
                        "caption": (f"One figure reports the NOx directly; two others give it as the share of burned "
                                    f"ammonia times the ammonia used, with a {feed_nh3:.0f} ppm feed. Agreement "
                                    "confirms the feed and the readings."),
                        "rows": [{"°C": t, "NOx reported (ppm)": round(direct[t], 1),
                                  "Share × NH3 used (ppm)": round(share[t] * (feed_nh3 - nh3[t]), 1)}
                                 for t in sorted(set(direct) & set(share) & set(nh3))]})
    return out


def main() -> None:
    spec = read_yaml(HERE / "sources.yaml")  # keeps the species key NO a string
    pack = load_pack(PACK)
    points = load_points(spec)
    defaults = pack.network.parameters()
    target = pack.spec.metrics[YARDSTICK["metric"]].target
    k_def = {pw: float(np.exp(defaults[f"{pw}.ln_k_ref"])) for pw in PATHWAYS}
    summary = {
        "run_date": date.today().isoformat(),
        "pack": PACK,
        "sources": {sid: {k: s[k] for k in ("cite", "url", "conditions", "ghsv_per_h")}
                    | {"id": sid, "pressure_kPa": s.get("pressure_kPa", 101.325)} for sid, s in spec["sources"].items()},
        "figures": [{k: f[k] for k in ("id", "source", "label", "page", "feed")} for f in spec["figures"]],
        "reactor": "isothermal plug flow, ideal (no mass-transfer limits)",
        "pathways": PATHWAYS,
        "t_ref_C": round(pack.network.t_ref - 273.15, 1),
        "t_ref_F": round((pack.network.t_ref - 273.15) * 9 / 5 + 32, 1),
        "yardstick": YARDSTICK | {"target": target.value, "target_label": target.label,
                                  "ghsv_per_h": pack.protocol(YARDSTICK["protocol"]).step(YARDSTICK["step"]).ghsv_per_h},
        "defaults": {
            "params": {f"{pw}.{k}": round(defaults[f"{pw}.{k}"], 4) for pw in PATHWAYS for k in ("ln_k_ref", "Ea_kJ")},
            "k_ref_per_s": {pw: round(v, 3) for pw, v in k_def.items()},
            "yardstick": yardstick(pack, defaults),
        },
        "catalysts": {},
    }

    for entry in spec["datasets"]:
        ds = Dataset(spec, entry, points, pack)
        result = {"name": ds.info["name"], "form": ds.info["form"], "source": entry["source"],
                  "published_Ea_kJ": ds.source.get("published_Ea_kJ", {}).get(ds.cat, {}),
                  "published_model": ds.source.get("published_model", "published model"),
                  "fitted": ds.fit, "switched_off": list(ds.off), "pathways": entry["fit"],
                  "analyzer": ds.analyzer.model_dump(), "holdout_test": entry["holdout"],
                  "range_C": [min(r["T_C"] for r in ds.rows), max(r["T_C"] for r in ds.rows)],
                  "checks": checks(ds, points)}
        for label in ("full", "holdout"):
            use = [r for r in ds.rows if label == "full" or not ds.held_out(r["step"], r["T_C"])]
            fit = fit_kinetics(pack, ds.protocol, ds.observations(use), fit=ds.fit, fixed=ds.off, n_starts=8)
            k = {pw: float(np.exp(fit.params[f"{pw}.ln_k_ref"])) for pw in entry["fit"]}
            res = {
                "params": {n: round(fit.params[n], 4) for n in ds.fit},
                "std": {n: round(fit.std[n], 4) for n in ds.fit},
                "poorly_determined": fit.poorly_determined,
                "chi2_reduced": round(fit.chi2_reduced, 3),
                "n_points": fit.n_points,
                "k_ref_per_s": {pw: round(v, 3) for pw, v in k.items()},
                "panels": panels(ds, fit, label == "holdout"),
                "yardstick": yardstick(pack, fit.params, fit, ds.fit),
            }
            res["errors_fit"] = errors(res["panels"], True)
            res["errors_predicted"] = errors(res["panels"], False)
            result[label] = res
            print(f"{ds.id} {label}: chi2 {res['chi2_reduced']}, params {res['params']}, poor {res['poorly_determined']}")
            for kind in ("errors_fit", "errors_predicted"):
                for run, e in res[kind].items():
                    print(f"   {kind} {run}: mean |d| {e['mean_abs']}, max {e['max_abs']}, in band {e['in_band']}/{e['n']}")
        result["alternatives"] = []
        for alt in entry.get("alternatives", []):  # the same data under a different model choice
            names = [n for n in ds.fit if n not in alt.get("drop", [])] + alt.get("add", [])
            film = alt.get("film", False)
            fit = fit_kinetics(pack, ds.protocol, ds.observations(ds.rows, film), fit=names,
                               fixed=ds.off | alt.get("fixed", {}), n_starts=8)
            pans = panels(ds, fit, False, names, band=False, film=film)
            result["alternatives"].append({"label": alt["label"], "note": alt.get("note", ""),
                                           "chi2_reduced": round(fit.chi2_reduced, 3), "n_params": len(names),
                                           "poorly_determined": fit.poorly_determined,
                                           "params": {n: round(fit.params[n], 4) for n in names},
                                           "std": {n: round(fit.std[n], 4) for n in names},
                                           "fixed": alt.get("fixed", {}), "film": film,
                                           "errors_fit": errors(pans, True),
                                           "curves": {pan["id"]: pan["curve"]["value"] for pan in pans},
                                           "yardstick": yardstick(pack, fit.params, fit, names)})
            print(f"   alternative '{alt['label']}': chi2 {fit.chi2_reduced:.2f}, params "
                  f"{result['alternatives'][-1]['params']}, poor {fit.poorly_determined}")
        result["n_params"] = len(ds.fit)
        summary["catalysts"][ds.cat] = result

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
