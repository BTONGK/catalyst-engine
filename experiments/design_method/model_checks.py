"""Checks of the design method's non-atomistic steps, and the summary the Manual's Methodology tab reads.

    uv run python experiments/design_method/model_checks.py

Reads results/atomistic.json (atomistic_checks.py), the shipped example screen, the UMA benchmark in
experiments/uma_binding/ and the reaction packs, then writes results/summary.json:

  values   single numbers the write-up quotes (report.yaml fills {name} with them)
  tables   rows for the tables on the page
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cantera as ct
import numpy as np

from catalyst_engine.chem.formula import element_counts
from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.design import screening, uma_worker
from catalyst_engine.packs import PACKS_DIR, list_packs, load_pack
from catalyst_engine.units import R

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLE = next((REPO / "examples" / "screens").iterdir())
values: dict[str, float | int | str] = {}
tables: dict[str, list[dict]] = {}


def r(x: float, n: int = 3) -> float:
    return float(round(x, n))


# ---------------------------------------------------------------- the example screen: quality and the face fix
job = screening.read_job(EXAMPLE)
results = screening.read_results(EXAMPLE)
status = screening.read_status(EXAMPLE)
pack = load_pack(job["request"]["pack"])
relax_rows = [x for x in results if x["site"] != "error"]
ads_rows = [x for x in relax_rows if x["adsorbate"] != "*"]
values.update(
    example_materials=len(job["materials"]), example_relaxations=len(relax_rows),
    example_converged=sum(bool(x.get("converged")) for x in relax_rows),
    example_unconverged=sum(not x.get("converged") for x in relax_rows),
    example_anomalies=sum(bool(x.get("anomaly")) for x in ads_rows),
    example_minutes=r((status.get("finished", 0) - status.get("started", 0)) / 60, 1),
    example_errors=sum(x["site"] == "error" for x in results),
)
unconverged = [x for x in relax_rows if not x.get("converged")]
summary = screening.binding_summary(results)
values["example_unconverged_were_minimum"] = sum(
    1 for x in unconverged if x["adsorbate"] != "*" and summary[x["material"]]["binding"][x["adsorbate"]]["site"] == x["site"])
names = {m["id"]: m["formula"] for m in job["materials"]}
if not unconverged:
    values["convergence_note"] = f"In the example screen all {len(relax_rows)} relaxations reached the force limit."
else:
    which = ", ".join(f"{names[x['material']]} {x['site']}" for x in unconverged)
    held = values["example_unconverged_were_minimum"] == 0
    values["convergence_note"] = (
        f"In the example screen {values['example_converged']} of {len(relax_rows)} relaxations reached the force limit. "
        f"The {'one' if len(unconverged) == 1 else len(unconverged)} that stopped at the step limit ({which}) "
        + ("sat above its material's best spot, so the ranking stays the same." if held
           else "set its material's binding energy; the thorough setting allows more steps."))

# surface geometry per material: face, top layer, packing
face_rows = []
before = {}
for line in (HERE / "data" / "example_before_face_fix.jsonl").read_text().splitlines():
    x = json.loads(line)
    if x["site"] == "clean":
        before.setdefault(x["material"], {})["facet"] = x["facet"]
    elif x["adsorbate"] == "O" and not x.get("anomaly"):
        e = before.setdefault(x["material"], {}).get("E")
        before[x["material"]]["E"] = x["E_bind"] if e is None else min(e, x["E_bind"])
for m in job["materials"]:
    bulk = uma_worker.bulk_from(m)
    hkl = uma_worker.densest_plane(bulk)
    slab = uma_worker.build_slab(bulk, hkl)
    z = slab.positions[:, 2]
    top = [s for s, zz in zip(slab.get_chemical_symbols(), z) if zz > z.max() - 0.6]
    density, mismatch = uma_worker.plane_layers(bulk, hkl)
    row = {"material": m["formula"], "face": list(hkl), "top_layer": " ".join(f"{e}{top.count(e)}" for e in sorted(set(top))),
           "packing_per_A2": r(density), "composition_gap": r(mismatch, 2), "atoms": len(slab),
           "E_after": summary.get(m["id"], {}).get("binding", {}).get("O", {}).get("E"),
           "face_before": before.get(m["id"], {}).get("facet"), "E_before": before.get(m["id"], {}).get("E")}
    face_rows.append(row)
tables["faces"] = face_rows
values["alloys_changed"] = sum(1 for x in face_rows if x["face_before"] != x["face"])
values["metals_unchanged"] = sum(1 for x in face_rows if x["face_before"] == x["face"]
                                 and x["E_before"] is not None and x["E_after"] is not None
                                 and abs(x["E_before"] - x["E_after"]) < 1e-3)

# ---------------------------------------------------------------- automatic pipeline against the hand-built benchmark
bench = json.loads((REPO / "experiments" / "uma_binding" / "results" / "summary.json").read_text())["metals"]["binding"]
bench_o = {b["metal"]: b["E"] for b in bench if b["adsorbate"] == "O"}
rows = []
for m in job["materials"]:
    if m["formula"] in bench_o and m["id"] in summary:
        e = summary[m["id"]]["binding"]["O"]["E"]
        rows.append({"material": m["formula"], "E_app": r(e), "E_benchmark": r(bench_o[m["formula"]]),
                     "difference": r(e - bench_o[m["formula"]])})
tables["benchmark"] = rows
d = np.array([x["difference"] for x in rows])
values.update(benchmark_n=len(rows), benchmark_mae=r(np.abs(d).mean()), benchmark_max=r(np.abs(d).max()),
              benchmark_mean=r(d.mean()))
corr = np.corrcoef([x["E_app"] for x in rows], [x["E_benchmark"] for x in rows])[0, 1]
values["benchmark_r2"] = r(corr ** 2, 4)

# ---------------------------------------------------------------- gas references (the UMA benchmark's gas.py)
gas = [line.split(",") for line in (REPO / "experiments" / "uma_binding" / "results" / "gas.csv").read_text().splitlines()[1:]]
tables["gas"] = [{"quantity": q, "uma_eV": float(u), "reference_eV": float(ref), "reference": src,
                  "difference": r(float(u) - float(ref))} for task, q, u, ref, src in gas if task == "oc20"]
values["gas_max_gap"] = r(max(abs(x["difference"]) for x in tables["gas"]))

# ---------------------------------------------------------------- atomistic checks (atomistic_checks.py)
atom = json.loads((HERE / "results" / "atomistic.json").read_text())
lat = atom["lattice"]
tables["lattice"] = [{"material": x["material"], "source": x["source"], "oc20_pct": r(x["oc20_pct"], 2),
                      "omat_pct": r(x["omat_pct"], 2)} for x in lat]
values["lattice_oc20_max_pct"] = r(max(abs(x["oc20_pct"]) for x in lat), 1)
values["lattice_omat_max_pct"] = r(max(abs(x["omat_pct"]) for x in lat), 1)
values["lattice_omat_mean_pct"] = r(float(np.mean([abs(x["omat_pct"]) for x in lat])), 1)
slopes = {}
for sym in sorted({x["material"] for x in atom["strain"]}):
    pts = [(x["strain_pct"], x["E"]) for x in atom["strain"] if x["material"] == sym]
    slopes[sym] = float(np.polyfit(*zip(*pts), 1)[0])
tables["strain"] = [{"material": x["material"], "strain_pct": x["strain_pct"], "E": r(x["E"])} for x in atom["strain"]]
values["strain_slope_max"] = r(max(abs(s) for s in slopes.values()), 3)  # eV per % strain
values["strain_slopes"] = " and ".join(f"{v:+.3f} eV for {k}" for k, v in sorted(slopes.items(), key=lambda kv: kv[1]))
# every input sits below UMA's own lattice by a similar amount; the part that cancels against the catalyst to beat is
# the shared gap, the part that stays is each material's gap minus the reference's
ref_formula = next(m["formula"] for m in job["materials"] if m.get("is_reference"))
ref_gap = next(x["omat_pct"] for x in lat if x["material"] == ref_formula)
rel = np.array([x["omat_pct"] - ref_gap for x in lat if x["material"] != ref_formula])
values["lattice_gap_rms"] = r(float(np.sqrt(np.mean(rel ** 2))), 2)
values["lattice_gap_max_rel"] = r(float(np.abs(rel).max()), 2)
values["strain_error_est"] = r(values["strain_slope_max"] * values["lattice_gap_rms"], 3)
values["strain_error_max"] = r(values["strain_slope_max"] * values["lattice_gap_max_rel"], 3)

size = atom["slab_size"]
tables["slab_size"] = [{k: (r(v) if k == "E" else v) for k, v in x.items()} for x in size]
thick, wide = [], []
for sym in sorted({x["material"] for x in size}):
    base = next(x for x in size if x["material"] == sym and x["thickness_A"] == 7 and x["width_A"] == 7)
    thick += [abs(x["E"] - base["E"]) for x in size if x["material"] == sym and x["width_A"] == 7]
    wide += [x["E"] - base["E"] for x in size if x["material"] == sym and x["width_A"] > 7]
values["slab_thickness_max_gap"] = r(max(thick))
values["slab_width_gap_min"] = r(min(abs(g) for g in wide), 3)
values["slab_width_gap_max"] = r(max(abs(g) for g in wide), 3)
values["slab_width_spread"] = r(max(wide) - min(wide), 3)  # how much the coverage effect differs between cases
values["slab_size_max_gap"] = r(max(values["slab_thickness_max_gap"], values["slab_width_spread"]), 3)

sites = atom["sites"]
tables["sites"] = [{"material": x["material"], "sites_tried": x["n_sites"], "E_standard": r(x["E_standard"]),
                    "site_standard": x["site_standard"], "E_all": r(x["E_all"]), "site_all": x["site_all"],
                    "gain": r(x["E_standard"] - x["E_all"])} for x in sites]
values["sites_same_minimum"] = sum(1 for x in tables["sites"] if x["gain"] < 0.01)
values["sites_n"] = len(sites)
values["sites_max_gain"] = r(max(x["gain"] for x in tables["sites"]))

tol = atom["tolerance"]
tables["tolerance"] = [{"material": x["material"], "E_standard": r(x["E_standard"]), "E_tight": r(x["E_tight"]),
                        "difference": r(x["E_tight"] - x["E_standard"])} for x in tol]
values["tolerance_max_gap"] = r(max(abs(x["difference"]) for x in tables["tolerance"]))

# the same code on the same structure gives the same number: this run against the example screen
repro = []
for x in sites:
    m = next(m for m in job["materials"] if m["formula"] == x["material"])
    if m["id"] in summary:
        repro.append(abs(summary[m["id"]]["binding"]["O"]["E"] - x["E_standard"]))
values["repro_max_gap"] = float(f"{max(repro):.2g}") if repro else None

seg = atom["segregation"]
tables["segregation"] = [{"material": x["material"], "face": x["face"], "surface": x["surface"],
                          "top_layer": " ".join(f"{k}{v}" for k, v in x["top_layer"].items()), "E": r(x["E"]),
                          "site": x["site"]} for x in seg]
spread = {}
for x in seg:
    spread.setdefault(x["material"], []).append(x["E"])
values["seg_min"] = r(min(max(v) - min(v) for v in spread.values()), 2)
values["seg_max"] = r(max(max(v) - min(v) for v in spread.values()), 2)
values["seg_n"] = len(spread)
shifts = [abs(x["E_after"] - x["E_before"]) for x in face_rows
          if x["face_before"] != x["face"] and x["E_after"] is not None and x["E_before"] is not None]
values["face_fix_max_shift"] = r(max(shifts), 2) if shifts else 0.0
values["face_fix_max_material"] = max((x for x in face_rows if x["face_before"] != x["face"]),
                                      key=lambda x: abs(x["E_after"] - x["E_before"]))["material"]

# ---------------------------------------------------------------- the error bar
values.update(uma_mae_id=0.052, uma_mae_ood=0.070, uma_adsorbml=66.8, uma_structures_million=459)
setup = [values["sites_max_gain"], values["tolerance_max_gap"], values["slab_size_max_gap"], values["strain_error_est"]]
values["setup_quadrature"] = r(math.sqrt(sum(s * s for s in setup)), 3)
values["uma_sigma_est"] = r(values["uma_mae_ood"] * math.sqrt(math.pi / 2), 3)  # SD of a normal error with that MAE
values["sigma_combined"] = r(math.sqrt(values["uma_sigma_est"] ** 2 + values["setup_quadrature"] ** 2), 3)
values["sigma_assumed"] = pack.spec.screening.sigma_eV

# ---------------------------------------------------------------- the volcano: sensitivity of the ranking
preds, _ = screening.predict(pack, job, results)
spec = pack.spec.screening
ref = next(p for p in preds if p.is_reference)
done = [p for p in preds if p.descriptor is not None]


def ranking(s) -> list[str]:
    return [p.formula for p in sorted(done, key=lambda p: s.barrier_shift(p.descriptor, ref.descriptor))]


def spearman(a: list[str], b: list[str]) -> float:
    ra = {k: i for i, k in enumerate(a)}
    rb = {k: i for i, k in enumerate(b)}
    n = len(a)
    return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in a) / (n * (n * n - 1))


base = ranking(spec)
rows = []
for opt in (0.0, 0.1, 0.2, 0.3, 0.4):
    for weak, strong in ((0.9, 1.0), (0.5, 0.5), (1.0, 1.0), (0.5, 1.0), (1.0, 0.5)):
        s = spec.model_copy(update={"optimum_shift_eV": opt, "slope_weak": weak, "slope_strong": strong})
        order = ranking(s)
        rows.append({"optimum_shift_eV": opt, "slope_weak": weak, "slope_strong": strong, "leader": order[0],
                     "top3": ", ".join(order[:3]), "spearman": r(spearman(base, order), 3)})
tables["volcano_sensitivity"] = rows
slope_rows = [x for x in rows if x["optimum_shift_eV"] == spec.optimum_shift_eV]
values["volcano_spearman_slopes_min"] = r(min(x["spearman"] for x in slope_rows), 2)
values["volcano_spearman_all_min"] = r(min(x["spearman"] for x in rows), 2)
values["volcano_leaders"] = ", ".join(sorted({x["leader"] for x in rows}))
values["volcano_top3_base"] = ", ".join(base[:3])
t_design = job["request"]["design_temperature_C"]
values["kT_design"] = r(screening.K_B * (t_design + 273.15), 4)
values["design_T"] = t_design
values["factor_01"] = r(math.exp(0.1 / (screening.K_B * (t_design + 273.15))), 1)
values["factor_01_400"] = r(math.exp(0.1 / (screening.K_B * 673.15)), 1)
values["sigma_factor"] = r(math.exp(spec.sigma_eV / (screening.K_B * (t_design + 273.15))), 1)
values["example_leader"] = next(p.formula for p in preds if p.rank == 1)
values["example_ranking"] = ", ".join(p.formula for p in preds if p.rank)
overlap = [p for p in done if not p.is_reference and p.rate_range[0] <= 1 <= p.rate_range[1]]
values["example_within_error"] = ", ".join(p.formula for p in overlap)
tables["example_ranking"] = [{"rank": p.rank, "material": p.formula, "descriptor_eV": r(p.descriptor),
                              "shift_eV": r(p.shift), "rate_ratio": float(f"{p.rate_ratio:.3g}"),
                              "range_low": float(f"{p.rate_range[0]:.2g}"), "range_high": float(f"{p.rate_range[1]:.2g}"),
                              "metric_at_design": r(p.metric_at_design)} for p in preds if p.rank]

# ---------------------------------------------------------------- the reactor
# (1) the integrator against the exact plug-flow answer for one first-order reaction with volume change
co = load_pack(job["request"]["pack"])
step = screening.design_step(co)
y = step.feed_fractions(co.network.species, co.spec.water)
temps = np.linspace(380.0, 560.0, 13)
params = co.network.parameters()
cond = Conditions(temps, np.full(13, step.pressure_kPa * 1000.0), np.full(13, step.ghsv_per_h), np.tile(y, (13, 1)))
out = outlet(co.network, params, cond)
pw = spec.pathways[0]
i_co = co.network.index("CO")
eps = y[i_co] * (-0.5)  # moles of gas per mole fed, at full conversion of CO: CO + 1/2 O2 -> CO2
k = np.exp(params[f"{pw}.ln_k_ref"] - params[f"{pw}.Ea_kJ"] * 1000 / R * (1 / temps - 1 / co.network.t_ref))
q = step.ghsv_per_h / 3600 * temps / 273.15 * 101325 / (step.pressure_kPa * 1000)  # actual gas flow per catalyst volume
# exact first-order PFR with volume change: k tau = -(1 + eps) ln(1 - X) - eps X; solve for X
x_exact = []
for kt in k / q:
    lo, hi = 0.0, 1.0 - 1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if -(1 + eps) * math.log(1 - mid) - eps * mid < kt else (lo, mid)
    x_exact.append(0.5 * (lo + hi))
x_exact = np.array(x_exact)
x_num = co.metrics(cond.feed, out)["co_conversion"]
values["reactor_exact_max_gap"] = float(f"{np.abs(x_num - x_exact).max():.1g}")
values["reactor_exact_pct"] = 100 * values["reactor_exact_max_gap"]
tables["reactor_exact"] = [{"T_C": r(t - 273.15, 1), "X_engine": r(a, 5), "X_exact": r(b, 5)}
                           for t, a, b in zip(temps, x_num, x_exact)]

# (2) element balance and (3) equilibrium, for every pack
bal_rows, eq_rows = [], []
for pid in list_packs(PACKS_DIR):
    p = load_pack(pid)
    st = screening.design_step(p)
    yy = st.feed_fractions(p.network.species, p.spec.water)
    tk = np.array(st.kelvin())
    n = len(tk)
    c = Conditions(tk, np.full(n, st.pressure_kPa * 1000.0), np.full(n, st.ghsv_per_h), np.tile(yy, (n, 1)))
    o = outlet(p.network, screening.fast_params(p), c)
    formulas = [p.spec.species[s].formula for s in p.network.species]
    elements = sorted({e for f in formulas for e in element_counts(f)})
    a = np.array([[element_counts(f).get(e, 0) for f in formulas] for e in elements])
    e_in, e_out = a @ yy, (a @ o.T).T
    ratio = e_out / e_in  # every element changes by the same factor (the change in total moles) when atoms balance
    dev = np.abs(ratio / ratio.mean(axis=1, keepdims=True) - 1).max()
    bal_rows.append({"reaction": p.spec.name, "worst_element_gap": float(f"{dev:.1g}")})
    # equilibrium: packs whose species allow one independent reaction, every pathway reversible and screened
    reversible = all(pw_.equation and "<=>" in pw_.equation for pw_ in p.spec.pathways.values())
    if len(formulas) - np.linalg.matrix_rank(a) == 1 and reversible and set(p.spec.pathways) <= set(p.spec.screening.pathways):
        gas = ct.Solution(thermo="ideal-gas", species=[s for s in ct.Species.list_from_file("gri30.yaml")
                                                         if s.name in {p.spec.species[x].thermo_id for x in p.network.species}])
        hot = np.argsort(tk)[-3:]  # the hottest temperatures, where the fast catalyst sits at equilibrium
        gaps = []
        for j in hot:
            gas.TPX = tk[j], st.pressure_kPa * 1000.0, {p.spec.species[s].thermo_id: v for s, v in zip(p.network.species, yy) if v > 0}
            gas.equilibrate("TP")
            y_eq = np.array([gas[p.spec.species[s].thermo_id].X[0] for s in p.network.species])
            gaps.append(np.abs(o[j] - y_eq).max())
        eq_rows.append({"reaction": p.spec.name, "temperatures_C": ", ".join(f"{tk[j] - 273.15:.0f}" for j in hot),
                        "largest_mole_fraction_gap": float(f"{max(gaps):.1g}")})
tables["element_balance"] = bal_rows
tables["equilibrium"] = eq_rows
values["balance_worst"] = float(f"{max(x['worst_element_gap'] for x in bal_rows):.1g}")
values["balance_packs"] = len(bal_rows)
values["equilibrium_packs"] = len(eq_rows)
values["equilibrium_worst"] = float(f"{max(x['largest_mole_fraction_gap'] for x in eq_rows):.1g}") if eq_rows else None
# the same gaps as matching decimal places, for readers who skip scientific notation
values["balance_digits"] = math.floor(-math.log10(values["balance_worst"]))
values["equilibrium_digits"] = math.floor(-math.log10(values["equilibrium_worst"])) if eq_rows else None

# (4) calibration: each pack's catalyst to beat at its reference temperature, against the best possible
cal_rows = []
for pid in list_packs(PACKS_DIR):
    p = load_pack(pid)
    st = screening.design_step(p)
    one = st.model_copy(update={"temperatures_C": [p.network.t_ref - 273.15], "temperatures_F": None, "temperatures_K": None})
    best, refv = screening.metric_curves(p, one, [screening.fast_params(p), p.network.parameters()], p.spec.screening.metric)[:, 0]
    cal_rows.append({"reaction": p.spec.name, "reference": p.spec.screening.reference.name or p.spec.screening.reference.formula,
                     "T_ref_C": r(p.network.t_ref - 273.15, 0), "reference_metric": r(refv), "best_metric": r(best),
                     "share_of_best": r(refv / best, 3) if best else None})
tables["calibration"] = cal_rows
halfway = [x for x in cal_rows if x["share_of_best"] is not None and abs(x["share_of_best"] - 0.5) < 0.02]
values["calibrated_halfway"] = len(halfway)
values["calibration_packs"] = len(cal_rows)
others = [x for x in cal_rows if x not in halfway]
values["calibration_note"] = (
    f"{len(halfway)} of {len(cal_rows)} packs put the catalyst to beat at 50% of the best possible, within 2 points, "
    "at their anchor temperature." + ("" if not others else " " + "; ".join(
        f"{x['reaction']} sits at {100 * x['share_of_best']:.0f}% at {x['T_ref_C']:.0f} °C" for x in others)
        + ". These two packs came before the design app: their illustrative rates follow a conventional catalyst across "
        "the whole operating window, and their reference temperature anchors the rate constants in place of a "
        "light-off. The ranking within each reaction stays the same; their curves sit hotter or cooler accordingly."))

values["generated"] = atom["generated"]
values["atomistic_minutes"] = r(atom["seconds"] / 60, 0)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "summary.json").write_text(json.dumps({"generated": atom["generated"], "values": values,
                                                            "tables": tables}, indent=1, default=float))
print(json.dumps(values, indent=1, default=float))
