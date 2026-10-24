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

from catalyst_engine.chem import thermo
from catalyst_engine.chem.formula import element_counts
from catalyst_engine.chem.reactor import Conditions, outlet
from catalyst_engine.design import screening, uma_worker
from catalyst_engine.packs import PACKS_DIR, list_packs, load_pack
from catalyst_engine.units import R
from catalyst_engine.value.monolith import Monolith

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLE = next(p for p in sorted((REPO / "examples" / "screens").iterdir())  # the thermal example
               if (p / "job.json").exists() and json.loads((p / "job.json").read_text())["request"].get("kind") != "electro")
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
same_face = [x for x in face_rows if x["face_before"] == x["face"]
             and x["E_before"] is not None and x["E_after"] is not None]
values["metals_unchanged"] = len(same_face)
# since the earlier run, the nudge before each relaxation (step 4) moves these by a hundredth of an eV at most
values["metals_max_shift"] = r(max(abs(x["E_before"] - x["E_after"]) for x in same_face), 3)

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

# ---------------------------------------------------------------- the volcano's top across temperature
# (1) the rule against an exact two-step cycle, G <-> X* (adsorption), X* -> product (removal), solved at steady
# state for several mechanisms: how far its peak moves from the anchor to 200 K hotter, over the gas's own change
KB_J, H_EV, AMU = 1.380649e-23, 4.135668e-15, 1.660539e-27
co_pack = load_pack(job["request"]["pack"])
gas_id, n_gas = next(iter(co_pack.spec.screening.peak_gas.items()))
std = screening.design_step(co_pack)
x_gas = std.feed_fractions(co_pack.network.species, co_pack.spec.water)[co_pack.network.species.index(gas_id)]
p_gas = x_gas * std.pressure_kPa * 1000.0
mass = sum(n * {"O": 15.999, "N": 14.007, "C": 12.011, "H": 1.008}[el] for el, n in element_counts(gas_id).items())


def cycle_peak(t_k: float, alpha: float, beta: float, e1: float, offset: float) -> float:
    d = np.linspace(-3.0, 3.0, 60001)
    mu_gas = float(screening.gas_potential(co_pack, std, [t_k])[0])
    flux = p_gas * 6.7e-20 / math.sqrt(2 * math.pi * mass * AMU * KB_J * t_k)  # Hertz-Knudsen, per site
    ln_a = math.log(flux) - np.maximum(0.0, e1 + alpha * d) / (screening.K_B * t_k)
    ln_b = ln_a + (d + offset - mu_gas) / (screening.K_B * t_k)  # detailed balance with the gas
    ln_c = math.log(screening.K_B * t_k / H_EV) - np.maximum(0.0, 0.8 - beta * d) / (screening.K_B * t_k)
    ln_r = ln_a + ln_c - np.logaddexp(np.logaddexp(ln_a, ln_b), ln_c)
    return float(d[int(np.argmax(ln_r))])


t0 = co_pack.network.t_ref
gas_change = float(np.diff(screening.gas_potential(co_pack, std, [t0, t0 + 200.0]))[0])
cycle_rows = []
for case, alpha, beta, e1 in (("Adsorption free of barrier, removal slope 0.5", 0.0, 0.5, 0.0),
                              ("Adsorption free of barrier, removal slope 0.9", 0.0, 0.9, 0.0),
                              ("Adsorption with a barrier, slopes 0.5 and 0.5", 0.5, 0.5, 0.3),
                              ("Adsorption with a barrier, slopes 0.9 and 1.0 (the app's)", 0.9, 1.0, 0.6)):
    moved = cycle_peak(t0 + 200.0, alpha, beta, e1, -1.0) - cycle_peak(t0, alpha, beta, e1, -1.0)
    cycle_rows.append({"case": case, "peak_moved_eV": r(moved, 3), "gas_change_eV": r(gas_change, 3),
                       "follow": r(moved / gas_change, 2)})
tables["peak_cycle"] = cycle_rows
values["peak_follow_min"] = min(x["follow"] for x in cycle_rows)
values["peak_follow_max"] = max(x["follow"] for x in cycle_rows)
values["peak_follow_app"] = cycle_rows[-1]["follow"]
values["peak_follow_low"] = screening.PEAK_FOLLOW[1]
values["peak_gas_change_200"] = r(-gas_change, 2)
values["peak_check_T"] = round(t0 - 273.15)

# (2) every pack: the gas its top follows, and how far the top moves 100 K above its anchor, with its standard feed
move_rows = []
for pid in list_packs():
    pk = load_pack(pid)
    if pk.spec.screening is None:
        continue
    st_ = screening.design_step(pk)
    ta = pk.network.t_ref
    m = screening.peak_moves(pk, st_, [ta + 100.0])[0]
    move_rows.append({"reaction": pk.spec.name, "gas": " ".join(f"{n:+g} {g}" for g, n in pk.spec.screening.peak_gas.items()),
                      "anchor_C": round(ta - 273.15), "move_per_100K": r(m, 3)})
tables["peak_moves"] = sorted(move_rows, key=lambda x: x["move_per_100K"])
values["peak_move_largest"] = r(-min(x["move_per_100K"] for x in move_rows), 2)
values["peak_move_co"] = r(-screening.peak_moves(co_pack, std, [t0 + 100.0])[0], 2)

# (3) the example across its temperatures
_, ex_curves = screening.predict(pack, job, results)
labels = {p.material: p.formula for p in preds}
spans = screening.leader_spans(ex_curves, labels)
values["example_leader_spans"] = "; ".join(f"{n} from {a:.0f} to {b:.0f} °C" for n, a, b in spans)
ex_t = ex_curves["T_C"]
ratio_at = {p.formula: ex_curves["materials"][p.material]["ratio"] for p in done}
values["example_t_lo"], values["example_t_hi"] = ex_t[0], ex_t[-1]
values["example_pdpt_hot"] = float(f"{ratio_at['PdPt'][-1]:.2g}")


def one_in(v: float) -> str:
    """A small speed ratio in words: 1.9e-12 reads "1 in 520 billion"."""
    n = 1 / v
    for size, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1.0, "")):
        if n >= size:
            return f"1 in {float(f'{n / size:.2g}'):,g} {word}".strip()
    return f"{v:.2g}"


values["example_rh_hot"] = one_in(ratio_at["Rh"][-1])
values["example_rh_design"] = one_in(next(p for p in done if p.formula == "Rh").rate_ratio)
values["example_peak_hot"] = r(ex_curves["peak"][0] - ex_curves["peak"][-1], 2)  # toward stronger binding

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

# ---------------------------------------------------------------- the catalyst's form
# (1) a coated honeycomb against the exact answer: first order in CO with the film in series, k km / (k + km)
mono = screening.MONOLITH
check_step = step.model_copy(update={"temperatures_C": list(temps - 273.15), "temperatures_F": None,
                                     "temperatures_K": None})
k_wall = screening.wall_transfer(co, check_step, mono)[:, [x.id for x in co.network.pathways].index(pw)]
out = outlet(co.network, params, Conditions(temps, cond.p_pa, cond.ghsv_per_h, cond.feed,
                                            km_a=screening.wall_transfer(co, check_step, mono)))
x_film = co.metrics(cond.feed, out)["co_conversion"]
x_film_exact = []
for kt in k * k_wall / (k + k_wall) / q:
    lo, hi = 0.0, 1.0 - 1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if -(1 + eps) * math.log(1 - mid) - eps * mid < kt else (lo, mid)
    x_film_exact.append(0.5 * (lo + hi))
values["film_exact_pct"] = 100 * float(f"{np.abs(x_film - np.array(x_film_exact)).max():.1g}")
tables["film_exact"] = [{"T_C": r(t - 273.15, 1), "k_wall": r(kw, 0), "X_powder": r(a, 4), "X_engine": r(b, 4),
                         "X_exact": r(c, 4)} for t, kw, a, b, c in zip(temps, k_wall, x_num, x_film, x_film_exact)]

# (2) how much each part of the form moves hot SCR's catalyst to beat, and one ten times faster, at 1,050 °F:
# the honeycomb's film (the model), Hawthorn's entrance term (left out: the model's film is the fully developed
# one, the slowest), and diffusion into a 50 um coat at a hundredth and a tenth of the gas value (left out)
scr = load_pack("scr_hot")
s_step = screening.design_step(scr)
s_one = s_step.model_copy(update={"temperatures_C": [scr.network.t_ref - 273.15], "temperatures_F": None,
                                  "temperatures_K": None})
t_s = scr.network.t_ref
p_s = s_one.pressure_kPa * 1000.0
y_s = s_one.feed_fractions(scr.network.species, scr.spec.water)
gas_s = {scr.spec.species[s].thermo_id: float(v) for s, v in zip(scr.network.species, y_s) if v > 0}
keys_s = [scr.spec.species[scr.network.species[x.key]].thermo_id for x in scr.network.pathways]
km_s = screening.wall_transfer(scr, s_one, mono)
honeycomb = Monolith(mono["cpsi"], mono["wall_mm"])
q_act = s_one.ghsv_per_h / 3600 * t_s / 273.15 * 101325 / p_s  # actual gas flow per honeycomb volume, 1/s
# Hawthorn (1974): Sh = 2.98 (1 + 0.095 Re Sc d/L)^0.45; at a fixed space velocity Re Sc d/L = q d^2 / (OFA D), so the
# block's length drops out
graetz = np.array([q_act * honeycomb.hydraulic_diameter_m ** 2 / (honeycomb.open_frontal_area * thermo.diffusivity(
    sp, t_s, p_s, gas_s)) for sp in keys_s])
entrance = (1 + 0.095 * graetz) ** 0.45
base_s = scr.network.parameters()
faster = dict(base_s)
for x in scr.spec.screening.pathways:
    faster[f"{x}.ln_k_ref"] = base_s[f"{x}.ln_k_ref"] + math.log(10.0)
cases = [("Powder", None, None), ("Honeycomb, film (the model)", km_s, None),
         ("Honeycomb, film with the entrance term", km_s * entrance, None)]
for ratio in (0.1, 0.01):
    coat = Monolith(mono["cpsi"], mono["wall_mm"], washcoat_um=50.0, washcoat_de_ratio=ratio)
    cases.append((f"Honeycomb, film and a 50 µm coat at {ratio:g} of gas diffusion", km_s,
                  np.array([[coat.washcoat_phi2(sp, t_s, p_s, gas_s) for sp in keys_s]])))
form_rows = []
for label, km_case, wc in cases:
    cond_s = Conditions([t_s, t_s], [p_s, p_s], [s_one.ghsv_per_h] * 2, np.tile(y_s, (2, 1)),
                        km_a=None if km_case is None else np.tile(km_case, (2, 1)),
                        wc_phi2=None if wc is None else np.tile(wc, (2, 1)))
    o_s = outlet(scr.network, {k_: np.array([base_s[k_], faster[k_]]) for k_ in base_s}, cond_s)
    ref_x, fast_x = scr.metrics(cond_s.feed, o_s)[scr.spec.screening.metric]
    form_rows.append({"form": label, "reference": r(ref_x), "ten_times": r(fast_x), "gain": r(fast_x - ref_x)})
tables["form"] = form_rows
by_form = {x["form"]: x for x in form_rows}
powder, film, film_in = (by_form[c[0]] for c in cases[:3])
coat_fast, coat_slow = by_form[cases[3][0]], by_form[cases[4][0]]
values["form_T_C"] = r(t_s - 273.15, 0)
values["form_T_F"] = r((t_s - 273.15) * 9 / 5 + 32, 0)
values["form_film_drop_pts"] = r(100 * (powder["reference"] - film["reference"]), 1)
values["form_gain_powder_pts"] = r(100 * powder["gain"], 1)
values["form_gain_film_pts"] = r(100 * film["gain"], 1)
values["form_entrance_factor"] = r(float(entrance.max()), 2)
values["form_entrance_pts"] = r(100 * max(film_in["reference"] - film["reference"], film_in["ten_times"] - film["ten_times"]), 1)
values["form_coat_drop_lo_pts"] = r(100 * (film["reference"] - coat_fast["reference"]), 1)
values["form_coat_drop_hi_pts"] = r(100 * (film["reference"] - coat_slow["reference"]), 1)
values["form_cpsi"] = r(mono["cpsi"], 0)
values["form_channel_mm"] = r(honeycomb.hydraulic_diameter_m * 1000, 2)
# (3) published monoliths: the Literature check's apparent activation energies on Metkar's 400 cpsi cores
lit = json.loads((REPO / "experiments" / "literature_scr" / "results" / "summary.json").read_text())["catalysts"]
lit_rows = []
for cid in ("fe_zsm5", "cu_cha"):
    c = lit[cid]
    alt = {(a.get("film", False), a.get("washcoat", False)): a for a in c["alternatives"]}
    lit_rows.append({"catalyst": c.get("name", cid), "ideal": r(c["full"]["params"]["scr.Ea_kJ"], 0),
                     "film": r(alt[(True, False)]["params"]["scr.Ea_kJ"], 0),
                     "film_coat": r(alt[(True, True)]["params"]["scr.Ea_kJ"], 0),
                     "published": r(c["published_Ea_kJ"]["scr"], 0)})
tables["form_literature"] = lit_rows
values["form_lit_fe"] = "{ideal:.0f} kJ/mol in the ideal reactor, {film:.0f} with the film and {film_coat:.0f} with film and coat, against the thesis's {published:.0f}".format(**lit_rows[0])
values["form_lit_cu"] = "{ideal:.0f}, {film:.0f} and {film_coat:.0f} against {published:.0f}".format(**lit_rows[1])

# ---------------------------------------------------------------- relaxation: the nudge, and iron
tables["nudge"] = [{"material": x["material"], "change": r(x["E_nudged"] - x["E_unnudged"], 4),
                    "moved_A": r(x["moved_A"], 2)} for x in atom["nudge"]]
values["nudge_n"] = len(tables["nudge"])
values["nudge_max_change"] = r(max(abs(x["change"]) for x in tables["nudge"]), 3)
iron = atom["iron"]
values["iron_before"] = r(iron["E_NH3_before"], 2)
values["iron_after"] = r(iron["E_NH3_after"], 2)
values["iron_slab_drop"] = r(iron["slab_before"] - iron["slab_after"], 2)
tables["bain"] = [{"c_over_a": r(x["c_over_a"], 2), "oc20": r(x["oc20"], 3), "omat": r(x["omat"], 3)}
                  for x in iron["bain_eV_per_atom"]]
fcc = max(iron["bain_eV_per_atom"], key=lambda x: x["c_over_a"])  # c/a = sqrt(2): the bcc cell stretched into fcc
values["iron_fcc_oc20"] = r(fcc["oc20"], 2)
values["iron_fcc_omat"] = r(fcc["omat"], 2)

# ---------------------------------------------------------------- zeolites: ammonia in copper chabazite
zeo = json.loads((REPO / "experiments" / "zeolite_check" / "results" / "zeolite.json").read_text())
ZEO_SITES = {  # site: (label, published low, published high, source)
    "H-CHA": ("Acid site (H+, one Al)", -1.62, -1.62, "Nystrom et al. 2018, RPBE-D3"),
    "ZCu": ("Cu+ over a six-ring (one Al)", -1.3, -1.0, "Isapour et al. 2026, PBE+U+D3"),
    "Z2Cu": ("Cu2+ in a six-ring (two Al)", -1.5, -1.2, "Isapour et al. 2026, PBE+U+D3"),
    "ZCuOH": ("[CuOH]+ in an eight-ring (one Al)", None, None, ""),
}
modes = [m for m in ("oc20", "omat", "odac") if m in zeo]
zeo_rows, miss = [], {m: [] for m in modes}
for site, (label, lo, hi, src) in ZEO_SITES.items():
    row = {"site": label, **{m: r(zeo[m]["sites"][site]["E_bind"], 2) for m in modes},
           "published": "" if lo is None else (f"{lo:.2f}" if lo == hi else f"{lo:.1f} to {hi:.1f}"), "source": src}
    zeo_rows.append(row)
    if lo is not None:
        for m in modes:
            e = zeo[m]["sites"][site]["E_bind"]
            miss[m].append(max(0.0, lo - e, e - hi))  # distance outside the published range
tables["zeolite"] = zeo_rows
worst = {m: max(v) for m, v in miss.items()}
best_mode = min(worst, key=worst.get)
values["zeolite_best_mode"] = best_mode
values["zeolite_best_miss"] = r(worst[best_mode], 2)
values["zeolite_misses"] = ", ".join(f"{worst[m]:.2f} eV in the {m} mode" for m in modes)

# ---------------------------------------------------------------- oxides: molecules on oxide surfaces
ox = json.loads((REPO / "experiments" / "oxide_check" / "results" / "oxide.json").read_text())
OX_MODES = {"uma-s-1p1/oc20": "UMA-S 1.1, catalysis (the app's)", "uma-s-1p1/omat": "UMA-S 1.1, materials",
            "uma-s-1p2p1/oc22": "UMA-S 1.2.1, oxide catalysis", "uma-s-1p2p1/oc20": "UMA-S 1.2.1, catalysis",
            "uma-s-1p2p1/omat": "UMA-S 1.2.1, materials"}
# a model's MAE keeps the app's error bar when, with the setup terms above, it stays inside sigma_assumed
values["oxide_gate"] = r(math.sqrt(max(0.0, values["sigma_assumed"] ** 2 - values["setup_quadrature"] ** 2))
                         / math.sqrt(math.pi / 2), 2)


def oxide_scores(rows: list[dict]) -> dict:
    """Errors against OC22's DFT. Each adsorbate's median error is set aside first: it holds the gap between the
    model's and DFT's reference energies for that molecule, which the app's gas references remove."""
    by_ads: dict[str, list[dict]] = {}
    for x in rows:
        by_ads.setdefault(x["adsorbate"], []).append(x)
    resid, agree, pairs, ads_mae = [], 0, 0, {}
    for a, xs in by_ads.items():
        err = np.array([x["dE"] - x["dE_dft"] for x in xs])
        res = err - np.median(err)
        ads_mae[a] = float(np.abs(res).mean())
        resid += [(x["split"], float(e)) for x, e in zip(xs, res)]
        dft, mod = np.array([x["dE_dft"] for x in xs]), np.array([x["dE"] for x in xs])
        i, j = np.triu_indices(len(xs), 1)
        apart = np.abs(dft[i] - dft[j]) >= values["sigma_assumed"]  # surfaces DFT tells apart beyond the error bar
        pairs += int(apart.sum())
        agree += int((np.sign(dft[i] - dft[j]) == np.sign(mod[i] - mod[j]))[apart].sum())
    split = lambda s: float(np.mean([abs(e) for sp, e in resid if sp == s]))
    return {"n": len(rows), "mae_id": split("val_id"), "mae_ood": split("val_ood"),
            "mae": float(np.mean([abs(e) for _, e in resid])), "order": 100.0 * agree / max(pairs, 1),
            "worst_ads": max(ads_mae, key=ads_mae.get), "worst_ads_mae": max(ads_mae.values()), "ads": ads_mae}


ox_scores = {k: oxide_scores(ox["single"][k]) for k in OX_MODES if k in ox.get("single", {})}
tables["oxide"] = [{"mode": OX_MODES[k], "n": s["n"], "mae_id": r(s["mae_id"], 2), "mae_ood": r(s["mae_ood"], 2),
                    "order": r(s["order"], 0), "worst": f"{s['worst_ads']} ({s['worst_ads_mae']:.2f})"}
                   for k, s in ox_scores.items()]
ADS_ORDER = ["O", "H", "OH", "H2O", "HO2", "O2", "CO", "C", "N"]
tables["oxide_ads"] = [{"adsorbate": a, **{k.replace("uma-s-", "").replace("/", "_"): r(s["ads"][a], 2)
                                            for k, s in ox_scores.items() if a in s["ads"]}}
                       for a in ADS_ORDER if any(a in s["ads"] for s in ox_scores.values())]
app_mode, cand = "uma-s-1p1/oc20", "uma-s-1p2p1/oc22"
values["oxide_modes_n"] = ["one", "two", "three", "four", "five"][len(ox_scores) - 1]
values.update(oxide_n=ox_scores[app_mode]["n"], oxide_app_mae=r(ox_scores[app_mode]["mae"], 2),
              oxide_app_mae_ood=r(ox_scores[app_mode]["mae_ood"], 2), oxide_app_order=r(ox_scores[app_mode]["order"], 0))
if cand in ox_scores:
    values.update(oxide_cand_mae=r(ox_scores[cand]["mae"], 2), oxide_cand_mae_ood=r(ox_scores[cand]["mae_ood"], 2),
                  oxide_cand_order=r(ox_scores[cand]["order"], 0))
mat = "uma-s-1p1/omat"
values.update(oxide_mat_mae_id=r(ox_scores[mat]["mae_id"], 2), oxide_mat_mae_ood=r(ox_scores[mat]["mae_ood"], 2),
              oxide_mat_order=r(ox_scores[mat]["order"], 0), oxide_cand_mae_id=r(ox_scores[cand]["mae_id"], 2),
              oxide_app_mae_id=r(ox_scores[app_mode]["mae_id"], 2),
              oxide_mat_factor=r(ox_scores[app_mode]["mae_ood"] / ox_scores[mat]["mae_ood"], 0))
best_ox = min(ox_scores, key=lambda k: ox_scores[k]["mae_ood"])
values["oxide_best_mode"] = OX_MODES[best_ox]
values["oxide_best_mae_ood"] = r(ox_scores[best_ox]["mae_ood"], 2)
values["oxide_modes_pass"] = sum(1 for s in ox_scores.values() if s["mae_ood"] <= values["oxide_gate"])

if ox.get("relax"):
    tables["oxide_relax"] = []
    for k, rows in ox["relax"].items():
        s = oxide_scores(rows)
        tables["oxide_relax"].append({"mode": OX_MODES[k], "n": s["n"], "mae": r(s["mae"], 2),
                                      "single": r(oxide_scores([{**x, "dE": x["dE_single"]} for x in rows])["mae"], 2),
                                      "converged": sum(x["converged"] for x in rows)})
        values[f"oxide_relax_{k.split('/')[1]}_{k.split('/')[0][-3:]}"] = r(s["mae"], 2)

ANATASE_MEASURED = -1.23  # Koust et al. 2018: TPD, one NH3 on five-fold Ti of anatase (101) at zero coverage, +/- 0.1
NH3_COLS = {"anatase app face": "anatase_001", "anatase (101)": "anatase_101_app",
            "anatase (101), thicker": "anatase_101_app_thick", "v2o5 app face": "v2o5"}
nh3_rows, app_found = [], set()
for k, rows in ox["nh3"].items():
    best = {}  # the most stable intact NH3 per material and face, as the app keeps
    for x in rows:
        key = f"{x['material']} {x['face']}"
        if x["anomaly"]:
            continue
        if x["site_from"] == "app" and x["site"] == "top Ti" and x["site_bonds"] == 5:
            app_found.add(key)
        if key not in best or x["E_bind"] < best[key]["E_bind"]:
            best[key] = x
    far = [x["E_bind"] for x in rows if x["anomaly"] == "desorbed"]  # NH3 that drifted over 3 A off: near zero
    nh3_rows.append({"mode": OX_MODES[k], **{NH3_COLS[f]: r(x["E_bind"], 2) for f, x in best.items()},
                     "far": f"{min(far):.2f} to {max(far):.2f}" if far else ""})
tables["nh3"] = nh3_rows
app_row = next(x for x in nh3_rows if x["mode"] == OX_MODES[app_mode])
# a mode's gas reference agrees with its surfaces when NH3 that drifted off scores within the error bar of zero
consistent = [k for k, rows in ox["nh3"].items()
              if all(abs(x["E_bind"]) <= values["sigma_assumed"] for x in rows if x["anomaly"] == "desorbed")]
values["ref_consistent"] = " and ".join(OX_MODES[k] for k in consistent)
values["ref_offset_lo"] = r(min(-max(x["E_bind"] for x in ox["nh3"][k] if x["anomaly"] == "desorbed")
                            for k in ox["nh3"] if k not in consistent), 2)
values["ref_offset_hi"] = r(max(-min(x["E_bind"] for x in ox["nh3"][k] if x["anomaly"] == "desorbed")
                            for k in ox["nh3"] if k not in consistent), 2)
v2o5 = [x["v2o5"] for x in nh3_rows if any(x["mode"] == OX_MODES[k] for k in consistent)]
values.update(v2o5_nh3_lo=min(v2o5), v2o5_nh3_hi=max(v2o5), v2o5_nh3_app=app_row["v2o5"],
              v2o5_nh3_spread=r(max(v2o5) - min(v2o5), 2),
              anatase_builder_found=", ".join(sorted(app_found)) or "none",
              far_app=r(min(x["E_bind"] for x in ox["nh3"][app_mode] if x["anomaly"] == "desorbed"), 2),
              far_worst=r(min(x["E_bind"] for k in ox["nh3"] for x in ox["nh3"][k] if x["anomaly"] == "desorbed"), 2))
ana_rows = []
for k, rows in ox["anatase101"].items():
    by = {x["layers"]: x for x in rows}
    miss = by[4]["E_bind"] - ANATASE_MEASURED
    ana_rows.append({"mode": OX_MODES[k], "four": r(by[4]["E_bind"], 2), "five": r(by[5]["E_bind"], 2),
                     "miss": r(miss, 2), "N_Ti": by[4]["N_Ti_A"], "intact": not by[4]["anomaly"],
                     "match": abs(miss) <= values["sigma_assumed"] and k in consistent,
                     "far": next(x["far"] for x in nh3_rows if x["mode"] == OX_MODES[k])})
tables["anatase"] = ana_rows
best_ana = min(ana_rows, key=lambda x: abs(x["miss"]))
values.update(anatase_measured=ANATASE_MEASURED,
              anatase_app=next(x["four"] for x in ana_rows if x["mode"] == OX_MODES[app_mode]),
              anatase_app_miss=next(x["miss"] for x in ana_rows if x["mode"] == OX_MODES[app_mode]),
              anatase_misses=", ".join(f"{x['miss']:+.2f} eV ({x['mode']})" for x in ana_rows),
              anatase_best_mode=best_ana["mode"], anatase_best_miss=best_ana["miss"],
              anatase_layers_gap=r(max(abs(x["four"] - x["five"]) for x in ana_rows), 2),
              anatase_match=" and ".join(x["mode"] for x in ana_rows if x["match"]) or "one mode",
              anatase_mat_miss=next(x["miss"] for x in ana_rows if x["mode"] == OX_MODES[mat]),
              far_cand=r(-float(np.mean([x["E_bind"] for x in ox["nh3"][cand] if x["anomaly"] == "desorbed"])), 2))
# the worker's own (101) cut against the true one, where both kept NH3 whole
cut = [abs(n["anatase_101_app"] - a["four"]) for n, a in zip(nh3_rows, ana_rows) if "anatase_101_app" in n]
values["cut_gap"] = r(max(cut), 1)
values["v2o5_nh3_mat"] = next(x["v2o5"] for x in nh3_rows if x["mode"] == OX_MODES[mat])
values["cut_gap_app"] = r(abs(app_row["anatase_101_app"] - values["anatase_app"]), 2)

# ---------------------------------------------------------------- electrocatalysis: the hydrogen electrode on metals
el = json.loads((REPO / "experiments" / "electro_check" / "results" / "electro.json").read_text())
values.update(el["values"])
tables.update(el["tables"])
gate = values["oxide_gate"]  # the typical miss the error bar allows, as the oxide check set it
values["electro_verdict"] = (
    f"The catalysis mode reproduces published RPBE free energies of hydrogen within {values['her_mae_rpbe']:.2f} eV "
    f"and hydroxyl and oxygen within {values['orr_oh_mae']:.2f} and {values['orr_o_mae']:.2f} eV on average, orders the "
    f"metals as the measured exchange currents do ({values['her_spearman']:.2f}), and puts the OOH–OH gap at "
    f"{values['ooh_gap_mean']:.2f} eV against the universal {values['ooh_gap_published']:.1f}. "
    + (f"Hydrogen and hydroxyl stay within the {gate:.2f} eV the error bar allows, "
       if max(values["her_mae_rpbe"], values["orr_oh_mae"]) <= gate else
       f"Hydrogen or hydroxyl land beyond the {gate:.2f} eV the error bar allows, ")
    + (f"and so does oxygen. " if values["orr_o_mae"] <= gate else
       f"and oxygen sits just beyond it with an offset of {values['orr_o_offset']:+.2f} eV that every metal shares, so it "
       "cancels against the catalyst to beat. ")
    + "Hydrogen-evolution rankings carry the same error bar as the thermal screens. Read oxygen-reduction and "
    "oxygen-evolution overpotentials as the order of a shortlist, with the absolute values resting on platinum's "
    "corrections; the thermodynamic picture ranks silver above platinum for oxygen reduction, as Nørskov 2004's own "
    "table does before its authors set copper, silver and gold aside for their O2 activation barrier and their "
    "fate in acid.")

values["generated"] = atom["generated"]
values["atomistic_minutes"] = r(atom["seconds"] / 60, 0)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "summary.json").write_text(json.dumps({"generated": atom["generated"], "values": values,
                                                            "tables": tables}, indent=1, default=float))
print(json.dumps(values, indent=1, default=float))
