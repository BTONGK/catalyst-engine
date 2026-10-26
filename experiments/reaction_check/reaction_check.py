"""Does each reaction's ranking match published measurements? The check behind on_design_page.

    uv run python experiments/reaction_check/reaction_check.py

Reads the 16-metal UMA run (runs/metals-thermal), the published tables (data/published.yaml) and the rule fixed
before the results (PASS_RULE.md). For each thermal table: every metal's descriptor from the app's own binding
energies, its predicted speed against the pack's catalyst to beat at the table's temperature with the pack's
standard feed (the volcano's top moved there as on the Design page), and Spearman's rho against the published
order with a one-sided exact permutation p-value.

For the electrochemical reactions (runs/metals-electro, runs/metals-electro2 and runs/pblock-electro beside the
thermal run, which holds H, O, OH, N, CO and NO): each new intermediate's binding energy against a published DFT
table on the same metals,
after one constant offset per intermediate (the mean gap), then each reaction's limiting potentials against the
paper's own free energies put through the app's limiting-potential rule. The papers' gas references move to each
reaction's own zero with tabulated formation free energies, as the reaction files' shifts do. results/corrections.json
(corrections.py, in the UMA Python) holds the check on UMA's own zero-point and entropy corrections; the published
corrections in data/published.yaml stand in for UMA's wherever a source tabulates them.

Writes results/summary.json, which both Methodology tabs render.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from catalyst_engine.design import electro, screening
from catalyst_engine.packs import PACKS_DIR, list_packs, load_pack

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "metals-thermal"
ELECTRO_RUNS = [HERE / "runs" / r for r in ("metals-thermal", "metals-electro", "metals-electro2", "pblock-electro")]
OUT = HERE / "results" / "summary.json"
CORRECTIONS = HERE / "results" / "corrections.json"
R_KJ = 8.314462618e-3  # kJ/(mol K)
MIN_METALS, MIN_RHO, MAX_P, TOP = 5, 0.6, 0.05, 3
E_MIN_METALS, E_MAX_MEDIAN = 6, 0.20  # the electro rule: metals per comparison, typical miss after the offset (eV)
F_KJ = 96.485332  # kJ/mol per eV
# Formation free energies at 298.15 K (kJ/mol; NBS tables, Wagman et al. 1982), to move Elnabawy 2018's reference,
# HCOOH(g), to each reaction's own zero
DGF = {"H2O(l)": -237.129, "H2O(g)": -228.572, "CO(g)": -137.168, "HCOOH(g)": -351.0, "HCOOH(aq)": -372.3}
TO_CO_WATER = {  # COOH* + 1/2 H2 against CO(g) + liquid water; CO* against CO(g), from the paper's CO* + H2O(g) state
    "COOH": (DGF["HCOOH(g)"] - DGF["CO(g)"] - DGF["H2O(l)"]) / F_KJ,
    "CO": (DGF["HCOOH(g)"] - DGF["H2O(g)"] - DGF["CO(g)"]) / F_KJ}
TO_FORMIC_AQ = (DGF["HCOOH(g)"] - DGF["HCOOH(aq)"]) / F_KJ  # OCHO* or COOH* + 1/2 H2 against dissolved formic acid
ELNABAWY_TO = {
    "co2_to_co": TO_CO_WATER,
    "co_oxidation_electro": TO_CO_WATER,  # CO(g) and liquid water are this reaction's start
    "co2_to_formate": {"OCHO": TO_FORMIC_AQ},
    "formic_acid_oxidation": {"OCHO": TO_FORMIC_AQ},
    "formic_acid_oxidation_carboxyl": {"COOH": TO_FORMIC_AQ},
}
# Reactions whose potentials turn the run's metals to oxides: like oxygen evolution, they stay in the playground until
# an oxide check passes (PASS_RULE.md, batch 2), whatever their ranking gives
OXIDE_REACTIONS = {"water_oxidation_h2o2": 1.76}  # its equilibrium potential (V vs RHE)
# Intermediates the app already uses, accepted by the electrocatalysis checks (experiments/electro_check)
CHECKED_BEFORE = {"H", "O", "OH", "OOH"}


def ranks(values: list[float], higher_is_better: bool) -> np.ndarray:
    """Rank 1 = best, ties sharing their average rank."""
    v = np.array(values, dtype=float) * (-1 if higher_is_better else 1)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman_exact(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman's rho (Pearson on average ranks) and the one-sided exact p-value: the share of all n! orderings of b
    that agree with a at least as well. A dynamic program counts them over which ranks are already placed (2^n
    subsets) and the running sum of rank products, which twice the average ranks keep whole, so twelve metals take
    a second rather than 479 million orderings."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ac, bc = a - a.mean(), b - b.mean()
    rho = float(ac @ bc / math.sqrt((ac @ ac) * (bc @ bc)))
    ia, ib = np.rint(2 * a).astype(np.int64), np.rint(2 * b).astype(np.int64)
    n, observed = len(a), int(ia @ ib)
    top = int(np.sort(ia) @ np.sort(ib))
    counts = np.zeros((1 << n, top + 1), dtype=np.int64)  # [ranks placed, sum so far] -> orderings
    counts[0, 0] = 1
    for mask in range(1 << n):
        row = counts[mask]
        i = bin(mask).count("1")
        if i == n or not row.any():
            continue
        for j in range(n):
            if not mask & (1 << j):
                w = int(ia[i] * ib[j])
                counts[mask | (1 << j), w:] += row[:top + 1 - w]
    full = counts[-1]
    return rho, float(full[observed:].sum() / full.sum())


def published_scores(table: dict) -> tuple[dict[str, float], bool]:
    """The table's activity per metal and whether higher is better; Arrhenius tables become ln(rate) at its T."""
    if "arrhenius" in table:
        t = table["temperature_K"]
        return {m: lnA - e / (R_KJ * t) for m, (e, lnA) in table["arrhenius"].items()}, True
    return dict(table["values"]), bool(table["higher_is_better"])


def thermal_check(table: dict, summary: dict) -> dict:
    pack = load_pack(table["pack"])
    spec = pack.spec.screening
    anchor = spec.reference.formula
    t_k = table["temperature_K"] or pack.network.t_ref
    step = screening.design_step(pack)
    move = float(screening.peak_moves(pack, step, [t_k])[0])
    desc = {}
    for mat, s in summary.items():
        metal = mat.split(":", 1)[1]
        if all(s["binding"].get(a, {}).get("E") is not None for a in spec.descriptor):
            desc[metal] = float(sum(w * s["binding"][a]["E"] for a, w in spec.descriptor.items()))
    ref = desc[anchor]
    k_t = 8.617333e-5 * t_k
    shift = {m: spec.barrier_shift(x, ref, screening.PEAK_FOLLOW * move) for m, x in desc.items()}
    values, higher = published_scores(table)
    left = set(table.get("left_out", {})) | set(table.get("outside_run", {}))
    scored = [m for m in values if m in desc and m != anchor and m not in left]
    pub_rank = ranks([values[m] for m in scored], higher)
    pred_rank = ranks([-shift[m] for m in scored], True)
    rho, p = spearman_exact(pred_rank, pub_rank) if len(scored) >= 3 else (float("nan"), float("nan"))
    best = scored[int(np.argmin(pub_rank))]
    top = sorted(scored, key=lambda m: shift[m])[:TOP]
    passes = (table["role"] == "primary" and len(scored) >= MIN_METALS and rho >= MIN_RHO and p <= MAX_P
              and best in top)
    # how many tenfold steps separate the fastest and slowest scored metal, measured and predicted; for tables of
    # rates only (an order or a temperature carries no size, and a rate read as zero bounds the span from one side)
    sized = not (table.get("order_only") or table["unit"] == "K") and \
        ("arrhenius" in table or min(values[m] for m in scored) > 0)
    ln_rates = [values[m] if "arrhenius" in table else math.log(values[m]) for m in scored] if sized else None
    spans = {"measured_decades": None if ln_rates is None else round((max(ln_rates) - min(ln_rates)) / math.log(10), 1),
             "predicted_decades": round((max(shift[m] for m in scored) - min(shift[m] for m in scored))
                                        / (k_t * math.log(10)), 1)}
    rows = [{"metal": m, "descriptor_eV": round(desc[m], 3), "shift_eV": round(shift[m], 3),
             "ratio": float(np.exp(-shift[m] / k_t)), "predicted_rank": float(pr), "published": values[m],
             "published_rank": float(qr)} for m, pr, qr in zip(scored, pred_rank, pub_rank)]
    rows.sort(key=lambda r: r["published_rank"])
    return {"pack": table["pack"], "name": pack.spec.name, "role": table["role"], "source": table["source"],
            "short": table.get("short", ""), "via": table.get("via", ""), "quantity": table["quantity"],
            "unit": table["unit"],
            "temperature_K": t_k, "temperature_from": "table" if table["temperature_K"] else "pack reference",
            "anchor": anchor, "anchor_descriptor_eV": round(ref, 3), "peak_move_eV": round(move, 3),
            "descriptor": spec.descriptor, "n": len(scored), "rho": round(rho, 3), "p": sig(p),
            "published_best": best, "predicted_top": top, "best_in_top": best in top, "passes": passes,
            "left_out": sorted(left), **spans, "rows": rows}


def electro_binding() -> dict[str, dict[str, float]]:
    """Each metal's binding energy per adsorbate across the three runs (their clean slabs agree to 1e-5 eV)."""
    rows = [r for run in ELECTRO_RUNS for r in screening.read_results(run)]
    return {mid.split(":", 1)[1]: {a: b["E"] for a, b in s["binding"].items() if b["E"] is not None}
            for mid, s in screening.binding_summary(rows).items()}


def electro_source():
    """The published numbers electro_check.py already scores (one copy of each table)."""
    path = HERE.parent / "electro_check" / "electro_check.py"
    spec = importlib.util.spec_from_file_location("electro_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def intermediate_check(table: dict, column: str, name: str, binding: dict) -> dict:
    """One intermediate against one published table: UMA's binding energy plus one constant (the mean gap) against
    the paper's free energy, metal by metal."""
    j = table["columns"].index(column)
    metals = [m for m, row in table["values"].items() if name in binding.get(m, {}) and row[j] is not None]
    x = np.array([binding[m][name] for m in metals])
    y = np.array([table["values"][m][j] for m in metals])
    offset = float(np.mean(y - x))
    miss = x + offset - y
    rho, p = spearman_exact(ranks(list(x), False), ranks(list(y), False)) if len(metals) >= 3 else (float("nan"),) * 2
    median = float(np.median(np.abs(miss)))
    return {"table": table["table"], "short": table.get("short", table["table"]), "column": column,
            "intermediate": name, "n": len(metals),
            "offset_eV": round(offset, 3), "median_miss_eV": round(median, 3),
            "max_miss_eV": round(float(np.max(np.abs(miss))), 3), "rho": round(rho, 3), "p": sig(p),
            "passes": bool(len(metals) >= E_MIN_METALS and median <= E_MAX_MEDIAN and rho >= MIN_RHO and p <= MAX_P),
            "rows": [{"metal": m, "uma_eV": round(float(a), 3), "published_eV": float(b), "miss_eV": round(float(d), 3)}
                     for m, a, b, d in zip(metals, x, y, miss)]}


def published_free_energies(rid: str, spec: electro.ElectroSpec, tables: dict, ec) -> tuple[dict[str, dict[str, float]], str]:
    """Each metal's free energies on the reaction's own scale from the paper behind its check, and that paper; ({}, "")
    when the reaction has none."""
    names, shift = spec.adsorbates, {i.name: i.shift_eV for i in spec.intermediates}
    herron, elnabawy = tables["herron2015"], tables["elnabawy2018"]
    if rid in ELNABAWY_TO:  # against HCOOH(g) in the paper
        cols = {v: k for k, v in elnabawy["app"].items()}
        return {m: {a: row[elnabawy["columns"].index(cols[a])] + ELNABAWY_TO[rid][a] for a in names}
                for m, row in elnabawy["values"].items()}, elnabawy["short"]
    if set(names) <= set(herron["app"].values()):  # against N2 and H2 in the paper, as in the worker
        cols = {v: k for k, v in herron["app"].items()}
        return {m: {a: row[herron["columns"].index(cols[a])] + shift[a] for a in names}
                for m, row in herron["values"].items()}, herron["short"]
    if names == ["H"]:  # arXiv:2109.04219, Table S2, RPBE
        return {m: {"H": v[0] + shift["H"]} for m, v in ec.PUBLISHED_H.items()}, "arXiv:2109.04219, Table S2 (RPBE)"
    if names == ["OOH"]:  # Nørskov 2004: Table 2's O* through the Fig. 8 relation, with the caption's +0.18 eV
        return {m: {"OOH": ec.OOH_FROM_O[0] * row[1] + ec.OOH_FROM_O[1] + 0.18 + shift["OOH"]}
                for m, row in ec.TABLE_2.items()}, "Nørskov et al. 2004, Table 2 (O* through Fig. 8)"
    if names == ["OH"]:  # Nørskov 2004: Table 2's OH* with its footnote's free energy
        return {m: {"OH": row[0] + ec.FOOTNOTE["OH"] + shift["OH"]} for m, row in ec.TABLE_2.items()}, \
            "Nørskov et al. 2004, Table 2"
    return {}, ""


def reaction_check(rid: str, binding: dict, tables: dict, ec, passed: dict, uma_only: set[str]) -> dict:
    """A reaction's limiting potentials from UMA against those from the paper's own free energies, by the app's rule."""
    spec = electro.load_electro(rid)
    pub, source = published_free_energies(rid, spec, tables, ec)
    metals = [m for m in pub if all(a in binding.get(m, {}) for a in spec.adsorbates)]
    rows = []
    for m in metals:
        g = spec.free_energies({a: binding[m][a] for a in spec.adsorbates})
        u, k = spec.limiting_potential(g)
        u_pub, k_pub = spec.limiting_potential(pub[m])
        rows.append({"metal": m, "uma_V": round(u, 3), "published_V": round(u_pub, 3),
                     "uma_step": spec.step_names()[k], "published_step": spec.step_names()[k_pub]})
    if len(rows) >= 3:
        rho, p = spearman_exact(ranks([r["uma_V"] for r in rows], False), ranks([r["published_V"] for r in rows], False))
    else:
        rho, p = float("nan"), float("nan")
    new = [a for a in spec.adsorbates if a not in CHECKED_BEFORE]
    gaps = [a for a in new if not passed.get(a)]  # missed a check, or still waiting for a published table
    untabled = [a for a in new if a not in passed]
    needs = sorted(set(spec.adsorbates) & uma_only)
    ranking = bool(len(rows) >= E_MIN_METALS and rho >= MIN_RHO and p <= MAX_P)
    oxide = rid in OXIDE_REACTIONS
    return {"reaction": rid, "name": spec.name, "direction": spec.direction, "source": source, "n": len(rows),
            "rho": None if math.isnan(rho) else round(rho, 3), "p": None if math.isnan(p) else sig(p),
            "ranking_passes": ranking, "intermediates_new": new, "intermediates_failing": gaps,
            "intermediates_awaiting_table": untabled, "corrections_missing": needs, "oxide": oxide,
            "passes": ranking and not gaps and not needs and not oxide, "rows": rows}


def published_corrections(sources: list[dict]) -> dict[str, dict]:
    """Each published correction against the worker's gas reference for that intermediate: the adsorbate's ZPE - TS
    less its reference gases', from one source's own table (PASS_RULE.md, batch 2)."""
    from uma_worker import REFERENCE  # noqa: PLC0415  numpy and ASE only

    out = {}
    for s in sources:
        gas = {g: sum(v) for g, v in s["gases"].items()}
        for a, terms in s["adsorbates"].items():
            out[a] = {"eV": round(sum(terms) - sum(w * gas[g] for g, w in REFERENCE[a].items()), 3),
                      "short": s["short"], "surface": s["surface"], "temperature_K": s["temperature_K"]}
    return out


def joined(names: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    return " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]


def sig(x: float) -> float:
    """Three significant figures, so a p-value of 2.5e-7 keeps its size."""
    return float(f"{x:.3g}")


def p_text(p: float | None) -> str:
    """A p-value to read: 0.035, or 2.5 × 10⁻⁷ below a thousandth."""
    if p is None:
        return ""
    if p >= 1e-3:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant} × 10{str(int(exp)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def below_rule(n: int, rho: float | None, p: float | None, n_min: int) -> list[str]:
    """The parts of a ranking's score that sit outside the rule, in words."""
    out = [f"{n} metals, below {n_min}"] if n < n_min else []
    if rho is not None and rho < MIN_RHO:
        out.append(f"rho {rho:.2f}, below {MIN_RHO}")
    if p is not None and p > MAX_P:
        out.append(f"p {p:.3f}, above {MAX_P}")
    return out


def display(checks: list[dict], inter: list[dict], reactions: list[dict], corrections: dict,
            uma_only: set[str], published_c: dict[str, dict]) -> tuple[dict, dict]:
    """Single numbers and rows for both Methodology tabs, whose report.yaml files bring this summary in (`also`)
    and quote {name}."""
    values: dict = {"rc_min_metals": MIN_METALS, "rc_min_rho": MIN_RHO, "rc_max_p": MAX_P, "rc_top": TOP,
                    "rc_e_min_metals": E_MIN_METALS, "rc_e_max_median": E_MAX_MEDIAN}
    tables: dict = {}

    # thermal: one row per published table, then every metal behind them
    role = {"primary": "scored by the rule", "secondary": "second table, shown beside",
            "indicative": "indicative, shown beside"}
    rows = []
    for c in checks:
        if c["passes"]:
            verdict = "passes: on the Design page"
        elif c["role"] == "primary":
            reasons = below_rule(c["n"], c["rho"], c["p"], MIN_METALS)
            if not c["best_in_top"]:
                reasons.append(f"{c['published_best']} outside the predicted top {TOP}")
            verdict = "in the playground: " + "; ".join(reasons)
        else:
            verdict = "shown beside the scored table"
        rows.append({"reaction": c["name"], "table": c["short"], "role": role[c["role"]],
                     "T_K": f"{c['temperature_K']:.0f}" + (" (the pack's)" if c["temperature_from"] != "table" else ""),
                     "n": c["n"], "rho": c["rho"], "p": p_text(c["p"]),
                     "best": c["published_best"], "top": ", ".join(c["predicted_top"]), "verdict": verdict})
        key = f"rc_{c['pack']}_{c['role']}"
        values.update({f"{key}_n": c["n"], f"{key}_rho": c["rho"], f"{key}_p": c["p"],
                       f"{key}_best": c["published_best"], f"{key}_top": ", ".join(c["predicted_top"]),
                       f"{key}_T": round(c["temperature_K"], 1), f"{key}_anchor": c["anchor"]})
    tables["rc_thermal"] = rows
    tables["rc_thermal_metals"] = [
        {"reaction": c["name"], "table": c["short"], "metal": m["metal"], "descriptor_eV": m["descriptor_eV"],
         "shift_eV": m["shift_eV"], "predicted_rank": m["predicted_rank"], "published": m["published"],
         "published_rank": m["published_rank"]} for c in checks for m in c["rows"]]
    sized = [c for c in checks if c["measured_decades"] is not None]
    tables["rc_spans"] = [{"reaction": c["name"], "table": c["short"], "n": c["n"],
                           "measured": c["measured_decades"], "predicted": c["predicted_decades"],
                           "factor": round(c["predicted_decades"] / c["measured_decades"], 1)} for c in sized]
    values.update(rc_span_measured_lo=min(c["measured_decades"] for c in sized),
                  rc_span_measured_hi=max(c["measured_decades"] for c in sized),
                  rc_span_predicted_lo=min(c["predicted_decades"] for c in sized),
                  rc_span_predicted_hi=max(c["predicted_decades"] for c in sized),
                  rc_span_factor_lo=min(x["factor"] for x in tables["rc_spans"]),
                  rc_span_factor_hi=max(x["factor"] for x in tables["rc_spans"]))
    # where every shipped reaction stands: on the Design page with a passing check, on it from before the check
    # began and still awaiting a measured table, or in the playground
    checked = {c["pack"]: c["passes"] for c in checks if c["role"] == "primary"}
    tabled = {c["pack"] for c in checks}
    on_page, waiting_table, playground, untabled = [], [], [], []
    for pid in list_packs(PACKS_DIR):
        spec_ = load_pack(pid).spec
        if spec_.screening is None:
            continue
        if spec_.screening.on_design_page:
            (on_page if checked.get(pid) else waiting_table).append(spec_.name)
        else:
            playground.append(spec_.name)
            if pid not in tabled:
                untabled.append(spec_.name)
    values.update(rc_design_checked_names=joined(on_page), rc_design_checked_n=len(on_page),
                  rc_playground_untabled_names=joined(untabled), rc_playground_untabled_n=len(untabled),
                  rc_design_waiting_names=joined(waiting_table), rc_design_waiting_n=len(waiting_table),
                  rc_design_n=len(on_page) + len(waiting_table), rc_playground_n=len(playground),
                  rc_packs_n=len(on_page) + len(waiting_table) + len(playground))
    scored = [c for c in checks if c["role"] == "primary"]
    passed = [c for c in scored if c["passes"]]
    values.update(rc_thermal_tables=len(checks), rc_thermal_scored=len(scored), rc_thermal_passed=len(passed),
                  rc_thermal_passed_names=joined([c["name"] for c in passed]),
                  rc_thermal_failed_names=joined([c["name"] for c in scored if not c["passes"]]),
                  rc_thermal_failed_rho_lo=min(c["rho"] for c in scored if not c["passes"]),
                  rc_thermal_failed_rho_hi=max(c["rho"] for c in scored if not c["passes"]))

    # electro: each new intermediate against its published table
    rows = []
    for c in inter:
        reasons = below_rule(c["n"], c["rho"], c["p"], E_MIN_METALS)
        if c["median_miss_eV"] > E_MAX_MEDIAN:
            reasons.insert(0, f"typical miss {c['median_miss_eV']:.3f} eV, above {E_MAX_MEDIAN:.2f}")
        rows.append({"intermediate": c["intermediate"] + "*", "table": c["short"], "n": c["n"],
                     "offset_eV": c["offset_eV"], "median_miss_eV": c["median_miss_eV"],
                     "max_miss_eV": c["max_miss_eV"], "rho": c["rho"], "p": p_text(c["p"]),
                     "verdict": "passes" if c["passes"] else "misses: " + "; ".join(reasons)})
        key = f"rc_i_{c['table']}_{c['intermediate']}"
        values.update({f"{key}_n": c["n"], f"{key}_median": c["median_miss_eV"], f"{key}_max": c["max_miss_eV"],
                       f"{key}_offset": c["offset_eV"], f"{key}_rho": c["rho"], f"{key}_p": c["p"]})
    tables["rc_electro_intermediates"] = rows
    values.update(rc_i_checked=len(inter),
                  rc_i_passed_names=joined([c["intermediate"] + "*" for c in inter if c["passes"]]),
                  rc_i_missed_names=joined([c["intermediate"] + "*" for c in inter if not c["passes"]]),
                  rc_i_passed=sum(c["passes"] for c in inter))
    awaiting = list(dict.fromkeys(a + "*" for r in reactions for a in r["intermediates_awaiting_table"]))
    values.update(rc_i_awaiting=len(awaiting), rc_i_awaiting_names=joined(awaiting))

    # electro: each reaction's limiting potentials against the paper's own free energies
    rows = []
    for r in reactions:
        if r["passes"]:
            verdict = "passes: on the Design page"
        else:
            waiting = [a + "*" for a in r["intermediates_awaiting_table"]]
            if r["n"] == 0:
                reasons = [f"{joined(waiting)} {'awaits' if len(waiting) == 1 else 'await'} a published table, and so "
                           "does the ranking" if waiting else "the ranking awaits a published table"]
            elif waiting:
                reasons = [f"{joined(waiting)} {'awaits' if len(waiting) == 1 else 'await'} a published table"]
            else:
                reasons = []
            missed = [a + "*" for a in r["intermediates_failing"] if a not in r["intermediates_awaiting_table"]]
            if missed:
                reasons.append(f"{joined(missed)} {'misses its' if len(missed) == 1 else 'miss their'} check")
            if r["corrections_missing"]:
                reasons.append(f"the corrections for {joined([a + '*' for a in r['corrections_missing']])} rest on UMA "
                               "alone")
            if r["n"]:
                reasons += below_rule(r["n"], r["rho"], r["p"], E_MIN_METALS)
            if r["oxide"]:
                reasons.append(f"it runs above {OXIDE_REACTIONS[r['reaction']]:.2f} V, where the metals turn to oxides, "
                               "so it waits for an oxide check, as oxygen evolution does")
            verdict = "in the playground: " + "; ".join(reasons)
        rows.append({"reaction": r["name"], "table": r["source"] or "(a published table to come)", "n": r["n"],
                     "rho": r["rho"], "p": p_text(r["p"]), "verdict": verdict})
        key = f"rc_e_{r['reaction']}"
        values.update({f"{key}_n": r["n"], f"{key}_rho": r["rho"], f"{key}_p": r["p"]})
    tables["rc_electro_reactions"] = rows
    tables["rc_electro_metals"] = [{"reaction": r["name"], **m} for r in reactions for m in r["rows"]]
    passed = [r for r in reactions if r["passes"]]
    values.update(rc_e_checked=len(reactions), rc_e_passed=len(passed),
                  rc_e_passed_names=joined([r["name"] for r in passed]),
                  rc_e_waiting=len(reactions) - len(passed))

    # the zero-point and entropy corrections from UMA's own vibrations, on Pt(111)
    check, pt = corrections.get("check", {}), corrections.get("metals", {}).get("Pt", {})
    spread = corrections.get("spread", {})
    rows, zpe_gaps = [], []
    for a, row in pt.items():
        pub = check.get("published", {}).get(a)
        sk = check.get("skulason", {}).get(a, {}).get("published")
        tab = published_c.get(a, {}).get("eV")
        ref = next((v for v in (pub, sk, tab) if v is not None), None)
        if pub is not None:
            zpe_gaps.append(abs(row["correction"] + row["TS"] - pub))  # the adsorbate's entropy set aside
        if pub is not None:
            role = "checked against (the rule)"
        elif sk is not None:
            role = "shown beside (Skúlason 2012)"
        elif tab is not None:
            role = f"published ({published_c[a]['short']}, {published_c[a]['surface']}); the reactions take it"
        else:
            role = "UMA's value alone"
        rows.append({"intermediate": a + "*", "site": row["site"], "uma_eV": row["correction"],
                     "published_eV": ref, "gap_eV": None if ref is None else round(row["correction"] - ref, 3),
                     "role": role, "lowest_cm": round(min(row["frequencies_cm"]), 0), "spread_eV": spread.get(a)})
    tables["rc_corrections"] = rows
    if check:
        misses = check["misses"]
        worst = max(misses, key=lambda a: abs(misses[a]))
        sk_gaps = {a: v["uma"] - v["published"] for a, v in check.get("skulason", {}).items()}
        values.update(rc_corr_passes="passes" if check["passes"] else "misses",
                      rc_corr_max_miss=round(check["max_miss"], 2), rc_corr_median_miss=round(check["median_miss"], 2),
                      rc_corr_worst=worst + "*",
                      rc_corr_misses=joined([f"{a}* {v:+.2f}" for a, v in misses.items()]) + " eV",
                      rc_corr_entropy_aside_max=round(max(zpe_gaps), 2),
                      rc_corr_spread_max=round(max(spread.values()), 2),
                      rc_corr_skulason_gaps=joined([f"{a}* {v:+.2f}" for a, v in sk_gaps.items()]) + " eV",
                      rc_corr_skulason_max=round(max(abs(v) for v in sk_gaps.values()), 2),
                      rc_corr_uma_only=joined([a + "*" for a in sorted(uma_only, key=list(pt).index)]),
                      rc_corr_uma_only_n=len(uma_only))
        for a in ("OH", "OOH"):  # the soft motions behind the two largest misses
            values[f"rc_corr_ts_{a}"] = round(pt[a]["TS"], 2)
            values[f"rc_corr_soft_{a}"] = sum(f < 110 for f in pt[a]["frequencies_cm"])
        for a, c in published_c.items():  # the published corrections the reactions take in place of UMA's
            values[f"rc_corr_pub_{a}"] = c["eV"]
            values[f"rc_corr_pub_gap_{a}"] = round(pt[a]["correction"] - c["eV"], 2) if a in pt else None
        values["rc_corr_pub_names"] = joined([f"{a}* {c['eV']:.2f} eV" for a, c in published_c.items()])
    return values, tables


def electro_walk(binding: dict) -> dict:
    """The electro Methodology tab's loading walk-through: hydrogen peroxide production on platinum and copper, from
    this check's runs, through the app's own functions."""
    import uma_worker  # noqa: PLC0415  numpy and ASE only; the worker's gas reference for each adsorbate

    rid, ref, other = "h2o2_production", "Pt", "Cu"
    spec = electro.load_electro(rid)
    inter = spec.intermediates[0]
    out = {"ew_id": rid, "ew_name": spec.name, "ew_equation": spec.equation, "ew_electrons": spec.electrons,
           "ew_U_eq": spec.equilibrium_V, "ew_total": round(spec.total_eV, 3), "ew_ads": inter.name,
           "ew_correction": inter.correction_eV, "ew_shift": inter.shift_eV, "ew_compare": spec.compare_V,
           "ew_ideal": round(spec.ideal()[inter.name], 3), "ew_kT": round(electro.K_T, 4),
           "ew_reference": " ".join(f"{'+' if n > 0 else '−'} {abs(n):g} {g}" for g, n in
                                    uma_worker.REFERENCE[inter.name].items()).lstrip("+ "),
           "ew_ref": ref, "ew_other": other}
    hurdle = {}
    for m in (ref, other):
        g = spec.free_energies({inter.name: binding[m][inter.name]})
        steps, (u, k) = spec.steps(g), spec.limiting_potential(g)
        lo, hi = electro.limiting_range(spec, g)
        hurdle[m] = spec.hurdle(g, spec.compare_V)
        key = f"ew_{m.lower()}"
        out.update({f"{key}_E": round(binding[m][inter.name], 3), f"{key}_G": round(g[inter.name], 3),
                    f"{key}_step1": round(steps[0], 3), f"{key}_step2": round(steps[1], 3), f"{key}_UL": round(u, 3),
                    f"{key}_UL_step": k + 1, f"{key}_eta": round(spec.overpotential(g), 3),
                    f"{key}_lo": round(lo, 3), f"{key}_hi": round(hi, 3),
                    f"{key}_hurdle": round(hurdle[m], 3),
                    f"{key}_step2_at": round(spec.steps_at(g, spec.compare_V)[1], 3)})
    ratio = float(np.exp((hurdle[ref] - hurdle[other]) / electro.K_T))
    out["ew_other_one_in"] = f"{float(f'{1 / ratio:.2g}'):,.0f}"
    return out


def pairing_line(herron: dict) -> dict:
    """Herron 2015's N* + N* -> N2 barriers against N*'s free energy (its Table 1): the straight line a reaction file
    would carry as the pairing step's barrier, and the scatter about it."""
    pairs = herron["n_pairing"]
    j = herron["columns"].index("N")
    x = np.array([herron["values"][m][j] for m in pairs])
    y = np.array([pairs[m][1] for m in pairs])
    slope, intercept = np.polyfit(x, y, 1)
    rms = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
    return {"n": len(pairs), "metals": list(pairs), "slope": round(float(slope), 2),
            "intercept": round(float(intercept), 2), "rms_eV": round(rms, 2),
            "range_eV": [float(y.min()), float(y.max())]}


def main() -> None:
    sys.path.insert(0, str(HERE.parents[1] / "src" / "catalyst_engine" / "design"))  # the worker's gas references
    published = yaml.safe_load((HERE / "data" / "published.yaml").read_text())
    summary = screening.binding_summary(screening.read_results(RUN))
    status = screening.read_status(RUN)
    checks = [thermal_check(t, summary) for t in published["thermal"]]

    binding, ec = electro_binding(), electro_source()
    tables = {t["table"]: t for t in published["electro"]}
    inter = [intermediate_check(t, col, name, binding) for t in published["electro"] for col, name in t["app"].items()]
    passed = {}
    for c in inter:  # an intermediate scored by two tables passes when every table it meets passes
        passed[c["intermediate"]] = passed.get(c["intermediate"], True) and c["passes"]
    corrections = json.loads(CORRECTIONS.read_text()) if CORRECTIONS.exists() else {}
    published_c = published_corrections(published.get("corrections", []))
    published_corr = set(corrections.get("check", {}).get("published", {})) | \
        set(corrections.get("check", {}).get("skulason", {})) | set(published_c)
    uma_only = set() if corrections.get("check", {}).get("passes") else \
        {a for a in corrections.get("metals", {}).get("Pt", {}) if a not in published_corr}
    reactions = [reaction_check(rid, binding, tables, ec, passed, uma_only) for rid in electro.list_electro()
                 if rid not in ("hydrogen_evolution", "oxygen_reduction", "oxygen_evolution")]

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.update({
        "rule": {"min_metals": MIN_METALS, "min_rho": MIN_RHO, "max_p": MAX_P, "top": TOP,
                 "electro_min_metals": E_MIN_METALS, "electro_max_median_eV": E_MAX_MEDIAN},
        "thermal_run": {"folder": str(RUN.relative_to(HERE)), "state": status.get("state"),
                        "metals": sorted(m.split(":", 1)[1] for m in summary), "model": "UMA-S 1.1, oc20 mode"},
        "thermal": checks,
        "electro_runs": [str(r.relative_to(HERE)) for r in ELECTRO_RUNS],
        "electro_intermediates": inter,
        "electro_reactions": reactions,
        "corrections": {k: corrections.get(k) for k in ("rule", "check", "spread")} if corrections else {},
        "corrections_uma_only": sorted(uma_only),
        "corrections_published": published_c,
        "n_pairing": pairing_line(tables["herron2015"]),
    })
    out["values"], out["tables"] = display(checks, inter, reactions, corrections, uma_only, published_c)
    out["values"].update(electro_walk(binding))
    nn = out["n_pairing"]
    out["values"].update(rc_nn_n=nn["n"], rc_nn_metals=joined(nn["metals"]), rc_nn_slope=nn["slope"],
                         rc_nn_intercept=nn["intercept"], rc_nn_rms=nn["rms_eV"], rc_nn_lo=nn["range_eV"][0],
                         rc_nn_hi=nn["range_eV"][1])
    out["values"]["rc_run_metals"] = len(out["thermal_run"]["metals"])
    out["values"]["rc_run_metal_list"] = joined(out["thermal_run"]["metals"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for c in checks:
        verdict = "PASS" if c["passes"] else ("shown" if c["role"] != "primary" else "stays in the playground")
        print(f"{c['pack']:24} {c['role']:10} n={c['n']:2} rho={c['rho']:+.2f} p={c['p']:.4f} best={c['published_best']:3} "
              f"top={','.join(c['predicted_top'])} -> {verdict}")
    for c in inter:
        print(f"{c['table']:13} {c['intermediate']:6} n={c['n']:2} offset={c['offset_eV']:+.2f} median={c['median_miss_eV']:.2f} "
              f"max={c['max_miss_eV']:.2f} rho={c['rho']:+.2f} p={c['p']:.4f} -> {'PASS' if c['passes'] else 'miss'}")
    for r in reactions:
        rho = "  -  " if r["rho"] is None else f"{r['rho']:+.2f}"
        print(f"{r['reaction']:24} n={r['n']:2} rho={rho} p={r['p']} ranking={'pass' if r['ranking_passes'] else 'miss'} "
              f"failing={r['intermediates_failing']} uma_only={r['corrections_missing']} -> {'PASS' if r['passes'] else 'playground'}")


if __name__ == "__main__":
    sys.exit(main())
