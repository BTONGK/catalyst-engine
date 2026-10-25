"""How close does the app's electrochemistry come to published DFT and measurement?

    uv run python experiments/electro_check/electro_check.py [screen folder]

Reads a finished electrochemical screen of pure metals that holds H, O, OH and OOH (the shipped example,
examples/electro/screens/*oxygen-reduction*, unless a folder is given), scores it against published sources and
writes results/summary.json, the numbers behind report.yaml on the electrocatalysis Manual's Methodology tab:

  1. hydrogen: dG_H = dE_H + 0.24 eV against the RPBE and PBE values of Table S2 in "Revisiting trends in the
     exchange current for hydrogen evolution" (arXiv:2109.04219). Its 3 x 3 close-packed slabs hold H at 1/9
     monolayer on the metals that bind it weakly (dG_H > 0) and at a full monolayer, averaged per atom, on the rest
     (its SI). The app places one H on a 3 x 3 patch, so the weak binders compare like for like and the rest carry
     the step from one H to a full layer. Then the order of the metals against measured exchange currents (its
     Table S3-1, the Nørskov 2005 data);
  2. the app's arithmetic on the paper's own numbers: the 13 metals of Table 2 in Nørskov et al., J. Phys. Chem. B
     2004, through its footnote's free energies (G_OH = dE_OH - 0.25 eV, G_O = dE_O + 0.01 eV), against the
     table's dG1 and dG2 at 1.23 V and its O2-splitting barrier E_a;
  3. hydroxyl and oxygen: dE_OH and dE_O against water and hydrogen gas, the same Table 2 (RPBE, 1/4 monolayer on
     the close-packed face);
  4. hydroperoxyl: the OOH-OH binding-energy gap against 3.13 eV on metals (Nørskov 2004, Fig. 8 caption:
     dE_OOH = 0.5 dE_O + 3.18 eV and dE_OH = 0.5 dE_O + 0.05 eV) and 3.20 eV on oxides (Man et al., ChemCatChem
     2011, Fig. 2: binding energies, fit MAE 0.17 eV);
  5. the whole pipeline: platinum's limiting potential for oxygen reduction against the paper's 0.78 V; the
     two-route ranking at the comparison potential on UMA's binding energies and on the paper's own (Table 2 with
     the Fig. 8 OOH* relation); the paper's account of experiment, which sets Cu, Ag and Au aside and finds Pt and
     Pd first, then Ir and Rh; the likely range on platinum against a grid over the error box; and platinum's
     hydrogen free energy;
  6. the error bar: the gate and UMA's misses on oxides come from experiments/design_method (the thermal error
     budget and its oxide check), so both Methodology tabs state the same numbers.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from catalyst_engine.design import electro, screening

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
THERMAL = REPO / "experiments" / "design_method" / "results" / "summary.json"

# arXiv:2109.04219, Table S2: dG_H (eV) with RPBE and PBE, no van der Waals term; its SI puts H at 1/9 monolayer where
# dG_H > 0 and at a full monolayer where dG_H < 0
PUBLISHED_H = {"Ag": (0.565, 0.416), "Au": (0.574, 0.418), "Co": (-0.202, -0.355), "Cu": (0.188, 0.040),
               "Ir": (-0.081, -0.208), "Ni": (-0.231, -0.388), "Pd": (-0.123, -0.286), "Pt": (-0.038, -0.192),
               "Rh": (-0.116, -0.270), "Ru": (-0.180, -0.330), "Mo": (-0.321, -0.453), "Re": (-0.154, -0.293)}
# its Table S3-1: measured log10 of the exchange current density (A/cm2), the Nørskov 2005 data, one or more per metal
MEASURED_LOG_J0 = {"Pt": [-3.1, -2.63, -3.34], "Ir": [-3.7, -3.46], "Pd": [-3.0, -3.0], "Rh": [-3.6, -3.22],
                   "Ni": [-5.21, -5.2], "Co": [-5.32], "W": [-5.9], "Cu": [-5.37], "Mo": [-7.07], "Re": [-2.87],
                   "Nb": [-6.8], "Au": [-6.6, -6.8, -5.4], "Ag": [-5.0, -7.85]}
# Nørskov et al. 2004, Table 2 (RPBE, quarter monolayer, close-packed face): dE_OH and dE_O in eV against H2O and H2,
# then what the paper builds from them at U0 = 1.23 V: dG0 (1/2 O2 -> O*), dG1 (O* -> HO*), dG2 (HO* -> H2O) and
# E_a, the O2-splitting barrier
TABLE_2 = {"Ag": (0.72, 2.12, -0.33, -0.43, 0.76, 0.93), "Au": (1.49, 2.75, 0.30, -0.29, -0.01, 2.06),
           "Co": (-0.08, -0.22, -2.67, 1.11, 1.56, -3.29), "Cu": (0.37, 1.20, -1.25, 0.14, 1.11, -0.73),
           "Fe": (-0.88, -0.90, -3.35, 0.99, 2.36, -4.51), "Ir": (0.63, 1.00, -1.45, 0.60, 0.85, -1.09),
           "Mo": (-0.61, -1.62, -4.07, 1.98, 2.09, -5.81), "Ni": (0.13, 0.34, -2.11, 0.76, 1.35, -2.28),
           "Pd": (0.92, 1.53, -0.92, 0.36, 0.56, -0.14), "Pt": (1.05, 1.57, -0.88, 0.45, 0.43, -0.06),
           "Rh": (0.34, 0.44, -2.01, 0.87, 1.14, -2.10), "Ru": (-0.01, -0.05, -2.50, 1.01, 1.49, -2.98),
           "W": (-0.80, -2.06, -4.51, 2.23, 2.28, -6.60)}
FOOTNOTE = {"OH": -0.25, "O": 0.01}  # Table 2's footnote: G = dE + this at 0 V, set to reproduce Table 1's platinum
OOH_FROM_O = (0.5, 3.18)  # Fig. 8 caption: dE_OOH = 0.5 dE_O + 3.18 eV, beside dE_OH = 0.5 dE_O + 0.05 eV
OOH_GAP_METALS = OOH_FROM_O[1] - 0.05  # the OOH-OH binding-energy gap those two relations give on metals
OOH_GAP_OXIDES, OOH_GAP_OXIDES_MAE = 3.20, 0.17  # Man et al. 2011, Fig. 2: dE_OOH = dE_OH + 3.20 eV on oxides
PT_ORR_LIMIT = 0.78  # Nørskov 2004: Pt(111), the highest potential where every step runs downhill (V vs RHE)
# Nørskov 2004 on experiment: "If we exclude Cu, Ag, and Au, then Pt and Pd stand out as the metals with the smallest
# overpotentials, followed by Ir and Rh."
SET_ASIDE, PAPER_ORDER = ("Cu", "Ag", "Au"), (("Pt", "Pd"), ("Ir", "Rh"))


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def names(metals: list[str]) -> str:
    return ", ".join(metals)


def follows_paper(order: list[str]) -> bool:
    """Whether an order, with Cu, Ag and Au set aside, starts with Pt and Pd and goes on with Ir and Rh."""
    kept = [m for m in order if m not in SET_ASIDE]
    head = [set(group) for group in PAPER_ORDER]
    return [set(kept[:2]), set(kept[2:4])] == head


def stronger(gap: float) -> str:
    """UMA minus published, in words: a lower binding energy binds more strongly."""
    return f"{abs(gap):.2f} eV more {'strongly' if gap < 0 else 'weakly'}"


def grid_gap(spec: electro.ElectroSpec, g: dict[str, float], n: int = 31) -> tuple[float, bool]:
    """How close a grid over the error box comes to the exact likely range, and whether it stays inside."""
    lo, hi = electro.limiting_range(spec, g)
    axes = [np.linspace(g[a] - electro.SIGMA_EV, g[a] + electro.SIGMA_EV, n) for a in spec.adsorbates]
    ul = [spec.limiting_potential(dict(zip(spec.adsorbates, v)))[0] for v in itertools.product(*axes)]
    return max(min(ul) - lo, hi - max(ul)), lo - 1e-9 <= min(ul) and max(ul) <= hi + 1e-9


def main(folder: Path) -> None:
    job = screening.read_job(folder)
    results = screening.read_results(folder)
    summary = screening.binding_summary(results)
    sym = {m["id"]: m["formula"] for m in job["request"]["materials"]}
    e = {sym[mid]: {a: b["E"] for a, b in s["binding"].items() if b["E"] is not None} for mid, s in summary.items()
         if mid in sym}
    corr_h = electro.load_electro("hydrogen_evolution").intermediates[0].correction_eV
    orr = electro.load_electro("oxygen_reduction")
    thermal = json.loads(THERMAL.read_text())

    # 1. hydrogen
    her = []
    for m in sorted(e):
        if "H" in e[m] and m in PUBLISHED_H:
            dg = e[m]["H"] + corr_h
            her.append({"metal": m, "uma": round(dg, 3), "rpbe": PUBLISHED_H[m][0], "pbe": PUBLISHED_H[m][1],
                        "coverage": "1/9 monolayer" if PUBLISHED_H[m][0] > 0 else "full monolayer",
                        "gap_rpbe": round(dg - PUBLISHED_H[m][0], 3), "gap_pbe": round(dg - PUBLISHED_H[m][1], 3),
                        "log_j0": None if m not in MEASURED_LOG_J0 else round(float(np.median(MEASURED_LOG_J0[m])), 2)})
    with_j0 = [r for r in her if r["log_j0"] is not None]
    low = [r for r in her if r["rpbe"] > 0]
    full = [r for r in her if r["rpbe"] < 0]

    # 2. the app's arithmetic on the paper's own Table 2
    table2 = []
    for m, (d_oh, d_o, _dg0, dg1, dg2, ea) in TABLE_2.items():
        st = orr.steps_at({"OOH": 0.0, "O": d_o + FOOTNOTE["O"], "OH": d_oh + FOOTNOTE["OH"]}, 1.23)
        table2.append({"metal": m, "d_oh": d_oh, "d_o": d_o, "dg1": dg1, "dg1_app": round(st[2], 3), "dg2": dg2,
                       "dg2_app": round(st[3], 3), "ea": ea, "ea_app": round(orr.activation.barrier({"O": d_o}), 3)})
    table2_gap = max(abs(r[k] - r[f"{k}_app"]) for r in table2 for k in ("dg1", "dg2", "ea"))
    pt2 = next(r for r in table2 if r["metal"] == "Pt")
    pt2_step = orr.step_names()[2 if pt2["dg1_app"] >= pt2["dg2_app"] else 3]

    # 3. hydroxyl and oxygen against Table 2
    orr_metals = []
    for m in sorted(e):
        if {"OH", "O"} <= set(e[m]) and m in TABLE_2:
            orr_metals.append({"metal": m, "uma_oh": round(e[m]["OH"], 3), "pub_oh": TABLE_2[m][0],
                               "uma_o": round(e[m]["O"], 3), "pub_o": TABLE_2[m][1],
                               "gap_oh": round(e[m]["OH"] - TABLE_2[m][0], 3),
                               "gap_o": round(e[m]["O"] - TABLE_2[m][1], 3)})

    # 4. hydroperoxyl
    ooh = [{"metal": m, "oh": round(e[m]["OH"], 3), "ooh": round(e[m]["OOH"], 3), "gap": round(e[m]["OOH"] - e[m]["OH"], 3)}
           for m in sorted(e) if {"OH", "OOH"} <= set(e[m])]
    ooh_mean = float(np.mean([r["gap"] for r in ooh]))

    # 5. the whole pipeline, on UMA's binding energies and on the paper's own
    pipeline = {}
    for sid in ("oxygen_reduction", "hydrogen_evolution"):
        spec = electro.load_electro(sid)
        req = {**job["request"], "pack": sid, "conditions": {"compare_V": spec.compare_V}}
        preds, _ = electro.predict(spec, {**job, "request": req}, results)
        pipeline[sid] = {"compare_V": spec.compare_V, "ranking": [p.formula for p in preds if p.rank is not None],
                         "materials": {p.formula: {
                             "rank": p.rank, "limiting_V": round(p.limiting_V, 3), "overpotential_V": round(p.overpotential_V, 3),
                             "range": [round(x, 3) + 0.0 for x in p.limiting_range], "limiting_step": spec.step_names()[p.limiting_step],
                             "limiting_route": p.limiting_route, "route": p.route, "hurdle_eV": round(p.hurdle_eV, 3),
                             "barrier_eV": None if p.barrier_eV is None else round(p.barrier_eV, 3),
                             "free_energies": {k: round(v, 3) for k, v in p.free_energies.items()}}
                             for p in preds if p.complete},
                         "incomplete": {p.formula: p.missing for p in preds if not p.complete}}
    o, h = pipeline["oxygen_reduction"], pipeline["hydrogen_evolution"]
    u_cmp = o["compare_V"]
    paper_g = {m: orr.free_energies({"OH": d_oh, "O": d_o, "OOH": OOH_FROM_O[0] * d_o + OOH_FROM_O[1]})
               for m, (d_oh, d_o, *_rest) in TABLE_2.items()}
    paper_h = {m: orr.hurdle(g, u_cmp) for m, g in paper_g.items()}
    common = [m for m in o["ranking"] if m in paper_h]
    paper_order = sorted(common, key=lambda m: paper_h[m])
    orr_routes = [{"metal": m, "rank": v["rank"], "limit": v["limiting_V"], "lo": v["range"][0], "hi": v["range"][1],
                   "limiting_route": v["limiting_route"], "step": v["limiting_step"], "barrier": v["barrier_eV"],
                   "hurdle": v["hurdle_eV"], "route": v["route"],
                   "paper_hurdle": round(paper_h[m], 3) if m in paper_h else None}
                  for m, v in sorted(o["materials"].items(), key=lambda kv: kv[1]["rank"])]
    kept = [m for m in o["ranking"] if m not in SET_ASIDE]
    kept_paper = [m for m in paper_order if m not in SET_ASIDE]
    pt = o["materials"]["Pt"]
    gap, inside = grid_gap(orr, pt["free_energies"])
    # platinum against the volcano line: the O* the Fig. 8 relation gives for its OH*, against its own
    corr = {i.name: i.total_correction for i in orr.intermediates}
    x_pt = pt["free_energies"]["OH"]
    o_line = (x_pt - corr["OH"] - 0.05) / 0.5 + corr["O"]

    # 6. the error bar, from the thermal error budget and its oxide check
    tv = thermal["values"]
    omat = {r["adsorbate"]: r["1p1_omat"] for r in thermal["tables"]["oxide_ads"]}
    oc20 = {r["adsorbate"]: r["1p1_oc20"] for r in thermal["tables"]["oxide_ads"]}
    oer = [omat[a] for a in ("O", "OH", "HO2")]

    gaps_r = np.array([r["gap_rpbe"] for r in her])
    gaps_p = np.array([r["gap_pbe"] for r in her])
    rank = {m: v["rank"] for m, v in o["materials"].items()}
    values = {
        "her_n": len(her), "her_mae_rpbe": round(float(np.abs(gaps_r).mean()), 3),
        "her_mae_pbe": round(float(np.abs(gaps_p).mean()), 3), "her_max_rpbe": round(float(np.abs(gaps_r).max()), 3),
        "her_offset_rpbe": round(float(gaps_r.mean()), 3),
        "her_low_n": len(low), "her_low_metals": names([r["metal"] for r in low]),
        "her_low_mae": round(float(np.mean([abs(r["gap_rpbe"]) for r in low])), 3),
        "her_low_max": round(float(np.max([abs(r["gap_rpbe"]) for r in low])), 3),
        "her_full_n": len(full), "her_full_mae": round(float(np.mean([abs(r["gap_rpbe"]) for r in full])), 3),
        "her_full_offset": round(float(np.mean([r["gap_rpbe"] for r in full])), 3),
        "her_full_words": stronger(float(np.mean([r["gap_rpbe"] for r in full]))),
        "her_spearman": round(spearman([-abs(r["uma"]) for r in with_j0], [r["log_j0"] for r in with_j0]), 2),
        "her_spearman_published": round(spearman([-abs(r["rpbe"]) for r in with_j0], [r["log_j0"] for r in with_j0]), 2),
        "her_j0_n": len(with_j0), "her_pt": next(r["uma"] for r in her if r["metal"] == "Pt"),
        "table2_n": len(table2), "table2_values": 3 * len(table2), "table2_max_gap": round(table2_gap, 3),
        "table2_pt_limit": round(1.23 - max(pt2["dg1_app"], pt2["dg2_app"]), 2), "table2_pt_step": pt2_step,
        "table2_pt_dg1": pt2["dg1_app"], "table2_pt_dg2": pt2["dg2_app"],
        "orr_n": len(orr_metals), "orr_oh_mae": round(float(np.mean([abs(r["gap_oh"]) for r in orr_metals])), 3),
        "orr_o_mae": round(float(np.mean([abs(r["gap_o"]) for r in orr_metals])), 3),
        "orr_oh_max": round(float(np.max([abs(r["gap_oh"]) for r in orr_metals])), 3),
        "orr_o_max": round(float(np.max([abs(r["gap_o"]) for r in orr_metals])), 3),
        "orr_oh_offset": round(float(np.mean([r["gap_oh"] for r in orr_metals])), 3),
        "orr_o_offset": round(float(np.mean([r["gap_o"] for r in orr_metals])), 3),
        "orr_o_words": stronger(float(np.mean([r["gap_o"] for r in orr_metals]))),
        "orr_oh_spearman": round(spearman([r["uma_oh"] for r in orr_metals], [r["pub_oh"] for r in orr_metals]), 2),
        "orr_o_spearman": round(spearman([r["uma_o"] for r in orr_metals], [r["pub_o"] for r in orr_metals]), 2),
        "ooh_n": len(ooh), "ooh_gap_mean": round(ooh_mean, 2), "ooh_gap_sd": round(float(np.std([r["gap"] for r in ooh])), 2),
        "ooh_gap_metals": round(OOH_GAP_METALS, 2), "ooh_gap_oxides": OOH_GAP_OXIDES,
        "ooh_gap_oxides_mae": OOH_GAP_OXIDES_MAE, "ooh_vs_metals": round(ooh_mean - OOH_GAP_METALS, 2),
        "ooh_vs_oxides": round(ooh_mean - OOH_GAP_OXIDES, 2),
        "pt_orr_limit": pt["limiting_V"], "pt_orr_lo": pt["range"][0], "pt_orr_hi": pt["range"][1],
        "pt_orr_step": pt["limiting_step"], "pt_orr_route": pt["limiting_route"], "pt_orr_limit_published": PT_ORR_LIMIT,
        "pt_barrier": pt["barrier_eV"], "pt_hurdle": pt["hurdle_eV"],
        "range_grid_gap": round(gap, 3), "range_grid_inside": "inside" if inside else "outside",
        "pt_line_V": round(orr.volcano_limit(x_pt), 2), "pt_o_tighter": round(o_line - pt["free_energies"]["O"], 2),
        "paper_pt_o_line": round((TABLE_2["Pt"][0] - 0.05) / 0.5, 2), "paper_pt_o": TABLE_2["Pt"][1],
        "orr_compare_V": u_cmp, "orr_ranking": names(o["ranking"]), "orr_top3": names(o["ranking"][:3]),
        "orr_ranking_paper": names(paper_order),
        "orr_spearman_paper": round(spearman([o["materials"][m]["hurdle_eV"] for m in common], [paper_h[m] for m in common]), 2),
        "orr_set_aside": names(kept), "orr_set_aside_paper": names(kept_paper),
        "orr_follows_paper": "which opens with Pt and Pd, then Ir and Rh, as the paper reports from experiment"
                             if follows_paper(o["ranking"]) else
                             "which departs from the opening the paper reports from experiment, Pt and Pd, then Ir and Rh",
        "orr_set_aside_both": (f"both orders read {names(kept)}" if kept == kept_paper else
                               f"UMA's order reads {names(kept)} and the paper's numbers give {names(kept_paper)}"),
        "orr_paper_follows_paper": follows_paper(paper_order),
        "ag_rank": rank.get("Ag"), "au_rank": rank.get("Au"), "cu_rank": rank.get("Cu"),
        "ag_rank_paper": paper_order.index("Ag") + 1, "au_rank_paper": paper_order.index("Au") + 1,
        "ag_route": o["materials"]["Ag"]["route"], "ag_hurdle": o["materials"]["Ag"]["hurdle_eV"],
        "ag_barrier": o["materials"]["Ag"]["barrier_eV"], "ag_limit": o["materials"]["Ag"]["limiting_V"],
        "ag_step": o["materials"]["Ag"]["limiting_step"], "ag_paper_hurdle": round(paper_h["Ag"], 2),
        "au_barrier": o["materials"]["Au"]["barrier_eV"], "au_hurdle": o["materials"]["Au"]["hurdle_eV"],
        "orr_barriers": ", ".join(f"{m} {v['barrier_eV']:+.2f}" for m, v in
                                  sorted(o["materials"].items(), key=lambda kv: kv[1]["barrier_eV"])),
        "orr_complete_n": len(o["materials"]), "orr_materials_n": len(o["materials"]) + len(o["incomplete"]),
        "her_ranking": names(h["ranking"]), "pt_her_eta": h["materials"]["Pt"]["overpotential_V"],
        "pt_her_lo": h["materials"]["Pt"]["range"][0], "pt_her_hi": h["materials"]["Pt"]["range"][1],
        "gate": tv["oxide_gate"], "sigma": electro.SIGMA_EV,
        "oxide_omat_o": omat["O"], "oxide_omat_oh": omat["OH"], "oxide_omat_ooh": omat["HO2"],
        "oxide_omat_lo": min(oer), "oxide_omat_hi": max(oer), "oxide_oc20_lo": min(oc20[a] for a in ("O", "OH", "HO2")),
        "oxide_oc20_hi": max(oc20[a] for a in ("O", "OH", "HO2")),
        "screen": folder.name, "task": job["task"], "generated": job["request"]["created"][:10],
    }
    out = {"generated": values["generated"], "values": values,
           "tables": {"her": her, "table2": table2, "orr_metals": orr_metals, "ooh": ooh, "orr_routes": orr_routes},
           "pipeline": pipeline}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(values, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = next(p for p in sorted(electro.EXAMPLES.iterdir()) if "oxygen-reduction" in p.name)
    main(target)
