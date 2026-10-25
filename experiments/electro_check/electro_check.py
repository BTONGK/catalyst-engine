"""How close does the app's electrochemistry come to published DFT and measurement?

    uv run python experiments/electro_check/electro_check.py [screen folder]

Reads a finished electrochemical screen of pure metals that holds H, O, OH and OOH (the shipped example,
examples/electro/screens/*oxygen-reduction*, unless a folder is given), scores its binding energies against three
published sources and writes results/summary.json, the numbers behind report.yaml on the electrocatalysis Manual's
Methodology tab:

  1. hydrogen: dG_H = dE_H + 0.24 eV against the RPBE and PBE values of Table S2 in "Revisiting trends in the
     exchange current for hydrogen evolution" (arXiv:2109.04219; 3 x 3 close-packed slabs, 1/9 monolayer), and the
     order of the metals against measured exchange currents (its Table S3-1, the Nørskov 2005 data);
  2. hydroxyl and oxygen: dE_OH and dE_O against water and hydrogen gas, Table 2 of Nørskov et al., J. Phys. Chem.
     B 2004 (RPBE, 1/4 monolayer on the close-packed face);
  3. hydroperoxyl: the OOH-OH gap against the universal 3.2 eV of Man et al., ChemCatChem 2011 (fit MAE 0.17 eV);
  4. the whole pipeline on platinum: the app's limiting potential for oxygen reduction against the paper's 0.78 V,
     and its hydrogen free energy against the near-zero value that makes platinum the best HER metal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from catalyst_engine.design import electro, screening

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# arXiv:2109.04219, Table S2: dG_H (eV) with RPBE and PBE, no van der Waals term
PUBLISHED_H = {"Ag": (0.565, 0.416), "Au": (0.574, 0.418), "Co": (-0.202, -0.355), "Cu": (0.188, 0.040),
               "Ir": (-0.081, -0.208), "Ni": (-0.231, -0.388), "Pd": (-0.123, -0.286), "Pt": (-0.038, -0.192),
               "Rh": (-0.116, -0.270), "Ru": (-0.180, -0.330), "Mo": (-0.321, -0.453), "Re": (-0.154, -0.293)}
# its Table S3-1: measured log10 of the exchange current density (A/cm2), the Nørskov 2005 data, one or more per metal
MEASURED_LOG_J0 = {"Pt": [-3.1, -2.63, -3.34], "Ir": [-3.7, -3.46], "Pd": [-3.0, -3.0], "Rh": [-3.6, -3.22],
                   "Ni": [-5.21, -5.2], "Co": [-5.32], "W": [-5.9], "Cu": [-5.37], "Mo": [-7.07], "Re": [-2.87],
                   "Nb": [-6.8], "Au": [-6.6, -6.8, -5.4], "Ag": [-5.0, -7.85]}
# Nørskov et al. 2004, Table 2: (dE_OH, dE_O) in eV against H2O and H2, RPBE, quarter monolayer, close-packed face
PUBLISHED_OH_O = {"Ag": (0.72, 2.12), "Au": (1.49, 2.75), "Co": (-0.08, -0.22), "Cu": (0.37, 1.20), "Fe": (-0.88, -0.90),
                  "Ir": (0.63, 1.00), "Mo": (-0.61, -1.62), "Ni": (0.13, 0.34), "Pd": (0.92, 1.53), "Pt": (1.05, 1.57),
                  "Rh": (0.34, 0.44), "Ru": (-0.01, -0.05), "W": (-0.80, -2.06)}
OOH_GAP, OOH_GAP_MAE = 3.20, 0.17  # Man et al. 2011: dE_OOH = dE_OH + 3.20 eV, MAE of the fit
PT_ORR_LIMIT = 0.78  # Nørskov 2004: the highest potential where every ORR step on Pt(111) runs downhill (V vs RHE)


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(folder: Path) -> None:
    job = screening.read_job(folder)
    summary = screening.binding_summary(screening.read_results(folder))
    sym = {m["id"]: m["formula"] for m in job["request"]["materials"]}
    e = {sym[mid]: {a: b["E"] for a, b in s["binding"].items() if b["E"] is not None} for mid, s in summary.items()
         if mid in sym}
    corr_h = electro.load_electro("hydrogen_evolution").intermediates[0].correction_eV

    her = []
    for m in sorted(e):
        if "H" in e[m] and m in PUBLISHED_H:
            dg = e[m]["H"] + corr_h
            her.append({"metal": m, "uma": round(dg, 3), "rpbe": PUBLISHED_H[m][0], "pbe": PUBLISHED_H[m][1],
                        "gap_rpbe": round(dg - PUBLISHED_H[m][0], 3), "gap_pbe": round(dg - PUBLISHED_H[m][1], 3),
                        "log_j0": None if m not in MEASURED_LOG_J0 else round(float(np.median(MEASURED_LOG_J0[m])), 2)})
    with_j0 = [r for r in her if r["log_j0"] is not None]
    orr = []
    for m in sorted(e):
        if {"OH", "O"} <= set(e[m]) and m in PUBLISHED_OH_O:
            orr.append({"metal": m, "uma_oh": round(e[m]["OH"], 3), "pub_oh": PUBLISHED_OH_O[m][0],
                        "uma_o": round(e[m]["O"], 3), "pub_o": PUBLISHED_OH_O[m][1],
                        "gap_oh": round(e[m]["OH"] - PUBLISHED_OH_O[m][0], 3), "gap_o": round(e[m]["O"] - PUBLISHED_OH_O[m][1], 3)})
    ooh = [{"metal": m, "oh": round(e[m]["OH"], 3), "ooh": round(e[m]["OOH"], 3), "gap": round(e[m]["OOH"] - e[m]["OH"], 3)}
           for m in sorted(e) if {"OH", "OOH"} <= set(e[m])]

    # the whole pipeline on platinum, the catalyst to beat of both reactions
    pipeline = {}
    for sid in ("oxygen_reduction", "hydrogen_evolution"):
        spec = electro.load_electro(sid)
        req = {**job["request"], "pack": sid, "conditions": {"compare_V": spec.compare_V}}
        preds, _ = electro.predict(spec, {**job, "request": req}, screening.read_results(folder))
        pt = next(p for p in preds if p.formula == "Pt")
        pipeline[sid] = {"limiting_V": round(pt.limiting_V, 3), "overpotential_V": round(pt.overpotential_V, 3),
                         "limiting_step": spec.step_names()[pt.limiting_step],
                         "free_energies": {k: round(v, 3) for k, v in pt.free_energies.items()},
                         "ranking": [p.formula for p in preds if p.rank is not None],
                         "incomplete": {p.formula: p.missing for p in preds if p.missing}}

    gaps_r = np.array([r["gap_rpbe"] for r in her])
    gaps_p = np.array([r["gap_pbe"] for r in her])
    values = {
        "her_n": len(her), "her_mae_rpbe": round(float(np.abs(gaps_r).mean()), 3),
        "her_mae_pbe": round(float(np.abs(gaps_p).mean()), 3), "her_max_rpbe": round(float(np.abs(gaps_r).max()), 3),
        "her_offset_rpbe": round(float(gaps_r.mean()), 3),
        "her_spearman": round(spearman([-abs(r["uma"]) for r in with_j0], [r["log_j0"] for r in with_j0]), 2),
        "her_spearman_published": round(spearman([-abs(r["rpbe"]) for r in with_j0], [r["log_j0"] for r in with_j0]), 2),
        "her_j0_n": len(with_j0), "her_pt": next(r["uma"] for r in her if r["metal"] == "Pt"),
        "orr_n": len(orr), "orr_oh_mae": round(float(np.mean([abs(r["gap_oh"]) for r in orr])), 3),
        "orr_o_mae": round(float(np.mean([abs(r["gap_o"]) for r in orr])), 3),
        "orr_oh_max": round(float(np.max([abs(r["gap_oh"]) for r in orr])), 3),
        "orr_o_max": round(float(np.max([abs(r["gap_o"]) for r in orr])), 3),
        "orr_oh_offset": round(float(np.mean([r["gap_oh"] for r in orr])), 3),
        "orr_o_offset": round(float(np.mean([r["gap_o"] for r in orr])), 3),
        "orr_oh_spearman": round(spearman([r["uma_oh"] for r in orr], [r["pub_oh"] for r in orr]), 2),
        "orr_o_spearman": round(spearman([r["uma_o"] for r in orr], [r["pub_o"] for r in orr]), 2),
        "ooh_n": len(ooh), "ooh_gap_mean": round(float(np.mean([r["gap"] for r in ooh])), 2),
        "ooh_gap_sd": round(float(np.std([r["gap"] for r in ooh])), 2), "ooh_gap_published": OOH_GAP,
        "ooh_gap_published_mae": OOH_GAP_MAE,
        "pt_orr_limit": pipeline["oxygen_reduction"]["limiting_V"], "pt_orr_limit_published": PT_ORR_LIMIT,
        "pt_orr_step": pipeline["oxygen_reduction"]["limiting_step"],
        "pt_her_eta": pipeline["hydrogen_evolution"]["overpotential_V"],
        "orr_ranking": ", ".join(pipeline["oxygen_reduction"]["ranking"]),
        "her_ranking": ", ".join(pipeline["hydrogen_evolution"]["ranking"]),
        "orr_incomplete": ", ".join(f"{m} ({', '.join(a)})" for m, a in pipeline["oxygen_reduction"]["incomplete"].items()) or "none",
        "screen": folder.name, "task": job["task"], "generated": job["request"]["created"][:10],
    }
    values["gate"] = 0.11  # the typical miss the thermal error budget allows (experiments/design_method)
    values["sigma"] = electro.SIGMA_EV
    out = {"generated": values["generated"], "values": values, "tables": {"her": her, "orr_metals": orr, "ooh": ooh},
           "pipeline": pipeline}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(values, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = next(p for p in sorted(electro.EXAMPLES.iterdir()) if "oxygen-reduction" in p.name)
    main(target)
