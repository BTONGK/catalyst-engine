"""Summarize the UMA test run against known behavior. Reads results/*.csv and the relaxed structures; writes
results/summary.json (read by the app's Atomistic check page) and results/summary.md.

    /opt/miniconda3/bin/python3 experiments/uma_binding/analyze.py
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# Electronic formation energies of the gas molecules from the OC20 reference species (see oxides.py)
FORMATION = {"NH3": -0.815, "NO": 3.647}
# Bond-counting slopes for AHx versus A binding (Abild-Pedersen et al., PRL 99, 016105, 2007)
SCALING = {("NH", "N"): 2 / 3, ("NH2", "N"): 1 / 3, ("OH", "O"): 1 / 2}
# Orderings every d-band picture agrees on: (binds more strongly, binds more weakly)
ORDERINGS = [("Ru", "Rh"), ("Rh", "Pd"), ("Pd", "Ag"), ("Ni", "Cu"), ("Cu", "Ag"), ("Pt", "Au"), ("Ir", "Pt")]
METALS = ["Ru", "Rh", "Pd", "Ag", "Ir", "Pt", "Au", "Ni", "Cu"]
ADSORBATES = ["N", "NH", "NH2", "NH3", "O", "OH", "NO"]
# Oxide cases in oxides.py: chart label, what is known, the known binding energy range in eV where one exists,
# and the source key in report.yaml
OXIDE_CASES = {
    ("anatase TiO2(101)", "NH3", "Ti5c"): ("NH3 on TiO2 (Ti site)", "binds 1.23 ± 0.1 eV (TPD); 1.20 eV (optB88-vdW DFT)",
                                           (-1.33, -1.13), "koust"),
    ("anatase TiO2(101)", "NH3", "Ti5c, full monolayer (per molecule)"): (
        "NH3 on TiO2, full layer", "0.28 eV weaker per molecule than an isolated molecule", None, "koust"),
    ("anatase TiO2(101)", "NO", "Ti5c"): ("NO on TiO2 (Ti site)", "weakly held", None, "topsoe"),
    ("V2O5(001)", "NH3", "V Lewis"): ("NH3 on V2O5 (V site)", "binds about 0.1–0.3 eV (cluster RPBE)", (-0.3, -0.1), "hu_berlin"),
    ("V2O5(001)", "NO", "V Lewis"): ("NO on V2O5 (V site)", "weakly held", None, "topsoe"),
    ("V2O5(001)", "NH3", "V-OH Bronsted"): ("NH3 on V2O5 (V–OH site)", "proton transfers; NH4+ binds strongly", None, "yin"),
}
# Relaxed structure for each single-molecule case and the molecule's atom count, for the closest contact
CONTACT_FILES = {
    ("anatase TiO2(101)", "NH3", "Ti5c"): ("anatase_NH3", 4),
    ("anatase TiO2(101)", "NO", "Ti5c"): ("anatase_NO", 2),
    ("V2O5(001)", "NH3", "V Lewis"): ("v2o5_NH3_lewis", 4),
    ("V2O5(001)", "NO", "V Lewis"): ("v2o5_NO_lewis", 2),
}
MONOLAYER_SHIFT_KNOWN = 0.28


def read_csvs(pattern: str) -> list[dict]:
    return [r for f in sorted(RESULTS.glob(pattern)) for r in csv.DictReader(f.open())]


def contact_text(task: str, spec: tuple[str, int] | None) -> str | None:
    """N to its nearest surface atom in the relaxed structure, e.g. 'N–Ti 2.28 Å'."""
    path = RESULTS / "structures" / f"{task}_{spec[0]}.extxyz" if spec else None
    if path is None or not path.exists():
        return None
    from ase.io import read  # ASE ships with fairchem; imported here so the metal summary runs with numpy alone

    a = read(path)
    n, slab = len(a) - spec[1], list(range(len(a) - spec[1]))
    d = a.get_distances(n, slab, mic=True)
    return f"N–{a[slab[int(np.argmin(d))]].symbol} {d.min():.2f} Å"


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(slope), float(intercept), float(r2)


def metals_summary() -> dict:
    rows = read_csvs("metals_*.csv")
    best: dict[tuple[str, str], tuple[float, str]] = {}
    flagged = []
    for r in rows:
        if r["adsorbate"] == "*":
            continue
        if r["anomaly"]:
            flagged.append({k: r[k] for k in ("metal", "adsorbate", "site", "anomaly")})
            continue
        key, e = (r["metal"], r["adsorbate"]), float(r["E_ads_eV"])
        if key not in best or e < best[key][0]:
            best[key] = (e, r["site"])
    metals = [m for m in METALS if (m, "N") in best]

    scaling = []
    for (ahx, a), expected in SCALING.items():
        pts = [{"metal": m, "x": best[(m, a)][0], "y": best[(m, ahx)][0]} for m in metals if (m, a) in best and (m, ahx) in best]
        slope, intercept, r2 = fit(np.array([p["x"] for p in pts]), np.array([p["y"] for p in pts]))
        scaling.append({"x": a, "y": ahx, "slope": slope, "intercept": intercept, "expected": expected, "r2": r2, "points": pts})

    orderings = []
    for strong, weak in ORDERINGS:
        for a in ("N", "O"):
            if (strong, a) in best and (weak, a) in best:
                es, ew = best[(strong, a)][0], best[(weak, a)][0]
                orderings.append({"strong": strong, "weak": weak, "element": a, "e_strong": es, "e_weak": ew, "agrees": es < ew})

    molecules = [{"metal": m, **{name: best[(m, name)][0] - FORMATION[name] for name in FORMATION if (m, name) in best}}
                 for m in metals]
    return {
        "adsorbates": ADSORBATES,
        "metals": metals,
        "binding": [{"metal": m, "adsorbate": a, "E": best[(m, a)][0], "site": best[(m, a)][1]}
                    for m in metals for a in ADSORBATES if (m, a) in best],
        "scaling": scaling,
        "orderings": orderings,
        "molecules": molecules,
        "flagged": flagged,
        "relaxations": len(rows),
        "converged": sum(r["converged"] == "True" for r in rows),
    }


def oxides_summary() -> dict:
    rows = [r for r in read_csvs("oxides_*.csv") if r["system"] != "gas"]
    e = {(r["task"], r["system"], r["adsorbate"], r["site"]): float(r["E_bind_eV"]) for r in rows}
    notes = {(r["system"], r["adsorbate"], r["site"]): r["note"] for r in rows if "proton" in r["note"]}
    cases = []
    for key, (label, known, known_range, source) in OXIDE_CASES.items():
        cases.append({
            "label": label, "system": key[0], "molecule": key[1], "site": key[2],
            "oc20": e.get(("oc20", *key)), "omat": e.get(("omat", *key)),
            "known": known, "known_range": known_range, "source": source,
            "contact_oc20": contact_text("oc20", CONTACT_FILES.get(key)),
            "contact_omat": contact_text("omat", CONTACT_FILES.get(key)),
            "note": notes.get(key, ""),
        })
    shift = {}
    for task in ("oc20", "omat"):
        iso = e.get((task, "anatase TiO2(101)", "NH3", "Ti5c"))
        full = e.get((task, "anatase TiO2(101)", "NH3", "Ti5c, full monolayer (per molecule)"))
        if iso is not None and full is not None:
            shift[task] = full - iso
    return {
        "cases": cases,
        "monolayer_shift": shift,
        "monolayer_shift_known": MONOLAYER_SHIFT_KNOWN,
        "relaxations": len(rows),
        "converged": sum(r["converged"] == "True" for r in rows),
    }


def as_floats(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [{k: float(v) if k in keys else v for k, v in r.items()} for r in rows]


def markdown(s: dict) -> str:
    m, ox = s["metals"], s["oxides"]
    best = {(b["metal"], b["adsorbate"]): b for b in m["binding"]}
    out = ["# UMA test run: summary", "", "## Metals (catalysis mode, OC20 references: 1/2 N2, H2O - H2, 1/2 H2)", "",
           "| Metal | " + " | ".join(m["adsorbates"]) + " |", "|---" * (len(m["adsorbates"]) + 1) + "|"]
    for metal in m["metals"]:
        cells = [f"{best[(metal, a)]['E']:+.2f} ({best[(metal, a)]['site']})" if (metal, a) in best else "–" for a in m["adsorbates"]]
        out.append(f"| {metal} | " + " | ".join(cells) + " |")
    out += ["", "### Scaling relations (slope of AHx binding against A binding)", "",
            "| Pair | UMA slope | Bond-counting slope | R² |", "|---|---|---|---|"]
    out += [f"| {f['y']} vs {f['x']} | {f['slope']:.2f} | {f['expected']:.2f} | {f['r2']:.2f} |" for f in m["scaling"]]
    out += ["", "### Periodic-table orderings (stronger binder listed first)", "",
            "| Pair | Element | UMA agrees |", "|---|---|---|"]
    out += [f"| {o['strong']} > {o['weak']} | {o['element']} | {'yes' if o['agrees'] else 'NO'} ({o['e_strong']:+.2f} vs {o['e_weak']:+.2f}) |"
            for o in m["orderings"]]
    out.append(f"\n{sum(o['agrees'] for o in m['orderings'])} of {len(m['orderings'])} orderings match.")
    out += ["", "### NH3 and NO binding relative to the gas molecule", "", "| Metal | NH3 (eV) | NO (eV) |", "|---|---|---|"]
    out += [f"| {r['metal']} | {r.get('NH3', float('nan')):+.2f} | {r.get('NO', float('nan')):+.2f} |" for r in m["molecules"]]
    if m["flagged"]:
        out += ["", "### Placements left out (adsorbate desorbed or broke apart)", ""]
        out += [f"- {r['metal']} {r['adsorbate']} from {r['site']}: {r['anomaly']}" for r in m["flagged"]]

    out += ["", "## Oxides in SCR catalysts (binding energy relative to the gas molecule, eV)", "",
            "| Case | Catalysis mode (oc20) | Materials mode (omat) | Known | Contact after relaxing (oc20 / omat) | Note |",
            "|---|---|---|---|---|---|"]
    for c in ox["cases"]:
        vals = " | ".join(f"{c[t]:+.2f}" if c[t] is not None else "–" for t in ("oc20", "omat"))
        contact = " / ".join(c[f"contact_{t}"] or "–" for t in ("oc20", "omat"))
        out.append(f"| {c['label']} | {vals} | {c['known']} | {contact} | {c['note']} |")
    for task, shift in ox["monolayer_shift"].items():
        out.append(f"\n{task}: full-monolayer NH3 binds {shift:+.2f} eV per molecule relative to an isolated molecule "
                   f"(experiment and DFT: +{ox['monolayer_shift_known']:.2f}).")

    if s["gas"]:
        out += ["", "## Gas molecules in each mode (gas.py)", "", "| Mode | Quantity | UMA (eV) | Reference (eV) | Reference |",
                "|---|---|---|---|---|"]
        out += [f"| {r['task']} | {r['quantity']} | {r['uma_eV']:+.3f} | {r['reference_eV']:+.3f} | {r['reference']} |" for r in s["gas"]]
    if s["lattice"]:
        out += ["", "## Bulk lattice in each mode (lattice.py)", "", "| Crystal | Mode | Axis | Experiment (Å) | UMA (Å) | Change |",
                "|---|---|---|---|---|---|"]
        out += [f"| {r['crystal']} | {r['task']} | {r['axis']} | {r['experiment_A']:.3f} | {r['uma_A']:.3f} | {r['change_pct']:+.1f}% |"
                for r in s["lattice"]]
    return "\n".join(out) + "\n"


def main() -> None:
    summary = {
        "generated": date.today().isoformat(),
        "metals": metals_summary(),
        "oxides": oxides_summary(),
        "gas": as_floats(read_csvs("gas.csv"), ("uma_eV", "reference_eV")),
        "lattice": as_floats(read_csvs("lattice.csv"), ("experiment_A", "uma_A", "change_pct")),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    text = markdown(summary)
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
