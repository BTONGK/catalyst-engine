"""Candidate materials from the Materials Project (materialsproject.org): search by elements, fetch crystal
structures, and turn them into atomistic jobs and 3D views.

The API key comes from MP_API_KEY, PMG_MAPI_KEY or ~/.config/.pmgrc.yaml (the pymatgen convention). It travels only
in the request header, to api.materialsproject.org. Structures are cached in <data dir>/materials/.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
import yaml
from ase import Atoms
from ase.build import bulk as ase_bulk
from ase.io import write

from catalyst_engine.design import zeolites

API = "https://api.materialsproject.org/materials/summary/"
FIELDS = ("material_id,formula_pretty,energy_above_hull,formation_energy_per_atom,band_gap,is_metal,nsites,"
          "density,symmetry,elements,theoretical")


class MaterialsProjectError(RuntimeError):
    pass


def api_key() -> str | None:
    for var in ("MP_API_KEY", "PMG_MAPI_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    rc = Path.home() / ".config" / ".pmgrc.yaml"
    if rc.exists():
        return (yaml.safe_load(rc.read_text()) or {}).get("PMG_MAPI_KEY")
    return None


def _cache_dir() -> Path:
    return Path(os.environ.get("CE_DATA_DIR", Path(__file__).resolve().parents[3] / "data")) / "materials"


def _get(params: dict) -> list[dict]:
    key = api_key()
    if not key:
        raise MaterialsProjectError("Searching needs a Materials Project API key. Add PMG_MAPI_KEY to ~/.config/.pmgrc.yaml.")
    try:
        r = requests.get(API, headers={"X-API-KEY": key}, params=params, timeout=40)
    except requests.RequestException as exc:
        raise MaterialsProjectError(f"Materials Project is unreachable: {type(exc).__name__}") from None
    if r.status_code != 200:
        raise MaterialsProjectError(f"Materials Project answered {r.status_code}: {r.text[:200]}")
    return r.json().get("data", [])


def _row(d: dict) -> dict:
    sym = d.get("symmetry") or {}
    return {
        "material_id": d["material_id"],
        "formula": d["formula_pretty"],
        "e_hull": float(d.get("energy_above_hull") or 0.0),
        "formation_energy": d.get("formation_energy_per_atom"),
        "band_gap": d.get("band_gap"),
        "is_metal": d.get("is_metal"),
        "nsites": d.get("nsites"),
        "density": d.get("density"),
        "crystal_system": sym.get("crystal_system", ""),
        "space_group": sym.get("symbol", ""),
        "elements": d.get("elements") or re.findall(r"[A-Z][a-z]?", d["formula_pretty"]),
        "theoretical": d.get("theoretical"),
    }


def chemical_systems(elements: list[str], max_elements: int, add_oxygen: bool) -> list[str]:
    """Every combination of the elements up to `max_elements`, each with oxygen added when asked."""
    base = [e for e in dict.fromkeys(elements) if not (add_oxygen and e == "O")]
    out = []
    for n in range(1, max_elements + 1):
        for combo in itertools.combinations(sorted(base), n):
            out.append("-".join(sorted(combo + (("O",) if add_oxygen else ()))))
    return out


def search(elements: list[str], max_elements: int = 2, add_oxygen: bool = False, max_e_hull: float = 0.05,
           max_atoms: int = 40, limit: int = 60) -> list[dict]:
    """Materials made only of the given elements (plus oxygen when asked), most stable first."""
    systems = chemical_systems(elements, max_elements, add_oxygen)
    if not systems:
        return []
    rows = []
    for chunk in range(0, len(systems), 40):
        rows += _get({"chemsys": ",".join(systems[chunk:chunk + 40]), "energy_above_hull_max": max_e_hull,
                      "nsites_max": max_atoms, "_fields": FIELDS, "_sort_fields": "energy_above_hull",
                      "_limit": limit})
    out = sorted((_row(d) for d in rows), key=lambda r: (r["e_hull"], r["nsites"] or 0))
    return out[:limit]


def most_stable(formula: str) -> dict:
    data = _get({"formula": formula, "_fields": FIELDS, "_sort_fields": "energy_above_hull", "_limit": 1})
    if not data:
        raise MaterialsProjectError(f"Materials Project search for {formula} came back empty")
    return _row(data[0])


def structure(material_id: str) -> dict:
    """{lattice, species, frac} for a material, from the cache or the API."""
    path = _cache_dir() / f"{material_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = _get({"material_ids": material_id, "_fields": "material_id,structure"})
    if not data:
        raise MaterialsProjectError(f"Materials Project returned an empty structure for {material_id}")
    st = data[0]["structure"]
    out = {"lattice": st["lattice"]["matrix"],
           "species": [max(s["species"], key=lambda x: x["occu"])["element"] for s in st["sites"]],
           "frac": [s["abc"] for s in st["sites"]]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def element_entry(symbol: str) -> dict:
    """A job entry for a pure element in its reference crystal (ASE's table), usable without the API."""
    return {"id": f"element:{symbol}", "label": symbol, "formula": symbol, "ase_bulk": symbol, "e_hull": 0.0,
            "elements": [symbol], "source": "Reference crystal"}


def job_entry(row: dict) -> dict:
    """A job entry for a Materials Project material: its search row plus its crystal structure."""
    return {"id": row["material_id"], "label": f"{row['formula']} ({row['material_id']})", "formula": row["formula"],
            "e_hull": row["e_hull"], "elements": row["elements"], "source": "Materials Project",
            **structure(row["material_id"])}


def atoms_of(entry: dict) -> Atoms:
    if entry.get("ase_bulk"):
        return ase_bulk(entry["ase_bulk"])
    return Atoms(entry["species"], scaled_positions=entry["frac"], cell=entry["lattice"], pbc=True)


def frameworks(max_atoms: int = 72) -> list[dict]:
    """Silica frameworks (zeolites) in the Materials Project: SiO2 crystals lighter than 2.1 g/cm3 (quartz and the
    other dense forms weigh 2.2 and up) with at most `max_atoms` in the cell. Named ones first, then measured
    structures, then the smallest cells."""
    data = _get({"formula": "SiO2", "_fields": "material_id,nsites,density,energy_above_hull,theoretical,symmetry",
                 "_limit": 1000})
    rows = [{"material_id": d["material_id"], "nsites": d["nsites"], "density": d["density"],
             "e_hull": d["energy_above_hull"], "experimental": not d["theoretical"],
             "space_group": d["symmetry"]["symbol"], "name": zeolites.KNOWN.get(d["material_id"], "")}
            for d in data if d["density"] < 2.1 and d["nsites"] <= max_atoms]
    return sorted(rows, key=lambda r: (not r["name"], not r["experimental"], r["nsites"], r["e_hull"]))


def zeolite_entry(framework: dict, form: str, metal: str = "Cu") -> dict:
    """A job entry for one active site (zeolites.FORMS) on a framework row from frameworks(): every starting
    arrangement of the site, and the index of the atom molecules bind to."""
    z = atoms_of(structure(framework["material_id"]))
    if not zeolites.is_framework(z):
        raise ValueError(f"{framework['material_id']} holds silicon with other than four oxygen; pick another.")
    variants, site = zeolites.build(z, form, metal)
    ion = {"H": "H", "M+": f"{metal}+", "M2+": f"{metal}2+", "MOH+": f"[{metal}OH]+"}[form]
    code = zeolites.code(framework["material_id"]) or framework["material_id"]
    first = variants[0]
    pore = 2 * zeolites.clearance(z, zeolites.pore_center(z)) - 2.7  # less an oxygen's radius on each side
    return {"id": f"zeolite:{framework['material_id']}:{form}:{metal if form != 'H' else ''}",
            "label": f"{ion} {framework['name'] or 'SiO2 framework ' + framework['material_id']}, "
                     f"Si/Al {zeolites.si_al(first):.0f}",
            "formula": f"{ion}-{code}", "kind": "framework", "framework": framework["material_id"], "form": form,
            "e_hull": framework["e_hull"], "elements": sorted(set(first.get_chemical_symbols())),
            "source": "Materials Project framework, sites built here", "nsites": len(first),
            "lattice": first.cell.array.tolist(), "site": site, "pore_A": round(float(pore), 1),
            "variants": [{"species": v.get_chemical_symbols(), "frac": v.get_scaled_positions(wrap=False).tolist()}
                         for v in variants]}


def cif_text(atoms: Atoms) -> str:
    buf = io.BytesIO()
    write(buf, atoms, format="cif")
    return buf.getvalue().decode()


def lattice_summary(atoms: Atoms) -> str:
    a, b, c, al, be, ga = np.round(atoms.cell.cellpar(), 2)
    return f"a {a} Å, b {b} Å, c {c} Å; angles {al}°, {be}°, {ga}°"
