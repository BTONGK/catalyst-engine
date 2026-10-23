"""Build a reaction pack from a short form: the wanted reaction, side reactions, the gas mix, the catalyst to beat
and the molecule whose grip ranks materials. The pack lands in <data dir>/reactions/<id>/ with one design protocol,
and its rate constant is calibrated so the catalyst to beat gets halfway to the best possible result at the
temperature the user names."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from catalyst_engine.chem import thermo
from catalyst_engine.chem.network import parse_equation
from catalyst_engine.design import screening
from catalyst_engine.design.spec import suggest_peak_gas
from catalyst_engine.packs import PACKS_DIR, load_pack, user_packs_dir

# Common names for gases whose thermo id differs from the formula people type
THERMO_ALIASES = {"HCHO": "CH2O", "Ar": "AR"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "reaction"


def thermo_id(species: str) -> str | None:
    """The species' id in the thermodynamic database (GRI-Mech 3.0), or None when it is outside it."""
    for cand in (species, species.upper(), THERMO_ALIASES.get(species)):
        if cand and thermo.has_species(cand):
            return cand
    return None


def known_gases() -> list[str]:
    return sorted(thermo._gas().species_names)


def species_in(equations: list[str]) -> list[str]:
    """Every species named in the equations, reactants of the first equation first."""
    out: list[str] = []
    for eq in equations:
        stoich, _ = parse_equation(eq)
        out += [s for s, nu in sorted(stoich.items(), key=lambda kv: kv[1]) if s not in out]
    return out


def reactants(equation: str) -> list[str]:
    stoich, _ = parse_equation(equation)
    return [s for s, nu in stoich.items() if nu < 0]


def products(equation: str) -> list[str]:
    stoich, _ = parse_equation(equation)
    return [s for s, nu in stoich.items() if nu > 0]


def suggest_descriptor(equation: str) -> list[str]:
    """A first guess at the adsorbate whose binding sets the rate: N for nitrogen chemistry, O for oxidations and
    oxygen-carrying reactions, C and O for carbon hydrogenation, H for the rest."""
    names = " ".join(parse_equation(equation)[0])
    if "N" in re.sub(r"Na|Ni|Nb|Nd", "", names):
        return ["N"]
    if "C" in names and "H2" in reactants(equation):
        return ["C", "O"] if "O" in names else ["C"]
    if "O" in names:
        return ["O"]
    return ["H"]


def create_reaction(*, name: str, category: str, summary: str, wanted: str, side: list[str], metric: dict,
                    descriptor: list[str], reference_formula: str, reference_name: str, suggested: list[str],
                    feed: dict[str, str], pressure_bar: float, ghsv_per_h: float, temperatures_C: list[float],
                    half_C: float, Ea_kJ: float = 80.0, basis: str = "", root: Path | None = None) -> str:
    """Write, check and calibrate a new pack; returns its id. Raises ValueError with a plain message on any
    problem, leaving the reactions folder as it was."""
    if not name.strip():
        raise ValueError("Give the reaction a name.")
    pack_id = slug(name)
    root = root or user_packs_dir()
    if (PACKS_DIR / pack_id).exists() or (root / pack_id).exists():
        raise ValueError(f"A reaction called {name!r} already exists. Pick another name.")
    if not (min(temperatures_C) <= half_C <= max(temperatures_C)):
        raise ValueError("Pick a halfway temperature inside the temperature range.")
    equations = [wanted, *side]
    try:
        names = species_in(equations)
    except ValueError as exc:
        raise ValueError(f"Check the equations: {exc}. Write them like CO + H2O <=> CO2 + H2.") from None
    names += [s for s in feed if s not in names]
    species = {}
    for s in names:
        tid = thermo_id(s)
        if tid is None:
            raise ValueError(f"{s} is outside the app's gas data (GRI-Mech 3.0). It covers these gases: "
                             f"{', '.join(known_gases())}.")
        species[s] = {"formula": s, **({"cantera": tid} if tid != s else {})}
    wanted_reactants = reactants(wanted)
    key = metric.get("denominator", metric["species"])[0] if metric["kind"] == "yield" else metric["species"][0]
    key = key if key in wanted_reactants else wanted_reactants[0]
    pathways = {"main": {"equation": wanted, "role": "desired",
                         "rate": {"key": key, "orders": {key: 1}, "ln_k_ref": 0.0, "Ea_kJ": float(Ea_kJ)}}}
    for i, eq in enumerate(side, start=1):
        r = reactants(eq)
        pathways[f"side_{i}"] = {"equation": eq, "role": "side",
                                 "rate": {"key": r[0], "orders": {r[0]: 1}, "ln_k_ref": -4.0, "Ea_kJ": float(Ea_kJ) + 30}}
    weights = {a: round(1.0 / len(descriptor), 3) for a in descriptor}
    peak_gas = suggest_peak_gas(weights, set(species), {s for s, v in feed.items() if v})
    metric_spec = {"kind": metric["kind"], "species": metric["species"],
                   **({"denominator": metric["denominator"]} if metric["kind"] == "yield" else {}),
                   "description": metric.get("description") or (
                       f"Share of the {metric['species'][0]} converted." if metric["kind"] == "conversion"
                       else f"Moles of {metric['species'][0]} made per mole of {metric['denominator'][0]} fed.")}
    pack = {
        "id": pack_id, "name": name.strip(), "version": 1, "status": "draft-built-in-app",
        "summary": summary.strip(), "category": category,
        "description": summary.strip() + " Built in the app; the rate constant is calibrated so the catalyst to "
                       f"beat reaches half the best possible result at {half_C:g} °C.",
        "reference_temperature_K": round(half_C + 273.15, 2),
        **({"water": "H2O"} if "H2O" in species else {}),
        "species": species, "pathways": pathways, "metrics": {"main": metric_spec},
        "screening": {"descriptor": weights, "pathways": ["main"],
                      "metric": "main", "reference": {"formula": reference_formula.strip(), "name": reference_name.strip()},
                      **({"peak_gas": peak_gas} if peak_gas else {}),
                      "suggested_elements": suggested, "basis": basis.strip() or
                      f"The surface's grip on {', '.join(descriptor)} sets the rate; the top of the volcano sits at "
                      f"the catalyst to beat."},
    }
    protocol = {"id": "design-v1", "pack": pack_id, "version": 1, "status": "draft",
                "description": "Conditions entered when the reaction was built in the app.",
                "analyzer": {"name": "Simulated analyzer", "measured": sorted(set(metric["species"])),
                             "sigma_abs_ppm": 10, "sigma_rel": 0.02},
                "steps": [{"id": "screen", "purpose": "Performance against temperature", "feed": feed,
                           "ghsv_per_h": float(ghsv_per_h), "pressure_kPa": float(pressure_bar) * 100.0,
                           "temperatures_C": [float(t) for t in temperatures_C]}]}
    folder = root / pack_id
    try:
        (folder / "protocols").mkdir(parents=True)
        _write(folder, pack, protocol)
        loaded = load_pack(folder)
        step = loaded.protocols["design-v1"].steps[0]
        for key_, value in screening.calibrate(loaded, step, half_C).items():
            pack["pathways"][key_.split(".")[0]]["rate"]["ln_k_ref"] = round(value, 3)
        for pid, pw in pack["pathways"].items():
            if pw["role"] == "side":
                pw["rate"]["ln_k_ref"] = round(pack["pathways"]["main"]["rate"]["ln_k_ref"] - 4.0, 3)
        _write(folder, pack, protocol)
        load_pack(folder)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise ValueError(str(exc)) from None
    return pack_id


def _write(folder: Path, pack: dict, protocol: dict) -> None:
    header = "# Built in the app's Reactions page. Edit freely; the app reloads it on the next page view.\n\n"
    (folder / "pack.yaml").write_text(header + yaml.safe_dump(pack, sort_keys=False, allow_unicode=True))
    (folder / "protocols" / "design-v1.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True))


def delete_reaction(pack_id: str, root: Path | None = None) -> None:
    """Remove a reaction built in the app (the shipped library stays as is)."""
    folder = (root or user_packs_dir()) / pack_id
    if folder.is_dir() and (folder / "pack.yaml").exists():
        shutil.rmtree(folder)
