"""Element facts for ranking candidate materials beyond activity: how common each element is, and which are
precious metals."""

from __future__ import annotations

from ase.data import atomic_names, atomic_numbers

# Abundance in Earth's crust, mg/kg (CRC Handbook of Chemistry and Physics, "Abundance of elements in the Earth's
# crust and in the sea"). A rough guide to supply and price: below about 0.1 mg/kg an element is scarce.
CRUST_MG_PER_KG = {
    "O": 461000, "Si": 282000, "Al": 82300, "Fe": 56300, "Ca": 41500, "Na": 23600, "Mg": 23300, "K": 20900,
    "Ti": 5650, "H": 1400, "P": 1050, "Mn": 950, "F": 585, "Ba": 425, "Sr": 370, "S": 350, "C": 200, "Zr": 165,
    "Cl": 145, "V": 120, "Cr": 102, "Rb": 90, "Ni": 84, "Zn": 70, "Ce": 66.5, "Cu": 60, "Nd": 41.5, "La": 39,
    "Y": 33, "Co": 25, "Sc": 22, "Li": 20, "Nb": 20, "N": 19, "Ga": 19, "Pb": 14, "B": 10, "Th": 9.6, "Pr": 9.2,
    "Sm": 7.05, "Gd": 6.2, "Dy": 5.2, "Er": 3.5, "Yb": 3.2, "Hf": 3.0, "Cs": 3.0, "Be": 2.8, "U": 2.7, "Sn": 2.3,
    "Eu": 2.0, "Ta": 2.0, "As": 1.8, "Ge": 1.5, "Ho": 1.3, "W": 1.25, "Mo": 1.2, "Tb": 1.2, "Tl": 0.85, "Lu": 0.8,
    "Tm": 0.52, "I": 0.45, "In": 0.25, "Sb": 0.2, "Cd": 0.15, "Hg": 0.085, "Ag": 0.075, "Se": 0.05, "Pd": 0.015,
    "Pt": 0.005, "Bi": 0.0085, "Au": 0.004, "Os": 0.0015, "Ir": 0.001, "Ru": 0.001, "Rh": 0.001, "Te": 0.001,
    "Re": 0.0007,
}
PRECIOUS = {"Ru", "Rh", "Pd", "Ag", "Os", "Ir", "Pt", "Au", "Re"}
# Pure elements the app builds from ASE's reference crystals (fcc, bcc, hcp or bct), most used in catalysis first
METALS = ["Pt", "Pd", "Rh", "Ru", "Ir", "Os", "Au", "Ag", "Cu", "Ni", "Co", "Fe", "Mo", "W", "Re", "Cr", "V", "Nb",
          "Ta", "Ti", "Zr", "Zn", "Al", "Sn", "In", "Ce", "La", "Y"]


def name(symbol: str) -> str:
    return atomic_names[atomic_numbers[symbol]]


def scarcest(elements: list[str]) -> tuple[str, float | None]:
    """The least abundant element of a material and its crustal abundance (mg/kg), None when untabulated."""
    known = [(CRUST_MG_PER_KG[e], e) for e in elements if e in CRUST_MG_PER_KG]
    if not known:
        return elements[0], None
    value, el = min(known)
    return el, value


def supply_label(elements: list[str]) -> str:
    """Plain words for how easy the elements are to get."""
    if any(e in PRECIOUS for e in elements):
        return "Precious metal"
    _, value = scarcest(elements)
    if value is None:
        return "Unrated"
    if value >= 10:
        return "Abundant"
    if value >= 0.1:
        return "Uncommon"
    return "Scarce"
