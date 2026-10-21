"""Chemical formulas: element counts and molar masses."""

from __future__ import annotations

import re
from collections import Counter

# g/mol, IUPAC standard atomic weights (abridged to elements used by gas-phase packs so far).
ATOMIC_MASS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
    "Cl": 35.45, "F": 18.998, "Ar": 39.95, "He": 4.0026,
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)")


def element_counts(formula: str) -> Counter[str]:
    """Parse a formula such as 'NH3', 'CH2O' or 'Ca(OH)2' into element counts."""
    stack: list[Counter[str]] = [Counter()]
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"cannot parse formula {formula!r}")
        pos = m.end()
        element, count, opening, closing, group_count = m.groups()
        if element:
            stack[-1][element] += int(count or 1)
        elif opening:
            stack.append(Counter())
        elif closing:
            if len(stack) == 1:
                raise ValueError(f"unbalanced parenthesis in {formula!r}")
            group = stack.pop()
            for el, n in group.items():
                stack[-1][el] += n * int(group_count or 1)
    if pos != len(formula) or len(stack) != 1 or not stack[0]:
        raise ValueError(f"cannot parse formula {formula!r}")
    return stack[0]


def molar_mass(formula: str) -> float:
    counts = element_counts(formula)
    missing = set(counts) - set(ATOMIC_MASS)
    if missing:
        raise KeyError(f"no atomic mass for {sorted(missing)}")
    return sum(ATOMIC_MASS[el] * n for el, n in counts.items())
