"""How a reaction pack ranks candidate materials: a binding-energy descriptor and a Sabatier volcano.

The descriptor is a weighted sum of adsorbate binding energies that the atomistic model computes on each
material's surface. Activity peaks where the descriptor sits at the volcano's optimum: bind more weakly and
activation slows, bind more strongly and the surface clogs. The pack anchors the optimum on a reference
catalyst computed in the same run, so the model's constant offsets cancel.

The optimum holds at the pack's reference temperature with its standard feed. Elsewhere it follows the free
energy of the gas the descriptor's adsorbates exchange with (`peak_gas`): heat lowers that free energy, the
surface empties more easily, and the optimum moves toward stronger binding (design/screening.py, peak_moves).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Adsorbates the atomistic worker can place; their geometries and gas references live in uma_worker.py.
ADSORBATES = {
    "H": "hydrogen atom",
    "C": "carbon atom",
    "N": "nitrogen atom",
    "O": "oxygen atom",
    "OH": "hydroxyl",
    "OOH": "hydroperoxyl",
    "CO": "carbon monoxide",
    "NO": "nitric oxide",
    "NH3": "ammonia",
}


# Gases each adsorbate can exchange with, in order of preference: {gas: molecules per adsorbate}, products
# negative. A reaction uses the first whose supplying gases (positive) it feeds and whose other gases it knows.
PEAK_GAS_CHOICES = {
    "O": [{"O2": 0.5}, {"H2O": 1.0, "H2": -1.0}, {"N2O": 1.0, "N2": -1.0}, {"CO2": 1.0, "CO": -1.0}],
    "N": [{"N2": 0.5}, {"NH3": 1.0, "H2": -1.5}, {"NO": 1.0, "O2": -0.5}],
    "C": [{"CH4": 1.0, "H2": -2.0}, {"CO2": 1.0, "H2O": -2.0, "H2": 2.0}, {"CO": 1.0, "H2O": -1.0, "H2": 1.0}],
    "H": [{"H2": 0.5}],
    "OH": [{"H2O": 1.0, "H2": -0.5}],
    "OOH": [{"H2O": 2.0, "H2": -1.5}],
    "CO": [{"CO": 1.0}],
    "NO": [{"NO": 1.0}],
    "NH3": [{"NH3": 1.0}],
}


def suggest_peak_gas(descriptor: dict[str, float], species: set[str], feed: set[str]) -> dict[str, float]:
    """The gases a reaction's descriptor exchanges with, weighted like the descriptor: per adsorbate, the first
    choice whose supplying gases are in the feed, else the first whose gases the reaction knows. Empty when an
    adsorbate has no such gas, so its optimum holds still."""
    total: dict[str, float] = {}
    for ads, weight in descriptor.items():
        choices = [c for c in PEAK_GAS_CHOICES.get(ads, []) if set(c) <= species]
        fed = [c for c in choices if all(g in feed for g, n in c.items() if n > 0)]
        pick = (fed or choices or [None])[0]
        if pick is None:
            return {}
        for gas, n in pick.items():
            total[gas] = total.get(gas, 0.0) + weight * n
    return {g: round(n, 4) for g, n in total.items() if abs(n) > 1e-9}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceSpec(_Strict):
    formula: str  # the most stable Materials Project entry with this formula
    name: str = ""  # plain name for display, e.g. "Platinum"


class ScreeningSpec(_Strict):
    descriptor: dict[str, float]  # adsorbate -> weight; the descriptor is the weighted sum of binding energies (eV)
    pathways: list[str]  # pathways whose barrier follows the descriptor
    metric: str  # pack metric that ranks candidates (higher is better)
    reference: ReferenceSpec
    optimum_shift_eV: float = 0.0  # volcano top relative to the reference's descriptor; positive = weaker binding
    slope_weak: float = Field(0.9, gt=0)  # barrier rise per eV of descriptor on the weak-binding side
    slope_strong: float = Field(1.0, gt=0)  # barrier rise per eV of descriptor on the strong-binding side
    sigma_eV: float = Field(0.15, gt=0)  # uncertainty of each binding energy
    # gas species -> molecules per unit of descriptor: what the descriptor's adsorbates exchange with; the optimum
    # follows its free energy away from the reference temperature (empty: the optimum holds still)
    peak_gas: dict[str, float] = Field(default_factory=dict)
    suggested_elements: list[str] = Field(default_factory=list)
    basis: str = ""  # where the descriptor and the optimum come from, in plain words

    @model_validator(mode="after")
    def _check(self) -> "ScreeningSpec":
        unknown = set(self.descriptor) - set(ADSORBATES)
        if unknown:
            raise ValueError(f"descriptor names unknown adsorbates {sorted(unknown)}; known: {sorted(ADSORBATES)}")
        if not self.descriptor or not self.pathways:
            raise ValueError("a screening spec needs a descriptor and at least one pathway")
        return self

    @property
    def sigma_descriptor(self) -> float:
        return self.sigma_eV * sum(w * w for w in self.descriptor.values()) ** 0.5

    def barrier_shift(self, descriptor: float, reference: float, peak_move: float = 0.0) -> float:
        """Extra barrier (eV) of a material over the reference: the volcano's height difference, with the optimum
        moved by `peak_move` (eV, negative = toward stronger binding) from where the pack sets it."""
        top = reference + self.optimum_shift_eV + peak_move

        def depth(x: float) -> float:
            return self.slope_weak * max(x - top, 0.0) + self.slope_strong * max(top - x, 0.0)

        return depth(descriptor) - depth(reference)
