"""How a reaction pack ranks candidate materials: a binding-energy descriptor and a Sabatier volcano.

The descriptor is a weighted sum of adsorbate binding energies that the atomistic model computes on each
material's surface. Activity peaks where the descriptor sits at the volcano's optimum: bind more weakly and
activation slows, bind more strongly and the surface clogs. The pack anchors the optimum on a reference
catalyst computed in the same run, so the model's constant offsets cancel.
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
    "CO": "carbon monoxide",
    "NO": "nitric oxide",
    "NH3": "ammonia",
}


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

    def barrier_shift(self, descriptor: float, reference: float) -> float:
        """Extra barrier (eV) of a material over the reference: the volcano's height difference."""
        top = reference + self.optimum_shift_eV

        def depth(x: float) -> float:
            return self.slope_weak * max(x - top, 0.0) + self.slope_strong * max(top - x, 0.0)

        return depth(descriptor) - depth(reference)
