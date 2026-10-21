"""Honeycomb monolith geometry, external mass transfer and pressure drop (square channels).

Fully developed laminar flow in square channels: Sherwood number 2.98 (constant
wall concentration) and Darcy friction f*Re = 56.9 (Shah and London).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from catalyst_engine.chem import thermo

SHERWOOD_SQUARE = 2.98
DARCY_FRE_SQUARE = 56.9


@dataclass(frozen=True)
class Monolith:
    cpsi: float  # cells per square inch
    wall_mm: float

    @property
    def pitch_m(self) -> float:
        return 0.0254 / np.sqrt(self.cpsi)

    @property
    def hydraulic_diameter_m(self) -> float:
        d = self.pitch_m - self.wall_mm / 1000.0
        if d <= 0:
            raise ValueError("wall is thicker than the cell pitch")
        return d

    @property
    def open_frontal_area(self) -> float:
        return (self.hydraulic_diameter_m / self.pitch_m) ** 2

    @property
    def surface_per_volume(self) -> float:
        """Geometric surface area per monolith volume, m2/m3."""
        return 4.0 * self.hydraulic_diameter_m / self.pitch_m**2

    def km_a(self, species: str, t_k: float, p_pa: float, gas: Mapping[str, float]) -> float:
        """Volumetric external mass-transfer coefficient (1/s) for a species (Cantera id)."""
        d = thermo.diffusivity(species, t_k, p_pa, gas)
        return SHERWOOD_SQUARE * d / self.hydraulic_diameter_m * self.surface_per_volume

    def pressure_drop_pa(
        self, length_m: float, face_velocity_m_s: float, t_k: float, p_pa: float, gas: Mapping[str, float]
    ) -> float:
        mu = thermo.viscosity(t_k, p_pa, gas)
        v_channel = face_velocity_m_s / self.open_frontal_area
        return DARCY_FRE_SQUARE / 2.0 * mu * v_channel * length_m / self.hydraulic_diameter_m**2
