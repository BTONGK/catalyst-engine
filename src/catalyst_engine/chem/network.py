"""Reaction networks defined as data.

A pack declares species and pathways (balanced equations plus a rate-law form).
This module validates that declaration and compiles it into arrays that the
reactor integrates for many catalysts and conditions at once.

Rate-law form for pathway j, giving the consumption rate of its key reactant
in mol per m3 of catalyst per second:

    r_j = k_j(T) * prod_i C_i^n_ij * prod_c theta_c * (1 + sum_h K_h C_h)^-m_j * (1 - Q_j/K_eq,j)

    k_j(T)   = exp(ln_k_ref) * exp(-Ea/R (1/T - 1/T_ref))
    theta_c  = K_c C_c / (1 + K_c C_c)            (saturating dependence on species c)
    K(T)     = exp(ln_K_ref) * exp(-dH/R (1/T - 1/T_ref))
    last factor only for reversible pathways ('<=>'), with K_eq from Cantera thermo.

Concentrations are mol/m3, so first-order k is in 1/s (m3 gas per m3 catalyst per s).
Parameters are named '<pathway>.<name>', e.g. 'scr.ln_k_ref' or 'scr.cov.NH3.dH_kJ'.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catalyst_engine.chem import thermo
from catalyst_engine.chem.formula import element_counts
from catalyst_engine.units import R

PARAM_BOUNDS = {
    "ln_k_ref": (-30.0, 30.0),
    "Ea_kJ": (5.0, 300.0),
    "ln_K_ref": (-15.0, 30.0),
    "dH_kJ": (-300.0, 0.0),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpeciesSpec(_Strict):
    formula: str
    cantera: str | None = None  # id in the thermo database; defaults to the formula

    @property
    def thermo_id(self) -> str:
        return self.cantera or self.formula


class AdsorptionSpec(_Strict):
    ln_K_ref: float  # ln(K / (m3/mol)) at the pack's reference temperature
    dH_kJ: float = 0.0  # adsorption enthalpy, kJ/mol (negative: binds less at high T)
    fit: bool = False


class RateSpec(_Strict):
    key: str | None = None  # reactant whose consumption the law gives; default: first order species
    orders: dict[str, float] = Field(default_factory=dict)
    coverage: dict[str, AdsorptionSpec] = Field(default_factory=dict)
    inhibition: dict[str, AdsorptionSpec] = Field(default_factory=dict)
    inhibition_exponent: float = 1.0
    ln_k_ref: float
    Ea_kJ: float
    fit: list[Literal["ln_k_ref", "Ea_kJ"]] = Field(default_factory=lambda: ["ln_k_ref", "Ea_kJ"])


class PathwaySpec(_Strict):
    equation: str
    role: Literal["desired", "side"]
    rate: RateSpec
    note: str | None = None


_TERM = re.compile(r"^\s*(\d*\.?\d+)?\s*([A-Za-z][A-Za-z0-9()]*)\s*$")


def parse_equation(equation: str) -> tuple[dict[str, float], bool]:
    """'4 NH3 + 4 NO + O2 -> 4 N2 + 6 H2O' -> ({species: nu}, reversible). Products positive."""
    if "<=>" in equation:
        lhs, rhs = equation.split("<=>")
        reversible = True
    elif "->" in equation:
        lhs, rhs = equation.split("->")
        reversible = False
    else:
        raise ValueError(f"equation needs '->' or '<=>': {equation!r}")
    stoich: Counter[str] = Counter()
    for side, sign in ((lhs, -1.0), (rhs, 1.0)):
        for term in side.split("+"):
            m = _TERM.match(term)
            if not m:
                raise ValueError(f"cannot parse term {term!r} in {equation!r}")
            stoich[m.group(2)] += sign * float(m.group(1) or 1.0)
    return {s: nu for s, nu in stoich.items() if nu != 0.0}, reversible


class NetworkSpec(_Strict):
    species: dict[str, SpeciesSpec]
    pathways: dict[str, PathwaySpec]
    reference_temperature_K: float
    water: str | None = None  # species ids used for wet/dry and O2-corrected bases
    oxygen: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "NetworkSpec":
        errors: list[str] = []
        for sid, sp in self.species.items():
            try:
                element_counts(sp.formula)
            except ValueError as exc:
                errors.append(f"species {sid}: {exc}")
            if not thermo.has_species(sp.thermo_id):
                errors.append(f"species {sid}: {sp.thermo_id!r} not in thermo database")
        for ref in (self.water, self.oxygen):
            if ref is not None and ref not in self.species:
                errors.append(f"basis species {ref!r} is not declared")
        for pid, pw in self.pathways.items():
            if "." in pid:
                errors.append(f"pathway id {pid!r} may not contain '.'")
            try:
                stoich, _ = parse_equation(pw.equation)
            except ValueError as exc:
                errors.append(f"pathway {pid}: {exc}")
                continue
            unknown = set(stoich) - set(self.species)
            if unknown:
                errors.append(f"pathway {pid}: undeclared species {sorted(unknown)}")
                continue
            balance: Counter[str] = Counter()
            for s, nu in stoich.items():
                for el, n in element_counts(self.species[s].formula).items():
                    balance[el] += nu * n
            off = {el: round(v, 9) for el, v in balance.items() if abs(v) > 1e-9}
            if off:
                errors.append(f"pathway {pid}: elements not balanced {off}")
            rate = pw.rate
            named = set(rate.orders) | set(rate.coverage) | set(rate.inhibition)
            if named - set(self.species):
                errors.append(f"pathway {pid}: rate names undeclared species {sorted(named - set(self.species))}")
            key = rate.key or next(iter(rate.orders), None)
            if key is None or stoich.get(key, 0.0) >= 0.0:
                errors.append(f"pathway {pid}: key species {key!r} must be a reactant")
        if errors:
            raise ValueError("invalid network:\n  " + "\n  ".join(errors))
        return self


@dataclass
class _Ads:
    species: int
    ln_name: str
    dh_name: str


@dataclass
class _Pathway:
    id: str
    role: str
    key: int
    key_nu: float
    reversible: bool
    thermo_stoich: dict[str, float]
    coverage: list[_Ads] = field(default_factory=list)
    inhibition: list[_Ads] = field(default_factory=list)
    inhibition_exponent: float = 1.0


@dataclass
class Network:
    """A compiled network: species order, stoichiometric matrix and parameter layout."""

    spec: NetworkSpec
    species: list[str]
    nu: np.ndarray  # (n_pathways, n_species)
    orders: np.ndarray  # (n_pathways, n_species)
    pathways: list[_Pathway]
    defaults: dict[str, float]
    fitted: list[str]  # parameters the pack marks as fitted by default

    @property
    def t_ref(self) -> float:
        return self.spec.reference_temperature_K

    def index(self, species: str) -> int:
        return self.species.index(species)

    @classmethod
    def compile(cls, spec: NetworkSpec) -> "Network":
        species = list(spec.species)
        idx = {s: i for i, s in enumerate(species)}
        n_p, n_s = len(spec.pathways), len(species)
        nu = np.zeros((n_p, n_s))
        orders = np.zeros((n_p, n_s))
        pathways: list[_Pathway] = []
        defaults: dict[str, float] = {}
        fitted: list[str] = []
        for j, (pid, pw) in enumerate(spec.pathways.items()):
            stoich, reversible = parse_equation(pw.equation)
            for s, v in stoich.items():
                nu[j, idx[s]] = v
            for s, n in pw.rate.orders.items():
                orders[j, idx[s]] = n
            key = pw.rate.key or next(iter(pw.rate.orders))
            rate = pw.rate
            defaults[f"{pid}.ln_k_ref"] = rate.ln_k_ref
            defaults[f"{pid}.Ea_kJ"] = rate.Ea_kJ
            fitted += [f"{pid}.{name}" for name in rate.fit]

            def ads(group: str, entries: Mapping[str, AdsorptionSpec]) -> list[_Ads]:
                out = []
                for s, a in entries.items():
                    base = f"{pid}.{group}.{s}"
                    defaults[f"{base}.ln_K_ref"] = a.ln_K_ref
                    defaults[f"{base}.dH_kJ"] = a.dH_kJ
                    if a.fit:
                        fitted.extend([f"{base}.ln_K_ref", f"{base}.dH_kJ"])
                    out.append(_Ads(idx[s], f"{base}.ln_K_ref", f"{base}.dH_kJ"))
                return out

            pathways.append(
                _Pathway(
                    id=pid,
                    role=pw.role,
                    key=idx[key],
                    key_nu=abs(stoich[key]),
                    reversible=reversible,
                    thermo_stoich={spec.species[s].thermo_id: v for s, v in stoich.items()},
                    coverage=ads("cov", rate.coverage),
                    inhibition=ads("inh", rate.inhibition),
                    inhibition_exponent=rate.inhibition_exponent,
                )
            )
        return cls(spec, species, nu, orders, pathways, defaults, fitted)

    def parameters(self, overrides: Mapping[str, float] | None = None) -> dict[str, float]:
        """Defaults with overrides applied; rejects names the network does not have."""
        params = dict(self.defaults)
        for name, value in (overrides or {}).items():
            if name not in params:
                raise KeyError(f"unknown parameter {name!r}")
            params[name] = float(value)
        return params

    @staticmethod
    def bounds(name: str) -> tuple[float, float]:
        return PARAM_BOUNDS[name.rsplit(".", 1)[-1]]

    def constants(self, params: Mapping[str, np.ndarray | float], t_k: np.ndarray) -> "RateConstants":
        """Evaluate the temperature-dependent constants once per (catalyst, condition)."""
        t_k = np.asarray(t_k, dtype=float)
        inv = 1.0 / t_k - 1.0 / self.t_ref
        n_p = len(self.pathways)
        ln_k = np.empty(t_k.shape + (n_p,))
        ln_keq = np.full(t_k.shape + (n_p,), np.inf)
        cov: list[list[np.ndarray]] = []
        inh: list[list[np.ndarray]] = []

        def van_t_hoff(a: _Ads) -> np.ndarray:
            return np.exp(np.asarray(params[a.ln_name]) - np.asarray(params[a.dh_name]) * 1000.0 / R * inv)

        for j, pw in enumerate(self.pathways):
            ea = np.asarray(params[f"{pw.id}.Ea_kJ"])
            ln_k[..., j] = np.asarray(params[f"{pw.id}.ln_k_ref"]) - ea * 1000.0 / R * inv
            cov.append([van_t_hoff(a) for a in pw.coverage])
            inh.append([van_t_hoff(a) for a in pw.inhibition])
            if pw.reversible:
                temps = np.unique(t_k)
                table = {t: np.log(thermo.equilibrium_constant_c(pw.thermo_stoich, float(t))) for t in temps}
                ln_keq[..., j] = np.vectorize(table.__getitem__)(t_k)
        return RateConstants(self, ln_k, cov, inh, ln_keq)


@dataclass
class RateConstants:
    network: Network
    ln_k: np.ndarray  # (..., n_pathways)
    coverage_K: list[list[np.ndarray]]
    inhibition_K: list[list[np.ndarray]]
    ln_keq: np.ndarray  # (..., n_pathways), +inf when irreversible

    def take(self, idx: np.ndarray) -> "RateConstants":
        """The constants for a subset (or repetition) of a batch."""
        return RateConstants(
            self.network,
            self.ln_k[idx],
            [[k[idx] for k in group] for group in self.coverage_K],
            [[k[idx] for k in group] for group in self.inhibition_K],
            self.ln_keq[idx],
        )

    def key_rates(self, conc: np.ndarray) -> np.ndarray:
        """Consumption rate of each pathway's key species, mol/(m3_cat s). conc: (..., n_species).

        Power laws and the approach to equilibrium are evaluated in log space so a species that
        is exactly absent gives a zero forward rate and a finite reverse rate, never 0 * inf.
        """
        net = self.network
        c = np.maximum(conc, 0.0)
        ln_c = np.log(np.maximum(c, 1e-300))
        ln_fwd = self.ln_k + ln_c @ net.orders.T
        ln_q = ln_c @ net.nu.T
        r = np.exp(ln_fwd) - np.exp(ln_fwd + ln_q - self.ln_keq)
        for j, pw in enumerate(net.pathways):
            for a, kk in zip(pw.coverage, self.coverage_K[j]):
                kc = kk * c[..., a.species]
                r[..., j] *= kc / (1.0 + kc)
            if pw.inhibition:
                denom = 1.0 + sum(kk * c[..., a.species] for a, kk in zip(pw.inhibition, self.inhibition_K[j]))
                r[..., j] /= denom**pw.inhibition_exponent
        return r

    def species_rates(self, conc: np.ndarray, key_rates: np.ndarray | None = None) -> np.ndarray:
        """Net production rate of each species, mol/(m3_cat s)."""
        net = self.network
        r = self.key_rates(conc) if key_rates is None else key_rates
        extent = r / np.array([pw.key_nu for pw in net.pathways])
        return extent @ net.nu
