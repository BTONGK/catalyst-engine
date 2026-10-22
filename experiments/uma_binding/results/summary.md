# UMA test run: summary

## Metals (catalysis mode, OC20 references: 1/2 N2, H2O - H2, 1/2 H2)

| Metal | N | NH | NH2 | NH3 | O | OH | NO |
|---|---|---|---|---|---|---|---|
| Ru | -0.79 (hcp) | -1.14 (hcp) | -1.08 (fcc) | -1.51 (ontop) | -0.18 (hcp) | -0.01 (fcc) | +1.03 (ontop) |
| Rh | -0.18 (hcp) | -0.64 (fcc) | -0.83 (fcc) | -1.47 (ontop) | +0.56 (fcc) | +0.44 (bridge) | +1.20 (hcp) |
| Pd | +0.62 (fcc) | +0.08 (fcc) | -0.33 (fcc) | -1.31 (ontop) | +1.40 (fcc) | +0.92 (bridge) | +1.49 (fcc) |
| Ag | +3.21 (fcc) | +1.57 (fcc) | +0.26 (bridge) | -0.99 (ontop) | +2.24 (fcc) | +0.85 (fcc) | +3.34 (fcc) |
| Ir | +0.20 (hcp) | -0.37 (fcc) | -0.66 (fcc) | -1.49 (ontop) | +1.01 (fcc) | +0.79 (bridge) | +1.78 (ontop) |
| Pt | +0.61 (fcc) | -0.03 (fcc) | -0.44 (bridge) | -1.48 (ontop) | +1.60 (fcc) | +1.15 (bridge) | +1.92 (fcc) |
| Au | +2.95 (fcc) | +1.59 (fcc) | +0.45 (bridge) | -1.01 (ontop) | +2.79 (fcc) | +1.54 (bridge) | +3.49 (fcc) |
| Ni | -0.30 (fcc) | -0.90 (fcc) | -0.91 (bridge) | -1.40 (ontop) | +0.12 (fcc) | +0.13 (fcc) | +0.95 (fcc) |
| Cu | +1.66 (fcc) | +0.35 (fcc) | -0.32 (bridge) | -1.11 (ontop) | +1.07 (fcc) | +0.37 (fcc) | +2.59 (fcc) |

### Scaling relations (slope of AHx binding against A binding)

| Pair | UMA slope | Bond-counting slope | R² |
|---|---|---|---|
| NH vs N | 0.68 | 0.67 | 0.98 |
| NH2 vs N | 0.35 | 0.33 | 0.94 |
| OH vs O | 0.48 | 0.50 | 0.84 |

### Periodic-table orderings (stronger binder listed first)

| Pair | Element | UMA agrees |
|---|---|---|
| Ru > Rh | N | yes (-0.79 vs -0.18) |
| Ru > Rh | O | yes (-0.18 vs +0.56) |
| Rh > Pd | N | yes (-0.18 vs +0.62) |
| Rh > Pd | O | yes (+0.56 vs +1.40) |
| Pd > Ag | N | yes (+0.62 vs +3.21) |
| Pd > Ag | O | yes (+1.40 vs +2.24) |
| Ni > Cu | N | yes (-0.30 vs +1.66) |
| Ni > Cu | O | yes (+0.12 vs +1.07) |
| Cu > Ag | N | yes (+1.66 vs +3.21) |
| Cu > Ag | O | yes (+1.07 vs +2.24) |
| Pt > Au | N | yes (+0.61 vs +2.95) |
| Pt > Au | O | yes (+1.60 vs +2.79) |
| Ir > Pt | N | yes (+0.20 vs +0.61) |
| Ir > Pt | O | yes (+1.01 vs +1.60) |

14 of 14 orderings match.

### NH3 and NO binding relative to the gas molecule

| Metal | NH3 (eV) | NO (eV) |
|---|---|---|
| Ru | -0.70 | -2.61 |
| Rh | -0.65 | -2.45 |
| Pd | -0.49 | -2.15 |
| Ag | -0.17 | -0.31 |
| Ir | -0.67 | -1.86 |
| Pt | -0.66 | -1.73 |
| Au | -0.20 | -0.15 |
| Ni | -0.59 | -2.70 |
| Cu | -0.29 | -1.06 |

### Placements left out (adsorbate desorbed or broke apart)

- Au NO from ontop: desorbed

## Oxides in SCR catalysts (binding energy relative to the gas molecule, eV)

| Case | Catalysis mode (oc20) | Materials mode (omat) | Known | Contact after relaxing (oc20 / omat) | Note |
|---|---|---|---|---|---|
| NH3 on TiO2 (Ti site) | -0.64 | -1.16 | binds 1.23 ± 0.1 eV (TPD); 1.20 eV (optB88-vdW DFT) | N–Ti 2.40 Å / N–Ti 2.28 Å |  |
| NH3 on TiO2, full layer | -0.30 | -0.99 | 0.28 eV weaker per molecule than an isolated molecule | – / – |  |
| NO on TiO2 (Ti site) | -0.22 | -0.62 | weakly held | N–Ti 3.26 Å / N–Ti 2.43 Å |  |
| NH3 on V2O5 (V site) | -0.35 | -0.63 | binds about 0.1–0.3 eV (cluster RPBE) | N–V 2.78 Å / N–V 2.47 Å |  |
| NO on V2O5 (V site) | -0.54 | -0.66 | weakly held | N–O 2.86 Å / N–O 2.83 Å |  |
| NH3 on V2O5 (V–OH site) | -1.32 | -1.64 | proton transfers; NH4+ binds strongly | – / – | proton moved to N (NH4+ formed) |

oc20: full-monolayer NH3 binds +0.34 eV per molecule relative to an isolated molecule (experiment and DFT: +0.28).

omat: full-monolayer NH3 binds +0.17 eV per molecule relative to an isolated molecule (experiment and DFT: +0.28).

## Gas molecules in each mode (gas.py)

| Mode | Quantity | UMA (eV) | Reference (eV) | Reference |
|---|---|---|---|---|
| oc20 | H2 energy | -6.970 | -6.954 | OC20 reference |
| oc20 | N2 energy | -16.182 | -16.166 | OC20 reference |
| oc20 | H2O energy | -14.155 | -14.158 | OC20 reference |
| oc20 | NH3 formation | -0.843 | -0.815 | experiment |
| oc20 | NO formation | +3.574 | +3.647 | experiment |
| omat | H2 energy | -6.644 | -6.954 | OC20 reference |
| omat | N2 energy | -16.291 | -16.166 | OC20 reference |
| omat | H2O energy | -13.926 | -14.158 | OC20 reference |
| omat | NH3 formation | -1.065 | -0.815 | experiment |
| omat | NO formation | +3.392 | +3.647 | experiment |

## Bulk lattice in each mode (lattice.py)

| Crystal | Mode | Axis | Experiment (Å) | UMA (Å) | Change |
|---|---|---|---|---|---|
| anatase | oc20 | a | 3.784 | 3.846 | +1.6% |
| anatase | oc20 | b | 3.784 | 3.846 | +1.6% |
| anatase | oc20 | c | 9.514 | 9.818 | +3.2% |
| anatase | omat | a | 3.784 | 3.817 | +0.8% |
| anatase | omat | b | 3.784 | 3.817 | +0.8% |
| anatase | omat | c | 9.514 | 9.628 | +1.2% |
| V2O5 | oc20 | a | 11.512 | 11.221 | -2.5% |
| V2O5 | oc20 | b | 3.564 | 3.653 | +2.5% |
| V2O5 | oc20 | c | 4.368 | 7.224 | +65.4% |
| V2O5 | omat | a | 11.512 | 11.432 | -0.7% |
| V2O5 | omat | b | 3.564 | 3.651 | +2.4% |
| V2O5 | omat | c | 4.368 | 4.953 | +13.4% |
