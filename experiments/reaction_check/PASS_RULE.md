# Reaction check: the rule, fixed before the results (2026-10-25, 10:58 PDT)

Written while the 16-metal UMA run (runs/metals-thermal) was still going and before any pack's prediction was
computed, so the rule cannot be fitted to the outcome.

## Thermal reactions

- Prediction: the app's own binding energies (UMA-S 1.1, oc20 mode, standard setting, close-packed face of each
  pure metal: Pt Pd Rh Ir Ru Os Ni Co Fe Cu Ag Au Re Mo W Zn), through the pack's descriptor and volcano, give each
  metal's speed against the pack's catalyst to beat at the published measurement's temperature (the volcano's top
  moved there as on the Design page), with the pack's standard feed.
- Published comparison: one table of per-metal activities measured in one laboratory under one set of conditions
  (turnover rate, rate per metal atom, or the temperature that reaches a set rate), cited in data/published.yaml.
  A published DFT model's predicted rates count only where no measured table exists, and are labelled as such.
- Score: Spearman's rank correlation (rho) between predicted and published order over the metals both contain,
  leaving out the catalyst the pack sets the volcano's top on (its place is set by choice), with a one-sided exact
  permutation p-value (ties take average ranks).
- Pass: at least 5 scored metals besides that catalyst, rho >= 0.6, p <= 0.05, and the published best metal
  among the predicted top three.
- A reaction that passes joins the Design page. Any other reaction stays in the Reactions library and the
  playground, with a row in the Methodology tab's limits.

## Electrochemical reactions

- Each new intermediate's free energy from the worker (with the reaction's corrections) is compared with a published
  DFT table on the same metals, after removing one constant offset per intermediate (different functionals and
  corrections shift every metal alike). Pass for an intermediate: typical miss (median absolute) <= 0.20 eV over at
  least 6 metals, and Spearman rho >= 0.6 with p <= 0.05.
- A reaction joins the Electro design page when every intermediate on its path passes, and its predicted limiting
  potentials order the published metals with rho >= 0.6 (p <= 0.05) against the published table computed by the
  same rule (the paper's own free energies through the app's limiting-potential rule), with at least 6 metals.
- Any other reaction stays in the electro playground with a row in the electro Methodology tab's limits.

## Clarifications (2026-10-25, 11:38 PDT, written before any new pack's prediction was computed)

- The published best metal and the predicted top three are both taken over the scored metals, with the catalyst the
  pack sets the volcano's top on left out of both, so its set place cannot satisfy the test.
- For a reaction checked against published measurements, the pack's catalyst to beat is the metal those
  measurements find most active (ties go to the metal the secondary table ranks higher). The volcano's top sits on
  it, and it leaves the scoring.
- A table that gives only an order scores through its ranks. A table whose activity scale the source leaves
  unconfirmed (the temperatures in Bond's Table 7.3) is shown as indicative, outside the pass rule.
- A source temperature missing from the cited text (Grenoble 1981 through its review) means the prediction runs at
  the pack's reference temperature, stated beside the result.
- Metals outside the 16-metal run (Cr in Ganley's table) leave the comparison.
- Electro limiting potentials: the published side puts the paper's adsorbate free energies on the app's scale with
  tabulated gas thermochemistry (the same equilibrium potentials the reaction files use), then applies the app's
  limiting-potential rule over the electron steps.

## Free-energy corrections for the new electrochemical intermediates (2026-10-25, 12:02 PDT, written before computing them)

- Convention: the computational hydrogen electrode's own, as in the corrections the app already uses (Nørskov et al.
  2004 and 2005, Rossmeisl et al. 2007, Skúlason et al. 2012): each intermediate's correction is its zero-point
  energy minus temperature times entropy at 298.15 K, less the same for the gas molecules its binding energy is set
  against (the worker's reference for it). Heat-capacity terms stay out, as in those papers.
- Adsorbate side: harmonic frequencies from UMA-S 1.1 (oc20 mode) of the adsorbate's atoms on Pt(111), on the
  lowest-energy site of the reaction-check run, with the metal atoms held still; central finite differences of
  0.01 Angstrom; any frequency below 50 cm-1 (or imaginary) raised to 50 cm-1; vibrational entropy of the harmonic
  oscillators.
- Gas side: zero-point energies from measured fundamental frequencies (NIST CCCBDB), standard entropies at 298.15 K
  and 1 bar (NIST-JANAF); water at 0.035 bar, its vapour pressure, so its free energy matches the liquid's.
- Check: the same calculation for H, O, OH and OOH against the published values the app uses or cites (H 0.24 eV,
  Nørskov 2005; O 0.05, OH 0.35 and OOH 0.40 eV, Rossmeisl 2007 from Nørskov 2004). Pass: each within 0.10 eV and
  the median miss within 0.05 eV. Skúlason 2012's constants for N, NH, NH2, N2H and NNH2 (0.26, 0.57, 0.88, 0.80,
  1.11 eV) are shown beside the result; their text leaves open whether a water term sits inside them, so they stay
  outside the pass test.
- Pass: the new intermediates take UMA's values, rounded to 0.01 eV, and the existing reactions keep theirs.
  Otherwise each reaction whose intermediates lack a published correction stays in the electro playground, with a
  row in the electro Methodology tab's limits.
