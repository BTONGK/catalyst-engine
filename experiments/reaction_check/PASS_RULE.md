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

## Batch 2 (2026-10-25, 20:03 PDT)

Written after the UMA run for six new carbon intermediates (runs/metals-electro2: CHO, COH, CH2O, CHOH, CH3O and
CH2OH on the same 16 metals, oc20 mode, standard setting) finished, and before any of its binding energies was read,
compared or used in a prediction; the ten new thermal packs had no files yet.

Thermal:

- The ten new packs follow the thermal rule above as written. The sources reached before their predictions give
  no per-metal table measured in one laboratory for any of them, so each stays in the Reactions library and the
  playground, with a row in the Methodology tab's limits. A table found later joins through data/published.yaml
  under this same rule.

Electrochemical:

- CHO*, COH*, CH2OH*, CHOH* and CH3O* are each compared with Ferrin and Mavrikakis 2009, Table S3 of the
  supporting information (binding energies on the (111) faces of Rh, Ir, Ni, Pd, Pt, Cu, Ag and Au at 1/9
  monolayer, against the gas-phase species), by the intermediate rule above. Its one constant offset per
  intermediate absorbs the paper's gas references, its zero-point and entropy terms and the worker's references,
  each the same for every metal. Those five columns enter alone: the paper's CO, COOH, HCOO and OH columns stay
  outside, so the intermediates checked earlier keep their tables and verdicts (CO* misses against Elnabawy 2018).
- CH2O* has no table among the sources reached; a reaction through it stays in the playground.
- Ranking: the published side needs each paper's free energies on a scale tied to the reaction's gases. Ferrin
  2009's table sets each binding energy against a gas-phase radical whose energy in that paper's calculation sits
  outside the sources reached, so a reaction through these five intermediates awaits a published free-energy table
  for its ranking and stays in the playground. Tabulated radical thermochemistry would mix measured gas energies
  with the paper's calculated bonds, so it stays out of the published side.
- A reaction through OOH* or OH* alone (hydrogen peroxide oxidation, two-electron water oxidation) is ranked
  against Nørskov et al. 2004, Table 2, the way the two hydrogen peroxide reactions are.
- Two-electron water oxidation runs above 1.7 V, where the metals of the run turn to oxides. Like oxygen
  evolution, it stays in the playground until an oxide check passes, whatever its ranking gives.
- Corrections: a published zero-point and entropy correction enters where one source tabulates both the
  adsorbate's terms and the gas molecules' from calculated vibrations, so the correction against the worker's
  reference comes from that one source. COOH* and CO* take Peterson et al. 2010's Cu(211) values as Hansen et al.
  2013, Table S1, reproduces them, at that table's 18.5 °C, heat-capacity terms set aside as in the convention
  above; CO2 reduction to CO takes them too. OCHO* still awaits a published value. The six new carbon
  intermediates take UMA's own vibrations on Pt(111) by the method above, which serve the playground's starting
  point only.
- The ten electrochemical reactions of this batch (20:16 PDT, written before any of them had a file or a
  prediction): CO2 reduction to methane and to methanol, CO reduction to methane and to methanol, CO oxidation,
  methanol oxidation to CO2 and to formic acid, formic acid oxidation through carboxyl (COOH*), hydrogen peroxide
  oxidation, and two-electron water oxidation to hydrogen peroxide. The carboxyl route of formic acid oxidation
  ranks against Elnabawy et al. 2018's COOH* column, moved to dissolved formic acid as the formate route's is.
  Formaldehyde oxidation waits: the free energy of dissolved formaldehyde behind the tabulated standard potential
  of formaldehyde to methanol (0.232 V, so about -130.5 kJ/mol with methanol's -175.31) sits about 7.5 kJ/mol below
  the value the gas's free energy and its measured Henry's-law constant give, and the reaction joins once one value
  is settled.

## Batch 2: two tables found before the thermal predictions (2026-10-25, 20:52 PDT)

Written before any of the ten new thermal packs had a file or a prediction, and before the bar heights below were
measured. A search after the 20:03 note reached one laboratory's measurements for two packs of this batch:

- Propane combustion: G. Lee, M.Ch.E. thesis, University of Delaware, 2017 (advisor D. G. Vlachos), Figure D-1(c):
  turnover frequencies of propane total oxidation on seven alumina-supported metals (Ag, Co, Cu, Ni, Pd, Pt, Rh),
  0.5 % propane at an equivalence ratio of 0.5 (ten oxygen molecules per propane) in helium at 1 atm. The check
  reads the 340 °C bars, the temperature of the thesis's own volcano figures (Figures 3-1 and 3-2).
- Propylene combustion: the same thesis, Figure 4-4: turnover frequencies of propylene total oxidation on the same
  seven metals, 1 kPa propylene with 8 kPa oxygen at 1 atm. The check reads the 170 °C bars, the highest
  temperature shown, where every metal's rate is largest.
- Reading: both figures are bar charts with one fill colour per metal (the figures' legends). Each bar's height is
  measured from the thesis PDF rendered at 600 dpi, from the bar's fill colour, against the plot frame, rounded to
  0.01 s-1; bars that read the same at that resolution share a rank. The numbers go into data/published.yaml before
  either pack's prediction exists.
- Both tables enter as primary tables under the thermal rule above as written. Platinum is the most active metal in
  both figures, so it is each pack's catalyst to beat and leaves the scoring; the six other metals are scored.
- Descriptor for both packs: the average of the C and O binding energies. The thesis finds both binding energies
  set the rate (copper and platinum hold oxygen alike and differ through carbon), so the descriptor follows that
  finding; it was chosen from that statement, before any of UMA's numbers for these packs were computed.
- A master's thesis is a public record from one laboratory, examined by its committee and outside journal review;
  the Methodology tab says so beside the result.
- The thesis's methane panel (Figure D-1(a)) puts platinum ahead of palladium at four oxygen molecules per methane,
  and palladium ahead in fuel-rich feeds. The methane combustion pack (batch 1) keeps its top at palladium, the
  catalyst of lean-burn gas engines, and its row in the limits records this measurement.
- Clarification (20:53 PDT, before any bar was measured): both figures sit in the PDF as 150-dpi raster images, so
  rendering the page at 600 dpi adds no detail. Each bar is read at the image's own pixels instead: the height of
  the bar's top above the x axis, as a share of the frame's height, times the axis's top value. Bars whose tops sit
  within one native pixel of each other share a rank.

## Where each verdict shows (2026-10-26, written after the results, on the user's choice)

- Every reaction now appears on its Design page, with its verdict beside it in the list, under the choice, and above
  each screen's results: "checked" for a reaction that passes the rule above, "missed its check" for one that misses
  it, and "awaiting its check" for one whose measured table, intermediate table or published correction is still to
  come. This replaces the placement lines above (a pass joins the Design page, and the rest stay in the playground).
- The rule, its thresholds and every score stay as written above; only the place a reaction shows moves, so every
  verdict stays as the rule gave it. Each reaction file carries its verdict (check) and what the check found
  (check_note), and tests/test_checks.py holds both to results/summary.json.
- Electrochemical verdicts: a reaction misses when an intermediate on its path misses its table, when its ranking is
  scored on 6 or more metals and misses, or when it runs where the metals turn to oxides. It awaits when what remains
  is a table or a published correction still to come. Each intermediate whose correction comes from UMA's own
  vibrations carries correction_from: uma in its file, and the results page names it.
- Hydrogen evolution and oxygen reduction take "checked" from experiments/electro_check; oxygen evolution takes
  "missed its check" from the oxide check.
- Each rho now keeps six decimals until a page rounds it. Pass decisions always used the exact value; four displayed
  values move by 0.01: ammonia cracking 0.91 to 0.90, hydrogen peroxide production and oxidation 0.84 to 0.85, and
  CO electro-oxidation 0.96 to 0.97.

## Hydrogen peroxide decomposition (2026-10-31, 14:25 PDT, written before the package exists and before any prediction)

The user asked for H2O2 decomposition, 2 H2O2 -> 2 H2O + O2. No electrode drives it, so it joins the thermal
section as a package for the vapor: hydrogen peroxide in air, as in the exhaust of peroxide sterilizers.

- Table: D. W. McKee, Catalytic decomposition of hydrogen peroxide by metals and alloys of the platinum group,
  J. Catal. 14, 355 (1969), doi 10.1016/0021-9517(69)90326-1: the specific activity per unit surface area of the
  platinum-group metals for decomposing hydrogen peroxide in neutral water at 27 °C falls in the order
  Pt > Os > Ir > Pd > Ru > Rh. One laboratory, six metals, one set of conditions.
- Reading: the order comes from the publisher's abstract, as two web searches quote it. The full paper sits behind
  the publisher's access wall (ScienceDirect refused a fetch, and the Crossref, OpenAlex and Semantic Scholar records
  carry no abstract), so the table enters as an order (1 = most active), as Grenoble's water-gas shift order does,
  at 300.15 K.
- It enters as a primary table under the thermal rule as written. Platinum, the most active, is the package's
  catalyst to beat and leaves the scoring, so five metals are scored, the rule's minimum. At five metals a p of 0.05
  or less needs rho of 0.9 or more: at most one swap of neighbours.
- Descriptor: oxygen binding, {O: 1}, from Rankin and Greeley (ACS Catal. 2, 2664, 2012), who sort transition
  metals for hydrogen peroxide with the oxygen binding energy, and Ford et al. (Surf. Sci. 604, 1565, 2010), whose
  O-O scission barriers for H2O2 on twelve metals follow it. The top sits at platinum's value. The weak side keeps
  the generic 0.9 eV per eV: H2O2 splits into two OH, and OH binding moves half as fast as oxygen's (Abild-Pedersen
  et al., Phys. Rev. Lett. 99, 016105, 2007), so the splitting energy moves about one eV per eV of the descriptor,
  and the barrier 0.9 of that (Nørskov et al., J. Catal. 2002). The strong side keeps the generic 1.0. The top
  follows oxygen ({O2: 0.5}), as in the other oxidation packages.
- The package screens vapor in air; the table is measured in water, whose hydrogen bonds hold OH* more firmly and
  may move the order. The note beside the verdict says so. Whatever the ranking gives, the descriptor, the top and
  the slopes stay as written here.
