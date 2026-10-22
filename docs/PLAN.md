# Catalyst engine: plan

Source: *AI Catalyst Discovery Lab: Startup Analysis* (2026-10-21). This document is the
software plan for the company's computational layer. Updated 2026-10-21.

## The aim

Prove that AI finds much better catalysts, first for high-temperature SCR, then show the same
engine finds catalysts for other reactions faster each time.

## What the AlphaFold analogy requires

AlphaFold rested on three things:

| AlphaFold had | Catalysis equivalent | Status |
|---|---|---|
| A physics-grade structure model | Open universal atomistic models (UMA and peers) | Exists; download and fine-tune |
| The PDB: ~200k structures in one standard format | Measured performance under standard protocols, failures included, across reactions | Missing; this company generates it |
| CASP: blind, prospective benchmark | Pre-registered predictions scored against later lab results | Missing; built into the engine |

Catalyst performance depends on the recipe, the conditions and time on stream, not one
structure, which is why the dataset has to come from the company's own protocols. The engine's
job is to produce that dataset and that benchmark as a byproduct of each commercial campaign.

## Design rules

1. **A reaction is data.** Each reaction is a pack under `reactions/<id>/`: species, balanced
   pathways, rate-law forms, metrics, protocols, scenarios. The core never names a pack
   (enforced by `tests/test_packs.py`).
2. **One kinetics engine for any network.** Power-law, saturating (coverage), inhibited and
   equilibrium-limited rates, with thermodynamics from Cantera. The shared currency across
   reactions is per-pathway kinetics: `ln_k_ref`, `Ea`, adsorption constants, decay rates.
3. **Physics sits between the ML and the targets.** Models will predict kinetic parameters; the
   reactor model turns them into the ppm, slip and selectivity the gates are written in.
4. **Every concentration carries its basis** (wet, dry, O2-corrected).
5. **Data platform first**: append-only, provenance on every record, failures kept.
6. **Predictions are pre-registered**: hashed before the experiment, blinded arms, hit rates
   computed only from the log.
7. **Simulated data is quarantined** from real data and from any claim.
8. **Generality lives in the software; focus stays in the lab** (the Zymergen lesson).

## Proof sequence and how "works on other reactions" is measured

1. SCR: model-proposed formulations beat the reference catalyst (Gates 1 and 4).
2. Engine oxidation catalyst on the same rig and FTIR: the shared model reaches a hit in fewer
   experiments than SCR needed.
3. Electrocatalysis on public OCx24 data, retrospectively, before building an electrochemistry lab.
4. New assay module (electrochemical cell) for the expansion market.

Metrics, per reaction and across reactions:

- **Experiments to first hit** on a new reaction: shared model vs expert picks vs random.
- **Transfer gain**: for every new reaction, the log also records predictions from a model trained
  only on that reaction's data ("shadow model"). Both are scored against the same results, so
  cross-reaction learning is measured at no extra lab cost.
- **First-batch rank correlation**: how well the engine ranks candidates before any
  data from that reaction exists.
- **Calibration**: 90% intervals should contain about 90% of results, for every reaction.

Statistics: telling a 20% hit rate from a 5% one at alpha 0.05 and 80% power needs about
**76 formulations per arm**. Batch 1 (20-50 formulations) answers the kill question; the Gate 4
claim accumulates over about six batches of equal model and expert arms.

## Architecture

```
src/catalyst_engine/
  units.py            temperatures, concentration bases, ppm conversions
  packs.py            pack and protocol schemas, loader, protocol -> conditions, simulate
  chem/formula.py     element counts, molar masses
  chem/thermo.py      Cantera-backed K_eq, enthalpy, viscosity, diffusivity
  chem/network.py     reaction networks as data: validation (element balance), rate laws
  chem/reactor.py     plug-flow reactor: adaptive L-stable Rosenbrock, batched over conditions
  chem/metrics.py     conversion, outlet ppm on any basis, consumption ratios
  fit/kinetics.py     per-sample kinetic fits with uncertainty and identifiability flags
  fit/decay.py        aging fits, projections, extrapolation warnings
  value/              permits, tempering air, monolith mass transfer and pressure drop,
                      catalyst sizing, design scenarios
  cli.py              `ce` command
reactions/
  scr_hot/            pack, yardstick protocol, example turbine scenario
  oxidation_engine/   pack, screening protocol
```

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo, environment, tests | Done |
| 1 | Chemistry core, two packs, kinetic and aging fits, system value | Done (38 tests) |
| 2 | Data platform: formulations, samples, protocols, runs, raw files by hash, append-only; FTIR ingest with steady-state detection and checks; reference-run enforcement; hash-chained prediction log | Done 2026-10-21 (`src/catalyst_engine/data/`, `ce data`, the app's Lab data page) |
| 3 | Virtual rig: hidden "true" catalysts whose properties carry across reactions, producing FTIR-style files with noise, drift and failed runs | Started: simulated analyzer logs (bypass, settling holds, noise, failed and short runs) feed the demo store |
| 4 | Multi-task models (shared catalyst representation, a head per pathway), conformal calibration, batch planner scored through the reactor model, gate scorecards, transfer scorecard | |
| 5 | Atomistic descriptors (binding energies of N, O, NH3, CO and others on catalyst site types via UMA), Materials Project stability, OCx24 retrospective | Started: UMA test run done (see the app's Atomistic check page) |
| 6 | MCP server and agents, literature extraction into a low-trust tier, reports | Started: figure digitizer and the literature check (`experiments/literature_scr/`, the app's Literature check page) |

## Findings so far (from simulation with illustrative parameters)

- **The bench gas is set in permit units.** The analysis gives "15% O2" without a basis; read wet
  or dry it moves the 2 ppm limits by 11-55%. Decided 2026-10-21: dry, matching permits (40 CFR
  60.335) and the turbine NOx range, so a wet FTIR reading times 1.11 is ppmvd @ 15% O2.
- **Measure rate constants away from full conversion.** At 92% conversion a 0.5 ppm reading
  error moves k about 10%; near 63% about 5%. The yardstick adds a high space-velocity step.
- **The ammonia-oxidation step needs about 200 ppm NH3.** At 25 ppm the NO and N2O it forms sit
  near the FTIR noise floor and the fitter flags those pathways as unmeasurable; at 200 ppm all
  eight SCR parameters are recovered (N2O activation energy +/-6 instead of about +/-47 kJ/mol).
  To confirm on the real rig: first-order ammonia kinetics at 200 ppm.
- **Hot SCR on honeycombs is partly a mass-transfer problem.** At 1,150 F the NOx reaction is
  limited by diffusion to the wall while ammonia oxidation is not, so a catalyst needs a larger
  cut in ammonia oxidation than bench kinetics alone suggest, plus a tuned NH3/NOx ratio.
- **UMA ranks candidates well (atomistic check).** On nine metals it follows the textbook scaling
  rules and all 14 periodic-table orderings. Its materials mode matches measured NH3 binding on
  anatase (1.16 vs 1.23 ± 0.1 eV), and both modes rank the SCR acid sites correctly. Its NO
  energies on vanadia are artifacts. Details: the app's Atomistic check page.
- **The engine reproduces published zeolite SCR data (literature check, real measurements).** Digitized
  from Metkar's 2012 thesis (Fe-ZSM-5 and Cu-chabazite monoliths, 100-600 C), both catalysts fit the
  pack's own rate laws in the ideal reactor: NO conversion within 0.020 and 0.011 on average. Fitted below
  500 C, the model predicts 10 of 11 hotter points inside its 90% band, with central estimates on the
  optimistic side; the low-temperature fit leaves Fe-ZSM-5's Gate 1 call at 1,050 F open (0.88-0.96), and
  the full data settle it at 0.87-0.90. Apparent activation energies sit at 56-84% of the thesis's
  diffusion-resolved values. Details: the app's Literature check page.
- **Example scenario (all example inputs):** tempered design 58 m3 of catalyst, 34% tempering
  air, 623 kW of fans; hot target design 36 m3, no fans, more ammonia (ratio 1.30 vs 1.05).

## Open decisions and to-dos

- Bench feed O2 basis: dry (decided 2026-10-21; the catalysis lead can revisit before batch 1).
- Design space velocities from the reference catalyst's commissioning run.
- UMA access granted (2026-10-21). Have counsel read its Acceptable Use Policy line on
  "operation of critical infrastructure"; the atomistic backend stays swappable.
- Materials Project API key: saved 2026-10-21 in `~/.config/.pmgrc.yaml`, outside the repository.
- The catalysis lead (the founder, from 2026-10-21) reviews pack parameters, protocols and the starting design space.
- Rig logger export: confirm it can write the analyzer CSV the Lab data page ingests (or adapt the reader to it).
- Lab store backup: `data/lab/` lives outside git on this Mac; choose where it is backed up before batch 1.

## Deliberately not built

Foundation models of our own, a DFT engine, robot drivers before hardware is chosen, and a generic
all-domain platform. (A local app, `uv run ce ui`, was added on request so the project can be followed
without reading the code.)
