# catalyst-engine

The computational layer of a closed-loop catalyst discovery company: reaction packs, a
general kinetics engine, kinetic and aging fits, and system-value calculations. The plan
and its reasoning are in [docs/PLAN.md](docs/PLAN.md).

## The app

```sh
uv sync
uv run ce ui                      # opens http://localhost:8501
```

Seven pages: Overview (progress, next steps and decisions from `docs/status.yaml`), Atomistic check
(model checks from `experiments/*/report.yaml`), Reactions, Catalyst playground, Design scenarios,
Calculators and Fit lab. Every page reads the files on disk, so a new pack, an edited YAML file or a
new experiment summary shows up on the next page load.

## Command line

```sh
uv run pytest                     # 47 tests, ~20 s
uv run ce packs                   # reaction packs and whether each loads
uv run ce show scr_hot            # pathways, metrics, protocols, open questions
uv run ce simulate scr_hot        # run the yardstick protocol through the reactor model
uv run ce scenario reactions/scr_hot/scenarios/example_frame_turbine.yaml
uv run ce tempering --exhaust-f 1150
uv run ce permit --ppm 2 --cap-tpy 100
uv run ce fit scr_hot bench.csv --out fitted.yaml    # CSV: step, T_F, out_<species>_ppm, ...
uv run ce decay aging.csv                            # CSV: hours, activity[, sigma]
```

All rate parameters in the packs are illustrative until the catalysis lead's review and the rig replace them.
Example scenarios use example inputs throughout.

## Adding a reaction

Create `reactions/<id>/pack.yaml` and at least one `reactions/<id>/protocols/*.yaml`; no engine
code changes. Species name their Cantera id when it differs from the formula. Pathways are
balanced equations (`->`, or `<=>` for equilibrium-limited) with a rate law:

```yaml
pathways:
  scr:
    equation: 4 NH3 + 4 NO + O2 -> 4 N2 + 6 H2O
    role: desired
    rate:
      key: NO                 # the law gives this species' consumption rate
      orders: {NO: 1}
      coverage: {NH3: {ln_K_ref: 10.13, dH_kJ: -100}}
      ln_k_ref: 5.99          # at the pack's reference temperature
      Ea_kJ: 50
```

The loader checks element balance, species in the thermo database, metric species and protocol
feeds. `uv run pytest tests/test_packs.py` runs every pack through every protocol.

## Conventions

- GHSV is normal m3 of gas per m3 of catalyst per hour (0 C, 1 atm).
- Rates are per m3 of catalyst with concentrations in mol/m3; first-order k is in 1/s.
- Parameters are named `<pathway>.ln_k_ref`, `<pathway>.Ea_kJ`, `<pathway>.cov.<species>.ln_K_ref`.
- Analyzer readings are wet ppm; metrics state their basis (`ppmvd @ 15% O2` for permits).
- Lab data never lives in git (`data/` is ignored).
