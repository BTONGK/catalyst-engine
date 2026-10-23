# catalyst-engine

The computational layer of a closed-loop catalyst discovery company: reaction packs, a
general kinetics engine, kinetic and aging fits, and system-value calculations. The plan
and its reasoning are in [docs/PLAN.md](docs/PLAN.md).

## The app

```sh
uv sync
uv run ce ui                      # opens http://localhost:8501
```

The app designs catalysts on the computer. Pick a reaction and its conditions, choose materials (pure metals,
or alloys and oxides from the Materials Project), and a background run computes each surface's binding energies
with Meta FAIR Chemistry's UMA model. The app then ranks the materials against the catalyst to beat on the
reaction's Sabatier volcano, runs each one through the reactor model across the chosen temperatures, and shows
the surfaces in 3D. The Manual page (from `docs/manual.yaml`, with screenshots in `docs/manual/`) walks through
every step, has a plain-language primer on catalysis, and a Methodology tab: a method paper that states every assumption and the check behind each step (`experiments/design_method/`).

Pages, as grouped in the sidebar:

- **Start:** Home, Manual.
- **Design:** Design a catalyst (reaction, conditions, materials, run), Results (ranking, performance curves,
  volcano, 3D structures, CSV downloads), Materials (search the Materials Project or browse pure metals in 3D),
  Reactions (the library, a form that adds a new reaction, and each reaction's full detail).
- **Simulate:** Catalyst playground, Design scenarios, Calculators, Fit lab.
- **Check accuracy:** Model accuracy and Literature check (model checks from `experiments/*/report.yaml`,
  sorted by the report's `kind`).
- **Lab (optional):** Lab data (catalysts, runs, campaign scores and the prediction log).
- **Project:** Project status (progress, next steps and decisions from `docs/status.yaml`).

Every page reads the files on disk, so a new pack, an edited YAML file or a new experiment summary shows up on
the next page load.

### What a design run needs

- **UMA**, in its own Python: fairchem-core 2.23 and the `uma-s-1p1` checkpoint. The app looks for
  `/opt/miniconda3/bin/python3` and `~/uma/checkpoints/uma-s-1p1.pt`; set `CE_UMA_PYTHON` and
  `CE_UMA_CHECKPOINT` to point elsewhere. The worker (`src/catalyst_engine/design/uma_worker.py`) runs there as
  a separate process.
- **A Materials Project API key** (free, materialsproject.org) for alloys and oxides, read from `MP_API_KEY`,
  `PMG_MAPI_KEY` or `~/.config/.pmgrc.yaml`. Pure metals work right away.

Each screen is a folder in `data/screens/<id>/`: `job.json` (the request), `status.json`, `results.jsonl`
(every binding energy), `structures/*.cif` and `log.txt`. A finished example screen ships in
`examples/screens/` and lands in an empty `data/screens/` when the app opens. Reactions added in the app live
in `data/reactions/`, and fetched crystal structures are cached in `data/materials/`.

## Command line

```sh
uv run pytest                     # 111 tests, ~1 min
uv run ce packs                   # reaction packs and whether each loads
uv run ce show scr_hot            # pathways, metrics, protocols, open questions
uv run ce simulate scr_hot        # run the yardstick protocol through the reactor model
uv run ce scenario reactions/scr_hot/scenarios/example_frame_turbine.yaml
uv run ce tempering --exhaust-f 1150
uv run ce permit --ppm 2 --cap-tpy 100
uv run ce fit scr_hot bench.csv --out fitted.yaml    # CSV: step, T_F, out_<species>_ppm, ...
uv run ce decay aging.csv                            # CSV: hours, activity[, sigma]
uv run --with pymupdf python experiments/literature_scr/digitize.py   # published figures -> data/*.csv (sources: cache/)
uv run python experiments/literature_scr/replicate.py                 # fit them (~20 min), write results/summary.json
```

## Lab data and the prediction log

```sh
uv run ce data init                                  # the lab store, in data/lab (outside git)
uv run ce data add-formulation "Reference" --role reference --component Fe:active:1.5:wt% --component BEA:support:balance:
uv run ce data add-sample F-0001 --volume-ml 10
uv run ce data add-campaign "Batch 1" --pack scr_hot --rig bench-1 --reference F-0001 \
    --metric nox_conversion --step kinetic --temperature 1050 --temperature 1150 --margin 0.03
uv run ce data pick C-0001 F-0002 --arm model        # logged before the catalyst is made
uv run ce data ingest run.csv --campaign C-0001 --sample S-0002
uv run ce data score C-0001                          # hit rates per arm, once `ce data unblind C-0001`
uv run ce data verify                                # hash chain and raw files
uv run ce data anchor                                # head hash into docs/prediction_anchors.md; commit it
uv run ce data demo                                  # rebuild the simulated demo store
```

Every table is append-only; a correction is a new record that `supersedes` the old one. Raw analyzer
files are stored under their SHA-256 and set read-only. A campaign fixes the protocol version (by hash),
the rig, the reference and the hit rule before any pick, and refuses runs that differ. The lab store
refuses files with the simulated marker, so simulated data stays in `data/simulated`.

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

A `screening` block makes the reaction designable: the adsorbates whose binding energy ranks materials, the
pathways that follow it, the metric, the catalyst to beat and where the volcano's top sits relative to it:

```yaml
screening:
  descriptor: {O: 1}          # weights; {C: 0.5, O: 0.5} averages two binding energies
  pathways: [co_ox]
  metric: co_conversion
  reference: {formula: Pt, name: Platinum}
  optimum_shift_eV: 0.2       # the top binds 0.2 eV more weakly than the reference
  peak_gas: {O2: 0.5}         # the gas each O comes from; the top follows its free energy with temperature
  suggested_elements: [Pt, Pd, Rh, Ir, Ru, Au, Ag, Cu, Ni, Co]
```

The loader checks element balance, species in the thermo database, metric species and protocol
feeds. `uv run pytest tests/test_packs.py` runs every pack through every protocol.

The Reactions page's "Add a reaction" tab writes such a pack from a short form (`design/builder.py`) and
calibrates its rate constant so the catalyst to beat reaches half the best possible result at a temperature
the user names.

## Conventions

- GHSV is normal m3 of gas per m3 of catalyst per hour (0 C, 1 atm).
- Rates are per m3 of catalyst with concentrations in mol/m3; first-order k is in 1/s.
- Parameters are named `<pathway>.ln_k_ref`, `<pathway>.Ea_kJ`, `<pathway>.cov.<species>.ln_K_ref`.
- Analyzer readings are wet ppm; metrics state their basis (`ppmvd @ 15% O2` for permits).
- Lab data, screens and reactions built in the app stay outside git (`data/` is ignored).
