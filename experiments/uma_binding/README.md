# UMA test run: does the atomistic model track known catalyst behavior?

A first check of Meta's UMA model (`uma-s-1p1`, fairchem-core 2.23) before the atomistic layer
(Phase 5) is built on it. Two questions:

1. On metal surfaces, where UMA's catalysis mode was trained (OC20), does it reproduce the
   physics catalysis relies on?
2. On the oxides that make up commercial SCR catalysts (anatase TiO2 and V2O5), outside its
   training data, does it match experiment, and which UMA mode works better?

## Results

Open the app (`uv run ce ui`) and go to **Atomistic check**. The page shows the verdict, every
table and chart, the method and the sources. Its text lives in `report.yaml`; its numbers in
`results/summary.json`, which `analyze.py` writes. `results/summary.md` holds the same tables as
plain markdown.

## How to rerun

```sh
export TORCHDYNAMO_DISABLE=1   # until the Xcode command-line tools' C++ headers are repaired
PY=/opt/miniconda3/bin/python3
cd experiments/uma_binding
$PY metals.py --out results/metals_all.csv          # 9 metals x 7 fragments, catalysis mode
$PY oxides.py --task oc20 --out results/oxides_oc20.csv
$PY oxides.py --task omat --out results/oxides_omat.csv
$PY gas.py --out results/gas.csv                    # each mode's gas molecules vs references
$PY lattice.py --out results/lattice.csv            # each mode's bulk lattice vs experiment
$PY analyze.py                                      # writes results/summary.json and summary.md
```

The metal run here was split across three processes (`--metals Pt,Pd,Rh` and so on), which is
why `results/` holds three metal CSVs; `analyze.py` reads every `metals_*.csv`.

## Files

| File | What it does |
|---|---|
| `metals.py` | N, NH, NH2, NH3, O, OH, NO on nine close-packed metals |
| `oxides.py` | NH3 and NO on anatase TiO2(101) and V2O5, both UMA modes |
| `gas.py`, `lattice.py` | Reference checks: gas molecules and bulk lattices in each mode |
| `analyze.py` | Compares everything with known values; writes the summaries |
| `report.yaml` | The write-up shown in the app |
| `structures/` | Crystal structures from the Crystallography Open Database |
| `results/structures/` | Relaxed structures (extended XYZ; open with `ase gui`) |
