# catalyst-engine

- `docs/status.yaml` feeds the app's Project status page (recent work, next steps, phases, findings, decisions).
  Update it at the end of every piece of work, so the user can follow progress in the app.
- The engine core (`src/catalyst_engine/`, including `ui/`) never names a reaction pack;
  `tests/test_packs.py::test_core_never_names_a_pack` enforces this. Reaction knowledge goes in `reactions/`.
- App copy is affirmative: state what is true, with no negation words.
- Model checks live in `experiments/<name>/`: scripts, `results/`, and a `report.yaml` (the write-up) that
  the app renders with `results/summary.json`: the Model accuracy page by default, the Literature check page
  for `kind: literature`, the Manual's Methodology tab for `kind: methodology` (`experiments/design_method/`, whose
  text fills `{name}` from the summary's `values`). Put results there, not only in a README. Rerun
  `experiments/design_method/` after changing how the design worker builds surfaces or how screens predict.
- The repository is public. Published papers, theses and figure overlays stay in `experiments/<name>/cache/`
  (git-ignored); the repository holds the digitized numbers, the scripts and the citations.
- Materials Project API key: `~/.config/.pmgrc.yaml` (`PMG_MAPI_KEY`, read by pymatgen and mp-api). Keep it out of
  the repository and out of logs.
- Lab data lives in `data/lab` (real) and `data/simulated` (demo), both outside git; `CE_DATA_DIR` moves them
  and the tests point it at a temp folder. Hit rates and gate claims come from the lab store only. Rebuild the
  demo with `uv run ce data demo` after changing `reactions/<pack>/demo.yaml`.
- Commit only when asked.
- UMA (fairchem-core 2.23) lives in the conda Python at `/opt/miniconda3/bin/python3`, separate from this
  project's Python 3.12 venv. Checkpoint: `~/uma/checkpoints/uma-s-1p1.pt`. Run it with
  `TORCHDYNAMO_DISABLE=1` until the Xcode command-line tools' C++ headers are repaired.
- Catalyst design lives in `src/catalyst_engine/design/`. The UMA worker (`uma_worker.py`) runs as a subprocess in
  the conda Python (`CE_UMA_PYTHON` and `CE_UMA_CHECKPOINT` override the defaults) and writes one folder per screen
  in `data/screens/`. Reactions added in the app go to `data/reactions/`, and Materials Project structures are cached
  in `data/materials/`. A finished example screen ships in `examples/screens/` and is copied into an empty
  `data/screens/` on app start.
- The Manual page reads `docs/manual.yaml`; its step screenshots are `docs/manual/step-*.png`. Retake them when a
  page they show changes.
