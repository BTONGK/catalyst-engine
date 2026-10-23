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
- The worker relaxes with ASE's BFGS after a 0.03 Å nudge (`nudge`). Plain LBFGS stays balanced on a symmetric
  saddle, as the oc20 mode's clean bcc Fe slab showed. Each binding energy is set against the lower of the clean
  slab and the bare surface left under the molecule (`clean_reference`).
- The Design page's Advanced switch (`design_advanced`) exposes the catalyst form (powder or coated honeycomb, as
  `conditions["shape"]`; `screening.wall_transfer` feeds the reactor's film mass transfer), faces, the UMA mode
  (oc20 or omat), the catalyst to beat and the slab size; the simple form stays the default.
- The user ranks accuracy above features: a model option joins the app only after a check shows it reproduces
  published DFT or measurements within the error bar. What stays out, and why, goes in the Methodology tab's
  `limits` section (supports, promoters, zeolites, coat diffusion so far).
- Zeolites stay out of the app: `experiments/zeolite_check/` shows UMA's three modes miss published DFT for NH3 in
  Cu-chabazite by 0.38 eV or more. `design/zeolites.py` (numpy and ASE only, so the UMA Python loads it) and the
  worker's `framework` branch remain for that check and for a later model.
- Oxides: `experiments/oxide_check/` scores UMA modes against OC22's DFT (data in its git-ignored `cache/`, read by
  `extract_oc22.py` through a restricted unpickler) and measured NH3 on anatase (101). The materials mode (omat) is the
  app's advice for oxides; every mode sits beyond the error bar, so supports and promoters stay out. UMA-S 1.2.1
  (`~/uma/checkpoints/uma-s-1p2p1.pt`) serves that check only: its gas molecules sit 0.2–1.6 eV off its surfaces.
  The worker's surface cut gives anatase (101) a singly bonded top O; the check builds the true face itself.
- The Manual page reads `docs/manual.yaml`; its step screenshots are `docs/manual/step-*.png`. Retake them when a
  page they show changes.
