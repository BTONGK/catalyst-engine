# catalyst-engine

- `docs/status.yaml` feeds the app's Overview page (recent work, next steps, phases, findings, decisions).
  Update it at the end of every piece of work, so the user can follow progress in the app.
- The engine core (`src/catalyst_engine/`, including `ui/`) never names a reaction pack;
  `tests/test_packs.py::test_core_never_names_a_pack` enforces this. Reaction knowledge goes in `reactions/`.
- App copy is affirmative: state what is true, with no negation words.
- Model checks live in `experiments/<name>/`: scripts, `results/`, and a `report.yaml` (the write-up) that
  the app's Atomistic check page renders with `results/summary.json`. Put results there, not only in a README.
- Materials Project API key: `~/.config/.pmgrc.yaml` (`PMG_MAPI_KEY`, read by pymatgen and mp-api). Keep it out of
  the repository and out of logs.
- Commit only when asked.
- UMA (fairchem-core 2.23) lives in the conda Python at `/opt/miniconda3/bin/python3`, separate from this
  project's Python 3.12 venv. Checkpoint: `~/uma/checkpoints/uma-s-1p1.pt`. Run it with
  `TORCHDYNAMO_DISABLE=1` until the Xcode command-line tools' C++ headers are repaired.
