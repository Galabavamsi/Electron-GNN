# V4 Failed Experiment Archive

This folder contains the complete V4 verifier/refiner experiment history, preserved for traceability but excluded from the active production workflow.

## Why Archived

V4 did not outperform V3 hybrid reliably on the current tiny dataset and introduced significant decode/refinement instability.

## What Is Stored Here

- `evaluate_v4.py`
- `scripts/`:
  - `evaluate_v4.py`
  - `calibrate_v4_threshold.py`
  - `generate_v4_plots.py`
  - `compare_v3_v4.py`
- `train/`:
  - `train_v4.py`
  - `losses_v4.py`
- `utils/`:
  - `physics_verifiers.py`
  - `rational_refiner.py`
- `checkpoints/`:
  - V4 model checkpoints and smoke-test checkpoints
- `results/`:
  - V4 logs, threshold sweeps, and generated plots

## Active Production Path

Use V3 only:
- training: `train/train_v3_two_tower.py`
- inference: `utils/hybrid_inference.py`
- dashboard: `dashboard/app.py`

The active V3 recommendation file remains at:
- `results/model_selection.json`
