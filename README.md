# Confidence-guided Discovery Loop for REE

Code for a confidence-guided, multi-objective active-learning workflow for rare-earth-element (REE) separation. It jointly optimizes Nd recovery, Fe remaining, separation factor, and processing time.

This public version contains the **main confidence-guided workflow only**. The published entry point applies conformal-prediction bounds and dynamic safety gates before acquisition. Ablation scripts, ablation results, experimental data, and generated figures are intentionally excluded.

## Setup and run

Python 3.10 is recommended. Install dependencies, then provide your input workbook at `data/round1.xlsx` (or change `DATA_PATH` in `src/chem_active_learning/config.py`). The workbook must include `Con_pH`, `Con_time (h)`, `Nd_Recovery`, and `Fe_Remaining` columns.

```bash
pip install -r requirements.txt
python run_optimization.py
```

Results, plots, and the safe-region prediction table are written to `optimization_plots/`.

## Reproducibility settings

The main run calls `run_optimization_once(..., seed=100)`. Model initialization and training use deterministic PyTorch settings and these ensemble seeds:

| Item | Value |
| --- | --- |
| Ensemble size | 5 models per target |
| Base training seed | 20260403 |
| Nd member seeds | 20261403–20261407 |
| Fe member seeds | 20262403–20262407 |
| Conformal miscoverage (`alpha`) | 0.10 |
| Candidate batch size | 4 |
| pH / time search range | 3.0–6.0 / 2–16 h |
| Nd training | 500 epochs, lr 0.02 |
| Fe training | 3000 epochs, lr 0.0025 |

Dynamic gates are applied in every run. At round `r`, the Nd gate is `min(100 - 25 exp(-1.6r), 96)` %, and the Fe gate is `max(0.5 + 9.5 exp(-2r), 0.9)` %. No relaxed fallback is used if no safe candidate passes both conformal bounds.

## Repository layout

- `run_optimization.py` — main entry point.
- `src/chem_active_learning/` — configuration, preprocessing, DKL ensembles, conformal prediction, acquisition, and visualization.
- `scripts/` — optional data-preparation and plotting helpers.

## Notes

The repository is code-only. Do not commit experimental data or generated result files; both are ignored by Git.
