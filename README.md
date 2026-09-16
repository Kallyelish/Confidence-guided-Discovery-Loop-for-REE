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


## Repository layout

- `run_optimization.py` — main entry point.
- `src/chem_active_learning/` — configuration, preprocessing, DKL ensembles, conformal prediction, acquisition, and visualization.
- `scripts/` — optional data-preparation and plotting helpers.


