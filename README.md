# Confidence-guided Discovery Loop for REE

Code for a confidence-guided, multi-objective active-learning workflow for rare-earth-element (REE) separation. It jointly optimizes Nd recovery, Fe remaining, separation factor (SF), and processing time over a pH–time design space.

Each round, the workflow (1) trains deep-kernel Gaussian-process (DKL) ensembles for Nd recovery and Fe remaining, (2) calibrates their uncertainty with conformal prediction, (3) screens the candidate grid with round-dependent safety gates, and (4) selects the next experiments with q-Expected Hypervolume Improvement (qEHVI). A complete description of the method is in the manuscript.


## 1. System requirements

### Software dependencies

| Package | Version |
|---|---|
| Python | 3.10.12 |
| torch | 2.5.0+cu121 |
| botorch | 0.15.1 |
| gpytorch | 1.14.0 |
| numpy | 2.1.2 |
| pandas | 2.2.3 |
| scipy | 1.15.2 |
| openpyxl | 3.1.5 |
| matplotlib | >=3.9, <4 |
| scikit-learn | >=1.5, <2 |
| tqdm | >=4.66 (optional; progress bars only) |

All dependencies are listed in `requirements.txt`.

### Operating systems and tested versions

- Tested on: Ubuntu 22.04 (running under WSL2 on Windows 11), Python 3.10.12 with the package versions above.
- Native Windows and macOS are expected to work (pure Python; no OS-specific code) but have not been tested.
- No non-standard hardware is required.

## 2. Installation guide

```bash
git clone https://github.com/Kallyelish/Confidence-guided-Discovery-Loop-for-REE.git
cd Confidence-guided-Discovery-Loop-for-REE

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

CPU-only alternative (smaller download): replace the `torch` line in `requirements.txt` with `torch==2.5.0` and remove the `--extra-index-url` line, or run
`pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu` before installing the remaining packages.

**Typical install time on a normal desktop computer** (≈ 8 GB RAM, broadband connection): about 10–15 minutes with the build of PyTorch (≈ 2.5 GB download). Install time is dominated by the PyTorch download.

## 3. Demo

The real experimental data of the manuscript are provided in `data/` (`round1.xlsx`, `round2.xlsx`, `round3.xlsx`, one file per active-learning round).

1. Select the round to run in `src/chem_active_learning/config.py`. For the first round:
   ```python
   DATA_PATH = os.path.join("data", "round1.xlsx")   # default
   ROUND = 0
   ```
   For the next rounds use `round2.xlsx` with `ROUND = 1`, and `round3.xlsx` with `ROUND = 2`.
2. Run from the repository root:
   ```bash
   python run_optimization.py
   ```

**Expected output** (written to `optimization_plots/`):

- Console log: number of loaded data points, ensemble training progress, the round's dynamic Nd/Fe gate values, and a table of recommended experiments (pH, time, predicted Nd recovery, predicted Fe remaining, SF, acquisition score).
- `Full_Grid_Optimization_Results.xlsx` / `.csv` — predicted Nd recovery, Fe remaining, SF and log SF for every grid point inside the safe region (the points that pass the confidence gates).
- `Optimization_Tracking_Data.csv` — experimental and predicted values of the training points plus the recommended candidates, accumulated across rounds; `Surface_Contour_Data.csv` — predicted Nd/Fe/SF values over the full pH–time grid.
- Figures: `3D_Surface_{Nd,Fe,SF}.png`, `2D_Contour_{Nd,Fe,SF}.svg`, `Decision_Space_*.png`, `Pareto_Front_3D.png`, `TSNE_Nd_Latent.png`, `Training_Loss_Curve.png`, `Perf_Combined_SF_Nd_Fe.png`.

**Expected run time for the demo** on a normal desktop computer: about 8 minutes per round (training 5 Nd models × 500 epochs and 5 Fe models × 3000 epochs dominates). Training uses fixed seeds, so results are reproducible on the same software stack.

## 4. Instructions for use

### Run on your own data

1. Prepare an Excel workbook with one row per experiment and these columns:

   | Column | Meaning |
   |---|---|
   | `Con_pH` | pH of the experiment |
   | `Con_time (h)` | Reaction time (h) |
   | `Nd_Recovery` | Nd recovery (%) |
   | `Fe_Remaining` | Fe remaining (%) |

2. Set `DATA_PATH` in `src/chem_active_learning/config.py` to your workbook (default: `data/round1.xlsx`).
3. Set `ROUND` in `config.py` to the current active-learning round (starts at 0). The Nd gate becomes stricter and the Fe gate lower as `ROUND` increases.
4. Run `python run_optimization.py`.
5. Append the measured results to the workbook, increase `ROUND` by 1, and repeat.

If the workbook cannot be read, the script falls back to synthetic data and prints a warning; check the console output to make sure your data were loaded.

### Key settings (`src/chem_active_learning/config.py`)

| Setting | Default | Description |
|---|---|---|
| `PH_MIN`, `PH_MAX` | 3.0, 6.0 | pH search range |
| `T_MIN_H`, `T_MAX_H` | 2.0, 16.0 | Reaction-time range (h) |
| `Q_CANDIDATES` | 4 | Number of candidates |
| `CONFORMAL_ALPHA` | 0.1 | Conformal miscoverage level (90 % coverage) |
| `ENSEMBLE_SIZE` | 5 | Models per ensemble |

### Reproducing the results

To reproduce the multi-round results in the manuscript, run the workflow once per round with the corresponding data file and `ROUND` value (`round1.xlsx` → 0, `round2.xlsx` → 1, `round3.xlsx` → 2); each round takes about 8 minutes. Plotting helpers in `scripts/` regenerate the progression and surface figures from the saved outputs.

## Repository layout

- `run_optimization.py` — main entry point.
- `src/chem_active_learning/` — configuration, preprocessing, DKL ensembles, conformal prediction, acquisition, and visualization.
- `scripts/` — optional data-preparation and plotting helpers.
- `data/` — experimental data for rounds 1–3.

## License

Released under the MIT License (see `LICENSE`).
