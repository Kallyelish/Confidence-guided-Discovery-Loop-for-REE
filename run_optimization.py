"""Run the active-learning optimization workflow.

This is the GitHub-ready entry point corresponding to the original test.py.
The model code is split under src/chem_active_learning, but seeds and training
order are kept consistent with the original script.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from chem_active_learning.config import (
    COL_FE_PCT,
    COL_ND_PCT,
    COL_PH,
    COL_TIME,
    DATA_PATH,
    ROUND,
    prepare_runtime,
)
from chem_active_learning.data import compute_metrics
from chem_active_learning.optimization import run_optimization_once


def load_data():
    try:
        df_raw = pd.read_excel(DATA_PATH)
        df = compute_metrics(df_raw)
        print(f"Loaded {len(df)} data points.")
        return df
    except Exception:
        print("Warning: Using synthetic data.")
        n_syn = 50
        ph_syn = np.round(np.random.uniform(3, 6, n_syn), 1)
        time_syn = np.random.randint(2, 17, n_syn)
        nd_syn = np.clip(85 + 2 * ph_syn + 0.5 * time_syn + np.random.normal(0, 1, n_syn), 80, 99.9)
        fe_syn = np.clip(0.1 + 0.3 * ph_syn + 0.1 * time_syn + np.random.normal(0, 0.2, n_syn), 0.01, 3.0)
        df = pd.DataFrame({
            COL_PH: ph_syn,
            COL_TIME: time_syn,
            COL_ND_PCT: nd_syn,
            COL_FE_PCT: fe_syn,
        })
        return compute_metrics(df)


def main():
    prepare_runtime()
    df = load_data()
    run_optimization_once(df, seed=100, current_round=ROUND)


if __name__ == "__main__":
    main()
