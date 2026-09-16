"""Sample reproducible initial data from the final model surface predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "optimization_plots" / "Surface_Contour_Data.csv"
DEFAULT_OUTPUT = ROOT / "data" / "initial_data_from_surface_20.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample predicted surface points as an initial dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=20, help="Number of points to sample.")
    parser.add_argument("--seed", type=int, default=20260403, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    surface = pd.read_csv(args.input)
    required_columns = {"pH", "Time(h)", "Pred_Nd(%)", "Pred_Fe(%)", "SF_Ratio"}
    missing_columns = required_columns.difference(surface.columns)
    if missing_columns:
        raise ValueError(f"Missing required surface columns: {sorted(missing_columns)}")
    if args.n <= 0 or args.n > len(surface):
        raise ValueError(f"--n must be between 1 and {len(surface)}.")

    sampled = surface.sample(n=args.n, random_state=args.seed).copy()
    sf = sampled["SF_Ratio"].to_numpy(dtype=float)
    initial_data = pd.DataFrame({
        "Con_pH": sampled["pH"].to_numpy(),
        "Con_time (h)": sampled["Time(h)"].to_numpy(),
        "Nd_Recovery": sampled["Pred_Nd(%)"].to_numpy(),
        "Fe_Remaining": sampled["Pred_Fe(%)"].to_numpy(),
        "SF": sf,
        "log_SF": np.log(sf),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial_data.to_excel(args.output, index=False)
    initial_data.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Sampled {len(initial_data)} points with seed {args.seed}.")
    print(f"Excel: {args.output}")
    print(f"CSV:   {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
