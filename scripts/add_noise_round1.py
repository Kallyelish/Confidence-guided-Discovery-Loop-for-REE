"""Create a noisy copy of the round1 experimental dataset.

By default this script perturbs only the measured response columns:
Nd recovery and Fe remaining. The experimental condition columns, pH and time,
are kept unchanged so the generated file can be used as a response-noise
robustness test.

Example:
    python scripts/add_noise_round1.py
    python scripts/add_noise_round1.py --seed 42 --nd-sigma 1.0 --fe-sigma 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chem_active_learning.config import COL_FE_PCT, COL_ND_PCT, COL_PH, COL_TIME

FE_MIN = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add controlled random noise to data/round1.xlsx."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "round1.xlsx",
        help="Input Excel file. Default: data/round1.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "round1_noisy.xlsx",
        help="Output Excel file. Default: data/round1_noisy.xlsx",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260403,
        help="Random seed used to generate noise.",
    )
    parser.add_argument(
        "--nd-sigma",
        type=float,
        default=1.0,
        help="Gaussian noise standard deviation for Nd recovery percentage points.",
    )
    parser.add_argument(
        "--fe-sigma",
        type=float,
        default=0.05,
        help="Gaussian noise standard deviation for Fe remaining percentage points.",
    )
    parser.add_argument(
        "--noise-conditions",
        action="store_true",
        help="Also add noise to pH and time columns.",
    )
    parser.add_argument(
        "--ph-sigma",
        type=float,
        default=0.03,
        help="Gaussian noise standard deviation for pH if --noise-conditions is used.",
    )
    parser.add_argument(
        "--time-sigma",
        type=float,
        default=0.1,
        help="Gaussian noise standard deviation for time in hours if --noise-conditions is used.",
    )
    return parser.parse_args()


def add_noise(
    df: pd.DataFrame,
    seed: int,
    nd_sigma: float,
    fe_sigma: float,
    noise_conditions: bool,
    ph_sigma: float,
    time_sigma: float,
) -> pd.DataFrame:
    noisy = df.copy()
    rng = np.random.default_rng(seed)

    required_cols = [COL_ND_PCT, COL_FE_PCT]
    if noise_conditions:
        required_cols.extend([COL_PH, COL_TIME])

    missing = [col for col in required_cols if col not in noisy.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    nd = pd.to_numeric(noisy[COL_ND_PCT], errors="coerce")
    fe = pd.to_numeric(noisy[COL_FE_PCT], errors="coerce")

    noisy[COL_ND_PCT] = np.clip(
        nd + rng.normal(0.0, nd_sigma, size=len(noisy)),
        0.0,
        100.0,
    )
    noisy[COL_FE_PCT] = np.clip(
        fe + rng.normal(0.0, fe_sigma, size=len(noisy)),
        FE_MIN,
        None,
    )

    if "SF" in noisy.columns:
        noisy["SF"] = noisy[COL_ND_PCT] / np.maximum(noisy[COL_FE_PCT], FE_MIN)
    if "log_SF" in noisy.columns:
        sf = (
            noisy["SF"]
            if "SF" in noisy.columns
            else noisy[COL_ND_PCT] / np.maximum(noisy[COL_FE_PCT], FE_MIN)
        )
        noisy["log_SF"] = np.log(np.maximum(sf, 1e-12))

    if noise_conditions:
        ph = pd.to_numeric(noisy[COL_PH], errors="coerce")
        time_h = pd.to_numeric(noisy[COL_TIME], errors="coerce")

        noisy[COL_PH] = np.clip(
            ph + rng.normal(0.0, ph_sigma, size=len(noisy)),
            3.0,
            6.0,
        )
        noisy[COL_TIME] = np.clip(
            time_h + rng.normal(0.0, time_sigma, size=len(noisy)),
            2.0,
            16.0,
        )

    return noisy


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.input)
    noisy = add_noise(
        df=df,
        seed=args.seed,
        nd_sigma=args.nd_sigma,
        fe_sigma=args.fe_sigma,
        noise_conditions=args.noise_conditions,
        ph_sigma=args.ph_sigma,
        time_sigma=args.time_sigma,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    noisy.to_excel(args.output, index=False)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Seed:   {args.seed}")
    print(f"Noise:  {COL_ND_PCT} sigma={args.nd_sigma}, {COL_FE_PCT} sigma={args.fe_sigma}")
    if args.noise_conditions:
        print(f"        {COL_PH} sigma={args.ph_sigma}, {COL_TIME} sigma={args.time_sigma}")


if __name__ == "__main__":
    main()
