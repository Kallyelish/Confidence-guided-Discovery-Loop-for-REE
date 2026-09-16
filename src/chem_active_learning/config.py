"""Shared configuration for the active-learning optimization workflow."""

import os
import warnings

import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

SAVE_DIR = "optimization_plots"
DATA_PATH = os.path.join("data", "round1.xlsx")
COL_PH = "Con_pH"
COL_TIME = "Con_time (h)"
COL_ND_PCT = "Nd_Recovery"
COL_FE_PCT = "Fe_Remaining"

PH_MIN, PH_MAX = 3.0, 6.0
T_MIN_H, T_MAX_H = 2.0, 16.0
USE_LOG_TIME = True
Q_CANDIDATES = 4

ROUND = 0 # begin from 0
CONFORMAL_ALPHA = 0.1

ENSEMBLE_SIZE = 5
TRAIN_BASE_SEED = 20260403
ND_SEED_OFFSET = 1000
FE_SEED_OFFSET = 2000

ND_TRAIN_CFG = {
    "num_epochs": 500,
    "lr": 0.02,
    "weight_decay": 5e-5,
    "batch_size": 64,
    "max_inducing_points": 100,
    "kernel_lengthscale_min": 0.03,
    "noise_lower": 1e-4,
    "noise_upper": 0.2,
    "noise_init": 0.003,
    "smoothness_lambda_max": 2.0,
    "smoothness_start_ratio": 0.65,
    "smooth_fd_eps": 0.02,
}

FE_TRAIN_CFG = {
    "num_epochs": 3000,
    "lr": 0.0025,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_inducing_points": 100,
    "kernel_lengthscale_min": 2.5,
    "noise_lower": 1e-4,
    "noise_upper": 0.01,
    "noise_init": 0.008,
    "smoothness_lambda_max": 1e-2,
    "smoothness_start_ratio": 0.35,
    "smooth_fd_eps": 0.02,
}


def prepare_runtime():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Images will be saved to: {os.path.abspath(SAVE_DIR)}")
    torch.set_default_dtype(torch.float)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
