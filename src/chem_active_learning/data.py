"""Data preparation utilities."""

import random

import numpy as np
import pandas as pd
import torch

from .config import (
    COL_FE_PCT,
    COL_ND_PCT,
    COL_PH,
    COL_TIME,
    PH_MAX,
    PH_MIN,
    T_MAX_H,
    T_MIN_H,
    USE_LOG_TIME,
)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    cols = [COL_PH, COL_TIME, COL_ND_PCT, COL_FE_PCT]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[cols].isna().any().any():
        df = df.dropna(subset=cols)
    
    df["val_Nd"] = df[COL_ND_PCT]
    df["val_Fe"] = np.maximum(df[COL_FE_PCT], 1e-6) 
    df["val_Time"] = df[COL_TIME]
    df["pH"] = df[COL_PH]
    df["time_h"] = df[COL_TIME]
    return df

def make_training_tensors(df: pd.DataFrame):
    X = np.zeros((len(df), 2), dtype=float)
    X[:, 0] = df["pH"].to_numpy(dtype=float)
    t = df["time_h"].to_numpy(dtype=float)
    
    if USE_LOG_TIME:
        t = np.maximum(t, 1e-12)
        X[:, 1] = np.log(t)
    else:
        X[:, 1] = t

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-6
    X_norm = (X - x_mean) / x_std

    nd_norm = np.clip(df["val_Nd"].to_numpy(dtype=float) / 100.0, 1e-6, 1-1e-6)
    fe_norm = np.clip(df["val_Fe"].to_numpy(dtype=float) / 100.0, 1e-6, 1-1e-6)
    
    nd_logit = np.log(nd_norm / (1 - nd_norm))
    fe_logit = np.log(fe_norm / (1 - fe_norm))
    Y = np.stack([nd_logit, fe_logit], axis=1) 
    
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0) + 1e-6
    Y_std = (Y - y_mean) / y_std

    return (
        torch.tensor(X_norm, dtype=torch.float), 
        torch.tensor(Y_std, dtype=torch.float),
        (x_mean, x_std),
        (y_mean, y_std),
        torch.tensor(X, dtype=torch.float)
    )

def make_discrete_choices(x_stats, use_log_time=True):
    # Discrete Grid: pH step 0.1, Time step 1
    ph_vals = np.round(np.arange(PH_MIN, PH_MAX + 0.05, 0.1), 1)
    time_vals = np.arange(int(T_MIN_H), int(T_MAX_H) + 1, 1).astype(float)
    
    grid_ph, grid_time = np.meshgrid(ph_vals, time_vals)
    grid_flat = np.stack([grid_ph.flatten(), grid_time.flatten()], axis=1)
    
    X_raw = grid_flat.copy()
    if use_log_time:
        X_raw[:, 1] = np.log(np.maximum(X_raw[:, 1], 1e-12))
        
    x_mean, x_std = x_stats
    X_norm = (X_raw - x_mean) / x_std
    
    return (torch.tensor(X_norm, dtype=torch.float), 
            torch.tensor(X_raw, dtype=torch.float), 
            (grid_ph, grid_time))

