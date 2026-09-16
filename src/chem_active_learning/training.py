"""Training routines for the DKL ensembles."""

import random

import gpytorch
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .config import (
    FE_SEED_OFFSET,
    FE_TRAIN_CFG,
    ND_SEED_OFFSET,
    ND_TRAIN_CFG,
    ENSEMBLE_SIZE,
    TRAIN_BASE_SEED,
)
from .data import set_seed
from .models import DKLModel, DeepEnsembleGP, FeatureExtractor
from .visualization import plot_training_losses
def train_single_dkl(train_x, train_y, train_cfg, seed, progress_desc=None, show_progress=True):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    set_seed(seed)

    num_epochs = int(train_cfg["num_epochs"])

    perm_gen = torch.Generator().manual_seed(seed + 11)
    loader_gen = torch.Generator().manual_seed(seed + 23)

    if train_x.size(0) > int(train_cfg["max_inducing_points"]):
        idx = torch.randperm(train_x.size(0), generator=perm_gen)[:int(train_cfg["max_inducing_points"])]
        inducing_points = train_x[idx].clone()
    else:
        inducing_points = train_x.clone()
        
    feature_extractor = FeatureExtractor(input_dim=train_x.size(-1))
    model = DKLModel(
        inducing_points,
        feature_extractor,
        kernel_lengthscale_min=float(train_cfg["kernel_lengthscale_min"])
    )
    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(
            float(train_cfg["noise_lower"]),
            float(train_cfg["noise_upper"])
        )
    )
    likelihood.noise = float(train_cfg["noise_init"])
    
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))
    
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(
        dataset,
        batch_size=min(int(train_cfg["batch_size"]), len(dataset)),
        shuffle=True,
        generator=loader_gen
    )

    losses = []
    smooth_start_epoch = int(num_epochs * float(train_cfg["smoothness_start_ratio"]))
    smooth_ramp_denom = max(1, num_epochs - smooth_start_epoch)

    try:
        epoch_iter = range(num_epochs)
        if show_progress and tqdm is not None:
            epoch_iter = tqdm(epoch_iter, total=num_epochs, desc=progress_desc, leave=False)

        for epoch in epoch_iter:
            if epoch < smooth_start_epoch:
                smooth_w = 0.0
            else:
                smooth_w = float(train_cfg["smoothness_lambda_max"]) * ((epoch - smooth_start_epoch + 1) / smooth_ramp_denom)

            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                base_loss = -mll(output, y_batch)
                fd_eps = float(train_cfg.get("smooth_fd_eps", 0.02))
                if smooth_w > 0.0:
                    fd_terms = []
                    for dim in range(x_batch.size(1)):
                        delta = torch.zeros_like(x_batch)
                        delta[:, dim] = fd_eps
                        mu_plus = model(x_batch + delta).mean
                        mu_minus = model(x_batch - delta).mean
                        fd_grad = (mu_plus - mu_minus) / (2.0 * fd_eps)
                        fd_terms.append(fd_grad.pow(2))
                    smooth_penalty = torch.stack(fd_terms, dim=-1).sum(dim=-1).mean()
                else:
                    smooth_penalty = torch.tensor(0.0, device=x_batch.device)
                loss = base_loss + smooth_w * smooth_penalty

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(loader)
            losses.append(avg_epoch_loss)
            if show_progress and tqdm is not None and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{avg_epoch_loss:.4f}")
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    return model, likelihood, losses

def fit_ensemble(X, Y, n_models=3):
    print(f"  Training {n_models} DKL models per task...")
    all_losses_nd = []
    m_nd, l_nd = [], []
    for i in range(n_models):
        nd_seed = TRAIN_BASE_SEED + ND_SEED_OFFSET + i
        m, l, loss_history = train_single_dkl(
            X, Y[:, 0], ND_TRAIN_CFG,
            seed=nd_seed,
            progress_desc=f"Nd model {i + 1}/{n_models}",
            show_progress=True
        )
        m_nd.append(m); l_nd.append(l)
        all_losses_nd.append(loss_history)
        
    all_losses_fe = []
    m_fe, l_fe = [], []
    for i in range(n_models):
        fe_seed = TRAIN_BASE_SEED + FE_SEED_OFFSET + i
        m, l, loss_history = train_single_dkl(
            X, Y[:, 1], FE_TRAIN_CFG,
            seed=fe_seed,
            progress_desc=f"Fe model {i + 1}/{n_models}",
            show_progress=True
        )
        m_fe.append(m); l_fe.append(l)
        all_losses_fe.append(loss_history)

    plot_training_losses(all_losses_nd, all_losses_fe)
    return DeepEnsembleGP(m_nd, l_nd), DeepEnsembleGP(m_fe, l_fe)

