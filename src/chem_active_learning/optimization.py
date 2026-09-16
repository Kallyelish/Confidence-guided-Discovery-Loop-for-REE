"""One-round multi-objective active-learning optimization."""

import math
import os

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from .config import CONFORMAL_ALPHA, ENSEMBLE_SIZE, Q_CANDIDATES, SAVE_DIR, USE_LOG_TIME
from .conformal import ConformalPrediction
from .data import make_discrete_choices, make_training_tensors, set_seed
from .training import fit_ensemble
from .visualization import (
    plot_2d_contour_generic,
    plot_3d_surface_generic,
    plot_pareto_3d,
    plot_pareto_set_2d,
    plot_performance_metrics_all,
    plot_tsne_latent,
    save_surface_contour_data,
)
def run_optimization_once(df, seed, current_round=0, verbose=True):
    set_seed(seed)
    
    X_norm, Y_std, x_stats, y_stats, X_raw_train = make_training_tensors(df)
    
    if verbose: print(f"--- [Seed {seed}] Fitting VBLL/DKL Ensemble ---")
    model_nd, model_fe = fit_ensemble(X_norm, Y_std, n_models=ENSEMBLE_SIZE)
    model_list = ModelListGP(model_nd, model_fe) 

    choices_norm, choices_raw, (grid_ph, grid_time) = make_discrete_choices(x_stats, USE_LOG_TIME)

    # np_state = np.random.get_state()
    # torch_state = torch.get_rng_state()
    #
    # SHAP calculation is intentionally disabled for faster iterations.
    # plot_comprehensive_shap(
    #     model_nd, model_fe, choices_norm, choices_raw, y_stats, grid_ph, grid_time, filename_prefix="SHAP_Global"
    # )
    #
    # np.random.set_state(np_state)
    # torch.set_rng_state(torch_state)

    plot_tsne_latent(model_nd.models[0], X_norm, Y_std[:, 0], "Nd Recovery Features", "TSNE_Nd_Latent.png")
    
    def val_to_std_logit(val_pct, task_idx):
        val_frac = np.clip(val_pct / 100.0, 1e-6, 1-1e-6)
        val_logit = np.log(val_frac / (1 - val_frac))
        return (val_logit - y_stats[0][task_idx]) / y_stats[1][task_idx]
    
    with torch.no_grad():
        mu_nd_std = model_nd.posterior(choices_norm.unsqueeze(1)).mean.squeeze().numpy()
        mu_fe_std = model_fe.posterior(choices_norm.unsqueeze(1)).mean.squeeze().numpy()

    def rev_transform(val_std, task_idx):
        val_destd = val_std * y_stats[1][task_idx] + y_stats[0][task_idx]
        val_sig = 1.0 / (1.0 + np.exp(-val_destd))
        return val_sig * 100.0
        
    nd_vals_real = rev_transform(mu_nd_std, 0)
    fe_vals_real = rev_transform(mu_fe_std, 1)
    sf_vals_real = nd_vals_real / np.maximum(fe_vals_real, 0.5)

    save_surface_contour_data(grid_ph, grid_time, nd_vals_real, fe_vals_real, sf_vals_real)

    plot_3d_surface_generic(grid_ph, grid_time, nd_vals_real, "Nd (%)", "Nd Surface", "3D_Surface_Nd.png")
    plot_3d_surface_generic(grid_ph, grid_time, fe_vals_real, "Fe (%)", "Fe Surface", "3D_Surface_Fe.png")
    plot_3d_surface_generic(grid_ph, grid_time, sf_vals_real, "SF Ratio", "SF Surface (Full Range)", "3D_Surface_SF.png", enable_capping=False)

    plot_2d_contour_generic(grid_ph, grid_time, nd_vals_real, "Nd Recovery Rate (%)", "Nd 2D Projection", "2D_Contour_Nd.svg")
    plot_2d_contour_generic(grid_ph, grid_time, fe_vals_real, "Fe Impurity (%)", "Fe 2D Projection", "2D_Contour_Fe.png")
    plot_2d_contour_generic(grid_ph, grid_time, sf_vals_real, "SF Ratio", "SF 2D Projection", "2D_Contour_SF.png", enable_capping=False)

    cp_nd = ConformalPrediction(model_nd, X_norm, Y_std[:, 0], alpha=CONFORMAL_ALPHA)
    cp_fe = ConformalPrediction(model_fe, X_norm, Y_std[:, 1], alpha=CONFORMAL_ALPHA)
    fe_min_limit = 0.5
    nd_max_limit = 100.0
    current_nd_gate = min(nd_max_limit - (nd_max_limit - 75.0) * math.exp(-1.6 * current_round), 96.0)
    current_fe_gate = max(fe_min_limit + (10.0 - fe_min_limit) * math.exp(-2.0 * current_round), 0.9)

    if verbose:
        print(f"\n--- Round {current_round}: Dynamic Gates Applied ---")
        print(f"    ND_GATE Target: {current_nd_gate:.2f}% | FE_GATE Target: {current_fe_gate:.2f}%\n")

    nd_gate_std = val_to_std_logit(current_nd_gate, 0)
    fe_gate_std = val_to_std_logit(current_fe_gate, 1)
    nd_lower_std = cp_nd.predict_bound(choices_norm.unsqueeze(1), direction="lower").numpy()
    fe_upper_std = cp_fe.predict_bound(choices_norm.unsqueeze(1), direction="upper").numpy()
    safe_mask = ((nd_lower_std > nd_gate_std) & (fe_upper_std < fe_gate_std)).flatten()
    candidate_choices_norm = choices_norm[safe_mask]

    if len(candidate_choices_norm) == 0:
        raise RuntimeError("No points passed the safe-region gates. No relaxed fallback was applied.")
    if len(candidate_choices_norm) < Q_CANDIDATES:
        print(
            f"  Warning: Safe region small. Using only {len(candidate_choices_norm)} safe point(s); "
            "no relaxed fallback was applied."
        )

    current_q_candidates = min(Q_CANDIDATES, len(candidate_choices_norm))

    obs_nd = Y_std[:, 0]
    obs_fe_neg = -Y_std[:, 1]
    obs_time_neg = -X_norm[:, 1]
    
    obs_nd_real = df["val_Nd"].values
    obs_fe_real = df["val_Fe"].values
    obs_sf_real = obs_nd_real / np.maximum(obs_fe_real, 0.01)
    
    sf_mean = float(obs_sf_real.mean())
    sf_std = float(obs_sf_real.std() + 1e-6)
    obs_sf_std = torch.tensor((obs_sf_real - sf_mean) / sf_std, dtype=torch.float)

    train_obj = torch.stack([obs_nd, obs_fe_neg, obs_time_neg, obs_sf_std], dim=-1)

    partitioning = NondominatedPartitioning(ref_point=train_obj.min(dim=0).values - 0.1, Y=train_obj)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]))
    
    y_m_0, y_s_0 = float(y_stats[0][0]), float(y_stats[1][0])
    y_m_1, y_s_1 = float(y_stats[0][1]), float(y_stats[1][1])
    
    def objective_callable(samples, X):
        nd_std = samples[..., 0]
        fe_std = samples[..., 1]
        
        fe_neg = -fe_std
        time_neg = -X[..., 1]
        time_neg = time_neg.unsqueeze(0).expand(nd_std.shape)
        
        nd_real = 100.0 / (1.0 + torch.exp(-(nd_std * y_s_0 + y_m_0)))
        fe_real = 100.0 / (1.0 + torch.exp(-(fe_std * y_s_1 + y_m_1)))
        
        sf_real = nd_real / torch.clamp(fe_real, min=0.01)
        sf_std_obj = (sf_real - sf_mean) / sf_std
        
        return torch.stack([nd_std, fe_neg, time_neg, sf_std_obj], dim=-1)
    
    obj_tf = GenericMCMultiOutputObjective(objective=objective_callable)
    acq = qExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=partitioning.ref_point.tolist(),
        partitioning=partitioning,
        sampler=sampler,
        objective=obj_tf
    )
    
    cand_X_norm, acq_val = optimize_acqf_discrete(
        acq_function=acq,
        q=current_q_candidates,
        choices=candidate_choices_norm,
        max_batch_size=32,
        unique=True
    )
    
    cand_X_raw = cand_X_norm.detach().numpy() * x_stats[1] + x_stats[0]
    pH_cand = cand_X_raw[:, 0]
    time_cand_real = np.exp(cand_X_raw[:, 1]) if USE_LOG_TIME else cand_X_raw[:, 1]
    
    cand_raw_real = np.stack([pH_cand, time_cand_real], axis=1)

    print("Generating Decision Space Pareto Plots...")
    plot_pareto_set_2d(grid_ph, grid_time, safe_mask, None, 
                       "Decision Space Unsafe Region", 
                       "Decision_Space_Unsafe_Only.png",
                       show_candidates=False,
                       show_safe_region=False,
                       show_legend=False)
    plot_pareto_set_2d(grid_ph, grid_time, safe_mask, None, 
                       "Decision Space Safety Map", 
                       "Decision_Space_No_Optimal_Candidates.png",
                       show_candidates=False)
    plot_pareto_set_2d(grid_ph, grid_time, safe_mask, cand_raw_real, 
                       "Optimal Candidates in Decision Space", 
                       "Decision_Space_With_Optimal_Candidates.png",
                       show_candidates=True)

    with torch.no_grad():
        c_nd_std = model_nd.posterior(cand_X_norm.unsqueeze(1)).mean.squeeze()
        c_fe_std = model_fe.posterior(cand_X_norm.unsqueeze(1)).mean.squeeze()
        c_nd_real = rev_transform(c_nd_std.numpy(), 0)
        c_fe_real = rev_transform(c_fe_std.numpy(), 1)

    obs_stack = np.stack([df["val_Nd"].values, df["val_Fe"].values, df["val_Time"].values], axis=1)
    cand_stack = np.stack([c_nd_real, c_fe_real, time_cand_real], axis=1)
    
    plot_pareto_3d(obs_stack, cand_stack, "3-Obj Pareto Front", "Pareto_Front_3D.png")

    res_df = pd.DataFrame({
        "pH": np.round(pH_cand, 1),
        "Time(h)": np.round(time_cand_real).astype(int),
        "Pred_Nd(%)": np.round(c_nd_real, 2),
        "Pred_Fe(%)": np.round(c_fe_real, 3),
        "SF_Ratio": np.round(c_nd_real / np.maximum(c_fe_real, 0.01), 1),
        "Acq_Score": acq_val.detach().numpy()
    })
    
    print(">>> 正在计算 safe region 内数据点的最终预测并导出文件...")
    
    target_choices = candidate_choices_norm
    target_raw_X = target_choices.detach().numpy() * x_stats[1] + x_stats[0]
    
    with torch.no_grad():
        p_nd_std = model_nd.posterior(target_choices.unsqueeze(1)).mean.squeeze()
        p_fe_std = model_fe.posterior(target_choices.unsqueeze(1)).mean.squeeze()
        
        p_nd_real = rev_transform(p_nd_std.numpy(), 0)
        p_fe_real = rev_transform(p_fe_std.numpy(), 1)
        p_time_real = np.exp(target_raw_X[:, 1]) if USE_LOG_TIME else target_raw_X[:, 1]

    sf_real = p_nd_real / np.maximum(p_fe_real, 0.01)

    full_results_df = pd.DataFrame({
        "PH": np.round(target_raw_X[:, 0], 1),
        "time": np.round(p_time_real, 1),
        "Fe Remaining": np.round(p_fe_real, 3),
        "Nd Recovery": np.round(p_nd_real, 2),
        "SF": np.round(sf_real, 2),
        "log SF": np.round(np.log(sf_real), 4),
    })

    full_out_path = os.path.join(SAVE_DIR, "Full_Grid_Optimization_Results.xlsx")
    full_csv_path = os.path.join(SAVE_DIR, "Full_Grid_Optimization_Results.csv")
    full_results_df.to_excel(full_out_path, index=False)
    full_results_df.to_csv(full_csv_path, index=False)
    
    print(f"✅ Safe region 最终预测已保存至: {full_out_path}")
    print(f"✅ CSV 副本已保存至: {full_csv_path}")
    print(f"   数据总量: {len(full_results_df)} 行")
    
    print("Generating Combined Performance Metrics Plots & Saving Tracking Data...")
    plot_performance_metrics_all(
        model_nd, model_fe, X_norm, y_stats, df, res_df, 
        "Perf_Combined_SF_Nd_Fe.png",
        current_round=current_round
    )
    
    print("\n>>> Optimized Candidates")
    print(res_df.sort_values("Acq_Score", ascending=False).head(10).to_string(index=False))
    return res_df


