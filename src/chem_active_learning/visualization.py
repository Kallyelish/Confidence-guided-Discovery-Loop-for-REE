"""Plotting and export utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from torch import nn

from .config import SAVE_DIR, USE_LOG_TIME
def plot_training_losses(losses_nd, losses_fe):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i, loss in enumerate(losses_nd):
        plt.plot(loss, label=f'Model {i+1}')
    plt.title('Training Loss: Nd Recovery Task')
    plt.xlabel('Epoch')
    plt.ylabel('Negative ELBO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for i, loss in enumerate(losses_fe):
        plt.plot(loss, label=f'Model {i+1}')
    plt.title('Training Loss: Fe Impurity Task')
    plt.xlabel('Epoch')
    plt.ylabel('Negative ELBO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "Training_Loss_Curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_comprehensive_shap(model_nd, model_fe, X_norm, X_raw_data, y_stats, grid_ph, grid_time, filename_prefix="SHAP_Global"):
    print(f"\n>>> 正在生成全面的 SHAP 解释性图表 (使用连续热力图)...")
    try:
        import shap
    except ImportError:
        print("Warning: 缺少 'shap' 库，跳过 SHAP 绘图。")
        return

    X_norm_np = X_norm.numpy()
    
    X_real_np = X_raw_data.numpy().copy()
    if USE_LOG_TIME:
        X_real_np[:, 1] = np.exp(X_real_np[:, 1])
    
    feature_names = ["pH", "Time (h)"]

    def get_preds(X_np, model, task_idx):
        X_tensor = torch.tensor(X_np, dtype=torch.float)
        with torch.no_grad():
            post = model.posterior(X_tensor) 
            mu_std = post.mean.squeeze(-1).numpy()
        val_destd = mu_std * y_stats[1][task_idx] + y_stats[0][task_idx]
        val_sig = 1.0 / (1.0 + np.exp(-val_destd))
        return val_sig * 100.0

    def pred_nd(X_np): return get_preds(X_np, model_nd, 0)
    def pred_fe(X_np): return get_preds(X_np, model_fe, 1)
    def pred_sf(X_np): return pred_nd(X_np) / np.maximum(pred_fe(X_np), 0.01)

    bg_data = shap.kmeans(X_norm_np, 10) if len(X_norm_np) > 20 else X_norm_np

    targets = [
        ("Nd_Recovery", pred_nd),
        ("Fe_Impurity", pred_fe),
        ("SF_Ratio", pred_sf)
    ]

    shap_dict = {}
    
    for target_name, pred_fn in targets:
        display_label = target_name.split("_")[0]

        explainer = shap.KernelExplainer(pred_fn, bg_data)
        shap_values = explainer.shap_values(X_norm_np, silent=True)
        shap_dict[target_name] = shap_values

        # --- A: 柱状图 (Global Importance) ---
        plt.figure(figsize=(7, 4))
        shap.summary_plot(shap_values, X_real_np, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"Global Importance: {target_name.replace('_', ' ')}", fontsize=14, pad=15)
        plt.savefig(os.path.join(SAVE_DIR, f"{filename_prefix}_{target_name}_Bar.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # --- B: 依赖图 (Continuous Contourf & Unweighted Combined) ---
        # ---------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        Z_ph = shap_values[:, 0].reshape(grid_ph.shape)
        Z_time = shap_values[:, 1].reshape(grid_ph.shape)
        
        from scipy.ndimage import gaussian_filter
        smooth_sigma = 1.0  
        Z_ph_smooth = gaussian_filter(Z_ph, sigma=smooth_sigma)
        Z_time_smooth = gaussian_filter(Z_time, sigma=smooth_sigma)
        
        Z_combined = Z_ph_smooth + Z_time_smooth 
        
        if display_label == "Nd":
            percentile_cap = 90  
        else:
            percentile_cap = 95  

        vmax_ph = np.percentile(np.abs(Z_ph_smooth), percentile_cap) or 1.0
        cf0 = axes[0].pcolormesh(grid_time, grid_ph, Z_ph_smooth, cmap='coolwarm', 
                                 vmin=-vmax_ph, vmax=vmax_ph, shading='gouraud', antialiased=True)
        axes[0].set_title(f'[{display_label}] SHAP Impact of pH', weight='bold', fontsize=15)
        cb0 = fig.colorbar(cf0, ax=axes[0], extend='both')
        cb0.set_label(f'{display_label} pH Contribution', weight='bold')

        vmax_time = np.percentile(np.abs(Z_time_smooth), percentile_cap) or 1.0
        cf1 = axes[1].pcolormesh(grid_time, grid_ph, Z_time_smooth, cmap='coolwarm', 
                                 vmin=-vmax_time, vmax=vmax_time, shading='gouraud', antialiased=True)
        axes[1].set_title(f'[{display_label}] SHAP Impact of Time', weight='bold', fontsize=15)
        cb1 = fig.colorbar(cf1, ax=axes[1], extend='both')
        cb1.set_label(f'{display_label} Time Contribution', weight='bold')
        
        vmax_comb = np.percentile(np.abs(Z_combined), percentile_cap) or 1.0
        cf2 = axes[2].pcolormesh(grid_time, grid_ph, Z_combined, cmap='coolwarm', 
                                 vmin=-vmax_comb, vmax=vmax_comb, shading='gouraud', antialiased=True)
        axes[2].set_title(f'[{display_label}] Combined Impact (pH + Time)', weight='bold', fontsize=15)
        cb2 = fig.colorbar(cf2, ax=axes[2], extend='both')
        cb2.set_label(f'{display_label} Total SHAP Contribution', weight='bold')

        for ax in axes:
            ax.set_xlabel('Time (h)', weight='bold', fontsize=12)
            ax.set_ylabel('pH', weight='bold', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3, color='white')

        plt.suptitle(f"{display_label}: SHAP Dependence Analysis", fontsize=18, weight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"{filename_prefix}_{target_name}_2D_Dependence_Continuous.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # --- C: 瀑布图 ---
        preds = pred_fn(X_norm_np)
        best_idx = np.argmin(preds) if "Fe" in target_name else np.argmax(preds)
        
        plt.figure(figsize=(8, 5))
        exp = shap.Explanation(values=shap_values[best_idx], 
                               base_values=explainer.expected_value, 
                               data=X_real_np[best_idx], 
                               feature_names=feature_names)
        shap.plots.waterfall(exp, show=False)
        plt.title(f"Breakdown for Optimal Sample: {target_name} (Global Grid #{best_idx})", fontsize=14, pad=20)
        plt.savefig(os.path.join(SAVE_DIR, f"{filename_prefix}_{target_name}_Waterfall.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # --- D: 定制特征影响图 ---
    # ---------------------------------------------------------
    def plot_custom_impact(feature_idx, feature_name, filename):
        fig, ax = plt.subplots(figsize=(10, 4))
        target_keys = [t[0] for t in targets]
        y_ticks = []
        
        x_raw = X_real_np[:, feature_idx]
        unique_x = np.unique(x_raw)
        
        heatmap_data = np.zeros((len(target_keys), len(unique_x)))
        
        for i, target in enumerate(target_keys):
            s_vals = shap_dict[target][:, feature_idx] 
            y_ticks.append(target.replace("_", " "))
            
            for j, ux in enumerate(unique_x):
                heatmap_data[i, j] = np.mean(s_vals[x_raw == ux])
                
            row_max = np.abs(heatmap_data[i, :]).max()
            if row_max != 0:
                heatmap_data[i, :] = heatmap_data[i, :] / row_max
                
        vmax = 1.0 
        
        X_mesh, Y_mesh = np.meshgrid(np.arange(len(unique_x)+1), np.arange(len(target_keys)+1))
        cax = ax.pcolormesh(X_mesh, Y_mesh, heatmap_data, cmap='coolwarm', 
                            vmin=-vmax, vmax=vmax, shading='flat', antialiased=True)
        
        ax.set_yticks(np.arange(len(target_keys)) + 0.5)
        ax.set_yticklabels(y_ticks, fontsize=12, weight='bold')
        
        ax.set_xticks(np.arange(len(unique_x)) + 0.5)
        xtick_labels = [f"{v:.1f}" if feature_name=="pH" else f"{int(v)}" for v in unique_x]
        
        if len(unique_x) > 20:
            for idx in range(len(xtick_labels)):
                if idx % 2 != 0: xtick_labels[idx] = ""
        ax.set_xticklabels(xtick_labels, fontsize=11)
        
        ax.set_xlabel(f"Actual {feature_name} Values", fontsize=13, weight='bold')
        ax.set_title(f"Normalized SHAP Impact of {feature_name} Across Targets", fontsize=15, weight='bold')
        
        for spine in ax.spines.values(): spine.set_visible(False)
        
        cb = fig.colorbar(cax, ax=ax, pad=0.03)
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_ticklabels(['Strong Negative\n(Blue)', 'Neutral', 'Strong Positive\n(Red)'])
        cb.set_label("Relative Impact Direction (Row-Normalized)", weight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()

    plot_custom_impact(0, "pH", f"{filename_prefix}_Custom_Impact_pH_Heatmap.png")
    plot_custom_impact(1, "Time (h)", f"{filename_prefix}_Custom_Impact_Time_Heatmap.png")


def make_soft_diverging_cmap(name="soft_red_yellow_blue"):
    return LinearSegmentedColormap.from_list(
        name,
        [
            (0.00, "#6F7F99"),
            (0.43, "#EFE6C8"),
            (0.63, "#E7C66F"),
            (1.00, "#D7696F"),
        ],
        N=256,
    )


def plot_2d_contour_generic(grid_ph, grid_time, Z_values, z_label, title, filename, enable_capping=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    label_fontsize = 24
    tick_fontsize = 22
    Z_plot = Z_values.reshape(grid_ph.shape)
    
    if enable_capping:
        z_max_cap = np.quantile(Z_values, 0.95)
        Z_plot = np.clip(Z_plot, 0, z_max_cap)
    
    color_kwargs = {}
    if "Nd" in z_label:
        vmin = 5.0 * np.floor(np.nanmin(Z_plot) / 5.0)
        vmax = 5.0 * np.ceil(np.nanmax(Z_plot) / 5.0)
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        color_kwargs = {"vmin": vmin, "vmax": vmax}

    image = ax.imshow(
        Z_plot,
        extent=[
            float(np.min(grid_ph)),
            float(np.max(grid_ph)),
            float(np.min(grid_time)),
            float(np.max(grid_time)),
        ],
        origin="lower",
        aspect="auto",
        cmap=make_soft_diverging_cmap(),
        interpolation="bicubic",
        **color_kwargs,
    )
    cb = fig.colorbar(image, ax=ax, pad=0.07)
    cb.set_label(z_label, fontsize=label_fontsize, labelpad=18)
    if "Nd" in z_label:
        tick_start = 10.0 * np.ceil(color_kwargs["vmin"] / 10.0)
        tick_end = 10.0 * np.floor(color_kwargs["vmax"] / 10.0)
        cb.set_ticks(np.arange(tick_start, tick_end + 0.1, 10.0))
    cb.ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    cb.ax.tick_params(which="major", labelsize=tick_fontsize, length=5)
    cb.ax.tick_params(which="minor", length=3)
    
    ax.set_xlabel('pH', fontsize=label_fontsize)
    ax.set_ylabel('Time', fontsize=label_fontsize)
    ax.set_title("")
    ax.set_ylim(2, 16)
    ax.set_yticks(np.arange(2, 17, 2))
    ax.grid(False)
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", direction="out", length=5, labelsize=tick_fontsize)
    ax.tick_params(axis="x", which="minor", direction="out", length=3)
    
    save_name = os.path.splitext(filename)[0] + ".svg"
    save_path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(save_path, format="svg", bbox_inches='tight')
    plt.close()


def plot_performance_metrics_all(model_nd, model_fe, X_norm, y_stats, df_train, candidates_df, filename, current_round=0):
    model_nd.eval(); model_fe.eval()
    
    with torch.no_grad():
        post_nd = model_nd.posterior(X_norm)
        post_fe = model_fe.posterior(X_norm)
        mu_nd = post_nd.mean.squeeze(-1).numpy(); sigma_nd = post_nd.variance.sqrt().squeeze(-1).numpy()
        mu_fe = post_fe.mean.squeeze(-1).numpy(); sigma_fe = post_fe.variance.sqrt().squeeze(-1).numpy()

    def rev_transform_arr(val_std, mean, std):
        val_destd = val_std * std + mean
        val_sig = 1.0 / (1.0 + np.exp(-val_destd))
        return val_sig * 100.0

    mean_nd, std_nd = y_stats[0][0], y_stats[1][0]
    mean_fe, std_fe = y_stats[0][1], y_stats[1][1]

    pred_nd = rev_transform_arr(mu_nd, mean_nd, std_nd)
    lower_nd = rev_transform_arr(mu_nd - sigma_nd, mean_nd, std_nd)
    upper_nd = rev_transform_arr(mu_nd + sigma_nd, mean_nd, std_nd)
    
    pred_fe = rev_transform_arr(mu_fe, mean_fe, std_fe)
    lower_fe = rev_transform_arr(mu_fe - sigma_fe, mean_fe, std_fe)
    upper_fe = rev_transform_arr(mu_fe + sigma_fe, mean_fe, std_fe)

    pred_sf = pred_nd / np.maximum(pred_fe, 0.01)

    true_nd = df_train["val_Nd"].values
    true_fe = df_train["val_Fe"].values
    true_sf = true_nd / np.maximum(true_fe, 0.01)

    cand_nd = candidates_df["Pred_Nd(%)"].values
    cand_fe = candidates_df["Pred_Fe(%)"].values
    cand_sf = candidates_df["SF_Ratio"].values
    # 【新增】：提取推荐点的真实物理参数，用于标记唯一性
    cand_ph = candidates_df["pH"].values
    cand_time = candidates_df["Time(h)"].values

    # 【增强版保存逻辑】：保存用于后续多轮 Parity 对易绘图的唯一标识符
    history_df = pd.DataFrame({
        "Round": current_round,
        "Data_Type": "History",
        # 【新增】：用于标记唯一物理点的 ID（pH 和 Time）
        "pH_Raw": df_train["pH"].values,
        "Time_h_Raw": df_train["time_h"].values,
        "Exp_Nd(%)": true_nd,
        "Pred_Nd(%)": pred_nd,
        "Exp_Fe(%)": true_fe,
        "Pred_Fe(%)": pred_fe,
        "Exp_SF": true_sf,
        "Pred_SF": pred_sf
    })

    cand_df = pd.DataFrame({
        "Round": current_round,
        "Data_Type": "Candidate",
        "pH_Raw": cand_ph,
        "Time_h_Raw": cand_time,
        "Exp_Nd(%)": np.nan, 
        "Pred_Nd(%)": cand_nd,
        "Exp_Fe(%)": np.nan,
        "Pred_Fe(%)": cand_fe,
        "Exp_SF": np.nan,
        "Pred_SF": cand_sf
    })

    tracking_df = pd.concat([history_df, cand_df], ignore_index=True)
    tracking_path = os.path.join(SAVE_DIR, "Optimization_Tracking_Data.csv")

    if os.path.exists(tracking_path):
        existing_df = pd.read_csv(tracking_path)
        existing_df = existing_df[existing_df["Round"] != current_round] 
        tracking_df = pd.concat([existing_df, tracking_df], ignore_index=True)

    tracking_df.to_csv(tracking_path, index=False)
    print(f"✅ Round {current_round} tracking data with Point IDs saved to: {tracking_path}")

    # 下方原有的绘图代码保持不变...
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    def plot_row(row_idx, name, true_vals, pred_vals, cand_vals, lower=None, upper=None, maximize=True):
        ax_parity = axes[row_idx, 0]; ax_history = axes[row_idx, 1]
        
        d_min = min(true_vals.min(), pred_vals.min()) * 0.9
        d_max = max(true_vals.max(), pred_vals.max()) * 1.1
        ax_parity.plot([d_min, d_max], [d_min, d_max], 'k--', alpha=0.5)
        
        if lower is not None and upper is not None:
            yerr = [pred_vals - lower, upper - pred_vals]
            ax_parity.errorbar(true_vals, pred_vals, yerr=yerr, fmt='o', c='gray', alpha=0.5, capsize=3, label='Training Data')
        else:
            ax_parity.scatter(true_vals, pred_vals, c='gray', alpha=0.5, label='Training Data')
            
        ax_parity.scatter(cand_vals, cand_vals, marker='*', s=150, c='red', label='New Candidates', zorder=10)
        ax_parity.set_xlabel(f'Experimental {name}'); ax_parity.set_ylabel(f'Predicted {name}')
        ax_parity.set_title(f'{name}: Prediction Accuracy')
        ax_parity.legend(); ax_parity.grid(True, linestyle='--', alpha=0.3)

        x_train = np.random.normal(0, 0.03, len(true_vals))
        ax_history.scatter(x_train, true_vals, c='gray', alpha=0.5, s=50, label='History')
        
        x_cand = np.ones_like(cand_vals) + np.random.normal(0, 0.03, len(cand_vals))
        ax_history.scatter(x_cand, cand_vals, c='red', marker='*', s=150, label='Optimization')
        
        best = true_vals.max() if maximize else true_vals.min()
        ax_history.axhline(best, color='blue', linestyle='--', label=f'Best Exp: {best:.2f}')
            
        ax_history.set_xticks([0, 1]); ax_history.set_xticklabels(['Initial', 'New Batch'])
        ax_history.set_title(f'{name}: Optimization Progress'); ax_history.set_ylabel(name)
        ax_history.legend(); ax_history.grid(True, axis='y', linestyle='--', alpha=0.3)

    plot_row(0, "SF Ratio", true_sf, pred_sf, cand_sf, maximize=True)
    plot_row(1, "Nd Recovery (%)", true_nd, pred_nd, cand_nd, None, None, maximize=True)
    plot_row(2, "Fe Impurity (%)", true_fe, pred_fe, cand_fe, None, None, maximize=False)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
    plt.close()

def plot_tsne_latent(model, X_norm, Y, title, filename):
    model.eval()
    extractor = nn.Sequential(*list(model.feature_extractor.children())[:4])
    
    with torch.no_grad():
        features = extractor(X_norm).numpy()
    
    perp = min(30, len(features) - 1) if len(features) > 1 else 1
    if len(features) < 2: return

    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    z = tsne.fit_transform(features)
    
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(z[:, 0], z[:, 1], c=Y, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
    plt.colorbar(sc, label='Target Value (Std)')
    plt.title(f"T-SNE: {title}")
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_surface_generic(grid_ph, grid_time, Z_values, z_label, title, filename, enable_capping=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    title_suffix = ""
    Z_plot = Z_values
    
    if enable_capping:
        z_max_cap = np.quantile(Z_values, 0.95)
        Z_plot = np.clip(Z_values, 0, z_max_cap)
        title_suffix = f"\n(Capped at 95%: {z_max_cap:.1f})"

    surf = ax.plot_surface(grid_ph, grid_time, Z_plot.reshape(grid_ph.shape), 
                           cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
    
    ax.set_xlabel('pH')
    ax.set_ylabel('Time (h)')
    ax.set_zlabel(z_label)
    ax.set_title(title + title_suffix)
    fig.colorbar(surf, shrink=0.5, aspect=10, label=z_label)
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_surface_contour_data(grid_ph, grid_time, nd_values, fe_values, sf_values):
    plot_data = pd.DataFrame({
        "pH": grid_ph.flatten(),
        "Time(h)": grid_time.flatten(),
        "Pred_Nd(%)": np.asarray(nd_values).flatten(),
        "Pred_Fe(%)": np.asarray(fe_values).flatten(),
        "SF_Ratio": np.asarray(sf_values).flatten(),
    })
    save_path = os.path.join(SAVE_DIR, "Surface_Contour_Data.csv")
    plot_data.to_csv(save_path, index=False)
    print(f"✅ Surface and contour plot data saved to: {save_path}")

def plot_pareto_3d(obs_raw, cand_raw, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(obs_raw[:, 0], obs_raw[:, 1], obs_raw[:, 2], 
               c='gray', alpha=0.4, s=30, label='Observed')
    
    ax.scatter(cand_raw[:, 0], cand_raw[:, 1], cand_raw[:, 2], 
               c='red', marker='*', s=150, label='Candidates', depthshade=False)
    
    ax.set_xlabel('Nd Recovery (%)')
    ax.set_ylabel('Fe Impurity (%)')
    ax.set_zlabel('Time (h)')
    ax.set_title(title)
    
    plt.legend()
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_set_2d(grid_ph, grid_time, safe_mask, cand_raw, title, filename, show_candidates=True, show_safe_region=True, show_legend=True):
    plt.figure(figsize=(10, 6))
    
    flat_ph = grid_ph.flatten()
    flat_time = grid_time.flatten()
    safe_mask = np.asarray(safe_mask).flatten()
    
    unsafe_mask = ~safe_mask
    plt.scatter(flat_ph[unsafe_mask], flat_time[unsafe_mask], 
                c='lightgray', s=10, marker='s', alpha=0.12, label='Unsafe Region')
    
    if show_safe_region:
        plt.scatter(flat_ph[safe_mask], flat_time[safe_mask], 
                    c='lightgreen', s=15, marker='s', alpha=0.4, label='Safe Region')
    
    if show_candidates and cand_raw is not None and len(cand_raw) > 0:
        cand_raw = np.asarray(cand_raw)
        plt.scatter(cand_raw[:, 0], cand_raw[:, 1], 
                    c='red', marker='*', s=250, edgecolors='k', linewidth=1.5,
                    label='Pareto Optimal Candidates', zorder=10)
    
    plt.xlabel('pH')
    plt.ylabel('Time')
    if show_legend:
        plt.legend(loc='upper right')
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


