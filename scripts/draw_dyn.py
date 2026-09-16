import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def _build_truth_lookup(df):
    history = df[df["Data_Type"] == "History"].copy()
    history = history.sort_values("Round")
    truth_lookup = (
        history.dropna(subset=["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"])
        .drop_duplicates(subset=["pH_Raw", "Time_h_Raw"], keep="last")
        .set_index(["pH_Raw", "Time_h_Raw"])[["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]]
    )
    return truth_lookup


def _build_excel_truth_lookup(excel_path):
    if not os.path.exists(excel_path):
        return None

    excel_df = pd.read_excel(excel_path)
    required_cols = ["Con_pH", "Con_time (h)", "Nd_Recovery", "Fe_Remaining", "SF"]
    if any(col not in excel_df.columns for col in required_cols):
        return None

    truth_lookup = (
        excel_df.dropna(subset=["Con_pH", "Con_time (h)"])
        .drop_duplicates(subset=["Con_pH", "Con_time (h)"], keep="last")
        .rename(
            columns={
                "Con_pH": "pH_Raw",
                "Con_time (h)": "Time_h_Raw",
                "Nd_Recovery": "Exp_Nd(%)",
                "Fe_Remaining": "Exp_Fe(%)",
                "SF": "Exp_SF",
            }
        )
        .set_index(["pH_Raw", "Time_h_Raw"])[["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]]
    )
    return truth_lookup


def _build_prediction_lookup(df, round_id):
    history = df[(df["Data_Type"] == "History") & (df["Round"] == round_id)].copy()
    prediction_lookup = (
        history.drop_duplicates(subset=["pH_Raw", "Time_h_Raw"], keep="last")
        .set_index(["pH_Raw", "Time_h_Raw"])[["Pred_Nd(%)", "Pred_Fe(%)", "Pred_SF"]]
    )
    return prediction_lookup


def _attach_truth(candidate_df, truth_lookup):
    candidate_df = candidate_df.drop(
        columns=["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"],
        errors="ignore",
    )
    merged = candidate_df.merge(
        truth_lookup.reset_index(),
        on=["pH_Raw", "Time_h_Raw"],
        how="left",
    )
    return merged


def _fill_final_round_truth_from_excel(candidate_df, fallback_truth_lookup, final_round):
    if fallback_truth_lookup is None or fallback_truth_lookup.empty:
        return candidate_df, 0

    candidate_df = candidate_df.copy()
    missing_mask = (
        (candidate_df["Round"] == final_round)
        & candidate_df[["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]].isna().any(axis=1)
    )
    if not missing_mask.any():
        return candidate_df, 0

    fallback_df = fallback_truth_lookup.reset_index()
    fallback_rows = candidate_df.loc[missing_mask, ["pH_Raw", "Time_h_Raw"]].merge(
        fallback_df,
        on=["pH_Raw", "Time_h_Raw"],
        how="left",
    )

    filled_count = 0
    for col in ["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]:
        available = fallback_rows[col].notna()
        if available.any():
            target_index = candidate_df.loc[missing_mask].index[available]
            candidate_df.loc[target_index, col] = fallback_rows.loc[available, col].values
            filled_count += int(available.sum())

    return candidate_df, filled_count


def _attach_final_prediction(candidate_df, prediction_lookup):
    candidate_df = candidate_df.copy()
    base_pred_cols = ["Pred_Nd(%)", "Pred_Fe(%)", "Pred_SF"]
    for col in base_pred_cols:
        if col not in candidate_df.columns:
            candidate_df[col] = np.nan

    # Preserve the candidate's original round prediction before attaching
    # the latest-round prediction for comparison and arrow rendering.
    rename_map = {col: f"{col}_first" for col in base_pred_cols}
    candidate_df = candidate_df.rename(columns=rename_map)

    merged = candidate_df.merge(
        prediction_lookup.reset_index(),
        on=["pH_Raw", "Time_h_Raw"],
        how="left",
        suffixes=("", "_latest"),
    )

    for col in base_pred_cols:
        first_col = f"{col}_first"
        if col not in merged.columns:
            merged[col] = np.nan
        if first_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[first_col])

    return merged


def _format_point_list(df_points):
    if df_points.empty:
        return ""
    labels = [
        f"(pH={row['pH_Raw']}, time={row['Time_h_Raw']})"
        for _, row in df_points[["pH_Raw", "Time_h_Raw"]].drop_duplicates().iterrows()
    ]
    return ", ".join(labels)


def _print_plotted_candidates(candidate_df, all_rounds):
    print("实际画出的 candidate 点:")
    for r in all_rounds:
        r_df = candidate_df[candidate_df["Round"] == r].copy()
        if r_df.empty:
            print(f"  Round {r}: 无 candidate")
            continue

        plotted_df = r_df[
            r_df[["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]].notna().all(axis=1)
        ][["pH_Raw", "Time_h_Raw"]].drop_duplicates()
        skipped_df = r_df[
            r_df[["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]].isna().any(axis=1)
        ][["pH_Raw", "Time_h_Raw"]].drop_duplicates()

        if plotted_df.empty:
            print(f"  Round {r}: 无可绘制 candidate")
        else:
            print(f"  Round {r}: {_format_point_list(plotted_df)}")

        if not skipped_df.empty:
            print(f"  Round {r} skipped: {_format_point_list(skipped_df)}")


def _style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=11, colors="#6C757D")
    ax.grid(True, linestyle="--", alpha=0.6, color="#E9ECEF", zorder=0)


def _draw_break_marks(ax_top, ax_bottom):
    kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8,
                  linestyle="none", color="#6C757D", mec="#6C757D", mew=1,
                  clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)


def _blend_with_white(hex_color, blend):
    rgb = np.array(to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = rgb * (1 - blend) + white * blend
    return tuple(np.clip(mixed, 0, 1))


def _round_color(round_idx, total_rounds, base_color="#1D3557"):
    if total_rounds <= 1:
        return base_color
    # Early rounds are lighter; later rounds get progressively darker.
    light_blend = 0.55 * (1 - round_idx / (total_rounds - 1))
    return _blend_with_white(base_color, light_blend)


def _round_palette(round_idx, total_rounds):
    palette = [
        "#54A24B",  # green
        "#EECA3B",  # yellow
        "#FF0000",  # bright red
        "#72B7B2",
        "#B279A2",
        "#4C78A8",
        "#FF9DA6",
    ]
    base_color = palette[round_idx % len(palette)]
    return _round_color(round_idx, total_rounds, base_color=base_color)


def plot_multi_round_parity_progress(
    csv_path="optimization_plots/Optimization_Tracking_Data.csv",
    save_dir="optimization_plots",
    truth_fallback_xlsx="cleaned_data_all.xlsx",
):
    if not os.path.exists(csv_path):
        print(f"找不到追踪数据文件: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"追踪数据为空: {csv_path}")
        return

    all_rounds = sorted(df["Round"].dropna().astype(int).unique().tolist())
    if not all_rounds:
        print("追踪数据里没有可用轮次。")
        return

    final_round = max(all_rounds)
    marker_list = ["o", "D", "h", "s", "v", "^", "p"]

    target_config = [
        ("Nd Recovery (%)", "Exp_Nd(%)", "Pred_Nd(%)", True),
        ("Fe Impurity (%)", "Exp_Fe(%)", "Pred_Fe(%)", False),
        ("SF Ratio", "Exp_SF", "Pred_SF", True),
    ]
    progress_ylabels = {
        "Exp_Nd(%)": "Nd Recovery Rate (%)",
        "Exp_Fe(%)": "Fe Residual Rate (%)",
    }

    truth_lookup = _build_truth_lookup(df)
    fallback_truth_lookup = _build_excel_truth_lookup(truth_fallback_xlsx)
    final_prediction_lookup = _build_prediction_lookup(df, final_round)
    candidate_all = df[df["Data_Type"] == "Candidate"].copy()
    candidate_truth = _attach_truth(candidate_all, truth_lookup)
    candidate_truth, fallback_fill_count = _fill_final_round_truth_from_excel(
        candidate_truth,
        fallback_truth_lookup,
        final_round,
    )
    candidate_final = _attach_final_prediction(candidate_truth, final_prediction_lookup)
    _print_plotted_candidates(candidate_final, all_rounds)

    initial_round = min(all_rounds)
    initial_history = df[(df["Data_Type"] == "History") & (df["Round"] == initial_round)].copy()
    history_shift = initial_history.merge(
        final_prediction_lookup.reset_index(),
        on=["pH_Raw", "Time_h_Raw"],
        how="left",
        suffixes=("_first", "_final"),
    )

    fig_left, axes_left = plt.subplots(3, 1, figsize=(9, 18), gridspec_kw={"hspace": 0.25})
    fig_right = plt.figure(figsize=(10, 18))
    right_grid = fig_right.add_gridspec(3, 1, hspace=0.25)
    axes_right = []
    for row in range(3):
        if row < 2:
            broken_grid = right_grid[row].subgridspec(2, 1, hspace=0.05)
            axes_right.append([
                fig_right.add_subplot(broken_grid[0]),
                fig_right.add_subplot(broken_grid[1]),
            ])
        else:
            axes_right.append([fig_right.add_subplot(right_grid[row])])

    unresolved_points = candidate_final[
        candidate_final[["Exp_Nd(%)", "Exp_Fe(%)", "Exp_SF"]].isna().any(axis=1)
    ][["pH_Raw", "Time_h_Raw"]].drop_duplicates()
    final_pred_missing_points = candidate_final[
        candidate_final[["Pred_Nd(%)", "Pred_Fe(%)", "Pred_SF"]].isna().any(axis=1)
    ][["pH_Raw", "Time_h_Raw"]].drop_duplicates()

    for i, (title, exp_col, pred_col, maximize) in enumerate(target_config):
        ax_left = axes_left[i]
        right_axes = axes_right[i]
        ax_right = right_axes[-1]

        hist_truth_vals = initial_history[exp_col].dropna().values
        hist_pred_vals = initial_history[pred_col].dropna().values
        cand_truth_vals = candidate_final[exp_col].dropna().values
        cand_pred_vals = candidate_final[pred_col].dropna().values
        hist_final_pred_col = pred_col.replace("Pred_", "Pred_") + "_final"
        hist_final_pred_vals = (
            history_shift[hist_final_pred_col].dropna().values
            if hist_final_pred_col in history_shift.columns
            else np.array([])
        )

        all_valid = np.concatenate(
            [
                arr
                for arr in [hist_truth_vals, hist_pred_vals, cand_truth_vals, cand_pred_vals, hist_final_pred_vals]
                if len(arr) > 0
            ]
        )
        if len(all_valid) == 0:
            continue

        val_range = all_valid.max() - all_valid.min()
        pad = max(val_range * 0.05, 1e-6)
        d_min = all_valid.min() - pad
        d_max = all_valid.max() + pad

        # Left: parity using final-round history prediction and final candidate prediction.
        ax_left.plot([d_min, d_max], [d_min, d_max], color="#8D99AE", linestyle="--", alpha=0.6, zorder=0)

        history_final_col = f"{pred_col}_final"
        if history_final_col in history_shift.columns:
            final_hist_df = history_shift[
                history_shift[exp_col].notna()
                & history_shift[history_final_col].notna()
            ].copy()
            ax_left.scatter(
                final_hist_df[exp_col],
                final_hist_df[history_final_col],
                color="#6C757D",
                alpha=0.9,
                s=42,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
                label=f"History (Round {final_round} Pred)",
            )

        for r in all_rounds:
            r_cands = candidate_final[
                (candidate_final["Round"] == r)
                & candidate_final[exp_col].notna()
                & candidate_final[pred_col].notna()
            ]
            if r_cands.empty:
                continue
            round_idx = all_rounds.index(r)
            marker = "*" if r == final_round else marker_list[r % len(marker_list)]

            ax_left.scatter(
                r_cands[exp_col],
                r_cands[pred_col],
                marker=marker,
                facecolor=_round_palette(round_idx, len(all_rounds)),
                s=220 if r == final_round else 70,
                zorder=5,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.95,
                label=f"Round {r} Candidate (Final Pred)",
            )

        ax_left.set_xlim(d_min, d_max)
        ax_left.set_ylim(d_min, d_max)
        ax_left.set_title(f"{title} - Parity", weight="bold", fontsize=14, pad=15)
        ax_left.set_xlabel("True Value", weight="bold", color="#495057")
        ax_left.set_ylabel("Predicted Value", weight="bold", color="#495057")
        _style_axes(ax_left)
        ax_left.legend(prop={"size": 10}, loc="upper left", frameon=True, edgecolor="none", facecolor="#F8F9FA", framealpha=0.8)

        # Right: one candidate column per round, all plotted with truth values.
        if not initial_history.empty:
            init_unique = initial_history.drop_duplicates(subset=["pH_Raw", "Time_h_Raw"], keep="last")
            x_init = np.zeros(len(init_unique))
            for progress_ax in right_axes:
                progress_ax.scatter(
                    x_init,
                    init_unique[exp_col],
                    color="#ADB5BD",
                    alpha=0.75,
                    s=70,
                    edgecolor="white",
                    linewidth=0.6,
                    label="Init Truth",
                    zorder=3,
                )

        for r_idx, r in enumerate(all_rounds):
            r_cands = candidate_final[(candidate_final["Round"] == r) & candidate_final[exp_col].notna()]
            if r_cands.empty:
                continue
            x_cand = np.full(len(r_cands), r_idx + 1)
            marker = "*" if r == final_round else marker_list[r % len(marker_list)]
            for progress_ax in right_axes:
                progress_ax.scatter(
                    x_cand,
                    r_cands[exp_col],
                    marker=marker,
                    facecolor=_round_palette(r_idx, len(all_rounds)),
                    s=220 if r == final_round else 70,
                    zorder=5,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.95,
                    label=f"Round {r} Candidate Truth",
                )

        best_truth = truth_lookup[exp_col].dropna()
        if not best_truth.empty:
            current_best = best_truth.max() if maximize else best_truth.min()
            for progress_ax in right_axes:
                progress_ax.axhline(
                    current_best,
                    color="#457B9D",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"Best Truth: {current_best:.2f}",
                )

        total_columns = len(all_rounds)
        for progress_ax in right_axes:
            progress_ax.set_xlim(-0.5, total_columns + 0.5)
            progress_ax.set_xticks([0] + [idx + 1 for idx in range(total_columns)])
            progress_ax.set_xticklabels(
                [str(idx) for idx in range(total_columns + 1)],
                weight="bold",
                color="#495057",
            )
            _style_axes(progress_ax)

        if exp_col == "Exp_Nd(%)":
            right_axes[0].set_ylim(65, d_max)
            right_axes[1].set_ylim(d_min, 55)
        elif exp_col == "Exp_Fe(%)":
            right_axes[0].set_ylim(11, d_max)
            right_axes[1].set_ylim(d_min, 6)
        else:
            ax_right.set_ylim(d_min, d_max)

        right_axes[0].set_title(f"{title} - Candidate Truth by Round", weight="bold", fontsize=14, pad=15)
        ax_right.set_xlabel("Number of Iteration", weight="bold", color="#495057")
        right_axes[0].set_ylabel(progress_ylabels.get(exp_col, "True Value"), weight="bold", color="#495057", y=0.0)
        right_axes[0].legend(prop={"size": 10}, loc="best", frameon=True, edgecolor="none", facecolor="#F8F9FA", framealpha=0.8)

        if len(right_axes) == 2:
            right_axes[0].tick_params(labelbottom=False)
            _draw_break_marks(right_axes[0], right_axes[1])

    os.makedirs(save_dir, exist_ok=True)
    left_path = os.path.join(save_dir, "perf_combined_left.png")
    right_path = os.path.join(save_dir, "perf_combined_right.png")

    fig_left.savefig(left_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig_right.savefig(right_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_left)
    plt.close(fig_right)

    print(f"左侧 parity 图已保存至: {left_path}")
    print(f"右侧 truth progress 图已保存至: {right_path}")
    if fallback_fill_count > 0:
        print(f"已从 {truth_fallback_xlsx} 为最后一轮 candidate 回填 {fallback_fill_count} 个真值字段。")
    if not unresolved_points.empty:
        print(
            f"注意: 仍有 {len(unresolved_points)} 个 candidate 点在 tracking 历史和 {truth_fallback_xlsx} 中都没有匹配到真值，图中已自动跳过: "
            f"{_format_point_list(unresolved_points)}"
        )
    if not final_pred_missing_points.empty:
        print(
            f"注意: 仍有 {len(final_pred_missing_points)} 个 candidate 点既没有匹配到 Round {final_round} 的最终预测，也没有 candidate 自带预测值，左图已自动跳过: "
            f"{_format_point_list(final_pred_missing_points)}"
        )


if __name__ == "__main__":
    plot_multi_round_parity_progress()
