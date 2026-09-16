import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator


SAVE_DIR = "optimization_plots"
DATA_PATH = os.path.join(SAVE_DIR, "Surface_Contour_Data.csv")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"


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


def plot_2d_contour_generic(grid_ph, grid_time, z_values, z_label, title, filename, enable_capping=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    label_fontsize = 22
    tick_fontsize = 20
    z_plot = z_values.reshape(grid_ph.shape)

    if enable_capping:
        z_max_cap = np.quantile(z_values, 0.95)
        z_plot = np.clip(z_plot, 0, z_max_cap)

    color_kwargs = {}
    if "Nd" in z_label:
        vmin = 5.0 * np.floor(np.nanmin(z_plot) / 5.0)
        vmax = 5.0 * np.ceil(np.nanmax(z_plot) / 5.0)
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        color_kwargs = {"vmin": vmin, "vmax": vmax}

    image = ax.imshow(
        z_plot,
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

    ax.set_xlabel("pH", fontsize=label_fontsize)
    ax.set_ylabel("Time", fontsize=label_fontsize)
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
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close()


def plot_3d_surface_generic(grid_ph, grid_time, z_values, z_label, title, filename, enable_capping=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    title_suffix = ""
    z_plot = z_values
    if enable_capping:
        z_max_cap = np.quantile(z_values, 0.95)
        z_plot = np.clip(z_values, 0, z_max_cap)
        title_suffix = f"\n(Capped at 95%: {z_max_cap:.1f})"

    surf = ax.plot_surface(
        grid_ph,
        grid_time,
        z_plot.reshape(grid_ph.shape),
        cmap=cm.viridis,
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )

    ax.set_xlabel("pH")
    ax.set_ylabel("Time (h)")
    ax.set_zlabel(z_label)
    ax.set_title(title + title_suffix)
    fig.colorbar(surf, shrink=0.5, aspect=10, label=z_label)

    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_plot_data(data_path=DATA_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Plot data not found: {data_path}. Run test.py once to generate it."
        )

    plot_data = pd.read_csv(data_path).sort_values(["Time(h)", "pH"])
    ph_values = np.sort(plot_data["pH"].unique())
    time_values = np.sort(plot_data["Time(h)"].unique())
    expected_rows = len(ph_values) * len(time_values)
    if len(plot_data) != expected_rows:
        raise ValueError(f"Incomplete grid data: expected {expected_rows} rows, got {len(plot_data)}.")

    grid_ph, grid_time = np.meshgrid(ph_values, time_values)
    return (
        grid_ph,
        grid_time,
        plot_data["Pred_Nd(%)"].to_numpy(),
        plot_data["Pred_Fe(%)"].to_numpy(),
        plot_data["SF_Ratio"].to_numpy(),
    )


def draw_surface_contour_plots(data_path=DATA_PATH):
    os.makedirs(SAVE_DIR, exist_ok=True)
    grid_ph, grid_time, nd_values, fe_values, sf_values = load_plot_data(data_path)

    plot_3d_surface_generic(grid_ph, grid_time, nd_values, "Nd (%)", "Nd Surface", "3D_Surface_Nd.png")
    plot_3d_surface_generic(grid_ph, grid_time, fe_values, "Fe (%)", "Fe Surface", "3D_Surface_Fe.png")
    plot_3d_surface_generic(grid_ph, grid_time, sf_values, "SF Ratio", "SF Surface (Full Range)", "3D_Surface_SF.png")

    plot_2d_contour_generic(grid_ph, grid_time, nd_values, "Nd Recovery Rate (%)", "Nd 2D Projection", "2D_Contour_Nd.svg")
    plot_2d_contour_generic(grid_ph, grid_time, fe_values, "Fe Remaining Rate (%)", "Fe 2D Projection", "2D_Contour_Fe.png")
    plot_2d_contour_generic(grid_ph, grid_time, sf_values, "SF Ratio", "SF 2D Projection", "2D_Contour_SF.png")
    print(f"Surface and contour plots saved to: {SAVE_DIR}")


if __name__ == "__main__":
    draw_surface_contour_plots()
