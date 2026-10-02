"""Plotting helpers shared by the paper figures (notebooks/main_figs.ipynb, sup_figs.ipynb)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator

from .agents import vector_field_against_population
from .paths import FIGURES_DIR

# Every figure is drawn at its printed width, the full text width of Physical Review X
# (revtex figure*), so its 9-pt labels print at 9 pt when included at width=\textwidth.
FIGURE_WIDTH = 7.0  # inches


def configure_paper_style() -> None:
    """Apply the paper's Matplotlib style (LaTeX text) and create the figure directory."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            # Constrained layout keeps every label inside the figure, so saved PDFs are
            # exactly FIGURE_WIDTH wide (no bbox_inches="tight" cropping).
            "figure.constrained_layout.use": True,
        }
    )
    FIGURES_DIR.mkdir(exist_ok=True)


def tsplot_freq(x, support, weights, *, bands=(99.9, 80.0, 60.0, 40.0, 20.0), color="r", ax=None):
    """Shade nested weighted-percentile bands of a distribution over time.

    ``weights[k, t]`` is the probability mass at ``support[k]`` at time ``x[t]``.
    ``bands`` are central coverages in percent (80 shades the 10th-90th
    percentiles). Each band has alpha 1.5 / len(bands), so overlapping bands
    darken the centre of the distribution. The black line is the mean.
    """
    ax = ax or plt.gca()
    bands = np.asarray(bands, dtype=float)
    grid = np.tile(np.asarray(support, dtype=float)[:, None], (1, weights.shape[1]))
    lower, upper = (
        np.percentile(grid, 50 + sign * bands / 2, axis=0, method="inverted_cdf", weights=weights)
        for sign in (-1, 1)
    )
    for low, high in zip(lower, upper, strict=True):
        ax.fill_between(x, low, high, alpha=min(1.5 / len(bands), 1.0), color=color, edgecolor=None)
    ax.plot(x, support @ weights, color="k")
    return ax


def tsplot_runs(
    x,
    y,
    *,
    ax=None,
    color="C0",
    percentile_band=(2.5, 97.5),
    max_lines=None,
    line_alpha=None,
    line_width=0.4,
    mean_color="k",
    mean_linewidth=1.5,
    band_color="k",
    band_linewidth=0.9,
    band_linestyle="--",
    band_alpha=0.7,
    rasterized=True,
    label_runs=False,
):
    """Overlay individual run trajectories with their mean and percentile band.

    Matches the two-agent cooperation figure: thin translucent lines for every
    run, a solid mean, and dashed lines at ``percentile_band``.  ``line_alpha``
    defaults to a value scaled to the number of drawn runs, so a few dozen runs
    stay visible while a few thousand do not saturate into a solid block.
    """
    if ax is None:
        ax = plt.gca()
    y = np.asarray(y, dtype=float)
    if y.shape[0] == len(x) and y.shape[1] != len(x):
        y = y.T
    mean = np.nanmean(y, axis=0)
    low = np.nanpercentile(y, percentile_band[0], axis=0)
    high = np.nanpercentile(y, percentile_band[1], axis=0)
    line_indices = (
        np.arange(y.shape[0])
        if max_lines is None or y.shape[0] <= max_lines
        else np.linspace(0, y.shape[0] - 1, max_lines, dtype=int)
    )
    if line_alpha is None:
        line_alpha = float(np.clip(20.0 / len(line_indices), 0.008, 0.35))
    ax.plot(
        x,
        y[line_indices].T,
        color=color,
        alpha=line_alpha,
        linewidth=line_width,
        zorder=1,
        rasterized=rasterized,
    )
    ax.plot(
        x,
        mean,
        color=mean_color,
        linewidth=mean_linewidth,
        zorder=5,
        label="Mean" if label_runs else None,
    )
    for values, percentile in zip((low, high), percentile_band, strict=True):
        ax.plot(
            x,
            values,
            color=band_color,
            linestyle=band_linestyle,
            linewidth=band_linewidth,
            alpha=band_alpha,
            zorder=5,
            label=rf"${percentile}\%$" if label_runs else None,
        )
    return ax


def plot_agents_and_avg_field_2x4(
    snapshots, titles, eta, *, b, c, grid_size=23, figsize=(FIGURE_WIDTH, 3.64)
):
    """Four population snapshots (top, colored by learning rate) above streamlines of
    the gradient field against each snapshot's population (bottom)."""
    eta_norm = mpl.colors.Normalize(vmin=0.01, vmax=0.39)
    fig, axs = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
    loc = FixedLocator([0.0, 0.5, 1.0])
    fmt = FixedFormatter(["0", "0.5", "1"])

    for col, (snapshot, title) in enumerate(zip(snapshots, titles, strict=True)):
        ax_top, ax_bottom = axs[:, col]
        ax_top.scatter(
            snapshot[:, 0],
            snapshot[:, 1],
            c=eta,
            cmap=plt.cm.viridis,
            norm=eta_norm,
            alpha=0.35,
            s=6,
        )
        ax_top.set_title(title, fontsize=8)  # long stage names; fits one line
        ax_top.tick_params(labelbottom=False, labelleft=col == 0)

        grid_x, grid_y, field_p, field_q = vector_field_against_population(
            snapshot, b=b, c=c, grid_size=grid_size
        )
        ax_bottom.streamplot(
            grid_x,
            grid_y,
            field_p,
            field_q,
            color="darkslategrey",
            density=1.1,
            linewidth=0.6,
            arrowsize=0.6,
            minlength=0.1,
            maxlength=3.0,
            integration_direction="both",
        )
        ax_bottom.set_xlabel(r"$p$")
        ax_bottom.tick_params(labelleft=col == 0)

        for ax in (ax_top, ax_bottom):
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_ylabel(r"$q$" if col == 0 else "")

    add_eta_colorbar(fig, axs)
    return fig, axs


def add_eta_colorbar(fig, axes):
    """The learning-rate colorbar shared by the two-row figures (Figs. 2, 3, S5)."""
    return fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mpl.colors.Normalize(0.01, 0.39)),
        ax=np.ravel(axes),
        location="right",
        shrink=0.9,
        pad=0.015,
        label=r"Learning Rate ($\eta$)",
    )


def plot_avg_coop_runs(
    values,
    x=None,
    percentile_band=(2.5, 97.5),
    max_lines=1000,
    color_by_initial=True,
    cmap_name="winter",
    show_colorbar=True,
    norm_range=(0.0, 1.0),
    figsize=(FIGURE_WIDTH, 3.2),
):
    values = np.asarray(values, dtype=float)
    valid_cols = ~np.all(np.isnan(values), axis=0)
    values = values[:, valid_cols]
    x_values = np.arange(values.shape[1]) if x is None else np.asarray(x)[valid_cols]
    mean = np.nanmean(values, axis=0)
    low = np.nanpercentile(values, percentile_band[0], axis=0)
    high = np.nanpercentile(values, percentile_band[1], axis=0)
    fig, ax = plt.subplots(figsize=figsize)
    line_indices = (
        np.arange(values.shape[0])
        if values.shape[0] <= max_lines
        else np.linspace(0, values.shape[0] - 1, max_lines, dtype=int)
    )
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(*norm_range)
    initial = values[:, 0]
    for line_idx in line_indices:
        color = cmap(norm(initial[line_idx])) if color_by_initial else None
        ax.plot(x_values, values[line_idx], alpha=0.35, linewidth=0.8, color=color)
    ax.plot(x_values, mean, linewidth=1.8, label="Mean", zorder=5, color="black")
    ax.plot(
        x_values, low, "--", linewidth=1.0, label=r"$2.5\%$", zorder=5, color="black", alpha=0.7
    )
    ax.plot(
        x_values, high, "--", linewidth=1.0, label=r"$97.5\%$", zorder=5, color="black", alpha=0.7
    )
    if color_by_initial and show_colorbar:
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.01).set_label(
            "Initial average cooperation"
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Average cooperation rate of pair")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    return fig, ax
