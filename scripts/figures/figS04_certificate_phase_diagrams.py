# Figure S4 (scripts/figures/figS04_certificate_phase_diagrams.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style

# Margins -U (ALLD certificate) and P - Q (TFT certificate) on two parameter grids.
with np.load(DATA_DIR / "convergence_phase_diagrams.npz") as data:
    sign_tolerance = float(data["sign_tolerance"])
    panels = [
        (data["p_max_values"], data["atom_masses"], data["mixture_alld"], data["mixture_tft"],
         "ALLD atom + uniform", r"$p_{\max}$", "ALLD atom mass", False),
        (data["beta_shapes"], data["beta_shapes"], data["full_beta_alld"], data["full_beta_tft"],
         r"Beta on $[\varepsilon,1-\varepsilon]$", r"$\alpha$", r"$\beta$", True),
    ]

configure_paper_style()
colors = ["#eeeeee", "C3", "#008b8b"]  # neither, ALLD, TFT
fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.6))
fig.get_layout_engine().set(wspace=0.08)
for ax, (x, y, alld_margin, tft_margin, title, xlabel, ylabel, log_axes) in zip(
    axes, panels, strict=True
):
    fields = [(-np.maximum(alld_margin, tft_margin), colors[0], -sign_tolerance),
              (alld_margin, colors[1], sign_tolerance),
              (tft_margin, colors[2], sign_tolerance)]
    for field, color, threshold in fields:
        if field.max() > threshold:
            ax.contourf(x, y, field, levels=[threshold, field.max() + 1], colors=[color],
                        alpha=0.7, antialiased=False)
    # Solid line: U = 0. Dashed line: P - Q = 0.
    for margin, style in zip((alld_margin, tft_margin), ["-", "--"], strict=True):
        if margin.min() < 0 < margin.max():
            ax.contour(x, y, margin, levels=[0], colors="black", linewidths=0.8, linestyles=style)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
    ax.set_title(title, pad=10)
    if log_axes:
        ticks = [0.1, 0.3, 1, 3, 10, 30]
        ax.set(xscale="log", yscale="log")
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    else:
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])

legend = fig.legend(
    handles=[
        Patch(facecolor=colors[i], edgecolor="k", label=label)
        for i, label in [(1, "ALLD certified"), (2, "TFT certified"), (0, "Neither certified")]
    ],
    loc="outside lower center", ncol=3, frameon=False,
)
for handle in legend.legend_handles:
    handle.set_alpha(0.7)
fig.savefig(FIGURES_DIR / "convergence_phase_diagrams.pdf")
