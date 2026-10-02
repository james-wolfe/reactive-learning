# Figure 2 (scripts/figures/fig02_agent_trajectories.py)
# Extracted from notebooks/main_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, add_eta_colorbar, configure_paper_style

configure_paper_style()
fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 3.5), sharex="col", sharey=True)
norm = plt.Normalize(vmin=0.01, vmax=0.39)
cmap = plt.get_cmap("viridis")

for column, (regime, title) in enumerate(
    [("uniform", "Uniform Learning Rates"), ("variable", "Variable Learning Rates")]
):
    with np.load(DATA_DIR / f"agent_{regime}.npz") as data:
        steps, agents, eta = data["steps"], data["agents"], data["eta"]
    for row, label in enumerate(["$p$", "$q$"]):
        ax = axes[row, column]
        lines = ax.plot(steps, agents[:, :, row], alpha=0.18, linewidth=0.35)
        for line, rate in zip(lines, eta, strict=True):
            line.set_color(cmap(norm(rate)))
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        if row == 0:
            ax.set_title(title, pad=9)
        else:
            ax.set_xlabel("Step")
        if column == 0:
            ax.set_ylabel(label)

add_eta_colorbar(fig, axes)
fig.savefig(FIGURES_DIR / "agent_based_dist.pdf", dpi=300)
