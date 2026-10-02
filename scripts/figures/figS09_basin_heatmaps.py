# Figure S9 (scripts/figures/figS09_basin_heatmaps.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style

with np.load(DATA_DIR / "basin.npz") as data:
    bc_ratios, alphas = data["bc_ratios"], data["alphas"]
    agent_counts, grids = data["agent_counts"], data["cooperation_fraction"]

shown = (bc_ratios >= 1.5 - 1e-9) & (bc_ratios <= 2.5 + 1e-9)
bc_shown = bc_ratios[shown]
half_step = float(np.mean(np.diff(bc_shown))) / 2
bc_edges = [bc_shown[0] - half_step, bc_shown[-1] + half_step]
# Rows run from alpha = 1e6 (a point mass at the mean rate, -log alpha = -infinity) to
# alpha = 0 (equal point masses at 0.01 and 0.39, -log alpha = +infinity); logs are base 10.
alpha_labels = [
    r"$-\infty$" if np.isclose(alpha, 1e6) else r"$\infty$" if np.isclose(alpha, 0.0)
    else rf"${int(np.round(np.log10(1 / alpha)))}$"
    for alpha in alphas[::-1]
]

configure_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.6), sharey=True)
for ax, num_agents, grid in zip(axes, agent_counts, grids, strict=True):
    image = ax.imshow(100 * grid[::-1, :][:, shown], origin="lower", aspect="auto", vmin=0.0,
                      vmax=100.0, extent=[*bc_edges, -0.5, len(alphas) - 0.5],
                      interpolation="nearest")
    ax.set_xlim(bc_edges)
    # Label every other b/c value so the labels do not collide.
    ax.set_xticks(bc_shown, [f"{r:.1f}" if k % 2 == 0 else "" for k, r in enumerate(bc_shown)])
    ax.set(title=rf"$N = {num_agents}$", xlabel=r"Reward ratio $b/c$")
axes[0].set_ylabel(r"$-\log \alpha$")
axes[0].set_yticks(np.arange(len(alphas)), alpha_labels)
fig.colorbar(image, ax=axes, shrink=0.92, pad=0.02,
             label=r"\% of populations ending in cooperation")
fig.savefig(FIGURES_DIR / "heatmaps.pdf", dpi=300)
