# Figure S7 (scripts/figures/figS07_gtft_q_shock.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style

with np.load(DATA_DIR / "gtft_q_shock.npz") as data:
    steps, agents, eta = data["steps"], data["agents"], data["eta"]
    regimes = data["rate_regimes"].tolist()
    p_gtft = 1 - float(data["epsilon"])
    q_gtft = p_gtft - float(data["c"]) / float(data["b"])
variable, uniform = regimes.index("variable"), regimes.index("uniform")

configure_paper_style()
fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.7))
norm, cmap = plt.Normalize(vmin=0.01, vmax=0.39), plt.cm.viridis
for ax, (label, k, gtft_value) in zip(axes, [("$p$", 0, p_gtft), ("$q$", 1, q_gtft)], strict=True):
    # With a shared rate, identical agents stay identical: one line is the whole population.
    ax.plot(steps, agents[uniform][:, 0, k], color="0.35", linestyle="--", linewidth=1.4,
            alpha=0.9, zorder=10, label="uniform learning rates")
    for i in np.argsort(eta[variable]):  # fastest learners drawn last, on top
        ax.plot(steps, agents[variable][:, i, k], color=cmap(norm(eta[variable][i])), alpha=0.36,
                linewidth=0.8, zorder=2)
    ax.axhline(gtft_value, color="0.6", linestyle=":", linewidth=1, zorder=0)
    ax.set(xlabel="step", ylabel=label)
    ax.grid(alpha=0.2)
axes[0].legend(loc="lower right", frameon=False)
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, label=r"Learning rate $\eta$",
             shrink=0.9, pad=0.02)
fig.savefig(FIGURES_DIR / "gtft_perturbation_trajectories_q_infinitesimal.pdf")
