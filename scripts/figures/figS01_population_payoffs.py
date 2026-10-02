# Figure S1 (scripts/figures/figS01_population_payoffs.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style


def individual_payoffs(agents, b, c):
    """Mean payoff b * (partner cooperation) - c * (own cooperation), self-play included."""
    p, q = agents.T
    r = p - q
    own = np.empty(len(agents))
    received = np.zeros(len(agents))
    for start in range(0, len(agents), 250):  # blocks avoid an N x N matrix
        block = slice(start, start + 250)
        r_i = r[block, None]
        cooperation = (q[block, None] + r_i * q[None, :]) / (1 - r_i * r[None, :])
        own[block] = cooperation.mean(axis=1)
        received += cooperation.sum(axis=0) / len(agents)
    return b * received - c * own


runs = {}
for regime in ("uniform", "variable"):
    with np.load(DATA_DIR / f"agent_{regime}.npz") as data:
        assert bool(data["self_interactions"])
        b, c = float(data["b"]), float(data["c"])
        payoffs = np.array([individual_payoffs(state, b, c) for state in data["agents"]])
        runs[regime] = data["steps"], payoffs, data["eta"]

configure_paper_style()
fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.8), sharey=True)
cmap, norm = plt.get_cmap("viridis"), plt.Normalize(vmin=0.01, vmax=0.39)
low = min(payoffs.min() for _, payoffs, _ in runs.values())
high = max(payoffs.max() for _, payoffs, _ in runs.values())
padding = 0.04 * (high - low)

for ax, (regime, (steps, payoffs, eta)) in zip(axes, runs.items(), strict=True):
    lines = ax.plot(steps, payoffs, linewidth=0.4, alpha=0.10, zorder=1, rasterized=True)
    for line, rate in zip(lines, eta, strict=True):
        line.set_color(cmap(norm(rate)))
    ax.plot(steps, payoffs.mean(axis=1), color="black", linewidth=1.5, zorder=3,
            label="Population mean")
    ax.set(title=f"{regime.capitalize()} learning rates", xlabel="Step",
           xlim=(steps[0], steps[-1]), ylim=(low - padding, high + padding))
    ax.set_xscale("symlog")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Payoff")
axes[0].legend(frameon=False)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.02,
             label=r"Learning rate $(\eta)$")
fig.savefig(FIGURES_DIR / "average_population_welfare.pdf")
