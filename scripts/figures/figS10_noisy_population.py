# Figure S10 (scripts/figures/figS10_noisy_population.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style, tsplot_runs

with np.load(DATA_DIR / "agent_noise.npz") as data:
    steps = data["steps"]
    runs = [(data["low_noise_agents"], float(data["low_noise_sd"])),
            (data["high_noise_agents"], float(data["high_noise_sd"]))]

configure_paper_style()
fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 3.5), sharex=True, sharey=True)
for column, (agents, noise_sd) in enumerate(runs):
    for row, color in enumerate(["C3", "C0"]):  # p (top), q (bottom)
        # agents has shape (saved step, agent, p/q); every agent is one thin line.
        tsplot_runs(steps, agents[:, :, row].T, color=color, ax=axes[row, column],
                    label_runs=(row, column) == (0, 0), line_alpha=0.03)
    axes[0, column].set_title(f"SD = {noise_sd:g}", pad=9)
    axes[1, column].set_xlabel("Step")
axes[0, 0].set_ylabel(r"$p$")
axes[1, 0].set_ylabel(r"$q$")
for ax in axes.ravel():
    ax.set_ylim(-0.05, 1.05)
axes[0, 0].legend(frameon=False, loc="lower center")

fig.savefig(FIGURES_DIR / "agent_based_dist_noisy_runs.pdf", dpi=300)
