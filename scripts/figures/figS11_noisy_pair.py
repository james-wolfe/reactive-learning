# Figure S11 (scripts/figures/figS11_noisy_pair.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import configure_paper_style, plot_avg_coop_runs

with np.load(DATA_DIR / "agent_pairs.npz") as data:
    cooperation, steps = data["average_cooperation"], data["steps"]

configure_paper_style()
# First 125 saved steps (steps 0-1240) of every replicate pair.
fig, ax = plot_avg_coop_runs(cooperation[:, :125], x=steps[:125], cmap_name="cool")
ax.set_ylim(-0.05, 1.05)
fig.savefig(FIGURES_DIR / "avg_coop_two_agents_noisy.pdf", dpi=300)
