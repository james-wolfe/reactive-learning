# Figure S2 (scripts/figures/figS02_lattice_dynamics.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style, tsplot_freq

configure_paper_style()
fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 3.5), sharex=True, sharey=True)
for column, (regime, title) in enumerate(
    [("uniform", "Uniform Learning Rates"), ("variable", "Variable Learning Rates")]
):
    with np.load(DATA_DIR / f"lattice_{regime}.npz") as data:
        steps, epsilon = data["steps"], float(data["epsilon"])
        # freq has shape (p point, q point, learning-rate grid, saved step).
        mass = data["freq"].sum(axis=2)
    grid = np.linspace(epsilon, 1 - epsilon, mass.shape[0])  # the lattice's grid points
    tsplot_freq(steps, grid, mass.sum(axis=1), color="C3", ax=axes[0, column])
    tsplot_freq(steps, grid, mass.sum(axis=0), color="C0", ax=axes[1, column])
    axes[0, column].set_title(title, pad=9)
    axes[1, column].set_xlabel("Step")
axes[0, 0].set_ylabel(r"$p$")
axes[1, 0].set_ylabel(r"$q$")

fig.savefig(FIGURES_DIR / "continuous_over_time.pdf", dpi=300)
