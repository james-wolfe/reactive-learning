# Figure S5 (scripts/figures/figS05_gtft_transition.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import configure_paper_style, plot_agents_and_avg_field_2x4

STEPS = (4500, 4600, 4650, 4750)

with np.load(DATA_DIR / "agent_variable.npz") as data:
    steps = data["steps"].tolist()
    snapshots = [data["agents"][steps.index(step)] for step in STEPS]
    eta, b, c = data["eta"], float(data["b"]), float(data["c"])

configure_paper_style()
fig, axs = plot_agents_and_avg_field_2x4(
    snapshots, [f"Step {step}" for step in STEPS], eta, b=b, c=c
)
fig.savefig(FIGURES_DIR / "steps_with_streams_GTFT.pdf")
