# Figure 3 (scripts/figures/fig03_stages_and_gradient_fields.py)
# Extracted from notebooks/main_figs.ipynb by scripts/sync_figure_scripts.py.
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import configure_paper_style, plot_agents_and_avg_field_2x4

STAGES = {
    5: "Collapse of forgiveness",
    300: "Transient unforgiving state",
    4609: "Move to GTFT",
    25000: "Stability of GTFT",
}

with np.load(DATA_DIR / "agent_variable.npz") as data:
    steps = data["steps"].tolist()
    snapshots = [data["agents"][steps.index(step)] for step in STAGES]
    eta, b, c = data["eta"], float(data["b"]), float(data["c"])

configure_paper_style()
fig, axs = plot_agents_and_avg_field_2x4(snapshots, STAGES.values(), eta, b=b, c=c)
fig.savefig(FIGURES_DIR / "steps_with_streams.pdf")
