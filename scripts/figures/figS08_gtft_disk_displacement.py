# Figure S8 (scripts/figures/figS08_gtft_disk_displacement.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style

METRICS = {"cooperation": "Average cooperation", "mean_p": "Mean $p$", "mean_q": "Mean $q$"}
COLORS = {"variable": "tab:blue", "uniform": "tab:orange"}

with np.load(DATA_DIR / "gtft_disk_displacement.npz") as data:
    steps, regimes = data["steps"], [str(regime) for regime in data["rate_regimes"]]
    histories = {regime: {key: data[key][i] for key in METRICS} for i, regime in enumerate(regimes)}
    final_agents, final_step = data["final_agents"], int(data["simulation_steps"])
    radius, epsilon = float(data["displacement_radius"]), float(data["epsilon"])
    p_gtft = 1 - epsilon
    q_gtft = p_gtft - float(data["c"]) / float(data["b"])

# Endpoint of every replicate: variable-rate populations back on p = 1 - epsilon;
# uniform-rate populations on q = epsilon with p < 1/2 (Theorem 1 then gives ALLD).
ATOL = 1e-10
for regime, agents in zip(regimes, final_agents, strict=True):
    if regime == "variable":
        passed = np.all(np.abs(agents[:, :, 0] - p_gtft) <= ATOL, axis=1)
        target = "every agent at p = 1 - epsilon"
    else:
        passed = np.all(np.abs(agents[:, :, 1] - epsilon) <= ATOL, axis=1) & np.all(
            agents[:, :, 0] < 0.5, axis=1
        )
        target = "every agent at q = epsilon and p < 1/2"
    print(f"{regime}: {passed.sum()}/{len(passed)} populations have {target} "
          f"at step {final_step} (atol={ATOL:g}).")

configure_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.8), sharex=True)
references = {"cooperation": 1.0, "mean_p": p_gtft, "mean_q": q_gtft}
for ax, (metric, title) in zip(axes, METRICS.items(), strict=True):
    for regime in ("variable", "uniform"):
        values = histories[regime][metric]  # (replicate, saved step)
        for row in values:
            ax.plot(steps, row, color=COLORS[regime], alpha=min(1.0, max(0.025, 6 / len(values))),
                    linewidth=0.8, zorder=5)
    ax.axhline(references[metric], color="0.4", linestyle=":", linewidth=1, zorder=3)
    ax.set(title=title, xlabel="step")
    ax.grid(alpha=0.2)
axes[0].set_ylabel("value")
fig.legend(
    handles=[Patch(facecolor=COLORS[regime], edgecolor="none", label=f"{regime} learning rates")
             for regime in ("variable", "uniform")],
    loc="outside lower center", ncol=2, frameon=False,
)
fig.savefig(FIGURES_DIR / f"gtft_perturbation_disk_displacement_r{radius:g}_lines.pdf")
