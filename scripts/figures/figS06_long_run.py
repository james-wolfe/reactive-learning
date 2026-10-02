# Figure S6 (scripts/figures/figS06_long_run.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style

with np.load(DATA_DIR / "agent_variable_long_run_n125_rate100.npz") as data:
    steps, agents, weights = data["steps"], data["agents"], data["weights"]
    b, c = float(data["b"]), float(data["c"])
    focal = int(data["special_count"])  # agents [:focal] started off the p = 1 - epsilon line
    requested = int(data["requested_steps"])
print(f"Completed {steps[-1]:,} / {requested:,} additional updates; N = {len(weights)}.")

# Weighted payoff-gradient magnitude, before learning-rate scaling or projection.
gradient_magnitudes = np.empty(agents.shape[:2])
for frame, state in enumerate(agents):
    p, q = state.T
    pi, qi, pj, qj = p[:, None], q[:, None], p[None, :], q[None, :]
    denom = (1 + pj * qi - qi * qj + pi * (-pj + qj)) ** 2
    common = c + b * (-pj + qj)
    gp = (-(pj * qi + qj - qi * qj) * common / denom) @ weights
    gq = (((qj - 1) + pi * (pj - qj)) * common / denom) @ weights
    gradient_magnitudes[frame] = np.hypot(gp, gq)

p, q = agents[-1].T
print("min p: ", p.min(), "| max p: ", p.max())
print("min q: ", q.min(), "| max q: ", q.max())

configure_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.6))
for coordinate, (ax, name) in enumerate(zip(axes[:2], ["p", "q"], strict=True)):
    ax.plot(steps, agents[:, focal:, coordinate], color="slategrey", alpha=0.3, lw=0.6)
    ax.plot(steps, agents[:, :focal, coordinate], color="tab:red", lw=1.4)
    ax.plot([], [], color="tab:red", label="Focal agent")
    ax.plot([], [], color="slategrey", label="Other agents")
    # Short dashes, so two whole dashes fit in the short legend sample below.
    ax.plot(steps, agents[:, :, coordinate] @ weights, color="black", ls=(0, (2.5, 1.5)), lw=1.4,
            label="Weighted avg.")
    ax.set_xscale("symlog", linthresh=1)
    ax.xaxis.get_major_locator().set_params(numticks=6)
    ax.set(xlabel="Additional updates", ylabel=rf"${name}$", title=rf"${name}$ trajectories")
# Lower-right corner, clear of the focal agent's rise near 10^2 updates.
axes[0].legend(loc="lower right", borderaxespad=0.2, handlelength=1.2, frameon=False)

ax = axes[2]
ax.plot(steps, gradient_magnitudes[:, focal:], color="slategrey", alpha=0.3, linewidth=0.6)
ax.plot(steps, gradient_magnitudes[:, :focal], color="tab:red", linewidth=1.4)
ax.set_xscale("symlog", linthresh=1)
ax.xaxis.get_major_locator().set_params(numticks=6)
ax.set_yscale("log")
ax.set_ylim(1e-15, None)
ax.yaxis.set_major_locator(ticker.FixedLocator([10.0**e for e in range(-1, -16, -2)]))
ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.set(xlabel="Additional updates", ylabel=r"$\|\nabla \Pi\|_2$", title="Gradient magnitude")
fig.savefig(FIGURES_DIR / "long_run_traj.pdf", dpi=300)
