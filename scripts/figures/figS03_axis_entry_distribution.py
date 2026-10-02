# Figure S3 (scripts/figures/figS03_axis_entry_distribution.py)
# Extracted from notebooks/sup_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, uniform

from reactive_learning.paths import DATA_DIR, FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style
from reactive_learning.transient_unforgiving import convergence_bounds

with np.load(DATA_DIR / "axis_entry.npz") as data:
    conditions, agents = data["conditions"], data["agents"]
    epsilon, b, c = (float(data[key]) for key in ("epsilon", "b", "c"))
    fraction, p_max = float(data["fraction_alld"]), float(data["p_max"])
    fraction_tft, beta_shapes = float(data["fraction_tft"]), data["beta_shapes"]
    target_margin = float(data["certificate_margin"])
    p_grid, survival = data["p_grid"], data["theoretical_survival"]

# Example distributions (bottom row), each certified by Theorem 1 with margin 0.001.
examples = (
    ("ALLD", uniform(loc=epsilon, scale=p_max - epsilon), fraction),
    ("TFT", beta(*beta_shapes, loc=epsilon, scale=1 - 2 * epsilon), fraction_tft),
)
print(f"ALLD example: {100 * fraction:.1f}% atom at epsilon + uniform on [epsilon, {p_max:.6f}]")
for expected, distribution, atom_mass in examples:
    bounds = convergence_bounds(distribution, b=b, c=c, epsilon=epsilon, atom_mass=atom_mass)
    print(f"{expected} example: P={bounds.positive:.9f}, Q={bounds.negative:.9f}, "
          f"U={-bounds.alld_margin:.9f}, P-Q={bounds.tft_margin:.9f}; "
          f"certificate: {bounds.outcome}")
    margin = bounds.alld_margin if expected == "ALLD" else bounds.tft_margin
    assert bounds.outcome == expected and np.isclose(margin, target_margin, atol=1e-10)

configure_paper_style()
fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 4.7), sharex="col")
axes[1, 1].sharey(axes[0, 1])

# Top row: empirical p at the first step with every agent on q = epsilon.
colors = {"Uniform learning rates": "C1", "Variable learning rates": "C0"}
bins = np.linspace(epsilon, 1 - epsilon, 11)
for condition, state in zip(conditions, agents, strict=True):
    label, color = str(condition), colors[str(condition)]
    weights = np.full(len(state), 1 / len(state))
    axes[0, 0].hist(state[:, 0], bins=bins, weights=weights, histtype="stepfilled",
                    color=color, alpha=0.18)
    axes[0, 0].hist(state[:, 0], bins=bins, weights=weights, histtype="step", color=color,
                    linewidth=1.5, label=label)
    sorted_p = np.sort(state[:, 0])
    thresholds = np.unique(np.r_[0.0, sorted_p, 1.0])
    empirical_survival = 1 - np.searchsorted(sorted_p, thresholds, side="left") / len(sorted_p)
    axes[0, 1].step(thresholds, empirical_survival, where="pre", linewidth=1.5, color=color,
                    label=label)

# Bottom row: densities of the continuous parts (atoms marked at the baseline).
# The grid approaches the Beta's integrable singularity at epsilon without evaluating it.
density_grid = np.unique(np.r_[
    epsilon + (1 - 2 * epsilon) * np.geomspace(1e-8, 1, 500),
    np.linspace(epsilon + 1e-8, 1 - epsilon, 2000),
    p_max,
    np.nextafter(p_max, np.inf),
])
labels = (
    rf"ALLD: {100 * fraction:.1f}\% atom + uniform",
    "TFT: " + (rf"{100 * fraction_tft:.2f}\% atom + " if fraction_tft else "")
    + rf"Beta$({beta_shapes[0]:.4f}, {beta_shapes[1]:g})$",
)
for (_, distribution, atom_mass), curve, color, label in zip(
    examples, survival, ("C3", "#008b8b"), labels, strict=True
):
    density = (1 - atom_mass) * distribution.pdf(density_grid)
    axes[1, 0].fill_between(density_grid, density, color=color, alpha=0.18)
    axes[1, 0].plot(density_grid, density, linewidth=1.5, color=color, label=label)
    if atom_mass:
        axes[1, 0].plot(epsilon, 0, "o", color=color, clip_on=False, zorder=5)
        axes[1, 0].annotate(
            rf"{100 * atom_mass:.1f}\% point mass at $\varepsilon$",
            xy=(epsilon, 0), xytext=(0.21, 0.17), textcoords="axes fraction", color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
            arrowprops={"arrowstyle": "->", "color": color, "shrinkA": 4, "shrinkB": 7},
        )
    axes[1, 1].plot(p_grid, curve, linewidth=1.5, color=color, label=label)

for row in range(2):
    axes[row, 0].set(xlabel=r"$p$" if row == 1 else "",
                     ylabel="Population fraction" if row == 0 else "Continuous density",
                     xlim=(-0.05, 1.05), ylim=(0, None))
    ax = axes[row, 1]
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.0002, 1.5), xlabel=r"$p$ threshold" if row == 1 else "",
           ylabel=r"Prob. $P\geq p$")
    ax.set_yticks([0, 1e-3, 1e-2, 1e-1, 1], ["0", "0.001", "0.01", "0.1", "1"])
axes[1, 0].set_ylim(0, 3)
for ax in axes.flat:
    ax.grid(alpha=0.25)
for ax in axes[:, 0]:
    ax.legend(loc="upper right", frameon=False)
fig.savefig(FIGURES_DIR / "axis_entry_p_distribution.pdf")
