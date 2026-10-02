# Figure 1 (scripts/figures/fig01_conceptual_comparison.py)
# Extracted from notebooks/main_figs.ipynb by scripts/sync_figure_scripts.py.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from reactive_learning.paths import FIGURES_DIR
from reactive_learning.plotting import FIGURE_WIDTH, configure_paper_style

# Illustrative values.
# Each arrow is rate * gradient; the two agents in a pair share a
# gradient. Agents: bottom-left, bottom-right, top-left, top-right.
STARTS = np.array([(0.63, 0.23), (0.75, 0.23), (0.30, 0.80), (0.42, 0.74)])
DIRECTIONS = np.array([(0, 1), (0, 1), (-0.12, -0.18), (-0.12, -0.18)])
GRADIENTS = np.repeat([1.65, 1.2], 2)  # bottom pair, top pair
UNIFORM_RATE = 0.2
VARIABLE_RATES = [0.165, 0.295, 0.25, 0.12]
NORM = mpl.colors.Normalize(0.09, 0.31)

configure_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.2))
fig.get_layout_engine().set(wspace=0.12)
for ax in axes:
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], box_aspect=1)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$q$", rotation=0, labelpad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def agent(ax, point, color):
    ax.scatter(*point, color=color, s=135, zorder=3, edgecolor="k", linewidth=0.6)


def arrow(ax, start, end, **kwargs):
    props = {"arrowstyle": "->", "linewidth": 1.44, "shrinkA": 7, "color": "k", **kwargs}
    ax.annotate("", xy=end, xytext=start, arrowprops=props)


imitation = [(0.2, 0.3), (0.8, 0.7), (0.27, 0.63), (0.54, 0.25)]
for point, color in zip(imitation, ["0.3", "0.3", "0.75", "0.75"], strict=True):
    agent(axes[0], point, color)
arrow(axes[0], imitation[0], imitation[1], shrinkB=6.3)
axes[0].set_title("Imitation", pad=9)

gradients = DIRECTIONS / np.hypot(*DIRECTIONS.T)[:, None] * GRADIENTS[:, None]
for ax, title, rates in [(axes[1], "Uniform Learning Rates", [UNIFORM_RATE] * 4),
                         (axes[2], "Variable Learning Rates", VARIABLE_RATES)]:
    for start, gradient, rate in zip(STARTS, gradients, rates, strict=True):
        agent(ax, start, plt.cm.viridis(NORM(rate)))
        arrow(ax, start, start + rate * gradient)
    ax.set_title(title, pad=9)

bar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=NORM),
                   cax=axes[2].inset_axes([1.12, 0, 0.06, 1]))
bar.set_ticks([NORM.vmin, NORM.vmax], labels=["Low", "High"])
bar.ax.tick_params(length=0)
bar.set_label(r"Learning Rate ($\eta$)", labelpad=-10)

fig.savefig(FIGURES_DIR / "cartoon.pdf")
