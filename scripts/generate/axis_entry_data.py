"""First whole-population entry into q = epsilon, and Theorem 1 examples (Fig. S3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta, uniform

from reactive_learning.axis_entry import find_first_axis_entry
from reactive_learning.paths import DATA_DIR
from reactive_learning.transient_unforgiving import convergence_bounds

EPSILON = 1e-3
B = 1.0
C = 0.5
CERTIFICATE_MARGIN = 1e-3
ALLD_ATOM_MASS = 0.35
TFT_BETA_SECOND_SHAPE = 1.0  # Left-endpoint divergence, finite density at the right endpoint.
CONDITIONS = (
    ("Uniform learning rates", DATA_DIR / "agent_uniform.npz"),
    ("Variable learning rates", DATA_DIR / "agent_variable.npz"),
)
OUTPUT_PATH = DATA_DIR / "axis_entry.npz"


def generate_data(output_path: Path = OUTPUT_PATH) -> None:
    entry_steps = []
    entry_agents = []
    source_files = []

    for label, path in CONDITIONS:
        with np.load(path) as data:
            step, agents = find_first_axis_entry(
                data["steps"],
                data["agents"],
                data["eta"],
                b=B,
                c=C,
                epsilon=EPSILON,
            )
        entry_steps.append(step)
        entry_agents.append(agents)
        source_files.append(path.name)
        print(
            f"{label}: first whole-population entry at step {step}; "
            f"max(q - epsilon) = {np.max(agents[:, 1] - EPSILON):.2e}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fraction_alld = ALLD_ATOM_MASS

    def alld_margin(p_max):
        distribution = uniform(loc=EPSILON, scale=p_max - EPSILON)
        return convergence_bounds(
            distribution, b=B, c=C, epsilon=EPSILON, atom_mass=fraction_alld
        ).alld_margin

    # Largest uniform endpoint retaining the desired margin with the fixed ALLD atom.
    interval = (EPSILON + C / B, 1 - EPSILON)
    p_max = brentq(lambda upper: alld_margin(upper) - CERTIFICATE_MARGIN, *interval)
    alld_p_max_cutoff = brentq(alld_margin, *interval)  # U = 0 is not certified.
    uniform_distribution = uniform(loc=EPSILON, scale=p_max - EPSILON)

    # Beta(a, 1) TFT example (no atom) at the same small TFT margin. Use exact moments
    # to solve the shape, then verify with the theorem's numerical integrals when plotting.
    length = 1 - 2 * EPSILON

    def tft_margin(first, second=TFT_BETA_SECOND_SHAPE):
        mean = length * first / (first + second)
        second_moment = length**2 * first * (first + 1) / ((first + second) * (first + second + 1))
        return B * second_moment + (B - C) * mean - C

    first = brentq(lambda first: tft_margin(first) - CERTIFICATE_MARGIN, 0.1, 100)
    beta_shapes = np.array([first, TFT_BETA_SECOND_SHAPE])
    fraction_tft = 0.0
    tft_distribution = beta(*beta_shapes, loc=EPSILON, scale=length)
    p_grid = np.unique(
        np.r_[
            0.0,
            EPSILON,
            np.nextafter(EPSILON, 1.0),
            np.linspace(EPSILON, 1 - EPSILON, 2000),
            p_max,
            1.0,
        ]
    )
    survival = np.stack(
        [
            fraction_alld * (p_grid <= EPSILON)
            + (1 - fraction_alld) * uniform_distribution.sf(p_grid),
            fraction_tft * (p_grid <= EPSILON) + (1 - fraction_tft) * tft_distribution.sf(p_grid),
        ]
    )
    np.savez_compressed(
        output_path,
        conditions=np.array([label for label, _ in CONDITIONS]),
        source_files=np.array(source_files),
        entry_steps=np.array(entry_steps),
        agents=np.stack(entry_agents),
        p_grid=p_grid,
        theoretical_survival=survival,
        fraction_alld=fraction_alld,
        p_max=p_max,
        fraction_tft=fraction_tft,
        beta_shapes=beta_shapes,
        alld_p_max_cutoff=alld_p_max_cutoff,
        certificate_margin=CERTIFICATE_MARGIN,
        b=B,
        c=C,
        epsilon=EPSILON,
    )


def main() -> None:
    generate_data()


if __name__ == "__main__":
    main()
