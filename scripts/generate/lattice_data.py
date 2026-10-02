"""Infinite-population (lattice) model, uniform and variable learning rates (Sec. S1.3, Fig. S2).

Each of 100 learning rates carries its own 50 x 50 grid of strategies on [epsilon, 1 - epsilon]^2.
The population starts uniform: every grid point of every learning rate has equal mass. Variable
rates are the 100 equally weighted values linspace(0.01, 0.39); uniform rates are all 0.2.
Runs last 25,000 steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reactive_learning.lattice import ReactiveLattice
from reactive_learning.paths import DATA_DIR

NUM_POINTS = 50
ETA_COUNT = 100
STEPS = 25001
EPSILON = 1e-3
REGIMES = {
    # regime: (learning rates, save every)
    "variable": (np.linspace(0.01, 0.39, ETA_COUNT), 100),
    "uniform": (np.full(ETA_COUNT, 0.2), 200),
}


def generate_lattice_data(output_dir: Path = DATA_DIR, regimes=tuple(REGIMES)) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = np.full((NUM_POINTS, NUM_POINTS, ETA_COUNT), 1 / (NUM_POINTS**2 * ETA_COUNT))
    for regime in regimes:
        eta, save_every = REGIMES[regime]
        lattice = ReactiveLattice(initial.copy(), eta=eta, epsilon=EPSILON)
        _, saved = lattice.run(steps=STEPS, steps_to_save=range(0, STEPS, save_every))
        np.savez_compressed(
            output_dir / f"lattice_{regime}.npz",
            steps=np.arange(0, STEPS, save_every),
            freq=saved,
            eta=eta,
            epsilon=EPSILON,
            b=1.0,
            c=0.5,
            initial_condition="uniform",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--regime", choices=sorted(REGIMES), action="append", dest="regimes")
    args = parser.parse_args()
    generate_lattice_data(args.output_dir, tuple(args.regimes or REGIMES))


if __name__ == "__main__":
    main()
