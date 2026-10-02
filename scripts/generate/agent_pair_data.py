"""Replicate isolated pairs of noisy gradient learners (Fig. S11)."""

from __future__ import annotations

import numpy as np

from reactive_learning.agents import WellMixedReactive, avg_cooperation_over_time
from reactive_learning.paths import DATA_DIR

B = 1.0
C = 0.5
EPSILON = 1e-3
SEED = 9
NUM_AGENTS = 2
N_RUNS = 1000
STEPS = 2001
STEPS_TO_SAVE = tuple(range(0, STEPS, 10))
ETA = 0.02
NOISE_SD = 0.001


def generate_data() -> None:
    """Generate the repeated noisy two-agent trajectory data file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    eta = np.full(NUM_AGENTS, ETA)
    runs = []

    for _ in range(N_RUNS):
        agents = np.clip(
            rng.random((NUM_AGENTS, 2)),
            EPSILON,
            1.0 - EPSILON,
        )
        system = WellMixedReactive(
            agents,
            eta,
            b=B,
            c=C,
            epsilon=EPSILON,
            self_interactions=False,
            noise_sd=NOISE_SD,
            rng=rng,
        )
        _, snapshots = system.run(steps=STEPS, steps_to_save=STEPS_TO_SAVE)
        runs.append(snapshots)

    np.savez_compressed(
        DATA_DIR / "agent_pairs.npz",
        steps=np.array(STEPS_TO_SAVE),
        agents=np.stack([[run[step] for step in STEPS_TO_SAVE] for run in runs]),
        average_cooperation=avg_cooperation_over_time(runs),
        eta=eta,
        noise_sd=NOISE_SD,
        b=B,
        c=C,
        epsilon=EPSILON,
        seed=SEED,
        self_interactions=False,
    )


def main() -> None:
    generate_data()


if __name__ == "__main__":
    main()
