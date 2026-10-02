"""Uniform-rate population with Gaussian noise added to every update (Fig. S10)."""

from __future__ import annotations

import numpy as np

from reactive_learning.agents import WellMixedReactive
from reactive_learning.paths import DATA_DIR

B = 1.0
C = 0.5
EPSILON = 1e-3
NUM_AGENTS = 2500
SEED = 29
STEPS = 2001
STEPS_TO_SAVE = tuple(range(0, STEPS, 10))
LOW_NOISE_SD = 0.01
HIGH_NOISE_SD = 0.05


def run_snapshots(
    agents: np.ndarray,
    eta: np.ndarray,
    *,
    noise_sd: float,
) -> dict[int, np.ndarray]:
    """Run one noisy population condition and return selected snapshots."""
    system = WellMixedReactive(
        agents,
        eta,
        b=B,
        c=C,
        epsilon=EPSILON,
        noise_sd=noise_sd,
        rng=np.random.RandomState(SEED),
    )
    _, snapshots = system.run(steps=STEPS, steps_to_save=STEPS_TO_SAVE)
    return snapshots


def generate_data() -> None:
    """Generate the low- and high-noise population data file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    agents = np.clip(
        np.random.RandomState(SEED).random((NUM_AGENTS, 2)),
        EPSILON,
        1.0 - EPSILON,
    )
    eta = np.full(NUM_AGENTS, 0.2)
    low_noise_snapshots = run_snapshots(agents.copy(), eta, noise_sd=LOW_NOISE_SD)
    high_noise_snapshots = run_snapshots(agents.copy(), eta, noise_sd=HIGH_NOISE_SD)

    np.savez_compressed(
        DATA_DIR / "agent_noise.npz",
        steps=np.array(STEPS_TO_SAVE),
        low_noise_agents=np.stack([low_noise_snapshots[step] for step in STEPS_TO_SAVE]),
        high_noise_agents=np.stack([high_noise_snapshots[step] for step in STEPS_TO_SAVE]),
        eta=eta,
        low_noise_sd=LOW_NOISE_SD,
        high_noise_sd=HIGH_NOISE_SD,
        b=B,
        c=C,
        epsilon=EPSILON,
        seed=SEED,
        self_interactions=True,
    )


def main() -> None:
    generate_data()


if __name__ == "__main__":
    main()
