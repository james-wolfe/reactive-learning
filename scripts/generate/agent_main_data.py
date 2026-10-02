"""Main populations with uniform and variable learning rates (Figs. 2, 3, S1, S5).

Also the input to axis_entry_data.py, agent_variable_long_run_data.py and
axis_entry_rate_sensitivity_data.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reactive_learning.agents import WellMixedReactive
from reactive_learning.paths import DATA_DIR

B = 1.0
C = 0.5
EPSILON = 1e-3
NUM_AGENTS = 2500
SEED = 29



def run_snapshots(
    agents: np.ndarray,
    eta: np.ndarray,
    *,
    steps: int,
    steps_to_save: list[int],
) -> dict[int, np.ndarray]:
    """Run the deterministic population model and return selected snapshots."""
    system = WellMixedReactive(
        agents,
        eta,
        b=B,
        c=C,
        epsilon=EPSILON,
        rng=np.random.RandomState(SEED),
    )
    _, snapshots = system.run(steps=steps, steps_to_save=steps_to_save)
    return snapshots


def save_snapshots(
    output_path: Path,
    snapshots: dict[int, np.ndarray],
    eta: np.ndarray,
) -> None:
    """Save one population trajectory."""
    saved_steps = np.array(sorted(snapshots))
    np.savez_compressed(
        output_path,
        steps=saved_steps,
        agents=np.stack([snapshots[step] for step in saved_steps]),
        eta=eta,
        noise_sd=0.0,
        b=B,
        c=C,
        epsilon=EPSILON,
        seed=SEED,
        self_interactions=True,
    )


def generate_data() -> None:
    """Generate the variable- and uniform-learning-rate population data files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(SEED)
    agents = np.clip(rng.random((NUM_AGENTS, 2)), EPSILON, 1.0 - EPSILON)
    variable_eta = 0.01 + 0.38 * rng.random(NUM_AGENTS)
    variable_steps = 25001
    variable_steps_to_save = sorted(
        set(range(0, variable_steps, 200))
        | set(range(0, 301))
        | set(range(3500, 4801, 50))
        | {4609}
    )
    variable_snapshots = run_snapshots(
        agents.copy(),
        variable_eta,
        steps=variable_steps,
        steps_to_save=variable_steps_to_save,
    )
    save_snapshots(DATA_DIR / "agent_variable.npz", variable_snapshots, variable_eta)

    uniform_eta = np.full(NUM_AGENTS, 0.2)
    uniform_steps = 15001
    uniform_steps_to_save = sorted(set(range(0, uniform_steps, 100)) | set(range(1, 101)))
    uniform_agents = np.clip(
        np.random.RandomState(SEED).random((NUM_AGENTS, 2)),
        EPSILON,
        1.0 - EPSILON,
    )
    uniform_snapshots = run_snapshots(
        uniform_agents,
        uniform_eta,
        steps=uniform_steps,
        steps_to_save=uniform_steps_to_save,
    )
    save_snapshots(DATA_DIR / "agent_uniform.npz", uniform_snapshots, uniform_eta)


def main() -> None:
    generate_data()


if __name__ == "__main__":
    main()
