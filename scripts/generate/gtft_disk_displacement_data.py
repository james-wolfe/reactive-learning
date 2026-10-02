"""GTFT populations after random fractions of agents move within an inward half-disk (Fig. S8)."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from reactive_learning.paths import DATA_DIR
from reactive_learning.perturbation import (
    displace_within_disk,
    draw_learning_rates,
    gtft_strategy,
    simulate_observables,
)

DISPLACEMENT_RADIUS = 0.01
DISPLACEMENT_PERCENT_RANGE = (1, 100)  # percent displaced, drawn uniformly per population
B, C, EPSILON = 1.0, 0.5, 1e-3
SEED, NUM_AGENTS, STEPS, SAVE_EVERY, N_REPLICATES = 9, 250, 500, 4, 500
SELF_INTERACTIONS = True
RATE_REGIMES = ("variable", "uniform")
OUTPUT_PATH = DATA_DIR / "gtft_disk_displacement.npz"


def run_trial(trial):
    regime_index, replicate = trial
    seed = SEED + 80_000 + regime_index * 100_000 + replicate
    rng = np.random.default_rng(seed)
    agents = np.tile(gtft_strategy(b=B, c=C, epsilon=EPSILON), (NUM_AGENTS, 1))
    low, high = DISPLACEMENT_PERCENT_RANGE
    if not 1 <= low <= high <= 100:
        raise ValueError("Displacement percentages must satisfy 1 <= low <= high <= 100")
    percent = low if low == high else rng.uniform(low, high)
    num_displaced = round(percent / 100 * NUM_AGENTS)
    # At 100%, every agent is displaced and no random draw is spent choosing them.
    selected = (
        np.arange(NUM_AGENTS)
        if num_displaced == NUM_AGENTS
        else rng.choice(NUM_AGENTS, size=num_displaced, replace=False)
    )
    agents[selected] = displace_within_disk(
        agents[selected], DISPLACEMENT_RADIUS, rng, epsilon=EPSILON
    )
    eta = draw_learning_rates(RATE_REGIMES[regime_index], NUM_AGENTS, rng)
    _, mean_p, mean_q, cooperation, final_agents = simulate_observables(
        agents,
        eta,
        steps=STEPS,
        save_every=SAVE_EVERY,
        b=B,
        c=C,
        epsilon=EPSILON,
        self_interactions=SELF_INTERACTIONS,
        seed=seed,
        return_final_agents=True,
    )
    return (mean_p, mean_q, cooperation), percent, num_displaced, final_agents


def generate_data(output_path: Path = OUTPUT_PATH) -> None:
    trials = [(i, j) for i in range(len(RATE_REGIMES)) for j in range(N_REPLICATES)]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(run_trial, trials)):
            rows.append(row)
            if (index + 1) % 50 == 0:
                print(f"Half-disk displacement: {index + 1}/{len(trials)} trials", flush=True)
    observables = np.asarray([row[0] for row in rows]).reshape(
        len(RATE_REGIMES), N_REPLICATES, 3, -1
    )
    percentages = np.array([row[1] for row in rows]).reshape(len(RATE_REGIMES), N_REPLICATES)
    counts = np.array([row[2] for row in rows]).reshape(len(RATE_REGIMES), N_REPLICATES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=np.arange(0, STEPS + 1, SAVE_EVERY),
        rate_regimes=np.array(RATE_REGIMES),
        mean_p=observables[:, :, 0],
        mean_q=observables[:, :, 1],
        cooperation=observables[:, :, 2],
        final_agents=np.stack([row[3] for row in rows]).reshape(
            len(RATE_REGIMES), N_REPLICATES, NUM_AGENTS, 2
        ),
        displacement_radius=DISPLACEMENT_RADIUS,
        displacement_percent_range=DISPLACEMENT_PERCENT_RANGE,
        displacement_percent=percentages,
        num_displaced=counts,
        b=B,
        c=C,
        epsilon=EPSILON,
        self_interactions=SELF_INTERACTIONS,
        seed=SEED,
        num_agents=NUM_AGENTS,
        simulation_steps=STEPS,
        save_every=SAVE_EVERY,
        num_replicates=N_REPLICATES,
    )


if __name__ == "__main__":
    generate_data()
