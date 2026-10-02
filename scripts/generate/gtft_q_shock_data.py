"""GTFT populations after every agent's q is raised by 1e-6 (Fig. S7)."""

from pathlib import Path

import numpy as np

from reactive_learning.agents import WellMixedReactive
from reactive_learning.paths import DATA_DIR
from reactive_learning.perturbation import draw_learning_rates, gtft_strategy

OUTPUT_PATH = DATA_DIR / "gtft_q_shock.npz"
B, C, EPSILON = 1.0, 0.5, 1e-3
SEED, NUM_AGENTS, STEPS = 9, 250, 400
Q_UP_MAGNITUDE = 1e-6
RATE_REGIMES = ("variable", "uniform")


def generate_data(output_path: Path = OUTPUT_PATH) -> None:
    steps = np.arange(STEPS + 1)
    frames, rates = [], []
    for regime, offset in zip(RATE_REGIMES, (2, 0), strict=True):
        seed = SEED + 60_000 + offset
        eta = draw_learning_rates(regime, NUM_AGENTS, np.random.default_rng(seed))
        agents = np.tile(gtft_strategy(b=B, c=C, epsilon=EPSILON), (NUM_AGENTS, 1))
        agents[:, 1] += Q_UP_MAGNITUDE
        np.clip(agents, EPSILON, 1 - EPSILON, out=agents)
        system = WellMixedReactive(
            agents, eta, b=B, c=C, epsilon=EPSILON, rng=np.random.default_rng(seed)
        )
        # run saves before updating, so append the final state explicitly.
        final, saved = system.run(STEPS, steps_to_save=steps[:-1])
        frames.append(np.stack([*saved.values(), final.copy()]))
        rates.append(eta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=steps,
        agents=np.stack(frames),
        eta=np.stack(rates),
        rate_regimes=np.array(RATE_REGIMES),
        b=B,
        c=C,
        epsilon=EPSILON,
        seed=SEED,
        num_agents=NUM_AGENTS,
        simulation_steps=STEPS,
        save_every=1,
        self_interactions=True,
        q_up_magnitude=Q_UP_MAGNITUDE,
    )


if __name__ == "__main__":
    generate_data()
