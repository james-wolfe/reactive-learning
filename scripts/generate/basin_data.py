"""Fraction of populations that reach cooperation, over b/c, learning-rate heterogeneity
and population size (Sec. III E, Fig. S9).

Rates are Beta(alpha, alpha) on [0.01, 0.39]; alpha = 0 means equal point masses at the
ends and alpha = 1e6 means every rate equals the mean, 0.2. 25 populations per cell, at
most 50,000 steps; the outcome rule is WellMixedReactive.classify_basin. Populations are
independent and run in parallel (--workers); the results do not depend on the worker count.
Gradients use the faster kernel-matrix form (WellMixedReactive.matmul_gradients).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from reactive_learning.agents import WellMixedReactive
from reactive_learning.paths import DATA_DIR

AGENT_COUNTS = (10, 100, 1000)
BC_RATIOS = np.arange(1.0, 3.1, 0.1)
ALPHAS = np.array([0.0, *np.logspace(-3, 3, 7), 1e6], dtype=float)
NUM_POPS = 25
STEPS = 50001
MIN_ETA = 0.01
MAX_ETA = 0.39


def uniform_to_beta_on_interval(
    samples: np.ndarray, alpha: float, low: float, high: float
) -> np.ndarray:
    from scipy.stats import beta

    samples = np.clip(samples, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return low + (high - low) * beta.ppf(samples, alpha, alpha)


def population_tasks(num_agents: int, seed: int = 9):
    """Yield (alpha index, b/c index, agents, eta, c) for every population with N agents.

    All random draws happen here, before any simulation, so the results do not depend
    on how the populations are distributed over worker processes.
    """
    rng = np.random.default_rng(seed)
    eta_draws = rng.random((num_agents, NUM_POPS))
    pop_samples = np.clip(rng.random((num_agents, 2, NUM_POPS)), 1e-3, 1.0 - 1e-3)
    for alpha_idx, alpha in enumerate(ALPHAS):
        for bc_idx, bc_ratio in enumerate(BC_RATIOS):
            for pop_idx in range(NUM_POPS):
                if np.isclose(alpha, 0.0):
                    eta = np.r_[
                        np.full(num_agents // 2, MIN_ETA),
                        np.full(num_agents - num_agents // 2, MAX_ETA),
                    ]
                elif np.isclose(alpha, 1e6):
                    eta = np.full(num_agents, (MIN_ETA + MAX_ETA) / 2.0)
                else:
                    eta = uniform_to_beta_on_interval(
                        eta_draws[:, pop_idx], alpha, MIN_ETA, MAX_ETA
                    )
                yield alpha_idx, bc_idx, pop_samples[:, :, pop_idx].copy(), eta, 1.0 / bc_ratio


def classify(task) -> tuple[int, float]:
    """Outcome (1 = cooperation) and final average cooperation of one population."""
    _, _, agents, eta, c = task
    system = WellMixedReactive(agents, eta, b=1.0, c=c, matmul_gradients=True)
    return system.classify_basin(steps=STEPS), system.avg_cooperation()


def run_sweep(num_agents: int, workers: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Grids (alpha, b/c) of the cooperating fraction and the mean final cooperation."""
    tasks = list(population_tasks(num_agents))
    outcomes = np.empty(len(tasks))
    cooperation = np.empty(len(tasks))
    # One linear-algebra thread per worker, so workers do not compete for cores and results
    # do not depend on the machine's core count.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (outcome, final) in enumerate(pool.map(classify, tasks)):
            outcomes[index], cooperation[index] = outcome, final
            if (index + 1) % 500 == 0 or index + 1 == len(tasks):
                print(f"N={num_agents}: {index + 1}/{len(tasks)} populations", flush=True)
    shape = (len(ALPHAS), len(BC_RATIOS), NUM_POPS)
    # Sum each cell's populations in order, as a serial loop would.
    totals = np.array([sum(cell) for cell in cooperation.reshape(-1, NUM_POPS)])
    return outcomes.reshape(shape).sum(axis=2) / NUM_POPS, totals.reshape(shape[:2]) / NUM_POPS


def generate_basin_data(workers: int | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grids = [run_sweep(num_agents, workers) for num_agents in AGENT_COUNTS]
    np.savez_compressed(
        DATA_DIR / "basin.npz",
        agent_counts=np.array(AGENT_COUNTS),
        bc_ratios=BC_RATIOS,
        alphas=ALPHAS,
        cooperation_fraction=np.stack([fraction for fraction, _ in grids]),
        average_cooperation=np.stack([average for _, average in grids]),
        num_populations=NUM_POPS,
        steps=STEPS,
        min_eta=MIN_ETA,
        max_eta=MAX_ETA,
        seed=9,
        b=1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: one per CPU core)")
    generate_basin_data(parser.parse_args().workers)


if __name__ == "__main__":
    main()
