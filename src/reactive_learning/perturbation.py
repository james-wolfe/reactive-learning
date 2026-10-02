"""Perturbations of a GTFT population (Sec. III D, Figs. S7 and S8)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .agents import WellMixedReactive, avg_cooperation

LearningRateRegime = Literal["variable", "uniform"]


def gtft_strategy(*, b: float, c: float, epsilon: float) -> np.ndarray:
    """Return generous tit-for-tat, (1 - epsilon, 1 - epsilon - c/b)."""
    p = 1.0 - epsilon
    return np.array([p, p - c / b])


def draw_learning_rates(
    regime: LearningRateRegime,
    num_agents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the variable or uniform learning rates used in the paper simulations."""
    if regime == "variable":
        return 0.01 + 0.38 * rng.random(num_agents)
    if regime == "uniform":
        return np.full(num_agents, 0.2)
    raise ValueError("regime must be 'variable' or 'uniform'")


def simulate_observables(
    initial_agents: np.ndarray,
    eta: np.ndarray,
    *,
    steps: int,
    save_every: int,
    b: float,
    c: float,
    epsilon: float,
    self_interactions: bool,
    seed: int,
    return_final_agents: bool = False,
) -> tuple[np.ndarray, ...]:
    """Simulate one population and save mean p, mean q and average cooperation (Fig. S8)."""
    if save_every <= 0:
        raise ValueError("save_every must be positive")

    saved_steps = np.arange(0, steps + 1, save_every)
    mean_p = np.empty(saved_steps.size)
    mean_q = np.empty(saved_steps.size)
    cooperation = np.empty(saved_steps.size)
    system = WellMixedReactive(
        initial_agents.copy(),
        eta.copy(),
        b=b,
        c=c,
        epsilon=epsilon,
        self_interactions=self_interactions,
        rng=np.random.default_rng(seed),
    )

    save_index = 0
    for step in range(steps + 1):
        if step % save_every == 0:
            mean_p[save_index] = system.agents[:, 0].mean()
            mean_q[save_index] = system.agents[:, 1].mean()
            cooperation[save_index] = avg_cooperation(system.agents)
            save_index += 1
        if step < steps:
            system.step()

    result = (saved_steps, mean_p, mean_q, cooperation)
    if return_final_agents:
        return (*result, system.agents.copy())
    return result


def displace_within_disk(
    agents: np.ndarray, radius: float, rng: np.random.Generator, *, epsilon: float
) -> np.ndarray:
    """Independently displace agents area-uniformly in the inward-facing half-disk.

    Angles span [pi/2, 3*pi/2], so p can only decrease. Strategies are then clipped
    to the model bounds. At GTFT the default radius 0.01 needs no clipping; larger
    radii can intersect other boundaries and lose area-uniformity after clipping.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    radial_distance = radius * np.sqrt(rng.random(len(agents)))
    theta = rng.uniform(np.pi / 2, 3 * np.pi / 2, len(agents))
    displaced = agents.copy()
    displaced[:, 0] += radial_distance * np.cos(theta)
    displaced[:, 1] += radial_distance * np.sin(theta)
    return np.clip(displaced, epsilon, 1 - epsilon)
