"""Locate the first step at which an entire population reaches q = epsilon."""

from __future__ import annotations

import numpy as np

from .agents import WellMixedReactive


def whole_population_on_axis(
    agents: np.ndarray,
    *,
    epsilon: float,
    atol: float = 1e-12,
) -> bool:
    """Return whether every agent lies on the clipped q boundary."""
    return bool(np.all(np.isclose(agents[:, 1], epsilon, rtol=0.0, atol=atol)))


def find_first_axis_entry(
    steps: np.ndarray,
    snapshots: np.ndarray,
    eta: np.ndarray,
    *,
    b: float,
    c: float,
    epsilon: float,
) -> tuple[int, np.ndarray]:
    """Refine saved snapshots one step at a time to find the first whole-population entry."""
    steps = np.asarray(steps, dtype=int)
    snapshots = np.asarray(snapshots, dtype=float)
    eta = np.asarray(eta, dtype=float)
    if snapshots.shape != (steps.size, eta.size, 2):
        raise ValueError("snapshots must have shape (n_steps, n_agents, 2)")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("saved steps must be strictly increasing")

    saved_on_axis = np.array(
        [whole_population_on_axis(state, epsilon=epsilon) for state in snapshots]
    )
    on_axis_indices = np.flatnonzero(saved_on_axis)
    if on_axis_indices.size == 0:
        raise ValueError("no saved snapshot has the whole population on q = epsilon")

    upper_index = int(on_axis_indices[0])
    if upper_index == 0:
        return int(steps[0]), snapshots[0].copy()

    lower_index = upper_index - 1
    current_step = int(steps[lower_index])
    upper_step = int(steps[upper_index])
    system = WellMixedReactive(
        agents=snapshots[lower_index].copy(),
        eta=eta.copy(),
        b=b,
        c=c,
        epsilon=epsilon,
    )

    while current_step <= upper_step:
        if whole_population_on_axis(system.agents, epsilon=epsilon):
            return current_step, system.agents.copy()
        system.step()
        current_step += 1

    raise RuntimeError("failed to reproduce the bracketed q = epsilon transition")
