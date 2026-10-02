"""Swap learning rates when a population first reaches q = epsilon (Sec. III B).

Each case restarts a main-run population at its first whole-population entry into the
unforgiving line (agent_uniform.npz at step 18, agent_variable.npz at step 270),
replaces the learning rates, and continues the full two-dimensional dynamics:

uniform_to_variable     uniform-rate entry state, rates redrawn from Uniform[0.01, 0.39];
                        stops once Theorem 1 certifies ALLD (U < 0).
switch_to_uniform_0.2   variable-rate entry state, every rate set to 0.2.
keep_variable_rates     control: variable-rate entry state, original rates.
small_rate_long         variable-rate entry state, every rate set to 0.025.

Output: data/generated/axis_entry_rate_sensitivity/<case>.npz. Pass --case to run one.
"""

from __future__ import annotations

import argparse

import numpy as np

from reactive_learning.agents import WellMixedReactive
from reactive_learning.axis_entry import whole_population_on_axis
from reactive_learning.paths import DATA_DIR
from reactive_learning.perturbation import draw_learning_rates
from reactive_learning.transient_unforgiving import empirical_convergence_bounds

OUTPUT_DIR = DATA_DIR / "axis_entry_rate_sensitivity"
SEED = 29

# Source run, entry step, rates afterwards, number of updates, and how often to save:
# one interval, or ((until, every), ...) segments that save more often early on.
CASES = {
    "uniform_to_variable": ("uniform", 18, "variable", 10_000, 100),
    "switch_to_uniform_0.2": ("variable", 270, 0.2, 5_000, 10),
    "keep_variable_rates": ("variable", 270, "original", 5_000, 10),
    "small_rate_long": ("variable", 270, 0.025, 75_000, ((24_000, 10), (75_000, 80))),
}


def sampling_schedule(max_steps: int, save_every) -> np.ndarray:
    """Return the increasing save points for an interval or ``((until, every), ...)``."""
    segments = ((max_steps, save_every),) if np.isscalar(save_every) else save_every
    points, start = [np.zeros(1, dtype=int)], 0
    for until, every in segments:
        until = min(int(until), max_steps)
        points.append(np.arange(start + every, until + 1, every, dtype=int))
        start = until
    schedule = np.unique(np.concatenate([*points, [max_steps]]))
    return schedule[schedule <= max_steps]


def generate_case(name: str) -> None:
    source, entry_step, rate_choice, max_steps, save_every = CASES[name]
    with np.load(DATA_DIR / f"agent_{source}.npz") as data:
        initial = data["agents"][data["steps"].tolist().index(entry_step)].copy()
        original_eta = data["eta"].copy()
        parameters = {key: float(data[key]) for key in ("b", "c", "epsilon")}
        assert bool(data["self_interactions"]) and float(data["noise_sd"]) == 0
    assert whole_population_on_axis(initial, epsilon=parameters["epsilon"], atol=0)
    if rate_choice == "variable":
        eta = draw_learning_rates("variable", len(initial), np.random.default_rng(SEED))
    elif rate_choice == "original":
        eta = original_eta.copy()
    else:
        eta = np.full(len(initial), float(rate_choice))

    # The low-memory gradient path (vectorized_threshold=0) matches the saved data.
    system = WellMixedReactive(initial.copy(), eta.copy(), **parameters, vectorized_threshold=0)
    epsilon = system.epsilon
    schedule = sampling_schedule(max_steps, save_every)
    states, first_boundary = [initial.copy()], -1
    for previous, target in zip(schedule[:-1], schedule[1:], strict=True):
        for step in range(previous + 1, target + 1):
            system.step()
            if first_boundary < 0 and np.all(system.agents[:, 0] == 1 - epsilon):
                first_boundary = step
        states.append(system.agents.copy())
        if target % 5000 == 0 or target == max_steps:
            print(f"{name}: {target:,} updates; mean p={system.agents[:, 0].mean():.6f}, "
                  f"mean q={system.agents[:, 1].mean():.6f}", flush=True)
        if (name == "uniform_to_variable" and target % 100 == 0
                and np.all(system.agents[:, 1] == epsilon)):
            margin = empirical_convergence_bounds(system.agents, **parameters, atol=0).alld_margin
            if margin > 0:
                print(f"{name}: Theorem 1 certifies ALLD at update {target} (-U = {margin:.4g})")
                break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / f"{name}.npz",
        agents=np.array(states),
        steps=schedule[: len(states)],
        eta=eta,
        original_eta=original_eta,
        source_file=f"agent_{source}.npz",
        source_step=entry_step,
        rate_choice=str(rate_choice),
        requested_steps=max_steps,
        first_all_p_boundary_step=first_boundary,
        self_interactions=True,
        noise_sd=0.0,
        vectorized_threshold=0,
        **parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(CASES), dest="cases",
                        help="run only this case (repeatable); default: all cases")
    for name in parser.parse_args().cases or CASES:
        generate_case(name)


if __name__ == "__main__":
    main()
