"""Main populations rerun without self-play (Sec. II B; verified in checks.ipynb).

Both regimes start from the main runs' initial strategies and learning rates (seed 29) and
differ from agent_main_data.py only in self_interactions=False. Every 1,000 updates, a run
stops once its outcome is established (reactive_learning.robustness): ALLD observed or
certified by Theorem 1, a permanent ALLD cohort that rules out cooperation, or the GTFT
regime held for 5,000 updates. Runs last at most 25,000 updates.
"""

import numpy as np

from reactive_learning.agents import WellMixedReactive
from reactive_learning.paths import DATA_DIR
from reactive_learning.robustness import endpoint_evidence, stopping_status

B, C, EPSILON, NUM_AGENTS, SEED = 1.0, 0.5, 1e-3, 2500, 29
MAX_STEPS, CHECK_EVERY, HOLD_STEPS = 25_000, 1_000, 5_000
OUTPUT_DIR = DATA_DIR / "self_interactions"


def generate_case(regime: str) -> None:
    rng = np.random.RandomState(SEED)
    initial = np.clip(rng.random((NUM_AGENTS, 2)), EPSILON, 1 - EPSILON)
    variable_eta = 0.01 + 0.38 * rng.random(NUM_AGENTS)
    eta = variable_eta if regime == "variable" else np.full(NUM_AGENTS, 0.2)
    parameters = dict(b=B, c=C, epsilon=EPSILON, self_interactions=False)
    system = WellMixedReactive(initial.copy(), eta, **parameters)

    steps, frames, gtft_since = [0], [initial.copy()], -1
    for step in range(CHECK_EVERY, MAX_STEPS + 1, CHECK_EVERY):
        system.run(CHECK_EVERY)
        steps.append(step)
        frames.append(system.agents.copy())
        evidence = endpoint_evidence(system.agents, **parameters)
        gtft_since = (step if gtft_since < 0 else gtft_since) if evidence["gtft_regime"] else -1
        status = stopping_status(evidence, step=step, gtft_since=gtft_since, hold_steps=HOLD_STEPS)
        print(f"{regime}: step {step:,}; {status}; ALLD={evidence['alld_fraction']:.2%}, "
              f"GTFT={evidence['gtft_fraction']:.2%}, C={evidence['cooperation']:.4f}", flush=True)
        if status != "unresolved":
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / f"{regime}_eps{EPSILON:g}_self0.npz",
        **parameters,
        **evidence,
        regime=regime,
        n=NUM_AGENTS,
        seed=SEED,
        noise_sd=0.0,
        eta=eta,
        steps=np.array(steps),
        agents=np.array(frames),
        status=status,
        gtft_since=gtft_since,
        check_every=CHECK_EVERY,
        hold_steps=HOLD_STEPS,
        requested_steps=MAX_STEPS,
    )


if __name__ == "__main__":
    for regime in ("uniform", "variable"):
        generate_case(regime)
