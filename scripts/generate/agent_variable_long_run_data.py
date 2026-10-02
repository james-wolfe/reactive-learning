"""Generate Figure S6: 125 weighted agents, 100x rates, 10^10 additional updates.

Starts from agent_variable.npz at its last saved step. The exceptional agent
retains weight 1/2500. Scaling up the rates changes the step-by-step dynamics, so this is
an exploratory run on a smaller population, not an exact prediction for the full one.
Requires a C compiler (cc). Does not overwrite existing output unless --resume is given.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from reactive_learning.long_run import (
    C_SOURCE,
    advance,
    checkpoint_steps,
    compile_kernel,
    sample_population,
    validate_kernel,
)
from reactive_learning.paths import DATA_DIR

INPUT_PATH = DATA_DIR / "agent_variable.npz"
OUTPUT_PATH = DATA_DIR / "agent_variable_long_run_n125_rate100.npz"
SAMPLE_SIZE = 125
RATE_MULTIPLIER = 100.0
STEPS = 10_000_000_000
SEED = 29
GRADIENT_TOLERANCE = 1e-14


def save_archive(path: Path, **data) -> None:
    """Write to a temporary file, then swap it in, so an interrupted save cannot
    corrupt the checkpoint."""
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    os.replace(temporary, path)


def generate_data(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    sample_size: int = SAMPLE_SIZE,
    rate_multiplier: float = RATE_MULTIPLIER,
    steps: int = STEPS,
    seed: int = SEED,
    exact: bool = False,
    resume: bool = False,
) -> None:
    """Sample the last saved main-run population, then run the updates (or resume a saved run)."""
    if steps < 1 or not np.isfinite(rate_multiplier) or rate_multiplier <= 0:
        raise ValueError("steps and rate-multiplier must be positive and finite")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must differ")
    if output_path.exists() and not resume:
        raise ValueError("output exists; use --resume or a different --output")
    if resume and not output_path.exists():
        raise ValueError("no output archive to resume")
    with np.load(input_path) as source:
        b, c, epsilon = (float(source[k]) for k in ("b", "c", "epsilon"))
        if float(source["noise_sd"]) != 0 or not bool(source["self_interactions"]):
            raise ValueError("This continuation requires zero noise and self interactions.")
        source_index = int(np.argmax(source["steps"]))
        source_step = int(source["steps"][source_index])
        full = source["agents"][source_index]
        full_eta = source["eta"]
        if (
            not np.all(np.isfinite(full))
            or not np.all(np.isfinite(full_eta))
            or np.any(full < epsilon)
            or np.any(full > 1 - epsilon)
        ):
            raise ValueError(
                "Source agents/rates must be finite and agents within clipping bounds."
            )
        agents, eta, weights, indices, special_count = sample_population(
            full,
            full_eta,
            sample_size,
            epsilon,
            seed,
        )
    eta *= rate_multiplier
    tolerance = 0.0 if exact else GRADIENT_TOLERANCE
    metadata = dict(
        eta=eta,
        weights=weights,
        source_indices=indices,
        special_count=special_count,
        source_step=source_step,
        source_population_size=len(full),
        source_path=str(input_path.resolve()),
        source_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(C_SOURCE.encode()).hexdigest(),
        b=b,
        c=c,
        epsilon=epsilon,
        self_interactions=True,
        noise_sd=0.0,
        seed=seed,
        rate_multiplier=rate_multiplier,
        gradient_tolerance=tolerance,
        sampling="one random agent per q stratum; all off-line agents retained; population weights",
    )
    saved_steps, frames, elapsed = [0], [agents.copy()], 0.0
    if resume:
        with np.load(output_path) as previous:
            for key, value in metadata.items():
                if not np.array_equal(previous[key], value):
                    raise ValueError(f"Resume settings/source differ: {key}")
            saved_steps = previous["steps"].tolist()
            frames = list(previous["agents"])
            agents = frames[-1].copy()
            elapsed = float(previous["elapsed_seconds"])
    if steps <= saved_steps[-1]:
        raise ValueError("--steps must exceed the last saved continuation step")
    schedule = checkpoint_steps(steps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reactive-long-run-") as directory:
        lib = compile_kernel(Path(directory))
        errors = validate_kernel(lib, agents, eta, weights, b, c, epsilon, tolerance)
        metadata["validation_max_abs_errors"] = errors
        print(
            f"N={len(eta)}, special={special_count}, weights sum={weights.sum():.16g}; "
            f"100-step reference errors (exact, moments): {errors}",
            flush=True,
        )
        for target in schedule[schedule > saved_steps[-1]]:
            started = time.perf_counter()
            count = int(target) - saved_steps[-1]
            advance(lib, agents, eta, weights, count, b, c, epsilon, tolerance)
            duration = time.perf_counter() - started
            elapsed += duration
            saved_steps.append(int(target))
            frames.append(agents.copy())
            save_archive(
                output_path,
                **metadata,
                steps=np.array(saved_steps),
                agents=np.array(frames),
                requested_steps=steps,
                elapsed_seconds=elapsed,
            )
            if target >= 10_000:
                print(
                    f"step {target:,}/{steps:,}; "
                    f"{count / max(duration, 1e-12):,.0f} updates/s; "
                    f"max |p-(1-eps)|={np.max(1 - epsilon - agents[:, 0]):.6g}",
                    flush=True,
                )
    print(f"Saved {output_path}; update time {elapsed:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH, dest="input_path")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, dest="output_path")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--exact", action="store_true", help="Use direct pairwise gradients.")
    parser.add_argument("--resume", action="store_true", help="Continue the existing checkpoint.")
    args = parser.parse_args()
    try:
        generate_data(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
