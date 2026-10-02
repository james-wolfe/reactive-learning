"""Theorem 1 certificate margins on two families of distributions on q = epsilon (Fig. S4).

Families: an ALLD atom plus a uniform distribution on [epsilon, p_max], and Beta(alpha, beta)
scaled onto [epsilon, 1 - epsilon], each on a 1001 x 1001 parameter grid. Before the file is
written, the Beta results are recomputed with twice as many quadrature points, and 25 grid
points are compared against scipy's adaptive integration.
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import betainc, betaln, roots_jacobi
from scipy.stats import beta as beta_distribution
from scipy.stats import uniform

from reactive_learning.paths import DATA_DIR
from reactive_learning.transient_unforgiving import convergence_bounds

B, C, EPSILON = 1.0, 0.5, 0.001
L = 1 - 2 * EPSILON
N_GRID = 1001
N_QUAD = 256
SIGN_TOL = 1e-8  # Margins this close to zero count as uncertified.

ATOM_MASSES = np.linspace(0, 1, N_GRID)
P_MAX_VALUES = np.linspace(EPSILON + 0.01, 1 - EPSILON, N_GRID)
SHAPES = np.geomspace(0.1, 30, N_GRID)
assert 0 < EPSILON < 0.5 and 0 < C / B < L


OUTPUT_PATH = DATA_DIR / "convergence_phase_diagrams.npz"


def classify(alld_margin, tft_margin):
    """0 = neither certified, 1 = ALLD, 2 = TFT."""
    assert not np.any((alld_margin > SIGN_TOL) & (tft_margin > SIGN_TOL))
    phase = np.zeros(np.shape(alld_margin), dtype=np.uint8)
    phase[alld_margin > SIGN_TOL] = 1
    phase[tft_margin > SIGN_TOL] = 2
    return phase


def beta_margins(shapes, upper, order):
    """Rows are beta, columns alpha. Returns margins -U and P-Q."""
    first = np.asarray(shapes)
    scale = upper - EPSILON
    split = min(C / (B * scale), 1.0)
    alld = np.empty((len(shapes), len(shapes)))
    tft = np.empty_like(alld)
    for row, second in enumerate(shapes):
        mean = first / (first + second)
        moment2 = first * (first + 1) / ((first + second) * (first + second + 1))
        tft[row] = B * scale**2 * moment2 + (B - C) * scale * mean - C
        negative = (
            C * betainc(first, second, split)
            - (B - C) * scale * mean * betainc(first + 1, second, split)
            - B * scale**2 * moment2 * betainc(first + 2, second, split)
        )
        if split == 1:
            alld[row] = negative
            continue
        nodes, weights = roots_jacobi(order, second - 1, 0)
        x = split + (1 - split) * (nodes + 1) / 2
        s = scale * x
        kernel = (B * s - C) * (s + 1) / (1 - L * s) ** 2
        density_factor = np.exp(
            (first[:, None] - 1) * np.log(x)
            + second * np.log((1 - split) / 2)
            - betaln(first, second)[:, None]
        )
        weighted_positive = (density_factor * kernel) @ weights
        alld[row] = negative - weighted_positive
    return alld, tft


def generate_data(output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uniform_bounds = [
        convergence_bounds(uniform(loc=EPSILON, scale=p_max - EPSILON), b=B, c=C, epsilon=EPSILON)
        for p_max in P_MAX_VALUES
    ]
    uniform_alld = np.array([v.alld_margin for v in uniform_bounds])
    uniform_tft = np.array([v.tft_margin for v in uniform_bounds])
    mass = ATOM_MASSES[:, None]
    mixture_alld = (1 - mass) * uniform_alld + mass * C
    mixture_tft = (1 - mass) * uniform_tft - mass * C

    full_alld, full_tft = beta_margins(SHAPES, 1 - EPSILON, N_QUAD)
    results = {
        "Atom + uniform": (mixture_alld, mixture_tft),
        "Full-edge Beta": (full_alld, full_tft),
    }
    phases = {name: classify(*margins) for name, margins in results.items()}
    print(f"Evaluated {N_GRID**2:,} parameter pairs in each of two families.")

    for name, upper, original in [
        ("Full-edge Beta", 1 - EPSILON, (full_alld, full_tft)),
    ]:
        refined = beta_margins(SHAPES, upper, 2 * N_QUAD)
        scaled_error = np.max(np.abs(refined[0] - original[0]) / (1 + np.abs(refined[0])))
        changed = np.count_nonzero(classify(*refined) != classify(*original))
        assert scaled_error < 1e-7, (name, scaled_error)
        assert changed == 0, (name, changed)
        # Replace the displayed margins with the refined calculation.
        results[name] = refined
        phases[name] = classify(*refined)
        print(f"{name}: doubling the quadrature order changes margins by at most "
              f"{scaled_error:.3g} (scaled) and changes {changed} classifications.")

    sample_indices = [0, N_GRID // 4, N_GRID // 2, 3 * N_GRID // 4, N_GRID - 1]
    max_reference_error = 0.0
    reference_warnings = 0
    for name, upper in [("Full-edge Beta", 1 - EPSILON)]:
        for i in sample_indices:
            for j in sample_indices:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", IntegrationWarning)
                    reference = convergence_bounds(
                        beta_distribution(SHAPES[j], SHAPES[i], loc=EPSILON, scale=upper - EPSILON),
                        b=B,
                        c=C,
                        epsilon=EPSILON,
                    )
                reference_warnings += sum(
                    issubclass(w.category, IntegrationWarning) for w in caught
                )
                values = np.array([results[name][0][i, j], results[name][1][i, j]])
                expected = np.array([reference.alld_margin, reference.tft_margin])
                np.testing.assert_allclose(values, expected, rtol=2e-7, atol=2e-8)
                max_reference_error = max(
                    max_reference_error, np.max(np.abs(values - expected) / (1 + np.abs(expected)))
                )

    # At alpha=beta=1, the full-edge Beta is exactly uniform.
    unit = beta_margins(np.array([1.0]), 1 - EPSILON, 2 * N_QUAD)
    reference = convergence_bounds(uniform(loc=EPSILON, scale=L), b=B, c=C, epsilon=EPSILON)
    np.testing.assert_allclose(
        [unit[0][0, 0], unit[1][0, 0]], [reference.alld_margin, reference.tft_margin], rtol=1e-8
    )
    print(
        f"Largest scaled discrepancy across 25 adaptive reference checks: {max_reference_error:.3g}"
    )
    print(
        f"Adaptive reference integration emitted {reference_warnings} tolerance warnings; "
        "all checked values agree within the asserted tolerance."
    )

    for name, phase in phases.items():
        shares = [100 * np.mean(phase == k) for k in (1, 2, 0)]
        print(f"{name}: ALLD {shares[0]:.2f}%, TFT {shares[1]:.2f}%, neither {shares[2]:.2f}%")
    np.savez_compressed(
        output_path,
        b=B,
        c=C,
        epsilon=EPSILON,
        sign_tolerance=SIGN_TOL,
        atom_masses=ATOM_MASSES,
        p_max_values=P_MAX_VALUES,
        beta_shapes=SHAPES,
        mixture_alld=mixture_alld,
        mixture_tft=mixture_tft,
        full_beta_alld=results["Full-edge Beta"][0],
        full_beta_tft=results["Full-edge Beta"][1],
        mixture_phase=phases["Atom + uniform"],
        full_beta_phase=phases["Full-edge Beta"],
        quadrature_order=2 * N_QUAD,
    )
    print("Saved margins, classifications and parameters in", output_path)


if __name__ == "__main__":
    generate_data()
