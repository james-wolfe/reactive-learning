"""Distribution certificates of Theorem 1 (proof in Supplement S3).

These are sufficient conditions for dynamics restricted to q = epsilon, with
arbitrary fixed positive learning rates and no upper step-size restriction.
Finite-time arrival of the whole population requires finitely many agents or
learning rates bounded away from zero.
They do not certify the unrestricted two-dimensional dynamics or empirical runs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .axis_entry import whole_population_on_axis


@dataclass(frozen=True)
class ConvergenceBounds:
    """Sign-separated integrals P, Q and certificate margins -U, P - Q.

    -alld_margin bounds I(p) from above for every p. When tft_margin is
    positive, it bounds I(p) from below; actual p gradients are epsilon * I(p).
    """

    positive: float
    negative: float
    alld_margin: float
    tft_margin: float

    @property
    def outcome(self) -> str:
        if self.alld_margin > 0:
            return "ALLD"
        if self.tft_margin > 0:
            return "TFT"
        return "undetermined"


def _validate_parameters(b, c, epsilon):
    if not (0 < epsilon < 0.5 and b > c > 0 and c / b < 1 - 2 * epsilon):
        raise ValueError("require 0 < epsilon < 1/2 and 0 < c/b < 1 - 2 epsilon")


def convergence_bounds(
    distribution, *, b: float, c: float, epsilon: float, atom_mass: float = 0.0
) -> ConvergenceBounds:
    """Integrate a continuous p distribution mixed with an atom at p = epsilon.

    `distribution` is a frozen scipy continuous distribution on the clipped edge.
    The continuous mass is 1 - atom_mass. Strictly positive margins certify the
    named endpoint under the theorem's assumptions; equality is inconclusive.
    """
    _validate_parameters(b, c, epsilon)
    low, high = distribution.support()
    if not (0 <= atom_mass <= 1 and epsilon <= low < high <= 1 - epsilon):
        raise ValueError("invalid mixture mass or support outside the clipped edge")
    cutoff = epsilon + c / b

    def integral(left, right, *, weighted=False):
        if right <= left:
            return 0.0

        def integrand(p):
            s = p - epsilon
            value = (b * s - c) * (s + 1) * distribution.pdf(p)
            return value / (1 - (1 - 2 * epsilon) * s) ** 2 if weighted else value

        return quad(
            integrand,
            left,
            right,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]

    positive = (1 - atom_mass) * integral(max(low, cutoff), high)
    negative = c * atom_mass - (1 - atom_mass) * integral(low, min(high, cutoff))
    weighted_positive = (1 - atom_mass) * integral(max(low, cutoff), high, weighted=True)
    return ConvergenceBounds(positive, negative, negative - weighted_positive, positive - negative)


def empirical_convergence_bounds(
    agents: np.ndarray, *, b: float, c: float, epsilon: float, atol: float = 1e-12
) -> ConvergenceBounds:
    """Evaluate Theorem 1's distribution inequalities for equally weighted agents.

    All agents must lie on q = epsilon (within atol). Integrals become averages
    over the agents, counting repeated strategies and agents exactly on a boundary.
    A positive margin tests the distribution condition only; the theorem's
    restriction to q = epsilon must be checked separately for a TFT conclusion.
    """
    _validate_parameters(b, c, epsilon)
    agents = np.asarray(agents, dtype=float)
    if agents.ndim != 2 or agents.shape[1] != 2 or len(agents) == 0:
        raise ValueError("agents must have nonempty shape (n_agents, 2)")
    if not np.isfinite(atol) or atol < 0 or not np.all(np.isfinite(agents)):
        raise ValueError("agents must be finite and atol finite and nonnegative")
    if np.any(agents < epsilon - atol) or np.any(agents > 1 - epsilon + atol):
        raise ValueError("agents must lie inside the clipped strategy square")
    if not whole_population_on_axis(agents, epsilon=epsilon, atol=atol):
        raise ValueError("the distribution inequalities require every q = epsilon")
    s = np.clip(agents[:, 0], epsilon, 1 - epsilon) - epsilon
    numerator = (b * s - c) * (s + 1)
    positive = float(np.maximum(numerator, 0).mean())
    negative = float(np.maximum(-numerator, 0).mean())
    weighted_positive = float((np.maximum(numerator, 0) / (1 - (1 - 2 * epsilon) * s) ** 2).mean())
    return ConvergenceBounds(positive, negative, negative - weighted_positive, positive - negative)
