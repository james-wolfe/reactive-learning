"""Outcome checks for the runs without self-play (self_interactions_data.py)."""

import numpy as np


def alld_margin(agents, *, b, c, epsilon, self_interactions):
    """Theorem 1 ALLD margin, minimized over the actual opponent pools.

    Require q exactly at its floor. Write s=p-epsilon and
    h(s)=(b*s-c)*(1+s). Each opponent contributes
    max(-h,0) - max(h,0)/(1-(1-2*epsilon)*s)**2 to -U.
    This contribution increases as s decreases. A positive worst-pool margin
    therefore persists as every p decreases, even with heterogeneous rates.

    On the line the ratio of the q and p gradient numerators is
    (1-epsilon-p_i*s)/(epsilon*(1+s)), positive and decreasing in s.
    Negative terms have s<c/b and positive terms s>c/b, so negative p
    gradients imply negative q gradients. Thus the line is invariant for this
    ALLD certificate in the unrestricted projected dynamics as well.
    """
    if not np.all(agents[:, 1] == epsilon):
        return float("nan")
    s = agents[:, 0] - epsilon
    h = (b * s - c) * (1 + s)
    contribution = np.maximum(-h, 0) - np.maximum(h, 0) / (1 - (1 - 2 * epsilon) * s) ** 2
    if self_interactions:
        return float(contribution.mean())
    return float(np.min((contribution.sum() - contribution) / (len(s) - 1)))


def endpoint_evidence(agents, *, b, c, epsilon, self_interactions):
    """Summarize whether a population has reached ALLD or is currently at GTFT.

    GTFT means >=99% on p=1-epsilon with q>epsilon, and cooperation >=95%.
    This allows the rare agents off that edge seen in the main variable-rate run;
    it does not prove that the population stays at GTFT.
    """
    margin = alld_margin(agents, b=b, c=c, epsilon=epsilon, self_interactions=self_interactions)
    p, q = agents.T
    r = p - q
    # Rowwise evaluation avoids allocating an N x N matrix.
    total = 0.0
    for i in range(len(agents)):
        values = (q[i] + r[i] * q) / (1 - r[i] * r)
        if not self_interactions:
            values[i] = 0
        total += values.sum()
    divisor = len(agents) * (len(agents) if self_interactions else len(agents) - 1)
    cooperation = total / divisor
    fraction = float(np.mean((p >= 1 - epsilon - 1e-10) & (q > epsilon + 1e-10)))
    return dict(
        **permanent_alld_evidence(
            agents, b=b, c=c, epsilon=epsilon, self_interactions=self_interactions
        ),
        alld_margin=margin,
        alld_observed=bool(np.all(agents == epsilon)),
        gtft_fraction=fraction,
        cooperation=cooperation,
        gtft_regime=bool(fraction >= 0.99 and cooperation >= 0.95),
    )


def permanent_alld_evidence(agents, *, b, c, epsilon, self_interactions):
    """Check, from the number of ALLD agents alone, that agents exactly at ALLD never leave.

    For a focal (e,e), an opponent (p,q) contributes (b*(p-q)-c)*t
    to G_p and (b*(p-q)-c)*(1-t) to G_q, t=e*p+(1-e)*q.
    ALLD opponents contribute -c*e and -c*(1-e), respectively.
    The largest positive G_p contribution occurs at p=1-e and the
    quadratic maximizer q_star below. The largest positive G_q contribution
    occurs at (p,q)=(1-e,e). These maxima bound arbitrary future opponents.

    If f is the fraction of ALLD opponents, both gradients at the corner
    are negative when f > max(M_p/(M_p+c*e), M_q/(M_q+c*(1-e))).
    Every member then stays exactly at ALLD after projection, inductively.
    Without self-play each member sees (k-1)/(N-1), rather than k/N.
    This certifies only the cohort, not convergence of the other agents.
    """
    if not (b > c > 0 and 0 < epsilon < 0.5):
        raise ValueError("require b > c > 0 and 0 < epsilon < 1/2")
    n = len(agents)
    if n < 2:
        raise ValueError("require at least two agents")
    e, high = epsilon, 1 - epsilon
    q_star = np.clip(((b * high - c) * high - b * e * high) / (2 * b * high), e, high)
    max_p = max(0.0, (b * (high - q_star) - c) * (e * high + high * q_star))
    max_q = max(0.0, (b * (high - e) - c) * (1 - 2 * e * high))
    opponent_threshold = max(max_p / (max_p + c * e), max_q / (max_q + c * high))
    count = int(np.count_nonzero(np.all(agents == e, axis=1)))
    fraction = count / n
    opponent_fraction = fraction if self_interactions else (count - 1) / (n - 1)
    gradient_upper = np.array([max_p, max_q]) * (1 - opponent_fraction)
    gradient_upper -= c * np.array([e, high]) * opponent_fraction
    certified = bool(count > 0 and np.max(gradient_upper) < -1e-12)
    # A locked agent cooperates with probability e against every opponent;
    # any other clipped strategy cooperates with probability at most 1-e.
    ceiling = fraction * e + (1 - fraction) * high if certified else high
    return dict(
        alld_fraction=fraction,
        permanent_alld_certified=certified,
        permanent_alld_fraction=fraction if certified else 0.0,
        permanent_alld_threshold=(
            opponent_threshold if self_interactions else (1 + (n - 1) * opponent_threshold) / n
        ),
        alld_corner_gradient_upper=gradient_upper,
        cooperation_ceiling=ceiling,
        cooperation_excluded=bool(certified and ceiling < 0.95),
    )


def stopping_status(evidence, *, step, gtft_since, hold_steps):
    """Label a run's state; a whole population at ALLD is reported separately from a
    group of agents that is permanently stuck at ALLD."""
    if evidence["alld_observed"]:
        return "ALLD observed"
    if evidence["alld_margin"] > 1e-8:
        return "ALLD certified"
    if evidence["cooperation_excluded"]:
        return "Cooperation excluded (permanent ALLD)"
    if evidence["gtft_regime"] and gtft_since >= 0 and step - gtft_since >= hold_steps:
        return "GTFT regime observed"
    return "unresolved"
