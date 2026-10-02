from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WellMixedReactive:
    agents: np.ndarray
    eta: np.ndarray
    b: float
    c: float
    epsilon: float = 1e-3
    self_interactions: bool = True
    noise_sd: float = 0.0
    rng: np.random.Generator | None = None
    vectorized_threshold: int = 2500
    # Kernel-matrix gradients (one N x N kernel and a matrix product): about 5x faster at
    # N = 1000, but rounded differently, so off by default to keep saved runs
    # reproducible bit for bit. Used by the basin sweep (basin_data.py).
    matmul_gradients: bool = False

    def __post_init__(self) -> None:
        self.agents = np.asarray(self.agents, dtype=float)
        self.eta = np.asarray(self.eta, dtype=float)
        if self.agents.ndim != 2 or self.agents.shape[1] != 2:
            raise ValueError("agents must have shape (n_agents, 2)")
        if self.eta.shape != (self.agents.shape[0],):
            raise ValueError("eta must have shape (n_agents,)")
        self.rng = self.rng or np.random.default_rng()

    @property
    def num_agents(self) -> int:
        return self.agents.shape[0]

    def compute_global_gradients(self) -> np.ndarray:
        if self.matmul_gradients:
            return self._compute_global_gradients_matmul()
        if self.num_agents <= self.vectorized_threshold:
            return self._compute_global_gradients_vectorized()
        return self._compute_global_gradients_low_memory()

    def _compute_global_gradients_vectorized(self) -> np.ndarray:
        p = self.agents[:, 0]
        q = self.agents[:, 1]
        p_i = p[:, None]
        q_i = q[:, None]
        p_j = p[None, :]
        q_j = q[None, :]
        denom = (1 + p_j * q_i - q_i * q_j + p_i * (-p_j + q_j)) ** 2
        common = self.c + self.b * (-p_j + q_j)
        grad_p_full = -((p_j * q_i + q_j - q_i * q_j) * common) / denom
        grad_q_full = ((q_j - 1) + p_i * (p_j - q_j)) * common / denom

        divisor = self.num_agents
        if not self.self_interactions:
            np.fill_diagonal(grad_p_full, 0.0)
            np.fill_diagonal(grad_q_full, 0.0)
            divisor -= 1

        return np.column_stack((grad_p_full.sum(axis=1), grad_q_full.sum(axis=1))) / divisor

    def _compute_global_gradients_matmul(self) -> np.ndarray:
        """The low-memory formula below, with its per-agent loop as one matrix product."""
        p = self.agents[:, 0]
        q = self.agents[:, 1]
        r = p - q
        u = self.b * r - self.c
        kernel = 1.0 / (1.0 - np.outer(r, r)) ** 2
        divisor = self.num_agents
        if not self.self_interactions:
            np.fill_diagonal(kernel, 0.0)
            divisor -= 1
        sum_r, sum_q, sum_u = (kernel @ np.column_stack((u * r, u * q, u))).T
        return np.column_stack((q * sum_r + sum_q, sum_u - sum_q - p * sum_r)) / divisor

    def _compute_global_gradients_low_memory(self) -> np.ndarray:
        p = self.agents[:, 0]
        q = self.agents[:, 1]
        r = p - q
        u = self.b * r - self.c
        weighted_r = u * r
        weighted_q = u * q
        divisor = self.num_agents if self.self_interactions else self.num_agents - 1

        grad_p = np.empty_like(p)
        grad_q = np.empty_like(q)

        for agent_idx, r_i in enumerate(r):
            kernel = 1.0 / (1.0 - r_i * r) ** 2
            if not self.self_interactions:
                kernel[agent_idx] = 0.0

            sum_r = kernel @ weighted_r
            sum_q = kernel @ weighted_q
            sum_u = kernel @ u
            grad_p[agent_idx] = (q[agent_idx] * sum_r + sum_q) / divisor
            grad_q[agent_idx] = (sum_u - sum_q - p[agent_idx] * sum_r) / divisor

        return np.column_stack((grad_p, grad_q))

    def step(self) -> None:
        gradients = self.compute_global_gradients()
        self.agents += self.eta[:, None] * gradients
        if self.noise_sd:
            self.agents += self.noise_sd * self.rng.standard_normal(gradients.shape)
        np.clip(self.agents, self.epsilon, 1.0 - self.epsilon, out=self.agents)

    def run(self, steps: int, steps_to_save=()) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        steps_to_save = set(steps_to_save)
        saved = {}
        for step in range(steps):
            if step in steps_to_save:
                saved[step] = self.agents.copy()
            self.step()
        return self.agents, saved

    def avg_cooperation(self) -> float:
        return float(avg_cooperation(self.agents))

    def classify_basin(self, steps: int, check_every: int = 100) -> int:
        for step in range(steps):
            last_agents = self.agents.copy() if step % check_every == 0 else None
            self.step()
            if step % check_every != 0:
                continue
            if np.mean(self.agents[:, 0]) > 0.99 or self.avg_cooperation() > 0.95:
                return 1
            if np.all(self.agents[:, 0] < 0.5):
                return 0
            q_at_floor = np.all(self.agents[:, 1] == self.epsilon)
            last_q_at_floor = np.all(last_agents[:, 1] == self.epsilon)
            if (
                q_at_floor
                and last_q_at_floor
                and np.max(self.agents[:, 0]) < np.max(last_agents[:, 0])
            ):
                return 0
            all_defect_mask = np.isclose(self.agents[:, 0], self.epsilon) & np.isclose(
                self.agents[:, 1], self.epsilon, atol=1e-8
            )
            threshold = ((self.b / self.c - 1) / (self.b / self.c - 0.5)) * self.num_agents
            if all_defect_mask.sum() > threshold:
                return 0
        return int(self.avg_cooperation() > 0.5)


def avg_cooperation(agents: np.ndarray) -> float:
    agents = np.asarray(agents, dtype=float)
    p = agents[:, 0]
    q = agents[:, 1]
    r = p - q
    numerator = q[None, :] * r[:, None] + q[:, None]
    denominator = 1.0 - r[:, None] * r[None, :]
    return float(np.mean(numerator / denominator))


def avg_cooperation_two_agents(agents: np.ndarray) -> float:
    agents = np.asarray(agents, dtype=float)
    if agents.shape != (2, 2):
        raise ValueError("agents must have shape (2, 2)")
    p1, q1 = agents[0]
    p2, q2 = agents[1]
    r1 = p1 - q1
    r2 = p2 - q2
    denom = 1.0 - r1 * r2
    if np.isclose(denom, 0.0):
        return np.nan
    return float(0.5 * ((q2 * r1 + q1) / denom + (q1 * r2 + q2) / denom))


def avg_cooperation_over_time(runs: list[dict[int, np.ndarray]]) -> np.ndarray:
    timesteps = sorted(set().union(*(run.keys() for run in runs)))
    values = np.full((len(runs), len(timesteps)), np.nan)
    for run_idx, run in enumerate(runs):
        for step_idx, step in enumerate(timesteps):
            if step in run:
                values[run_idx, step_idx] = avg_cooperation_two_agents(run[step])
    return values


def avg_gradient_at_point(p_i, q_i, agents: np.ndarray, b: float, c: float, tol: float = 1e-12):
    agents = np.asarray(agents, dtype=float)
    p_j = agents[:, 0][None, :]
    q_j = agents[:, 1][None, :]
    p_i = np.asarray(p_i)[..., None]
    q_i = np.asarray(q_i)[..., None]
    denom = (1 + p_j * q_i - q_i * q_j + p_i * (-p_j + q_j)) ** 2
    common = c + b * (-p_j + q_j)
    grad_p = -((p_j * q_i + q_j - q_i * q_j) * common) / denom
    grad_q = (((q_j - 1) + p_i * (p_j - q_j)) * common) / denom
    mask = denom > tol
    return np.where(mask, grad_p, 0.0).mean(axis=-1), np.where(mask, grad_q, 0.0).mean(axis=-1)


def vector_field_against_population(agents: np.ndarray, b: float, c: float, grid_size: int = 25):
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    grad_p, grad_q = avg_gradient_at_point(grid_x.ravel(), grid_y.ravel(), agents, b, c)
    field_p = grad_p.reshape(grid_x.shape)
    field_q = grad_q.reshape(grid_y.shape)
    return grid_x, grid_y, field_p, field_q
