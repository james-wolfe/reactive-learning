from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pair_gradients(lattice: np.ndarray, b: float, c: float) -> np.ndarray:
    p = lattice[..., 0]
    q = lattice[..., 1]
    p_i = p[:, :, None, None]
    q_i = q[:, :, None, None]
    p_j = p[None, None, :, :]
    q_j = q[None, None, :, :]
    denom = (1 + p_j * q_i - q_i * q_j + p_i * (-p_j + q_j)) ** 2
    common = c + b * (-p_j + q_j)
    grad_p = -((p_j * q_i + q_j - q_i * q_j) * common) / denom
    grad_q = ((q_j - 1) + p_i * (p_j - q_j)) * common / denom
    return np.stack((grad_p, grad_q), axis=-1)


@dataclass
class ReactiveLattice:
    freq: np.ndarray
    eta: np.ndarray
    epsilon: float = 1e-3
    b: float = 1.0
    c: float = 0.5

    def __post_init__(self) -> None:
        self.freq = np.asarray(self.freq, dtype=float)
        self.eta = np.asarray(self.eta, dtype=float)
        if self.freq.ndim != 3:
            raise ValueError("freq must have shape (n_points, n_points, n_eta)")
        if self.eta.shape != (self.freq.shape[-1],):
            raise ValueError("eta must have shape (n_eta,)")
        self.num_points = self.freq.shape[0]
        self.x = np.linspace(self.epsilon, 1.0 - self.epsilon, self.num_points)
        self.y = np.linspace(self.epsilon, 1.0 - self.epsilon, self.num_points)
        grid_x, grid_y = np.meshgrid(self.x, self.y, indexing="ij")
        self.lattice = np.stack((grid_x, grid_y), axis=-1)
        self.gradients = pair_gradients(self.lattice, self.b, self.c)

    def new_locations(self) -> np.ndarray:
        freq_total = self.freq.sum(axis=-1)
        weighted_gradients = np.tensordot(self.gradients, freq_total, axes=([2, 3], [0, 1]))
        locations = (
            self.lattice[..., None, :]
            + self.eta[None, None, :, None] * weighted_gradients[..., None, :]
        )
        return np.clip(locations, self.epsilon, 1.0 - self.epsilon)

    def step(self) -> None:
        updated = np.zeros_like(self.freq)
        locations = self.new_locations()

        for eta_idx in range(self.eta.size):
            mass = self.freq[..., eta_idx].ravel()
            p = locations[..., eta_idx, 0].ravel()
            q = locations[..., eta_idx, 1].ravel()

            p_idx = np.clip(np.searchsorted(self.x, p, side="right"), 1, self.num_points - 1)
            q_idx = np.clip(np.searchsorted(self.y, q, side="right"), 1, self.num_points - 1)
            p1, p2 = self.x[p_idx - 1], self.x[p_idx]
            q1, q2 = self.y[q_idx - 1], self.y[q_idx]
            dp = (p - p1) / (p2 - p1)
            dq = (q - q1) / (q2 - q1)

            np.add.at(updated[..., eta_idx], (p_idx - 1, q_idx - 1), (1 - dp) * (1 - dq) * mass)
            np.add.at(updated[..., eta_idx], (p_idx, q_idx - 1), dp * (1 - dq) * mass)
            np.add.at(updated[..., eta_idx], (p_idx - 1, q_idx), (1 - dp) * dq * mass)
            np.add.at(updated[..., eta_idx], (p_idx, q_idx), dp * dq * mass)

            total = updated[..., eta_idx].sum()
            if total:
                updated[..., eta_idx] /= total * self.eta.size

        self.freq = updated

    def run(self, steps: int, steps_to_save) -> tuple[np.ndarray, np.ndarray]:
        steps_to_save = list(steps_to_save)
        save_lookup = {step: idx for idx, step in enumerate(steps_to_save)}
        saved = np.zeros((*self.freq.shape, len(steps_to_save)))
        for step in range(steps):
            if step in save_lookup:
                saved[..., save_lookup[step]] = self.freq.copy()
            self.step()
        return self.freq, saved
