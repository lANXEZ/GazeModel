"""Turns tracker features into a screen position, learned from calibration.

The model is a small regularised ("ridge") regression: each screen coordinate is
a weighted sum of the features plus a few squared terms. That's enough to learn
your own eye geometry from ~13 calibration dots without overfitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

QUADRATIC = (0, 1, 3, 4)  # eye_h, eye_v, look_h, look_v also get squared / cross terms
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # regularisation strengths to try


class GazeEstimator:
    def __init__(self):
        self.mean = self.std = self.weights = self.bias = None
        self.error_px: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self.weights is not None

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        q = Z[:, QUADRATIC]
        i, j = np.triu_indices(len(QUADRATIC))
        return np.hstack([Z, q[:, i] * q[:, j]])

    @staticmethod
    def _ridge(A, Y, alpha):
        a_mean, y_mean = A.mean(axis=0), Y.mean(axis=0)
        Ac = A - a_mean
        W = np.linalg.solve(Ac.T @ Ac + alpha * len(A) * np.eye(A.shape[1]), Ac.T @ (Y - y_mean))
        return W, y_mean - a_mean @ W

    def fit(self, X, Y, groups) -> float:
        """X: features per sample, Y: screen (x, y) per sample, groups: calibration dot index.

        Picks the regularisation by leaving out one dot at a time and checking how well
        the rest predict it. That held-out error is an honest accuracy estimate.
        """
        X, Y, groups = np.asarray(X, float), np.asarray(Y, float), np.asarray(groups)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std < 1e-9, 1.0, std)
        A = self._design(X)

        best_error, best_alpha = np.inf, ALPHAS[-1]
        for alpha in ALPHAS:
            errors = []
            for g in np.unique(groups):
                held_out = groups == g
                W, b = self._ridge(A[~held_out], Y[~held_out], alpha)
                predicted = (A[held_out] @ W + b).mean(axis=0)
                errors.append(np.linalg.norm(predicted - Y[held_out][0]))
            if np.mean(errors) < best_error:
                best_error, best_alpha = float(np.mean(errors)), alpha

        self.weights, self.bias = self._ridge(A, Y, best_alpha)
        self.error_px = best_error
        return best_error

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self._design(features[None, :]) @ self.weights + self.bias)[0]

    def save(self, path: Path, screen_size: tuple[int, int]) -> None:
        np.savez(
            path, mean=self.mean, std=self.std, weights=self.weights, bias=self.bias,
            error_px=self.error_px, screen_size=np.array(screen_size),
        )

    def load(self, path: Path, screen_size: tuple[int, int]) -> bool:
        """Loads a saved calibration if it exists and was made for this screen size."""
        if not path.exists():
            return False
        data = np.load(path)
        if tuple(data["screen_size"]) != tuple(screen_size):
            return False
        self.mean, self.std, self.weights, self.bias = data["mean"], data["std"], data["weights"], data["bias"]
        self.error_px = float(data["error_px"])
        return True


class OneEuroFilter:
    """Smooths jitter when the gaze is still, but follows quickly when it jumps.

    Casiez et al., "1€ Filter", CHI 2012. Lower min_cutoff = smoother but laggier.
    """

    def __init__(self, min_cutoff=0.4, beta=0.005, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self._t is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = max(t - self._t, 1e-3)
        self._t = t
        self._dx = self._dx + self._alpha(self.d_cutoff, dt) * ((x - self._x) / dt - self._dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self._dx)
        self._x = self._x + self._alpha(cutoff, dt) * (x - self._x)
        return self._x
