"""Turns tracker features into a screen position, learned from calibration.

Two kinds of model, both small regularised ("ridge") regressions fitted to your calibration:

* learned: features -> screen (x, y) directly. Head position and rotation are inputs whose effect
  has to be learned, so the calibration must include some head movement (see ui.CALIBRATION_BLOCKS).
* geometric: gaze direction = head direction (from MediaPipe) + eye direction within the head. Only
  the eye-within-head part is learned (it's the same in any posture); a 3D ray from your eye position
  along that direction is then intersected with the screen. Head movement is handled by geometry.

Head values that barely changed during calibration are ignored: their effect can't be learned from
data where they didn't vary, and guessing makes the pointer jump wildly when you do move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .tracker import FaceResult

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # regularisation strengths to try
KEEP_FRACTION = 0.8  # per dot, keep the samples closest to the dot's typical value
MIN_SPREAD_CM = 1.0  # head position must vary at least this much (std) during calibration to be used
MIN_SPREAD_RAD = np.radians(3.0)  # same for head rotation (people nod ~1-2° just looking around)
CAMERA_ABOVE_SCREEN_MM = 8.0  # laptop webcams sit centred just above the top edge of the screen
FALLBACK_MM_PER_PX = 0.26  # if Windows doesn't know the screen's physical size (e.g. in a VM)
IDENTITY = np.array([1.0, 0.0, 1.0, 0.0])  # quick-fix correction: x scale, x shift, y scale, y shift


@dataclass(frozen=True)
class ScreenGeometry:
    size_px: tuple[int, int]
    size_mm: tuple[float, float]
    camera_above_mm: float = CAMERA_ABOVE_SCREEN_MM

    @property
    def cm_per_px(self) -> np.ndarray:
        mm = np.array(self.size_mm, float)
        if (mm <= 0).any():
            mm = np.array(self.size_px, float) * FALLBACK_MM_PER_PX
        return mm / np.array(self.size_px) / 10

    def to_camera(self, xy_px: np.ndarray) -> np.ndarray:
        """Screen pixels -> 3D points (cm) in the camera's frame (x right in the image, y up, screen at z=0)."""
        s = np.asarray(xy_px, float) * self.cm_per_px
        half_width = self.size_px[0] * self.cm_per_px[0] / 2
        # The camera faces you, so your right is its left.
        return np.stack([half_width - s[:, 0], -(s[:, 1] + self.camera_above_mm / 10), np.zeros(len(s))], axis=1)

    def to_screen(self, points: np.ndarray) -> np.ndarray:
        half_width = self.size_px[0] * self.cm_per_px[0] / 2
        s = np.stack([half_width - points[:, 0], -points[:, 1] - self.camera_above_mm / 10], axis=1)
        return s / self.cm_per_px


@dataclass(frozen=True)
class FeatureSet:
    """One way of describing a face as numbers; each gets its own calibration model."""

    label: str
    extract: Callable[[FaceResult], np.ndarray | None]
    quadratic: tuple[int, ...]  # feature indices that also get squared / cross terms
    head: dict[int, float] = field(default_factory=dict)  # head columns -> spread needed to be used
    interactions: tuple[tuple[int, int], ...] = ()  # (eye column, head column) products
    needs_deep: bool = False
    geometric: bool = False  # last 3 columns are the eye's 3D position; the model learns gaze angles
    head_angles: tuple[int, int] = (5, 6)  # (yaw, pitch) columns, used by geometric models


def _deep(f):
    return f.deep is not None


def _deep_in_head(f) -> np.ndarray:
    """Deep-model angles turned into eye-within-head angles in the geometric models' convention.

    Its first angle grows towards the screen's right (the camera's -x) and the second upwards;
    subtracting the head direction (yaw, -pitch) leaves the eyes' own rotation.
    """
    yaw, pitch = f.features[5], f.features[6]
    return np.array([-f.deep[0] - yaw, f.deep[1] + pitch])


# Column meanings of FaceResult.features:
# 0 eye_h, 1 eye_v, 2 eye_open, 3 look_h, 4 look_v, 5 yaw, 6 pitch, 7 head_x, 8 head_y, 9 head_z
HEAD = {5: MIN_SPREAD_RAD, 6: MIN_SPREAD_RAD, 7: MIN_SPREAD_CM, 8: MIN_SPREAD_CM, 9: MIN_SPREAD_CM}
# Distance stretches how far an eye movement travels on screen; head rotation adds to eye rotation.
EYE_X_HEAD = ((0, 9), (1, 9), (3, 9), (4, 9), (0, 5), (1, 6), (3, 5), (4, 6))

FEATURE_SETS = {
    "iris": FeatureSet("MediaPipe iris", lambda f: f.features, (0, 1, 3, 4), HEAD, EYE_X_HEAD),
    "deep": FeatureSet(
        "Deep model (L2CS-Net)",
        # deep 0, deep 1, yaw, pitch, head x, y, z
        lambda f: np.concatenate([f.deep, f.features[5:]]) if _deep(f) else None,
        (0, 1),
        {2: MIN_SPREAD_RAD, 3: MIN_SPREAD_RAD, 4: MIN_SPREAD_CM, 5: MIN_SPREAD_CM, 6: MIN_SPREAD_CM},
        ((0, 6), (1, 6), (0, 2), (1, 3)),
        needs_deep=True,
    ),
    "hybrid": FeatureSet(
        "Deep + iris",
        lambda f: np.concatenate([f.features, f.deep]) if _deep(f) else None,  # deep at 10, 11
        (0, 1, 3, 4, 10, 11),
        HEAD,
        EYE_X_HEAD + ((10, 9), (11, 9)),
        needs_deep=True,
    ),
    "geo_iris": FeatureSet(
        "Geometric iris",
        # eye_h .. pitch (0-6), then eye position x, y, z
        lambda f: np.concatenate([f.features[:7], f.eye_pos]),
        (0, 1),  # squared terms on more inputs extrapolate wildly in postures outside the calibration
        {5: MIN_SPREAD_RAD, 6: MIN_SPREAD_RAD},
        geometric=True,
    ),
    "geo_hybrid": FeatureSet(
        "Geometric deep + iris",
        # eye_h .. pitch (0-6), deep model's eye-within-head angles (7, 8), then eye position x, y, z
        lambda f: np.concatenate([f.features[:7], _deep_in_head(f), f.eye_pos]) if _deep(f) else None,
        (0, 1),
        {5: MIN_SPREAD_RAD, 6: MIN_SPREAD_RAD},
        needs_deep=True,
        geometric=True,
    ),
}


def samples_for(name: str, faces, Y, groups, min_dots: int):
    """The recorded samples a model can use (deep ones need deep angles), or None if they cover too few dots."""
    rows = [(x, y, g) for x, y, g in zip((FEATURE_SETS[name].extract(f) for f in faces), Y, groups) if x is not None]
    if len({g for _, _, g in rows}) < min_dots:
        return None
    X, Y, groups = zip(*rows)
    return np.array(X), np.array(Y), np.array(groups)


class GazeEstimator:
    def __init__(self, feature_set: FeatureSet, screen: ScreenGeometry):
        self.fs, self.screen = feature_set, screen
        self.mean = self.std = self.weights = self.bias = None
        self.correction = IDENTITY.copy()
        self.error_px: float | None = None  # leave-one-dot-out estimate from calibration
        self.validation_px: float | None = None  # measured on the separate validation dots

    @property
    def is_calibrated(self) -> bool:
        return self.weights is not None

    @property
    def best_error_px(self) -> float:
        return self.validation_px if self.validation_px is not None else self.error_px

    @property
    def ignored_head(self) -> list[int]:
        """Head columns left out because they didn't vary enough during calibration."""
        return [col for col in self.fs.head if np.isinf(self.std[col])]

    # --- the model ---

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        return X[:, :-3] if self.fs.geometric else X

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (self._inputs(X) - self.mean) / self.std  # ignored columns have std = inf, so become 0

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = self._standardise(X)
        q = Z[:, self.fs.quadratic]
        i, j = np.triu_indices(len(self.fs.quadratic))
        parts = [Z, q[:, i] * q[:, j]]
        if self.fs.interactions:
            parts.append(np.stack([Z[:, a] * Z[:, b] for a, b in self.fs.interactions], axis=1))
        return np.hstack(parts)

    def _head_direction(self, X: np.ndarray) -> np.ndarray:
        """Where the face points, as (horizontal, vertical) angles in the same convention as the gaze angles."""
        yaw, pitch = self.fs.head_angles
        return np.stack([X[:, yaw], -X[:, pitch]], axis=1)

    def _targets(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """What the regression learns: screen pixels, or (geometric) the eye-within-head angles towards the dot."""
        if not self.fs.geometric:
            return Y
        v = self.screen.to_camera(Y) - X[:, -3:]
        gaze = np.stack([np.arctan2(v[:, 0], v[:, 2]), np.arctan2(v[:, 1], np.hypot(v[:, 0], v[:, 2]))], axis=1)
        return gaze - self._head_direction(X)

    def _to_screen(self, X: np.ndarray, out: np.ndarray) -> np.ndarray:
        if not self.fs.geometric:
            return out
        gaze = out + self._head_direction(X)
        h, v = gaze[:, 0], gaze[:, 1]
        direction = np.stack([np.cos(v) * np.sin(h), np.sin(v), np.cos(v) * np.cos(h)], axis=1)
        eye = X[:, -3:]
        distance = -eye[:, 2] / np.maximum(direction[:, 2], 1e-6)  # how far along the ray the screen is
        return self.screen.to_screen(eye + direction * distance[:, None])

    @staticmethod
    def _ridge(A, T, alpha):
        a_mean, t_mean = A.mean(axis=0), T.mean(axis=0)
        Ac = A - a_mean
        W = np.linalg.solve(Ac.T @ Ac + alpha * len(A) * np.eye(A.shape[1]), Ac.T @ (T - t_mean))
        return W, t_mean - a_mean @ W

    def _drop_outliers(self, X, Y, groups):
        """Within each dot, drop samples far from the dot's median (eyes still arriving, glances away)."""
        Z = self._standardise(X)[:, self.fs.quadratic]
        keep = np.zeros(len(X), dtype=bool)
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            distance = np.linalg.norm(Z[idx] - np.median(Z[idx], axis=0), axis=1)
            keep[idx[np.argsort(distance)[: max(1, int(round(len(idx) * KEEP_FRACTION)))]]] = True
        return X[keep], Y[keep], groups[keep]

    def fit(self, X, Y, groups) -> float:
        """X: features per sample, Y: screen (x, y) per sample, groups: calibration dot index.

        Picks the regularisation by leaving out one dot at a time and checking how well
        the rest predict it.
        """
        X, Y, groups = np.asarray(X, float), np.asarray(Y, float), np.asarray(groups)
        inputs = self._inputs(X)
        self.mean = inputs.mean(axis=0)
        std = inputs.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        for col, min_spread in self.fs.head.items():
            if std[col] < min_spread:
                std[col] = np.inf
        self.std = std
        X, Y, groups = self._drop_outliers(X, Y, groups)
        A, T = self._design(X), self._targets(X, Y)

        best_error, best_alpha = np.inf, ALPHAS[-1]
        for alpha in ALPHAS:
            errors = []
            for g in np.unique(groups):
                held_out = groups == g
                W, b = self._ridge(A[~held_out], T[~held_out], alpha)
                predicted = self._to_screen(X[held_out], A[held_out] @ W + b).mean(axis=0)
                errors.append(np.linalg.norm(predicted - Y[held_out][0]))
            if np.mean(errors) < best_error:
                best_error, best_alpha = float(np.mean(errors)), alpha

        self.weights, self.bias = self._ridge(A, T, best_alpha)
        self.correction = IDENTITY.copy()
        self.error_px = best_error
        self.validation_px = None
        return best_error

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return self._to_screen(X, self._design(X) @ self.weights + self.bias)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_many(features[None, :])[0]

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        ax, bx, ay, by = self.correction
        raw = self._raw_predict(X)
        return np.stack([ax * raw[:, 0] + bx, ay * raw[:, 1] + by], axis=1)

    def validate(self, X, Y, groups) -> tuple[np.ndarray, np.ndarray]:
        """Scores the model on dots it wasn't trained on. Returns (dot positions, mean prediction per dot)."""
        X, Y, groups = np.asarray(X, float), np.asarray(Y, float), np.asarray(groups)
        predicted = self.predict_many(X)
        dots = np.unique(groups)
        means = np.array([predicted[groups == g].mean(axis=0) for g in dots])
        targets = np.array([Y[groups == g][0] for g in dots])
        self.validation_px = float(np.mean(np.linalg.norm(means - targets, axis=1)))
        return targets, means

    def fit_correction(self, X, Y, groups) -> float:
        """Quick fix: a shift and scale per axis that moves this model's predictions onto a few dots.

        Returns how far the pointer moves on average because of the new correction (pixels).
        """
        X, Y, groups = np.asarray(X, float), np.asarray(Y, float), np.asarray(groups)
        raw = self._raw_predict(X)
        dots = np.unique(groups)
        predicted = np.array([np.median(raw[groups == g], axis=0) for g in dots])
        targets = np.array([Y[groups == g][0] for g in dots])
        old = self.predict_many(X)
        correction = []
        for axis in (0, 1):
            p, t = predicted[:, axis], targets[:, axis]
            if np.ptp(p) > 1e-6 and np.ptp(t) > 1e-6:
                scale, shift = np.polyfit(p, t, 1)
                scale = float(np.clip(scale, 0.6, 1.6))
                shift = float(np.mean(t - scale * p))
            else:
                scale, shift = 1.0, float(np.mean(t - p))
            correction += [scale, shift]
        self.correction = np.array(correction)
        return float(np.mean(np.linalg.norm(self.predict_many(X) - old, axis=1)))

    # --- saving ---

    def state(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean, "std": self.std, "weights": self.weights, "bias": self.bias,
            "correction": self.correction, "error_px": np.array(self.error_px),
            "validation_px": np.array(np.nan if self.validation_px is None else self.validation_px),
        }

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.mean, self.std, self.weights, self.bias = state["mean"], state["std"], state["weights"], state["bias"]
        self.correction = state.get("correction", IDENTITY.copy())
        self.error_px = float(state["error_px"])
        validation = float(state["validation_px"])
        self.validation_px = None if np.isnan(validation) else validation


def save_calibration(path: Path, estimators: dict[str, GazeEstimator], screen: ScreenGeometry) -> None:
    arrays = {f"{name}.{key}": value for name, est in estimators.items() for key, value in est.state().items()}
    np.savez(path, screen_size=np.array(screen.size_px), **arrays)


def load_calibration(path: Path, screen: ScreenGeometry) -> dict[str, GazeEstimator]:
    """Loads saved calibrations if they exist and were made for this screen and this version of the models."""
    if not path.exists():
        return {}
    data = np.load(path)
    if tuple(data["screen_size"]) != tuple(screen.size_px):
        return {}
    estimators = {}
    for name, feature_set in FEATURE_SETS.items():
        prefix = f"{name}."
        state = {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}
        if not state:
            continue
        estimator = GazeEstimator(feature_set, screen)
        estimator.load_state(state)
        expected = estimator._design(np.zeros((1, len(estimator.mean) + 3 * feature_set.geometric))).shape[1]
        if estimator.weights.shape[0] == expected:  # skip calibrations saved by an older version
            estimators[name] = estimator
    return estimators


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
