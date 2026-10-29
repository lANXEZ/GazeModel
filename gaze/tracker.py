"""Webcam face + iris tracking with MediaPipe.

Each frame is turned into a handful of numbers ("features") describing where the
irises sit inside the eyes and how the head is turned. The calibration model in
estimator.py learns how those numbers map to a point on the screen.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# MediaPipe face-mesh landmark indices. Corners are listed image-left first
# (in the unmirrored camera image) so both eyes measure in the same direction.
EYES = (
    {"corners": (33, 133), "lids": (159, 145)},  # person's right eye
    {"corners": (362, 263), "lids": (386, 374)},  # person's left eye
)
IRISES = (range(468, 473), range(473, 478))  # centre point, then 4 points on the edge
EYE_CONTOURS = (
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
)

FEATURE_NAMES = (
    "eye_h",     # iris left/right inside the eye (average of both eyes)
    "eye_v",     # iris up/down inside the eye
    "eye_open",  # how open the eyelids are (they close a bit when you look down)
    "look_h",    # MediaPipe's own eye-direction scores ("blendshapes")
    "look_v",
    "yaw",       # head turned left/right
    "pitch",     # head tilted up/down
    "head_x",    # head position relative to the camera (cm)
    "head_y",
    "head_z",
)
BLINK_THRESHOLD = 0.6


@dataclass
class FaceResult:
    features: np.ndarray  # shape (len(FEATURE_NAMES),)
    blinking: bool
    points: np.ndarray  # (478, 2) landmark positions in pixels
    eyes: list[tuple[np.ndarray, np.ndarray, float]]  # (eye centre, iris centre, iris radius)


def _load_model() -> bytes:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model to {MODEL_PATH} ...")
        partial = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(MODEL_PATH)
    return MODEL_PATH.read_bytes()


def _measure_eye(points, eye, iris_centre):
    """Iris position relative to the eye corners, normalised by eye width."""
    a, b = points[eye["corners"][0]], points[eye["corners"][1]]
    width = np.linalg.norm(b - a)
    along = (b - a) / width  # image-left -> image-right along the eye
    down = np.array([-along[1], along[0]])  # perpendicular, pointing down the image
    offset = iris_centre - (a + b) / 2
    top, bottom = points[eye["lids"][0]], points[eye["lids"][1]]
    return offset @ along / width, offset @ down / width, np.linalg.norm(bottom - top) / width


class GazeTracker:
    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_buffer=_load_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceResult | None:
        timestamp_ms = max(timestamp_ms, self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        if not result.face_landmarks or len(result.face_landmarks[0]) < 478:
            return None

        h, w = frame_bgr.shape[:2]
        points = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]])
        irises = [
            (points[r.start], float(np.mean(np.linalg.norm(points[r.start + 1 : r.stop] - points[r.start], axis=1))))
            for r in IRISES
        ]

        measurements, eyes = [], []
        for eye in EYES:
            centre = points[list(eye["corners"])].mean(axis=0)
            # Pick the iris closest to this eye rather than trusting index order.
            iris_centre, radius = min(irises, key=lambda iris: np.linalg.norm(iris[0] - centre))
            measurements.append(_measure_eye(points, eye, iris_centre))
            eyes.append((centre, iris_centre, radius))
        eye_h, eye_v, eye_open = np.mean(measurements, axis=0)

        blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        look_h = (blend.get("eyeLookOutLeft", 0) + blend.get("eyeLookInRight", 0)) - (
            blend.get("eyeLookInLeft", 0) + blend.get("eyeLookOutRight", 0)
        )
        look_v = (blend.get("eyeLookUpLeft", 0) + blend.get("eyeLookUpRight", 0)) - (
            blend.get("eyeLookDownLeft", 0) + blend.get("eyeLookDownRight", 0)
        )
        blinking = max(blend.get("eyeBlinkLeft", 0), blend.get("eyeBlinkRight", 0)) > BLINK_THRESHOLD

        matrix = np.asarray(result.facial_transformation_matrixes[0])
        rotation = matrix[:3, :3]
        yaw = np.arctan2(rotation[0, 2], rotation[2, 2])
        pitch = np.arcsin(np.clip(-rotation[1, 2], -1.0, 1.0))
        head_x, head_y, head_z = matrix[:3, 3]

        features = np.array(
            [eye_h, eye_v, eye_open, look_h, look_v, yaw, pitch, head_x, head_y, head_z], dtype=np.float64
        )
        return FaceResult(features=features, blinking=blinking, points=points, eyes=eyes)

    def close(self):
        self._landmarker.close()


def draw_face(frame: np.ndarray, face: FaceResult | None) -> None:
    """Draws eye outlines, irises and an exaggerated eye-direction arrow onto the frame."""
    if face is None:
        return
    for contour in EYE_CONTOURS:
        cv2.polylines(frame, [face.points[contour].astype(np.int32)], True, (80, 220, 80), 1, cv2.LINE_AA)
    for eye_centre, iris_centre, radius in face.eyes:
        c = tuple(int(round(v)) for v in iris_centre)
        tip = iris_centre + (iris_centre - eye_centre) * 6
        cv2.circle(frame, c, max(int(round(radius)), 1), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.arrowedLine(frame, c, tuple(int(round(v)) for v in tip), (0, 120, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(frame, c, 2, (0, 0, 255), -1, cv2.LINE_AA)
