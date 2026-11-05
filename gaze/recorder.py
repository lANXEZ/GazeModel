"""Records everything seen during calibration and accuracy tests, so models can be compared offline.

Every camera frame while the dots are on screen is kept, including the ones the app skipped
(eyes still moving, blinks), together with where the dot was. Each recording is a folder:

    recordings/2026-11-05_143000/
        session.npz   one row per camera frame (landmarks, features, deep angles, face crops...)
        meta.json     screen size, dot layout and timings, the app's own results

Face crops are stored as JPEG bytes in one buffer plus offsets, so nothing needs pickle to load.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"
SEQUENCES = ("calibration", "validation", "quick")
PHASES = ("move", "settle", "record")  # what the dot was doing (see ui.DotSequence)


class SessionRecorder:
    def __init__(self):
        self.started = datetime.now()
        self._rows = []
        self.blendshape_names: tuple[str, ...] = ()

    def __len__(self):
        return len(self._rows)

    def add(self, sequence: str, snapshot, face, t: float, accepted: bool) -> None:
        """snapshot: (dot index, phase, dot start time, target (x, y), shown (x, y), block) from DotSequence."""
        if face is not None and face.blendshape_names:
            self.blendshape_names = face.blendshape_names
        self._rows.append((SEQUENCES.index(sequence), snapshot, face, t, accepted))

    def save(self, meta: dict) -> Path:
        folder = RECORDINGS_DIR / self.started.strftime("%Y-%m-%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        n = len(self._rows)
        nan = np.nan
        arrays = {
            "t": np.empty(n), "sequence": np.empty(n, np.int8), "dot": np.empty(n, np.int16),
            "phase": np.empty(n, np.int8), "block": np.empty(n, np.int8), "dot_start": np.empty(n),
            "target": np.empty((n, 2)), "eye_pos": np.full((n, 3), nan),
            "shown": np.empty((n, 2)), "accepted": np.zeros(n, bool), "face": np.zeros(n, bool),
            "blinking": np.zeros(n, bool), "features": np.full((n, 10), nan),
            "points": np.full((n, 478, 2), nan, np.float32), "blendshapes": np.full((n, 52), nan, np.float32),
            "head_matrix": np.full((n, 4, 4), nan), "deep": np.full((n, 2), nan), "deep_time": np.full(n, nan),
        }
        crops, offsets = [], [0]
        for i, (sequence, (dot, phase, dot_start, target, shown, block), face, t, accepted) in enumerate(self._rows):
            arrays["t"][i], arrays["sequence"][i], arrays["dot"][i], arrays["block"][i] = t, sequence, dot, block
            arrays["phase"][i], arrays["dot_start"][i] = PHASES.index(phase), dot_start
            arrays["target"][i], arrays["shown"][i], arrays["accepted"][i] = target, shown, accepted
            if face is not None:
                arrays["face"][i], arrays["blinking"][i] = True, face.blinking
                arrays["features"][i], arrays["points"][i] = face.features, face.points
                arrays["blendshapes"][i, : len(face.blendshapes)] = face.blendshapes
                arrays["head_matrix"][i], arrays["eye_pos"][i] = face.head_matrix, face.eye_pos
                if face.deep is not None:
                    arrays["deep"][i], arrays["deep_time"][i] = face.deep, face.deep_time
            jpeg = face.crop_jpeg if face is not None and face.crop_jpeg else b""
            crops.append(jpeg)
            offsets.append(offsets[-1] + len(jpeg))
        arrays["crop_bytes"] = np.frombuffer(b"".join(crops), np.uint8)
        arrays["crop_offsets"] = np.array(offsets, np.int64)

        np.savez_compressed(folder / "session.npz", **arrays)
        meta = {"created": self.started.isoformat(timespec="seconds"), "frames": n,
                "sequences": list(SEQUENCES), "phases": list(PHASES),
                "blendshape_names": list(self.blendshape_names), **meta}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return folder


def load_recording(folder: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(Path(folder) / "session.npz") as npz:
        data = {key: npz[key] for key in npz.files}
    meta = json.loads((Path(folder) / "meta.json").read_text(encoding="utf-8"))
    return data, meta


def decode_crop(data: dict[str, np.ndarray], i: int) -> np.ndarray | None:
    """The face crop of row i as an RGB image, or None if that frame had no face."""
    start, end = data["crop_offsets"][i], data["crop_offsets"][i + 1]
    if start == end:
        return None
    image = cv2.imdecode(data["crop_bytes"][start:end], cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
