"""Deep-learning gaze estimation with L2CS-Net (Abdelrahman et al., 2022, MIT licence).

A ResNet-50 trained on the Gaze360 dataset looks at a crop of your face and
outputs two gaze angles. Unlike the iris features it has seen thousands of
people, so it copes better with head movement and lighting. Its angles are fed
into the same calibration model as the iris features.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .tracker import face_crop

try:
    import torch
    import torchvision
    from safetensors.torch import load_file
    from torch import nn
except ImportError:  # the app still works without the deep model
    torch = None

# The original weights (Google Drive, .pkl) re-packaged by Py-Feat as .safetensors, which can only hold
# numbers, never code. Trained on Gaze360, whose licence allows research / personal use only, not commercial.
WEIGHTS_URL = "https://huggingface.co/py-feat/l2cs/resolve/main/l2cs_gaze360_resnet50.safetensors"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "l2cs" / "l2cs_gaze360_resnet50.safetensors"
NUM_BINS = 90  # Gaze360 model: 90 bins of 4°, covering -180° .. 180°
BIN_DEGREES = 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

if torch is not None:

    class L2CS(nn.Module):
        """ResNet-50 backbone with one classification head per gaze angle (same layer names as the original)."""

        def __init__(self, num_bins: int = NUM_BINS):
            super().__init__()
            resnet = torchvision.models.resnet50()
            self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            self.avgpool = resnet.avgpool
            self.fc_yaw_gaze = nn.Linear(2048, num_bins)
            self.fc_pitch_gaze = nn.Linear(2048, num_bins)
            self.fc_finetune = nn.Linear(2048 + 3, 3)  # unused, but present in the released weights

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = torch.flatten(self.avgpool(x), 1)
            return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


def _ensure_weights(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading deep gaze model (~95 MB) to {path} ...")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
        partial.replace(path)


class DeepGaze:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        _ensure_weights(weights_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            # Leave two cores for the camera, MediaPipe and the windows.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2))
        model = L2CS()
        model.load_state_dict(load_file(str(weights_path)))
        self.model = model.to(self.device, memory_format=torch.channels_last).eval()
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._bins = torch.arange(NUM_BINS, dtype=torch.float32, device=self.device)

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"

    def predict(self, frame_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Returns the two gaze angles (radians) for the face outlined by the MediaPipe landmarks."""
        return self.predict_crop(face_crop(frame_bgr, points))

    def predict_crop(self, face: np.ndarray) -> np.ndarray:
        """Gaze angles (radians) for a 224x224 RGB face crop from tracker.face_crop()."""
        # As in the authors' pipeline: 224x224 crop -> 448x448 -> ImageNet normalisation.
        face = cv2.resize(face, (448, 448), interpolation=cv2.INTER_LINEAR)
        with torch.inference_mode():
            x = torch.from_numpy(face).to(self.device).permute(2, 0, 1)[None].float().div_(255)
            x = ((x - self._mean) / self._std).contiguous(memory_format=torch.channels_last)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                outputs = self.model(x)
            angles = [(torch.softmax(o.float(), dim=1) @ self._bins[:, None]) * BIN_DEGREES - 180 for o in outputs]
            return np.radians(torch.cat(angles, dim=1)[0].cpu().numpy().astype(np.float64))


class DeepRunner:
    """Runs the deep model on its own thread so a slow (CPU-only) model never holds up the camera.

    The camera loop hands over each new face crop and immediately carries on. The thread always
    works on the newest crop, skipping any it couldn't get to, and keeps the latest result.
    """

    def __init__(self, deep: DeepGaze):
        self._deep = deep
        self._lock = threading.Lock()
        self._job: tuple[np.ndarray, float] | None = None  # (face crop, capture time)
        self._result: tuple[np.ndarray, float] | None = None  # (angles, capture time of their frame)
        self._wake = threading.Event()
        self._stopping = False
        self.rate = 0.0  # results per second
        self._thread = threading.Thread(target=self._run, name="deep-gaze", daemon=True)
        self._thread.start()

    def submit(self, crop: np.ndarray, captured_at: float) -> None:
        with self._lock:
            self._job = (crop, captured_at)
        self._wake.set()

    def latest(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            return self._result

    def _run(self):
        last = None
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                return
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            angles = self._deep.predict_crop(job[0])
            now = time.perf_counter()
            if last is not None:
                self.rate = 0.8 * self.rate + 0.2 / max(now - last, 1e-6)
            last = now
            with self._lock:
                self._result = (angles, job[1])

    def close(self):
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout=2)
