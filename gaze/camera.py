"""Reads the webcam and runs the trackers on a background thread so the windows stay responsive."""

from __future__ import annotations

import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .tracker import GazeTracker, draw_face, face_crop

MAX_READ_FAILURES = 30
# 720p is enough: both models shrink the face to ~256 px anyway, and 1080p made MediaPipe 3x slower.
CAPTURE_SIZE = (1280, 720)
PREVIEW_WIDTH = 960  # the preview window is small, so don't send it full-size frames
MAX_DEEP_AGE_S = 1.0  # ignore deep-model results older than this (e.g. after the face was lost)
CROP_JPEG_QUALITY = 95


class CameraWorker(QThread):
    frame_ready = Signal(QImage)  # annotated, mirrored preview frame
    face_ready = Signal(object, float)  # FaceResult or None, timestamp in seconds
    deep_status = Signal(bool, str)  # deep model running?, description
    failed = Signal(str)

    def __init__(self, camera_index: int = 0, use_deep: bool = True, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.use_deep = use_deep
        self.save_crops = False  # set by the app while a session is being recorded
        self._running = False

    def _load_deep(self):
        if not self.use_deep:
            self.deep_status.emit(False, "Deep model off (--no-deep).")
            return None
        try:
            from .deep import DeepGaze, DeepRunner

            deep = DeepGaze()
        except Exception as exc:
            self.deep_status.emit(False, f"Deep model unavailable: {exc}")
            return None
        note = "" if deep.device.type == "cuda" else " (no GPU, so it updates a few times per second)"
        self.deep_status.emit(True, f"Deep model running on {deep.device_name}{note}.")
        return DeepRunner(deep)

    def run(self):
        try:
            tracker = GazeTracker()
        except Exception as exc:
            self.failed.emit(f"Could not load the face model: {exc}")
            return
        deep = self._load_deep()  # a DeepRunner, or None

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            tracker.close()
            if deep is not None:
                deep.close()
            self.failed.emit(f"Could not open camera {self.camera_index}. Is another app using it?")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
        cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        failures, fps, last = 0, 0.0, time.perf_counter()
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures > MAX_READ_FAILURES:
                        self.failed.emit("The camera stopped sending frames.")
                        break
                    continue
                failures = 0

                now = time.perf_counter()
                face = tracker.process(frame, int(now * 1000))
                if face is not None and (deep is not None or self.save_crops):
                    crop = face_crop(frame, face.points)
                    if self.save_crops:
                        jpeg = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])[1]
                        face.crop_jpeg = jpeg.tobytes()
                    if deep is not None:
                        deep.submit(crop, now)  # runs in the background
                        latest = deep.latest()  # most recent finished result (from an earlier frame)
                        if latest is not None and now - latest[1] < MAX_DEEP_AGE_S:
                            face.deep, face.deep_time = latest
                self.face_ready.emit(face, now)

                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                h, w = frame.shape[:2]
                scale = PREVIEW_WIDTH / w
                preview = cv2.resize(frame, (PREVIEW_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)
                draw_face(preview, face, scale)
                preview = cv2.flip(preview, 1)  # mirror it, like looking in a mirror
                label = f"{fps:4.1f} FPS" + (f"  |  deep {deep.rate:4.1f}/s" if deep else "") + ("" if face else "  |  no face")
                cv2.putText(preview, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                ph, pw = rgb.shape[:2]
                self.frame_ready.emit(QImage(rgb.data, pw, ph, 3 * pw, QImage.Format.Format_RGB888).copy())
        finally:
            cap.release()
            tracker.close()
            if deep is not None:
                deep.close()

    def stop(self):
        self._running = False
        self.wait(3000)
