"""Reads the webcam and runs the tracker on a background thread so the windows stay responsive."""

from __future__ import annotations

import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .tracker import GazeTracker, draw_face

MAX_READ_FAILURES = 30


class CameraWorker(QThread):
    frame_ready = Signal(QImage)  # annotated, mirrored preview frame
    face_ready = Signal(object, float)  # FaceResult or None, timestamp in seconds
    failed = Signal(str)

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False

    def run(self):
        try:
            tracker = GazeTracker()
        except Exception as exc:
            self.failed.emit(f"Could not load the face model: {exc}")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            tracker.close()
            self.failed.emit(f"Could not open camera {self.camera_index}. Is another app using it?")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                self.face_ready.emit(face, now)

                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                last = now
                draw_face(frame, face)
                frame = cv2.flip(frame, 1)  # mirror it, like looking in a mirror
                label = f"{fps:4.1f} FPS" + ("" if face else "  |  no face")
                cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                self.frame_ready.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())
        finally:
            cap.release()
            tracker.close()

    def stop(self):
        self._running = False
        self.wait(3000)
