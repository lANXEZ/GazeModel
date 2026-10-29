"""Wires the camera thread, the calibration model and the windows together."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, Slot
from PySide6.QtGui import QGuiApplication

from .camera import CameraWorker
from .estimator import GazeEstimator, OneEuroFilter
from .ui import CalibrationScreen, GazePointer, PreviewWindow

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.npz"
MIN_CALIBRATION_DOTS = 6
FACE_LOST_HIDE_S = 0.4  # hide the pointer if the face is gone for this long


class GazeApp(QObject):
    def __init__(self, camera_index: int = 0, use_saved_calibration: bool = True):
        super().__init__()
        self.screen = QGuiApplication.primaryScreen()
        self.estimator = GazeEstimator()
        self.filter = OneEuroFilter()
        self.pointer = GazePointer()
        self.preview = PreviewWindow()
        self.calibration: CalibrationScreen | None = None
        self.pointer_enabled = True
        self._last_face_t = 0.0

        self.preview.calibrate_requested.connect(self.start_calibration)
        self.preview.pointer_toggled.connect(self.set_pointer_enabled)
        self.preview.smoothing_changed.connect(self.set_smoothing)

        self.worker = CameraWorker(camera_index)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.face_ready.connect(self.on_face)
        self.worker.failed.connect(self.preview.set_error)
        QGuiApplication.instance().aboutToQuit.connect(self.worker.stop)

        if use_saved_calibration and self.estimator.load(CALIBRATION_FILE, self._screen_size()):
            self.preview.set_status(
                f"Loaded saved calibration (expected error {self._describe_error()}). "
                "Press C to recalibrate if the pointer is off."
            )
        else:
            self.preview.set_status("Not calibrated yet. Sit comfortably, then press Calibrate (C).")

        self.preview.show()
        self.worker.start()

    def _screen_size(self) -> tuple[int, int]:
        size = self.screen.geometry().size()
        return size.width(), size.height()

    def _describe_error(self) -> str:
        err = self.estimator.error_px
        mm_per_px = self.screen.physicalSize().width() / self.screen.geometry().width()
        return f"about {err:.0f} px ≈ {err * mm_per_px / 10:.1f} cm"

    @Slot(object, float)
    def on_face(self, face, t: float):
        if self.calibration is not None:
            self.calibration.add_sample(face)
            return

        if face is not None:
            self._last_face_t = t
        elif t - self._last_face_t > FACE_LOST_HIDE_S:
            self.pointer.hide()
            self.filter.reset()
        if face is None or face.blinking or not self.estimator.is_calibrated or not self.pointer_enabled:
            return

        geo = self.screen.geometry()
        x, y = self.estimator.predict(face.features)
        x, y = self.filter(np.clip([x, y], 0, [geo.width() - 1, geo.height() - 1]), t)
        self.pointer.move_centre_to(QPointF(geo.x() + x, geo.y() + y))

    @Slot(bool)
    def set_pointer_enabled(self, on: bool):
        self.pointer_enabled = on
        if not on:
            self.pointer.hide()

    @Slot(float)
    def set_smoothing(self, min_cutoff: float):
        self.filter.min_cutoff = min_cutoff

    @Slot()
    def start_calibration(self):
        if self.calibration is not None:
            return
        self.pointer.hide()
        self.preview.hide()
        self.calibration = CalibrationScreen(self.screen)
        self.calibration.finished.connect(self._on_calibration_finished)
        self.calibration.cancelled.connect(self._end_calibration)
        self.calibration.start()

    def _end_calibration(self):
        self.preview.show()
        self.calibration.close()
        self.calibration.deleteLater()
        self.calibration = None
        self.filter.reset()

    def _on_calibration_finished(self, X, Y, groups):
        self._end_calibration()
        if len(np.unique(groups)) < MIN_CALIBRATION_DOTS:
            self.preview.set_error(
                "Calibration failed: your face wasn't detected for enough dots. "
                "Check the lighting and try again (C)."
            )
            return
        self.estimator.fit(X, Y, groups)
        self.estimator.save(CALIBRATION_FILE, self._screen_size())
        self.preview.set_status(f"Calibrated! Expected error {self._describe_error()}.")
