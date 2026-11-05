"""Wires the camera thread, the calibration models and the windows together."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, Slot
from PySide6.QtGui import QGuiApplication

from .camera import CAPTURE_SIZE, CameraWorker
from .estimator import (
    FEATURE_SETS, GazeEstimator, OneEuroFilter, ScreenGeometry, load_calibration, samples_for, save_calibration,
)
from .recorder import RECORDINGS_DIR, SessionRecorder
from .tracker import FEATURE_NAMES
from .ui import (
    QUICK_BLOCKS, VALIDATION_BLOCKS, DotSequence, GazePointer, PreviewWindow, ResultsScreen, calibration_blocks,
)

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.npz"
MIN_CALIBRATION_DOTS = 6
MIN_VALIDATION_DOTS = 3
MIN_QUICK_DOTS = 3
QUICK_TIMING = {"move_s": 0.5, "settle_s": 0.25, "record_s": 0.6}  # 4 dots in about 6 seconds
FACE_LOST_HIDE_S = 0.4  # hide the pointer if the face is gone for this long
HEAD_NAMES = {5: "head turn", 6: "head tilt", 7: "sideways position", 8: "height", 9: "distance"}


class GazeApp(QObject):
    def __init__(
        self, camera_index: int = 0, use_saved_calibration: bool = True, use_deep: bool = True, record: bool = False
    ):
        super().__init__()
        self.screen = QGuiApplication.primaryScreen()
        geo, physical = self.screen.geometry(), self.screen.physicalSize()
        self.geometry = ScreenGeometry((geo.width(), geo.height()), (physical.width(), physical.height()))
        self.estimators: dict[str, GazeEstimator] = {}  # one calibration model per feature set
        self.active: str | None = None  # the model driving the pointer
        self.deep_running = False
        self.deep_text = ""
        self.record_enabled = False
        self.recorder: SessionRecorder | None = None  # collects frames while dots are shown, if recording
        self._sequence_name = ""
        self._timings: dict[str, dict[str, float]] = {}
        self.filter = OneEuroFilter()
        self.pointer = GazePointer()
        self.preview = PreviewWindow()
        self.fullscreen: DotSequence | ResultsScreen | None = None
        self.pointer_enabled = True
        self._last_face_t = 0.0

        self.preview.calibrate_requested.connect(self.start_calibration)
        self.preview.validate_requested.connect(self.start_validation)
        self.preview.quick_requested.connect(self.start_quick_fix)
        self.preview.model_selected.connect(self.set_active_model)
        self.preview.pointer_toggled.connect(self.set_pointer_enabled)
        self.preview.smoothing_changed.connect(self.set_smoothing)
        self.preview.recording_toggled.connect(self.set_recording)

        self.worker = CameraWorker(camera_index, use_deep)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.face_ready.connect(self.on_face)
        self.worker.deep_status.connect(self.on_deep_status)
        self.worker.failed.connect(self.preview.set_error)
        QGuiApplication.instance().aboutToQuit.connect(self.worker.stop)

        if use_saved_calibration:
            self.estimators = load_calibration(CALIBRATION_FILE, self.geometry)
        self._refresh_models(select_best=True)
        if self.active is not None:
            self.preview.set_status(
                f"Loaded saved calibration ({self._describe(self.estimators[self.active].best_error_px)}). "
                "Press F for a quick fix after changing posture, or C to recalibrate."
            )
        else:
            self.preview.set_status("Not calibrated yet. Press Calibrate (C).")

        self.preview.record_box.setChecked(record)
        self.preview.show()
        self.worker.start()

    # --- helpers ---

    def _describe(self, error_px: float) -> str:
        return f"{error_px:.0f} px ≈ {error_px * self.geometry.cm_per_px[0]:.1f} cm"

    def _usable_models(self) -> list[str]:
        return [n for n in FEATURE_SETS if n in self.estimators and (self.deep_running or not FEATURE_SETS[n].needs_deep)]

    def _refresh_models(self, select_best: bool = False):
        usable = self._usable_models()
        if select_best or self.active not in usable:
            self.active = min(usable, key=lambda n: self.estimators[n].best_error_px) if usable else None
        self.preview.set_models(
            [(n, f"{FEATURE_SETS[n].label}  ({self._describe(self.estimators[n].best_error_px)})") for n in usable],
            self.active,
        )
        self.filter.reset()

    def _save(self):
        save_calibration(CALIBRATION_FILE, self.estimators, self.geometry)

    def _show_fullscreen(self, widget: DotSequence | ResultsScreen | None):
        """Swaps the current full-screen window (if any) for `widget`; None goes back to normal."""
        old, self.fullscreen = self.fullscreen, widget
        if widget is not None:
            self.pointer.hide()
            self.preview.hide()
            widget.start()
        else:
            self.preview.show()
            self.filter.reset()
        if old is not None:
            old.close()
            old.deleteLater()

    def _run_dots(self, name, blocks, on_finished, **kwargs):
        if self.record_enabled and self.recorder is None:
            self.recorder = SessionRecorder()
        self._sequence_name = name
        sequence = DotSequence(self.screen, blocks, **kwargs)
        self._timings[name] = sequence.timing
        sequence.finished.connect(on_finished)
        sequence.cancelled.connect(self._end_dots)
        self._show_fullscreen(sequence)

    def _end_dots(self, error: str = ""):
        """Back to normal after calibration / accuracy test / quick fix (finished, cancelled or failed)."""
        self._show_fullscreen(None)
        if error:
            self.preview.set_error(error)
        self._save_recording()

    def _save_recording(self):
        recorder, self.recorder = self.recorder, None
        if recorder is None or len(recorder) == 0:
            return
        folder = recorder.save({
            "screen_size": list(self.geometry.size_px),
            "screen_mm": list(self.geometry.size_mm),
            "device_pixel_ratio": self.screen.devicePixelRatio(),
            "camera_size": list(CAPTURE_SIZE),
            "deep_model": self.deep_text,
            "dot_timing_s": self._timings,
            "feature_names": list(FEATURE_NAMES),
            "app_results_px": {
                name: {"calibration": est.error_px, "validation": est.validation_px}
                for name, est in self.estimators.items()
            },
        })
        size_mb = sum(f.stat().st_size for f in folder.iterdir()) / 2**20
        self.preview.set_record_info(f"Recording saved: {folder.relative_to(RECORDINGS_DIR.parent)} ({size_mb:.0f} MB)")

    # --- live tracking ---

    @Slot(object, float)
    def on_face(self, face, t: float):
        if isinstance(self.fullscreen, DotSequence):
            snapshot = self.fullscreen.snapshot()
            accepted = self.fullscreen.add_sample(face)
            if self.recorder is not None and snapshot is not None:
                self.recorder.add(self._sequence_name, snapshot, face, t, accepted)
            return
        if self.fullscreen is not None:
            return

        if face is not None:
            self._last_face_t = t
        elif t - self._last_face_t > FACE_LOST_HIDE_S:
            self.pointer.hide()
            self.filter.reset()
        if face is None or face.blinking or self.active is None or not self.pointer_enabled:
            return
        features = FEATURE_SETS[self.active].extract(face)
        if features is None:
            return

        geo = self.screen.geometry()
        x, y = self.estimators[self.active].predict(features)
        x, y = self.filter(np.clip([x, y], 0, [geo.width() - 1, geo.height() - 1]), t)
        self.pointer.move_centre_to(QPointF(geo.x() + x, geo.y() + y))

    @Slot(bool, str)
    def on_deep_status(self, running: bool, text: str):
        self.deep_running = running
        self.deep_text = text
        self.preview.set_deep_status(running, text)
        self._refresh_models(select_best=True)

    @Slot(str)
    def set_active_model(self, name: str):
        self.active = name
        self.filter.reset()

    @Slot(bool)
    def set_pointer_enabled(self, on: bool):
        self.pointer_enabled = on
        if not on:
            self.pointer.hide()

    @Slot(bool)
    def set_recording(self, on: bool):
        self.record_enabled = on
        self.worker.save_crops = on  # face crops are only made while recording is on
        self.preview.set_record_info(
            f"Recording on: calibrations, accuracy tests and quick fixes are saved to {RECORDINGS_DIR.name}/ "
            "(incl. small face images)." if on else ""
        )

    @Slot(float)
    def set_smoothing(self, min_cutoff: float):
        self.filter.min_cutoff = min_cutoff

    # --- calibration, accuracy test and quick fix ---

    @Slot()
    def start_calibration(self):
        if self.fullscreen is None:
            self._run_dots("calibration", calibration_blocks(), self._on_calibration_dots)

    @Slot()
    def start_validation(self):
        if self.fullscreen is None and self._usable_models():
            self._run_dots("validation", VALIDATION_BLOCKS, self._on_validation_dots)

    @Slot()
    def start_quick_fix(self):
        if self.fullscreen is None and self._usable_models():
            self._run_dots("quick", QUICK_BLOCKS, self._on_quick_dots, banner="Quick fix: follow the dot", **QUICK_TIMING)

    def _on_calibration_dots(self, faces, Y, groups, blocks):
        if len(np.unique(groups)) < MIN_CALIBRATION_DOTS:
            self._end_dots(
                "Calibration failed: your face wasn't detected for enough dots. "
                "Check the lighting and try again (C)."
            )
            return
        # Every model learns from the same recording session, so they can be compared fairly.
        self.estimators = {}
        for name, feature_set in FEATURE_SETS.items():
            samples = samples_for(name, faces, Y, groups, MIN_CALIBRATION_DOTS)
            if samples is not None:
                self.estimators[name] = GazeEstimator(feature_set, self.geometry)
                self.estimators[name].fit(*samples)
        self._save()
        self._refresh_models(select_best=True)
        ignored = [HEAD_NAMES[c] for c in self.estimators["iris"].ignored_head]
        self.preview.set_status(
            f"Calibrated. Using {FEATURE_SETS[self.active].label} "
            f"(estimated error {self._describe(self.estimators[self.active].best_error_px)})."
            + (f" Your {', '.join(ignored)} hardly changed, so the learned models ignore it." if ignored else "")
        )
        self._run_dots("validation", VALIDATION_BLOCKS, self._on_validation_dots)

    def _on_validation_dots(self, faces, Y, groups, blocks):
        dots = np.unique(groups)
        if len(dots) < MIN_VALIDATION_DOTS:
            self._end_dots("Accuracy test failed: your face wasn't detected for enough dots (V to retry).")
            return
        results = {}
        for name in self._usable_models():
            samples = samples_for(name, faces, Y, groups, MIN_VALIDATION_DOTS)
            if samples is not None:
                model_dots, means = self.estimators[name].validate(*samples)
                error = self._describe(self.estimators[name].validation_px)
                results[name] = (FEATURE_SETS[name].label, model_dots, means, error)
        if not results:
            self._end_dots("Accuracy test failed: not enough usable samples (V to retry).")
            return
        self._save()
        self._refresh_models(select_best=True)

        best = self.estimators[self.active]
        self.preview.set_status(
            f"Accuracy test: using {FEATURE_SETS[self.active].label}, average error {self._describe(best.validation_px)}."
        )
        targets = np.array([Y[groups == g][0] for g in dots])
        ranked = dict(sorted(results.items(), key=lambda item: self.estimators[item[0]].validation_px))
        screen = ResultsScreen(self.screen, targets, ranked, self.active)
        screen.done.connect(lambda: self._show_fullscreen(None))
        self._show_fullscreen(screen)
        self._save_recording()

    def _on_quick_dots(self, faces, Y, groups, blocks):
        if len(np.unique(groups)) < MIN_QUICK_DOTS:
            self._end_dots("Quick fix failed: your face wasn't detected for enough dots (F to retry).")
            return
        moved = {}
        for name in self._usable_models():
            samples = samples_for(name, faces, Y, groups, MIN_QUICK_DOTS)
            if samples is not None:
                moved[name] = self.estimators[name].fit_correction(*samples)
        self._save()
        self._refresh_models()
        self._end_dots()
        if self.active in moved:
            self.preview.set_status(
                f"Quick fix applied to every model; {FEATURE_SETS[self.active].label} moved by about "
                f"{self._describe(moved[self.active])}. Press V to check accuracy."
            )
