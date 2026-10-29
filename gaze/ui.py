"""The three windows: camera preview, floating gaze pointer and calibration screen."""

from __future__ import annotations

import ctypes
import math
import sys
import time

import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QPixmap, QRadialGradient, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget

# --- Windows helpers: make a window ignore the mouse and stay above everything ---

GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010

if sys.platform == "win32":
    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.GetWindowLongW.restype = ctypes.c_long
    _user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    _user32.SetWindowLongW.restype = ctypes.c_long
    _user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND] + [ctypes.c_int] * 4 + [ctypes.c_uint]
    _user32.SetWindowPos.restype = wintypes.BOOL


def _make_click_through(widget: QWidget) -> None:
    if sys.platform != "win32":
        return
    hwnd = int(widget.winId())
    style = _user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    extra = WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    _user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | extra)


def _raise_to_top(widget: QWidget) -> None:
    if sys.platform != "win32":
        widget.raise_()
        return
    _user32.SetWindowPos(int(widget.winId()), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


# --- Gaze pointer ---------------------------------------------------------------


class GazePointer(QWidget):
    """A small glowing marker that floats above every window and lets clicks pass through.

    Moving a small window is much cheaper than repainting a transparent full-screen one.
    """

    SIZE = 64

    def __init__(self):
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(self.SIZE, self.SIZE)
        # Other "always on top" windows can cover us, so re-claim the top spot every second.
        self._topmost_timer = QTimer(self, interval=1000)
        self._topmost_timer.timeout.connect(lambda: _raise_to_top(self))

    def move_centre_to(self, global_pos: QPointF) -> None:
        self.move(round(global_pos.x() - self.SIZE / 2), round(global_pos.y() - self.SIZE / 2))
        if not self.isVisible():
            self.show()

    def showEvent(self, event):
        super().showEvent(event)
        _make_click_through(self)
        _raise_to_top(self)
        self._topmost_timer.start()

    def hideEvent(self, event):
        self._topmost_timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.SIZE / 2, self.SIZE / 2)
        glow = QRadialGradient(c, self.SIZE / 2)
        glow.setColorAt(0.0, QColor(255, 70, 70, 150))
        glow.setColorAt(0.5, QColor(255, 70, 70, 60))
        glow.setColorAt(1.0, QColor(255, 70, 70, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(c, self.SIZE / 2, self.SIZE / 2)
        p.setPen(QPen(QColor(255, 255, 255, 220), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, 12, 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 60, 60, 230))
        p.drawEllipse(c, 4, 4)


# --- Calibration screen ---------------------------------------------------------


def _calibration_targets(w: float, h: float, margin: float = 0.06) -> list[QPointF]:
    edges = (margin, 0.5, 1 - margin)
    # 3x3 grid walked in a snake so the dot never jumps across the whole screen...
    grid = [(x, y) for row, y in enumerate(edges) for x in (edges if row % 2 == 0 else edges[::-1])]
    # ...plus 4 points in between to help the curved parts of the mapping.
    inner = [(0.28, 0.28), (0.72, 0.28), (0.72, 0.72), (0.28, 0.72)]
    return [QPointF(fx * w, fy * h) for fx, fy in grid + inner]


class CalibrationScreen(QWidget):
    finished = Signal(object, object, object)  # features, screen targets, dot index per sample
    cancelled = Signal()

    MOVE_S = 0.9  # dot glides to its next spot (not recorded)
    SETTLE_S = 0.3  # eyes catch up (not recorded)
    RECORD_S = 1.0  # dot shrinks while samples are recorded

    def __init__(self, screen):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CursorShape.BlankCursor)
        g = screen.geometry()
        self._targets = _calibration_targets(g.width(), g.height())
        self._index = -1  # -1 = instructions screen
        self._phase_start = 0.0
        self._X, self._Y, self._groups = [], [], []
        self._face_visible = False
        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self._timer.start()

    def _begin(self):
        self._index = 0
        self._phase_start = time.perf_counter()

    def _elapsed(self) -> float:
        return time.perf_counter() - self._phase_start

    def add_sample(self, face) -> None:
        self._face_visible = face is not None
        if not 0 <= self._index < len(self._targets) or face is None or face.blinking:
            return
        if self._elapsed() >= self.MOVE_S + self.SETTLE_S:
            t = self._targets[self._index]
            self._X.append(face.features)
            self._Y.append((t.x(), t.y()))
            self._groups.append(self._index)

    def _tick(self):
        if self._index >= 0 and self._elapsed() >= self.MOVE_S + self.SETTLE_S + self.RECORD_S:
            self._index += 1
            self._phase_start = time.perf_counter()
            if self._index >= len(self._targets):
                self._timer.stop()
                self.finished.emit(np.array(self._X), np.array(self._Y), np.array(self._groups))
                return
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.cancelled.emit()
        elif event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._index < 0:
            self._begin()

    def mousePressEvent(self, event):
        if self._index < 0:
            self._begin()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 22, 28))
        p.setPen(QColor(220, 220, 230))

        if self._index < 0:
            p.setFont(QFont("Segoe UI", 16))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Gaze calibration\n\n"
                "A dot will move around the screen. Follow it with your eyes and\n"
                "look at its centre until it shrinks away.\n\n"
                "Keep your head still, in the position you'll use afterwards.\n\n"
                "Press Space (or click) to start  ·  Esc to cancel",
            )
        elif self._index < len(self._targets):
            t = self._elapsed()
            target = self._targets[self._index]
            prev = self._targets[self._index - 1] if self._index > 0 else QPointF(self.width() / 2, self.height() / 2)
            if t < self.MOVE_S:
                k = 0.5 - 0.5 * math.cos(math.pi * t / self.MOVE_S)  # ease in/out
                pos, radius = prev + (target - prev) * k, 16.0
            else:
                progress = min(max((t - self.MOVE_S - self.SETTLE_S) / self.RECORD_S, 0.0), 1.0)
                pos, radius = target, 16.0 - 10.0 * progress
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 90, 90))
            p.drawEllipse(pos, radius, radius)
            p.setBrush(QColor(20, 22, 28))
            p.drawEllipse(pos, 2.5, 2.5)
            p.setPen(QColor(130, 130, 145))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(
                self.rect().adjusted(0, 0, 0, -16),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                f"Dot {self._index + 1} / {len(self._targets)}",
            )

        if not self._face_visible:
            p.setPen(QColor(255, 120, 120))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(
                self.rect().adjusted(0, 16, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "Face not detected. Check the camera and the lighting on your face.",
            )


# --- Camera preview / control window ---------------------------------------------


class PreviewWindow(QWidget):
    calibrate_requested = Signal()
    pointer_toggled = Signal(bool)
    smoothing_changed = Signal(float)  # One Euro filter min_cutoff in Hz

    def __init__(self):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Gaze Tracker")

        self.video = QLabel("Starting camera…")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background: #111; color: #aaa;")
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video.setMinimumSize(320, 180)

        self.status = QLabel()
        self.status.setWordWrap(True)

        calibrate = QPushButton("Calibrate (C)")
        calibrate.clicked.connect(self.calibrate_requested)
        self.pointer_button = QPushButton("Pointer: on (P)")
        self.pointer_button.setCheckable(True)
        self.pointer_button.setChecked(True)
        self.pointer_button.toggled.connect(self._on_pointer_toggled)
        smoothing = QSlider(Qt.Orientation.Horizontal)
        smoothing.setRange(1, 10)
        smoothing.setValue(5)
        smoothing.valueChanged.connect(lambda v: self.smoothing_changed.emit(2.0 / v))

        controls = QHBoxLayout()
        controls.addWidget(calibrate)
        controls.addWidget(self.pointer_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Smoothing"))
        controls.addWidget(smoothing, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video, 1)
        layout.addLayout(controls)
        layout.addWidget(self.status)

        QShortcut(QKeySequence("C"), self).activated.connect(self.calibrate_requested)
        QShortcut(QKeySequence("P"), self).activated.connect(self.pointer_button.toggle)
        QShortcut(QKeySequence("Q"), self).activated.connect(self.close)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.close)

        self.resize(480, 400)
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - 24, area.bottom() - self.height() - 64)

    def _on_pointer_toggled(self, on: bool):
        self.pointer_button.setText(f"Pointer: {'on' if on else 'off'} (P)")
        self.pointer_toggled.emit(on)

    def set_frame(self, image: QImage):
        self.video.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

    def set_status(self, text: str):
        self.status.setStyleSheet("")
        self.status.setText(text)

    def set_error(self, text: str):
        self.status.setStyleSheet("color: #e05050;")
        self.status.setText(text)

    def closeEvent(self, event):
        super().closeEvent(event)
        QApplication.quit()
