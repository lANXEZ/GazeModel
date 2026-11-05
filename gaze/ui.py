"""The windows: camera preview, floating gaze pointer, calibration / validation dots and results."""

from __future__ import annotations

import ctypes
import math
import sys
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QPixmap, QRadialGradient, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

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


BACKGROUND = QColor(20, 22, 28)
MODEL_COLORS = {
    "iris": QColor(255, 150, 60), "deep": QColor(80, 200, 255), "hybrid": QColor(120, 230, 120),
    "geo_iris": QColor(255, 120, 200), "geo_hybrid": QColor(240, 220, 80),
}


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


# --- Calibration / validation dots ----------------------------------------------


def _snake_grid(xs, ys) -> list[tuple[float, float]]:
    """Grid walked row by row in a snake, so the dot never jumps across the whole screen."""
    return [(x, y) for row, y in enumerate(ys) for x in (xs if row % 2 == 0 else xs[::-1])]


def _jittered_grid(rng, cols=5, rows=4, margin=0.05) -> list[tuple[float, float]]:
    """One random spot inside each cell of a grid: even coverage, but new positions every block."""
    xs, ys = np.linspace(margin, 1 - margin, cols + 1), np.linspace(margin, 1 - margin, rows + 1)
    dots = []
    for r in range(rows):
        for c in range(cols) if r % 2 == 0 else reversed(range(cols)):
            dots.append((float(rng.uniform(xs[c], xs[c + 1])), float(rng.uniform(ys[r], ys[r + 1]))))
    return dots


# Positions are fractions of the screen size.
# Validation dots sit at fixed spots, between where calibration dots usually land.
VALIDATION_DOTS = [p if p != (0.5, 0.5) else (0.39, 0.61) for p in _snake_grid(*[(0.17, 0.5, 0.83)] * 2)]
QUICK_DOTS = [(0.5, 0.06), (0.94, 0.5), (0.5, 0.94), (0.06, 0.5)]  # top, right, bottom, left

# Calibrating in several postures lets the models see how head position changes things.
POSTURES = (
    ("Sit normally", "Sit the way you usually work."),
    ("Lean back", "Lean back in your chair, into a relaxed position."),
    ("Sit closer", "Sit up and move a little closer to the screen."),
    ("Shift sideways", "Move your head a little to one side, or tilt it slightly."),
)
SPACE_TO_START = "Press Space (or click) to start  ·  Esc to cancel"


def calibration_blocks(rng=None) -> list[tuple[str, list[tuple[float, float]]]]:
    rng = rng or np.random.default_rng()
    blocks = []
    for i, (title, instruction) in enumerate(POSTURES):
        intro = (
            ("Gaze calibration: 4 short rounds in different postures (about 3 minutes).\n"
             "Follow the dot with your eyes and look at its centre until it shrinks away.\n\n" if i == 0 else "")
            + f"Round {i + 1} of {len(POSTURES)}: {title}\n\n{instruction}\n"
            "Keep that position while you follow the dot (20 spots, about 40 seconds).\n\n" + SPACE_TO_START
        )
        blocks.append((intro, _jittered_grid(rng)))
    return blocks


VALIDATION_BLOCKS = [(
    "Accuracy test\n\n"
    "9 more dots. Sit however you like, for example somewhere between the postures you just used.\n"
    "This measures how accurate each model really is.\n\n"
    "Press Space (or click) to start  ·  Esc to skip",
    VALIDATION_DOTS,
)]
QUICK_BLOCKS = [("", QUICK_DOTS)]


class DotSequence(QWidget):
    """Full-screen dot that visits each target, in blocks; records the face while you look at it.

    Each block starts with its own instructions (skipped when they're empty, as in the quick fix).
    """

    finished = Signal(object, object, object, object)  # faces, screen target, dot index, block index per face
    cancelled = Signal()

    def __init__(self, screen, blocks, move_s=0.8, settle_s=0.3, record_s=0.9, banner=""):
        """move_s: dot glides to its next spot; settle_s: eyes catch up; record_s: samples are recorded."""
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CursorShape.BlankCursor)
        g = screen.geometry()
        self.move_s, self.settle_s, self.record_s = move_s, settle_s, record_s
        self._banner = banner
        self._targets, self._block_of, self._intros = [], [], [intro for intro, _ in blocks]
        for b, (_, dots) in enumerate(blocks):
            self._targets += [QPointF(fx * g.width(), fy * g.height()) for fx, fy in dots]
            self._block_of += [b] * len(dots)
        self._index = 0
        self._waiting = True  # showing a block's instructions
        self._phase_start = 0.0
        self._faces, self._Y, self._groups, self._blocks = [], [], [], []
        self._face_visible = False
        self._timer = QTimer(self, interval=16)
        self._timer.timeout.connect(self._tick)

    @property
    def timing(self) -> dict[str, float]:
        return {"move": self.move_s, "settle": self.settle_s, "record": self.record_s}

    def start(self):
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self._timer.start()
        if not self._intros[0]:
            self._begin()

    def _begin(self):
        self._waiting = False
        self._phase_start = time.perf_counter()

    def _elapsed(self) -> float:
        return time.perf_counter() - self._phase_start

    def _running(self) -> bool:
        return not self._waiting and self._index < len(self._targets)

    def _dot_now(self) -> tuple[QPointF, float, str]:
        """Where the dot is drawn right now, its radius, and its phase (move / settle / record)."""
        t = self._elapsed()
        target = self._targets[self._index]
        if t < self.move_s:
            same_block = self._index > 0 and self._block_of[self._index - 1] == self._block_of[self._index]
            prev = self._targets[self._index - 1] if same_block else QPointF(self.width() / 2, self.height() / 2)
            k = 0.5 - 0.5 * math.cos(math.pi * t / self.move_s)  # ease in/out
            return prev + (target - prev) * k, 16.0, "move"
        progress = min(max((t - self.move_s - self.settle_s) / self.record_s, 0.0), 1.0)
        return target, 16.0 - 10.0 * progress, "settle" if t < self.move_s + self.settle_s else "record"

    def snapshot(self):
        """(dot index, phase, dot start time, target (x, y), shown (x, y), block) for recordings, or None."""
        if not self._running():
            return None
        shown, _, phase = self._dot_now()
        target = self._targets[self._index]
        return (self._index, phase, self._phase_start, (target.x(), target.y()), (shown.x(), shown.y()),
                self._block_of[self._index])

    def add_sample(self, face) -> bool:
        """Records the face if the eyes should be on the dot right now. Returns whether it was used."""
        self._face_visible = face is not None
        if not self._running() or face is None or face.blinking:
            return False
        record_start = self._phase_start + self.move_s + self.settle_s
        # The deep model runs behind the camera (a lot on a CPU): skip samples whose deep
        # result was computed from a frame taken before the eyes had settled on this dot.
        if time.perf_counter() < record_start or (face.deep_time is not None and face.deep_time < record_start):
            return False
        t = self._targets[self._index]
        self._faces.append(face)
        self._Y.append((t.x(), t.y()))
        self._groups.append(self._index)
        self._blocks.append(self._block_of[self._index])
        return True

    def _tick(self):
        if self._running() and self._elapsed() >= self.move_s + self.settle_s + self.record_s:
            self._index += 1
            self._phase_start = time.perf_counter()
            if self._index >= len(self._targets):
                self._timer.stop()
                self.finished.emit(self._faces, np.array(self._Y), np.array(self._groups), np.array(self._blocks))
                return
            if self._block_of[self._index] != self._block_of[self._index - 1] and self._intros[self._block_of[self._index]]:
                self._waiting = True  # next block: show its instructions first
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.cancelled.emit()
        elif event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._waiting:
            self._begin()

    def mousePressEvent(self, event):
        if self._waiting:
            self._begin()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        p.setPen(QColor(220, 220, 230))

        if self._waiting:
            p.setFont(QFont("Segoe UI", 16))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._intros[self._block_of[self._index]])
        elif self._index < len(self._targets):
            pos, radius, _ = self._dot_now()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 90, 90))
            p.drawEllipse(pos, radius, radius)
            p.setBrush(BACKGROUND)
            p.drawEllipse(pos, 2.5, 2.5)
            p.setPen(QColor(130, 130, 145))
            p.setFont(QFont("Segoe UI", 11))
            in_block = [i for i, b in enumerate(self._block_of) if b == self._block_of[self._index]]
            rounds = f"Round {self._block_of[self._index] + 1} / {len(self._intros)}  ·  " if len(self._intros) > 1 else ""
            p.drawText(
                self.rect().adjusted(0, 0, 0, -16),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                f"{rounds}Dot {self._index - in_block[0] + 1} / {len(in_block)}",
            )
            if self._banner:
                p.setPen(QColor(200, 200, 210))
                p.setFont(QFont("Segoe UI", 13))
                p.drawText(QRectF(self.rect()).adjusted(0, 0, 0, -self.height() * 0.6),
                           Qt.AlignmentFlag.AlignCenter, self._banner)

        if not self._face_visible:
            p.setPen(QColor(255, 120, 120))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(
                self.rect().adjusted(0, 16, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "Face not detected. Check the camera and the lighting on your face.",
            )


class ResultsScreen(QWidget):
    """Accuracy map: where each model thought you were looking, for every validation dot."""

    done = Signal()

    def __init__(
        self, screen, targets: np.ndarray, results: dict[str, tuple[str, np.ndarray, np.ndarray, str]], best: str
    ):
        """results: model name -> (label, dots it was scored on, mean predicted point per dot, error text)."""
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(screen.geometry())
        self._targets, self._results, self._best = targets, results, best

    def start(self):
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def keyPressEvent(self, event):
        self.done.emit()

    def mousePressEvent(self, event):
        self.done.emit()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)

        for name, (_, dots, means, _) in self._results.items():
            color = MODEL_COLORS[name]
            for (tx, ty), (mx, my) in zip(dots, means):
                p.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(tx, ty), QPointF(mx, my))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                p.drawEllipse(QPointF(mx, my), 6, 6)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(235, 235, 240), 2))
        for tx, ty in self._targets:
            p.drawEllipse(QPointF(tx, ty), 10, 10)
            p.drawLine(QPointF(tx - 4, ty), QPointF(tx + 4, ty))
            p.drawLine(QPointF(tx, ty - 4), QPointF(tx, ty + 4))

        # Legend, best model first.
        p.setFont(QFont("Segoe UI", 13))
        y = 40.0
        p.setPen(QColor(235, 235, 240))
        p.drawText(QPointF(40, y), "Accuracy test   ◯ = where you looked   ● = where the model thought you looked")
        for name, (label, _, _, error) in self._results.items():
            y += 30
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(MODEL_COLORS[name])
            p.drawEllipse(QPointF(48, y - 6), 6, 6)
            p.setPen(QColor(235, 235, 240))
            p.drawText(QPointF(64, y), f"{label}: average error {error}" + ("   ← best, now driving the pointer" if name == self._best else ""))

        p.setPen(QColor(130, 130, 145))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(
            QRectF(self.rect()).adjusted(0, 0, 0, -16),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
            "Press any key or click to continue",
        )


# --- Camera preview / control window ---------------------------------------------


class PreviewWindow(QWidget):
    calibrate_requested = Signal()
    validate_requested = Signal()
    quick_requested = Signal()
    model_selected = Signal(str)
    pointer_toggled = Signal(bool)
    recording_toggled = Signal(bool)
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
        self.deep_status = QLabel("Loading deep model…")
        self.deep_status.setStyleSheet("color: #888;")
        self.record_info = QLabel()
        self.record_info.setStyleSheet("color: #888;")
        self.record_info.setWordWrap(True)
        self.record_info.hide()

        calibrate = QPushButton("Calibrate (C)")
        calibrate.clicked.connect(self.calibrate_requested)
        self.validate_button = QPushButton("Test accuracy (V)")
        self.validate_button.clicked.connect(self.validate_requested)
        self.validate_button.setEnabled(False)
        self.quick_button = QPushButton("Quick fix (F)")
        self.quick_button.setToolTip("4 quick dots to re-align the pointer after you change posture")
        self.quick_button.clicked.connect(self.quick_requested)
        self.quick_button.setEnabled(False)
        self.pointer_button = QPushButton("Pointer: on (P)")
        self.pointer_button.setCheckable(True)
        self.pointer_button.setChecked(True)
        self.pointer_button.toggled.connect(self._on_pointer_toggled)

        self.model_box = QComboBox()
        self.model_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_box.currentIndexChanged.connect(
            lambda i: i >= 0 and self.model_selected.emit(self.model_box.itemData(i))
        )
        smoothing = QSlider(Qt.Orientation.Horizontal)
        smoothing.setRange(1, 10)
        smoothing.setValue(5)
        smoothing.valueChanged.connect(lambda v: self.smoothing_changed.emit(2.0 / v))

        buttons = QHBoxLayout()
        buttons.addWidget(calibrate)
        buttons.addWidget(self.validate_button)
        buttons.addWidget(self.quick_button)
        buttons.addWidget(self.pointer_button)
        self.record_box = QCheckBox("Record (R)")
        self.record_box.setToolTip(
            "Save calibration and accuracy-test data (incl. small face images) to recordings/ for offline analysis"
        )
        self.record_box.toggled.connect(self.recording_toggled)
        buttons.addWidget(self.record_box)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Model"))
        settings.addWidget(self.model_box)
        settings.addSpacing(12)
        settings.addWidget(QLabel("Smoothing"))
        settings.addWidget(smoothing, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video, 1)
        layout.addLayout(buttons)
        layout.addLayout(settings)
        layout.addWidget(self.status)
        layout.addWidget(self.deep_status)
        layout.addWidget(self.record_info)

        QShortcut(QKeySequence("C"), self).activated.connect(self.calibrate_requested)
        QShortcut(QKeySequence("V"), self).activated.connect(self.validate_button.click)
        QShortcut(QKeySequence("F"), self).activated.connect(self.quick_button.click)
        QShortcut(QKeySequence("P"), self).activated.connect(self.pointer_button.toggle)
        QShortcut(QKeySequence("R"), self).activated.connect(self.record_box.toggle)
        QShortcut(QKeySequence("Q"), self).activated.connect(self.close)
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.close)

        self.resize(600, 470)
        area = QGuiApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - 24, area.bottom() - self.height() - 64)

    def _on_pointer_toggled(self, on: bool):
        self.pointer_button.setText(f"Pointer: {'on' if on else 'off'} (P)")
        self.pointer_toggled.emit(on)

    def set_models(self, models: list[tuple[str, str]], active: str | None):
        """models: (name, label shown in the box) for every calibrated model that can run now."""
        self.model_box.blockSignals(True)
        self.model_box.clear()
        for name, label in models:
            self.model_box.addItem(label, name)
        if active is not None:
            self.model_box.setCurrentIndex(max(self.model_box.findData(active), 0))
        self.model_box.blockSignals(False)
        self.validate_button.setEnabled(bool(models))
        self.quick_button.setEnabled(bool(models))

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

    def set_deep_status(self, running: bool, text: str):
        self.deep_status.setStyleSheet("color: #5a5;" if running else "color: #888;")
        self.deep_status.setText(text)

    def set_record_info(self, text: str):
        self.record_info.setText(text)
        self.record_info.setVisible(bool(text))

    def closeEvent(self, event):
        super().closeEvent(event)
        QApplication.quit()
