"""Webcam gaze tracker: live camera view + a gaze pointer that floats above every window."""

import argparse
import sys

# MediaPipe must be imported before PySide6: PySide6's import hook breaks the
# `six` package that MediaPipe pulls in (via matplotlib) if the order is reversed.
import mediapipe  # noqa: F401, I001
from PySide6.QtWidgets import QApplication

from gaze.app import GazeApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is the wrong camera)")
    parser.add_argument("--recalibrate", action="store_true", help="ignore the saved calibration")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # windows get hidden during calibration; closing the preview quits
    gaze = GazeApp(camera_index=args.camera, use_saved_calibration=not args.recalibrate)  # noqa: F841
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
