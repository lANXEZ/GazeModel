# GazeModel

Webcam eye-gaze tracking experiment. It shows the live camera feed with your eyes and irises
highlighted, and a red gaze pointer that floats **on top of every window** (clicks pass through it)
at the spot you're looking at.

## Setup (once)

Needs Python 3.12 (MediaPipe doesn't support 3.14 yet).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

The face model (~4 MB) is downloaded to `models/` on first run.

1. A small **Gaze Tracker** window opens in the bottom-right with your camera feed.
   Green = eye outline, yellow = iris, orange arrow = exaggerated eye direction.
2. Press **Calibrate (C)**. Follow the red dot with your eyes and look at its centre until it
   shrinks; there are 13 dots and it takes about 30 seconds.
3. The red gaze pointer appears. Switch to any other app; it stays on top.

| Key (in the Gaze Tracker window) | Action |
|---|---|
| `C` | Recalibrate |
| `P` | Show / hide the pointer |
| `Q` / `Esc` | Quit |

The **Smoothing** slider trades steadiness for lag. Calibration is saved to `calibration.npz`
and reused next time (`--recalibrate` ignores it, `--camera 1` picks another webcam).

## Tips for better accuracy

- Light your face from the front; avoid a bright window behind you.
- Calibrate in the posture you'll actually use, and keep your head roughly there afterwards.
  If you move or lean, recalibrate. It only takes 30 seconds.
- Glasses with strong reflections make iris tracking noisier.
- Expect roughly 2–5 cm of error on a laptop screen. That's good for "which region am I looking
  at", not for picking out single words. The status line shows the estimated error after
  calibrating.

## How it works

```
webcam ──► MediaPipe Face Landmarker ──► features ──► calibration model ──► smoothing ──► pointer
           478 face points incl. irises,    iris position     ridge regression    One Euro
           head pose, eye-direction scores  in each eye,      learned from        filter
                                            head yaw/pitch…   the 13 dots
```

| File | What it does |
|---|---|
| `gaze/tracker.py` | Runs MediaPipe on each frame and turns landmarks into ~10 gaze features |
| `gaze/estimator.py` | Calibration regression (features → screen x, y) and the jitter filter |
| `gaze/camera.py` | Reads the webcam on a background thread |
| `gaze/ui.py` | Preview window, always-on-top click-through pointer, calibration screen |
| `gaze/app.py` | Connects everything |

## Ideas for next steps

- Try a deep-learning gaze model (e.g. L2CS-Net on the GPU) and compare it against this one.
- Add a heat map that records where you looked over time.
- Dwell-to-click: trigger a click when you stare at one spot for ~1 second.
