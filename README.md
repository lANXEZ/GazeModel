# GazeModel

Webcam eye-gaze tracking experiment. It shows the live camera feed with your eyes and irises
highlighted, and a red gaze pointer that floats **on top of every window** (clicks pass through it)
at the spot you're looking at.

Five models are trained from the same calibration and compared on an accuracy test:

| Model | What it looks at | Head movement |
|---|---|---|
| **MediaPipe iris** | Where each iris sits inside the eye, plus head pose (CPU) | learned from calibration |
| **Deep model (L2CS-Net)** | A neural network trained on thousands of faces estimates gaze angles (GPU) | learned |
| **Deep + iris** | Both together; the calibration learns how much to trust each | learned |
| **Geometric iris** | Eye-in-head angles from the irises; a 3D ray from your eye is cast onto the screen | **geometry** |
| **Geometric deep + iris** | The same, using the deep model too | **geometry** |

The geometric models compute *gaze direction = head direction + eye direction within the head* and
intersect that ray with the screen, so moving your head (leaning back, sitting closer, turning)
is handled by geometry instead of having to be learned. The learned models only use a head value
(position, distance, rotation) if it actually varied during calibration; otherwise they ignore it
rather than guess, because a guessed effect throws the pointer far off when you do move.

## Setup (once)

Needs Python 3.12 (MediaPipe doesn't support 3.14 yet).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional, for the deep model (~3 GB, uses the NVIDIA GPU):
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

On first run the face model (~4 MB) and the deep gaze model (~95 MB) are downloaded to `models/`.

1. A small **Gaze Tracker** window opens in the bottom-right with your camera feed.
   Green = eye outline, yellow = iris, orange arrow = exaggerated eye direction.
   The bottom line says whether the deep model is running and on which device.
2. Press **Calibrate (C)**: 4 rounds of 20 dots, each in a different posture (sit normally, lean
   back, sit closer, shift sideways), about 3 minutes. Follow the red dot with your eyes and look at
   its centre until it shrinks. Calibrating in several postures teaches the models how your head
   position changes things.
3. An **accuracy test** follows: 9 new dots. Then a map shows where each model thought you were
   looking and its average error in cm. The best model drives the pointer automatically.
4. The red gaze pointer appears. Switch to any other app; it stays on top.
5. After changing posture, press **Quick fix (F)**: 4 dots (top, right, bottom, left) in about
   6 seconds re-align the pointer's position and scale without a full calibration.

| Key (in the Gaze Tracker window) | Action |
|---|---|
| `C` | Recalibrate in 4 postures (followed by the accuracy test) |
| `V` | Run the accuracy test again (e.g. after moving, to see how much it drifted) |
| `F` | Quick fix: 4 dots to re-align after changing posture |
| `P` | Show / hide the pointer |
| `Q` / `Esc` | Quit |

The **Model** box switches between the calibrated models. The **Smoothing** slider trades steadiness
for lag. Calibration is saved to `calibration.npz` and reused next time.

Options: `--recalibrate` ignores the saved calibration, `--camera 1` picks another webcam,
`--no-deep` skips the deep model (faster start-up, no GPU use), `--record` turns recording on.

## Recording sessions and offline analysis

Tick **Record (R)** (or start with `--record`) and every calibration / accuracy test is saved to
`recordings/<date-time>/` (~40–50 MB per session). It keeps every camera frame while the dots are
shown: landmarks, eye-direction scores, head pose, deep-model angles, where the dot was, and a small
224×224 JPEG of your face (the deep model's input, so it can also be used for fine-tuning later).
`recordings/` is git-ignored because it contains face images; delete it whenever you like.

Re-run the calibration and accuracy test on a recording without sitting through it again:

```powershell
.\.venv\Scripts\python.exe analyze.py                     # latest recording, same samples as the app
.\.venv\Scripts\python.exe analyze.py --recompute-deep    # re-run the deep model on the saved face crops
```

The first form reproduces the app's numbers exactly. `--recompute-deep` gives the deep model the
exact frame for each sample (useful when it ran on a CPU and lagged behind). For a multi-posture
calibration it also runs a **posture test**: calibrate on 3 postures, test on the 4th, which shows
how each model copes with a posture it wasn't calibrated in.

## Resource use (measured on an RTX 5060 Laptop GPU, 16-thread CPU)

| | Deep model on | `--no-deep` |
|---|---|---|
| CPU | ~1–1.5 cores (~6–9%) | ~0.8–1.4 cores (~5–8%) |
| RAM | ~1.4 GB (PyTorch + CUDA libraries) | ~0.7 GB |
| GPU | ~8–12% busy, ~180 MB video memory | not used |
| GPU power | ~13–17 W | ~4 W (idle) |

The deep model runs on its own background thread, always on the newest frame, so it never slows
the camera, the iris tracking or the pointer. The frame rate is set by the webcam (30 fps, or
20 fps in dim light). How far the deep model's answers lag behind the camera:

| Where it runs | Deep model per face | Its results lag by |
|---|---|---|
| NVIDIA GPU (RTX 5060 Laptop) | ~8–18 ms | ~0.03 s |
| CPU only, 8 threads | ~120 ms | ~0.26 s |
| CPU only, 2 threads (small VM) | ~300 ms | ~0.5 s |

## Running in a VM (e.g. VMware) without a GPU

It works: the deep model falls back to the CPU automatically (the preview says
"Deep model running on CPU"). Tips:

- Install the CPU-only PyTorch (~200 MB instead of ~3 GB):
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Connect the webcam to the VM (VMware Workstation: *VM → Removable Devices → camera → Connect*)
  and check the FPS in the preview.
- The pointer and calibration cover the **VM's** screen, so run the VM full screen.
- With few cores the deep model lags ~0.5 s, which makes it react slowly to quick glances. Compare
  the models in the accuracy test; if "MediaPipe iris" wins, `--no-deep` saves CPU.

## Tips for better accuracy

- Light your face from the front; avoid a bright window behind you.
- Follow each round's posture instruction during calibration; the variety is what makes the
  pointer survive moving around later. After a big posture change, press `F`.
- Raise the laptop so the camera is closer to eye level.
- Glasses with strong reflections make iris tracking noisier.

## How it works

```
                 ┌─► iris features + head pose (MediaPipe, CPU) ─┐
webcam ─► face ──┤                                               ├─► calibration model ─► smoothing ─► pointer
       landmarks └─► face crop ─► L2CS-Net (GPU) angles ─────────┘   learned, or geometric   One Euro
                                                                     (+ quick-fix correction) filter
```

| File | What it does |
|---|---|
| `gaze/tracker.py` | Runs MediaPipe on each frame and turns landmarks into ~10 gaze features |
| `gaze/deep.py` | L2CS-Net: face crop → ResNet-50 → gaze pitch / yaw |
| `gaze/estimator.py` | The 5 models (learned and geometric), quick-fix correction, jitter filter |
| `gaze/camera.py` | Reads the webcam and runs both trackers on a background thread |
| `gaze/ui.py` | Preview window, always-on-top pointer, calibration / accuracy-test screens |
| `gaze/recorder.py` | Saves / loads recordings of calibration and accuracy-test sessions |
| `gaze/app.py` | Connects everything |
| `analyze.py` | Re-runs calibration + accuracy test on a recording, for offline comparisons |

## Licences

- Code in this repo: yours.
- L2CS-Net code: MIT ([Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net)).
- L2CS-Net weights: re-packaged as `.safetensors` by [Py-Feat](https://huggingface.co/py-feat/l2cs).
  They were trained on Gaze360, whose licence **excludes commercial use**; fine for personal
  experiments and research.
