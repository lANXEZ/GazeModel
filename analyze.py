"""Re-runs calibration + accuracy test on a saved recording, so models can be compared offline.

    .\\.venv\\Scripts\\python.exe analyze.py                         (latest recording)
    .\\.venv\\Scripts\\python.exe analyze.py recordings\\2026-11-05_143000 --recompute-deep

If the calibration was done in several postures, it also checks each model on a posture it
wasn't calibrated in: calibrate on the other postures, test on the left-out one.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gaze.estimator import FEATURE_SETS, GazeEstimator, ScreenGeometry, samples_for
from gaze.recorder import PHASES, RECORDINGS_DIR, SEQUENCES, decode_crop, load_recording
from gaze.tracker import eye_position

MIN_CALIBRATION_DOTS = 6
MIN_TEST_DOTS = 3


def recompute_deep(data):
    """Runs the deep model on every saved face crop, so its angles match their frame exactly (no lag)."""
    from gaze.deep import DeepGaze

    model = DeepGaze()
    print(f"Recomputing deep angles on {model.device_name} for {int(data['face'].sum())} frames ...")
    deep = np.full_like(data["deep"], np.nan)
    for i in np.flatnonzero(data["face"]):
        crop = decode_crop(data, i)
        if crop is not None:
            deep[i] = model.predict_crop(crop)
    data["deep"], data["deep_time"] = deep, data["t"].copy()


def fill_missing(data, meta):
    """Older recordings lack some columns; derive them."""
    n = len(data["t"])
    if "block" not in data:
        data["block"] = np.zeros(n, np.int8)
    if "eye_pos" not in data:
        data["eye_pos"] = np.full((n, 3), np.nan)
        for i in np.flatnonzero(data["face"]):
            data["eye_pos"][i] = eye_position(data["points"][i].astype(float), data["features"][i, 9],
                                              tuple(meta["camera_size"]))


def faces_for(data, idx):
    return [
        SimpleNamespace(features=data["features"][i], eye_pos=data["eye_pos"][i],
                        deep=None if np.isnan(data["deep"][i]).any() else data["deep"][i])
        for i in idx
    ]


def split(data, rows):
    idx = np.flatnonzero(rows)
    return faces_for(data, idx), data["target"][idx], data["dot"][idx]


def score(screen, name, train, test):
    """(leave-one-dot-out estimate, test error) in px, or None if there's too little data."""
    train = samples_for(name, *train, MIN_CALIBRATION_DOTS)
    test = samples_for(name, *test, MIN_TEST_DOTS)
    if train is None:
        return None
    est = GazeEstimator(FEATURE_SETS[name], screen)
    est.fit(*train)
    if test is not None:
        est.validate(*test)
    return est.error_px, est.validation_px


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", type=Path, help="recording folder (default: the latest)")
    parser.add_argument("--recompute-deep", action="store_true", help="re-run the deep model on the saved face crops")
    args = parser.parse_args()

    folder = args.folder or max(RECORDINGS_DIR.iterdir(), key=lambda f: f.name)
    data, meta = load_recording(folder)
    fill_missing(data, meta)
    screen = ScreenGeometry(tuple(meta["screen_size"]), tuple(meta["screen_mm"]))
    cm = screen.cm_per_px[0]
    print(f"Recording {folder.name}: {meta['frames']} frames, {int(data['face'].sum())} with a face")

    if args.recompute_deep:
        recompute_deep(data)
        # No lag any more, so use every non-blink frame from the "record" phase of each dot.
        rows = data["face"] & ~data["blinking"] & (data["phase"] == PHASES.index("record"))
    else:
        rows = data["accepted"]  # exactly the samples the app used
    calib_rows = rows & (data["sequence"] == SEQUENCES.index("calibration"))
    test_rows = rows & (data["sequence"] == SEQUENCES.index("validation"))

    def fmt(px):
        return "-" if px is None else f"{px:5.0f} px {px * cm:4.1f} cm"

    print(f"Samples used: {calib_rows.sum()} calibration, {test_rows.sum()} accuracy test\n")
    print(f"{'model':24s} {'calibration est.':>18s} {'accuracy test':>18s} {'app said':>10s}")
    for name, feature_set in FEATURE_SETS.items():
        result = score(screen, name, split(data, calib_rows), split(data, test_rows))
        if result is None:
            print(f"{feature_set.label:24s} (not enough data)")
            continue
        app = meta.get("app_results_px", {}).get(name, {}).get("validation")
        print(f"{feature_set.label:24s} {fmt(result[0]):>18s} {fmt(result[1]):>18s} "
              f"{'-' if app is None else f'{app:.0f} px':>10s}")

    blocks = np.unique(data["block"][calib_rows])
    if len(blocks) > 1:
        print("\nPosture test: calibrate on the other postures, test on the one left out (its calibration dots)")
        print(f"{'model':24s} " + " ".join(f"{'left out: ' + str(b + 1):>14s}" for b in blocks) + f" {'average':>9s}")
        for name, feature_set in FEATURE_SETS.items():
            errors = []
            for b in blocks:
                held_out = data["block"] == b
                result = score(screen, name, split(data, calib_rows & ~held_out), split(data, calib_rows & held_out))
                errors.append(np.nan if result is None or result[1] is None else result[1] * cm)
            print(f"{feature_set.label:24s} " + " ".join(f"{e:11.1f} cm" for e in errors)
                  + f" {np.nanmean(errors):6.1f} cm")


if __name__ == "__main__":
    main()
