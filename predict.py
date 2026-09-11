"""Predict left vs right fist for each cue in an EEGMMIDB recording.

usage:
  python predict.py path/to/S001R04.edf

Prints one prediction (left/right, with P(right)) per T1/T2 cue in the file.

flags (all optional):
  --compare       show the annotated labels next to the predictions, plus agreement
  --out FILE      also save the predictions to a CSV
  --run N         give the run number if the file name doesn't contain it (e.x N=3 if the file is for executed left/right)
                    file name containing run number looks like this: S001R04

works on:
  - Left/right fist runs: 3, 7, 11 (executed, what the model was trained on) or 4, 8, 12 (imagined).
  - Full runs or clips, at any sampling rate, as long as each cue has at least 3.5 s of recording after it.

refuses:
  - Fists/feet runs (5, 6, 9, 10, 13, 14) and baseline runs (1, 2).
  - Files without T1/T2 cues or missing any of the 64 channels.

limitations:
  - Every cue gets left or right (binary prediction). The model cannot detect "no movement" or find trials on its own.
  - Accuracy on new people averages ~0.63, and for some it can be below chance.
  - Trained on executed movement only; imagined runs are a transfer.

model: filter-bank CSP (6 bands, 8-32 Hz) + shrinkage LDA, trained on 101 subjects' executed runs.
       Subjects 3, 19, 39, 68 and 87 were held out and never seen during training.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "fbcsp_model.npz"
EXECUTED_RUNS = {3, 7, 11}
IMAGINED_RUNS = {4, 8, 12}
CLASS_NAMES = np.array(["left", "right"])


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


def subject_number(edf_path):
    """Subject number from an EEGMMIDB file name (S001R03.edf), or None if the name doesn't follow it."""
    match = re.search(r"S(\d+)R\d+", Path(edf_path).name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_model(path=MODEL_PATH):
    """Load the exported parameters: band edges, window, channel order, CSP filters, LDA weights."""
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def check_run(edf_path, run=None):
    """Refuse runs where T1/T2 don't mean left/right fist. Returns a short description of the run."""
    if run is None:
        match = re.search(r"S\d+R(\d+)", Path(edf_path).name, re.IGNORECASE)
        if match is None:
            warn(
                "no run number in the file name; assuming a left/right fist run (set it with --run)")
            return "run unknown"
        run = int(match.group(1))
    if run in EXECUTED_RUNS:
        return f"run {run}, executed"
    if run in IMAGINED_RUNS:
        return f"run {run}, imagined (model trained on executed)"
    raise ValueError(
        f"run {run} is not a left/right fist run (use 3, 4, 7, 8, 11 or 12)")


def load_raw(edf_path, model):
    """Read the EDF, fix channel names, put channels in training order, match the sampling rate."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)

    # The CSP filters are 64-channel weight vectors, so every channel must exist, in the same order.
    ch_names = [str(c) for c in model["ch_names"]]
    missing = sorted(set(ch_names) - set(raw.ch_names))
    if missing:
        raise ValueError(
            f"EDF is missing {len(missing)} of the 64 channels the model needs: {missing}")
    raw.reorder_channels(ch_names)

    # Filters and the epoch window were designed at the training rate.
    sfreq = float(model["sfreq"])
    if raw.info["sfreq"] != sfreq:
        print(
            f"note: resampling {raw.info['sfreq']:g} Hz -> {sfreq:g} Hz", file=sys.stderr)
        raw.resample(sfreq, verbose="ERROR")
    return raw


def cue_events(raw, tmax, onsets=None):
    """One event per cue (all given the same code) plus the annotated label, kept aside."""
    # Cue times given by the user (seconds from file start) override the annotations.
    if onsets is not None:
        samples = raw.first_samp + \
            np.round(np.asarray(onsets) * raw.info["sfreq"]).astype(int)
        return np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)]), np.full(len(samples), "")

    ann = raw.annotations
    is_cue = np.isin(ann.description, ["T1", "T2"])

    # No cues means no trials: refuse rather than invent one.
    if not is_cue.any():
        raise ValueError("no T1/T2 cues in this file, so there is nothing to classify "
                         "(for a clip whose annotations were stripped, give the cue time with --cues)")

    # A cue shorter than the window (e.g. a trial cut by cropping) puts rest inside the window.
    short = is_cue & (ann.duration < tmax)
    if short.any():
        warn(f"{short.sum()} cue(s) last under {tmax:g} s, so their window includes rest "
             f"(onset {', '.join(f'{t:.2f}' for t in ann.onset[short])} s)")

    events, _ = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
    annotated = np.where(events[:, 2] == 1, "left", "right")
    events[:, 2] = 1   # the model sees "a cue happened here", never which one
    return events, annotated


def band_covariances(raw, events, model):
    """Per band: filter the continuous signal, cut 0.5-3.5 s epochs, one covariance per trial."""
    tmin, tmax = float(model["tmin"]), float(model["tmax"])
    covs, kept = [], None
    for lo, hi in model["bands"]:
        # Same filter call as training, applied to the continuous recording (edge transients fall outside epochs).
        filtered = raw.copy().filter(lo, hi, method="fir", phase="zero",
                                     fir_design="firwin", verbose="ERROR")
        epochs = mne.Epochs(filtered, events, event_id={"cue": 1}, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True, verbose="ERROR")
        X = epochs.get_data()
        covs.append(X @ X.transpose(0, 2, 1) / X.shape[-1])

        # Every band uses the same window, so the same cues survive; the features would misalign otherwise.
        assert kept is None or np.array_equal(kept, epochs.selection)
        kept = epochs.selection   # indices of cues whose window fit inside the recording
    return np.stack(covs, axis=1), kept


def predict_edf(edf_path, model=None, onsets=None, run=None):
    """Predicted label and P(right) for every usable cue in one EDF.

    onsets: optional cue times in seconds from the start of the file; overrides the annotations.
    run: optional run number, for files whose name doesn't contain one.
    Returns a dict: run (description), onset (s), label ("left"/"right"), p_right, annotated ("" if none).
    """
    model = load_model() if model is None else model
    if not Path(edf_path).is_file():
        raise FileNotFoundError(f"no such file: {edf_path}")
    run_info = check_run(edf_path, run)

    # Any valid recording is accepted; this only notes when the result isn't a test on an unseen person.
    subject = subject_number(edf_path)
    if subject is not None and "holdout" in model and subject not in model["holdout"]:
        warn(f"subject {subject} was in the training set, so this is not a test on an unseen person "
             f"(unseen subjects: {', '.join(map(str, model['holdout']))})")

    raw = load_raw(edf_path, model)
    tmax = float(model["tmax"])
    events, annotated = cue_events(raw, tmax, onsets)

    # Cues without 3.5 s of recording after them (or before the file start) have no full window.
    C, kept = band_covariances(raw, events, model)
    if len(kept) == 0:
        raise ValueError(
            f"no cue has {tmax:g} s of recording after it (file is {raw.times[-1]:.1f} s long)")
    if len(kept) < len(events):
        warn(
            f"skipped {len(events) - len(kept)} cue(s) without a full {tmax:g} s of recording after them")

    # CSP features: log variance of each spatial filter's output, log(w' C w), per band.
    # (n_bands, n_channels, n_filters)
    W = model["filters"]
    features = np.log(np.einsum("nbij,bik,bjk->nbk", C, W, W)
                      ).reshape(len(C), -1)

    # Shrinkage LDA: linear score, positive means right. P(right) is its logistic, as in sklearn.
    score = features @ model["coef"] + model["intercept"]

    return {
        "run": run_info,
        "onset": (events[kept, 0] - raw.first_samp) / raw.info["sfreq"],
        "label": CLASS_NAMES[(score > 0).astype(int)],
        "p_right": 1 / (1 + np.exp(-score)),
        "annotated": annotated[kept],
    }


def print_result(path, out, compare):
    """One header line per file, then one row per cue."""
    print(f"{Path(path).name}  ({out['run']}, {len(out['label'])} cues)")
    print(f"{'onset_s':>8}  {'pred':<5}  {'p_right':>7}" +
          ("  annotated" if compare else ""))
    for t, lab, p, ann in zip(out["onset"], out["label"], out["p_right"], out["annotated"]):
        print(f"{t:8.2f}  {lab:<5}  {p:7.3f}" +
              (f"  {ann}" if compare else ""))
    if compare:
        if all(out["annotated"]):
            print(f"agreement: {np.mean(out['label'] == out['annotated']):.3f} "
                  f"({len(out['label'])} cues; with this few trials, expect wide swings from chance)")
        else:
            print("agreement: n/a (cues given with --cues have no annotated label)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("edf", nargs="+", help="one or more EDF files")
    parser.add_argument("--compare", action="store_true",
                        help="also show the annotated labels (T1 = left, T2 = right) and the agreement; for checking only")
    parser.add_argument("--run", type=int, metavar="N",
                        help="run number, when the file name doesn't contain it (one file only)")
    parser.add_argument("--cues", type=float, nargs="+", metavar="SEC",
                        help="cue time(s) in seconds from file start; replaces the annotations (one file only)")
    parser.add_argument("--out", metavar="CSV",
                        help="also write every prediction to this CSV file")
    parser.add_argument("--model", default=MODEL_PATH, metavar="NPZ",
                        help="exported model (default: %(default)s)")
    args = parser.parse_args()

    if len(args.edf) > 1 and (args.run is not None or args.cues is not None):
        parser.error(
            "--run and --cues describe a single file; pass one EDF when using them")

    model = load_model(args.model)
    rows, failed = [], 0
    for path in args.edf:
        # A bad file is reported and skipped, so one failure doesn't stop a batch.
        try:
            out = predict_edf(path, model, args.cues, args.run)
        except Exception as e:
            print(f"error: {Path(path).name}: {e}\n", file=sys.stderr)
            failed += 1
            continue
        print_result(path, out, args.compare)
        rows += [[Path(path).name, f"{t:.2f}", lab, f"{p:.4f}", ann]
                 for t, lab, p, ann in zip(out["onset"], out["label"], out["p_right"], out["annotated"])]

    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows(
                [["file", "onset_s", "pred", "p_right", "annotated"]] + rows)
        print(f"saved {len(rows)} predictions to {args.out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
