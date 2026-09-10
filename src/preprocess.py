"""Preprocessing for left-vs-right fist decoding on EEGMMIDB.

Everything here is stateless — fixed constants, nothing learned from the data —
so it is safe to run before any train/test split. Anything that gets fit (CSP,
the classifier) lives in models.py and is fit inside the training fold only.
"""

import numpy as np
from mne import Epochs, events_from_annotations

from src.config import DATA_DIR
from src.data import load_run, run_family

# Exclusions from our own audit: 88/92/100 are 128 Hz with non-standard trial
# pacing; these two runs have anomalous durations.
EXCLUDED_SUBJECTS = {88, 92, 100}
EXCLUDED_RUNS = {(89, 3), (104, 8)}

# Mu and beta bands, where motor-imagery ERD lives. The band edges also remove
# drift and blinks below, 60 Hz line noise and most EMG above.
L_FREQ, H_FREQ = 8.0, 30.0

# Window relative to cue onset: skips the cue-evoked response and reaction lag.
# 2.0 s at 160 Hz = 321 samples.
TMIN, TMAX = 0.5, 2.5

CLASS_NAMES = ("left", "right")


def is_usable(subject, run):
    """Recording-level exclusions, applied identically in training and inference."""
    return subject not in EXCLUDED_SUBJECTS and (subject, run) not in EXCLUDED_RUNS


def epoch_run(subject, run, path=DATA_DIR):
    """Load one run, bandpass it, and cut it into labeled 2-second epochs."""
    # Guard against the T1/T2 trap: in fists-vs-feet runs the same strings mean
    # different conditions, so only left/right runs are allowed through.
    if run_family(run) not in ("executed_lr", "imagined_lr"):
        raise ValueError(f"run {run} is not a left/right fist run")

    # Filter the continuous recording rather than the epochs — FIR edge
    # transients would consume a large fraction of a 2 s window.
    raw = load_run(subject, run, path).filter(
        L_FREQ, H_FREQ, method="fir", phase="zero",
        fir_design="firwin", verbose="ERROR",
    )

    # The explicit relabeling step. T0 (rest) is left unmapped, so it yields no epochs.
    events, ids = events_from_annotations(raw, verbose="ERROR")
    event_id = {"left": ids["T1"], "right": ids["T2"]}

    # baseline=None: covariance-based models gain nothing from mean subtraction.
    # No artifact rejection in the baseline — deliberate, see README.
    return Epochs(
        raw, events, event_id=event_id, tmin=TMIN, tmax=TMAX,
        baseline=None, preload=True, verbose="ERROR",
    )


def build_dataset(subjects, runs, path=DATA_DIR):
    """Stack all runs into X, y, and the per-epoch metadata the eval splits need."""
    chunks, labels, subject_ids, run_ids = [], [], [], []

    for subject in subjects:
        for run in runs:
            if not is_usable(subject, run):
                continue
            epochs = epoch_run(subject, run, path)

            # Map MNE's internal event codes back to 0/1 via the names we assigned.
            code_to_label = {
                code: CLASS_NAMES.index(name)
                for name, code in epochs.event_id.items()
            }

            chunks.append(epochs.get_data(copy=False))
            labels.append(np.array([code_to_label[c] for c in epochs.events[:, 2]]))
            subject_ids.append(np.full(len(epochs), subject))
            run_ids.append(np.full(len(epochs), run))

    return (
        np.concatenate(chunks),
        np.concatenate(labels),
        np.concatenate(subject_ids),
        np.concatenate(run_ids),
    )