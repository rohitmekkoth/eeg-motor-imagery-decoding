from src.config import DATA_DIR

from mne.datasets import eegbci
from mne.io import read_raw_edf

BASELINE_RUNS = [1, 2]
EXECUTED_RUNS = [3, 7, 11]
IMAGINED_RUNS = [4, 8, 12]
FISTS_FEET_RUNS = [5, 9, 13, 6, 10, 14]


def run_family(run):
    if run in BASELINE_RUNS:
        return "baseline"
    if run in EXECUTED_RUNS:
        return "executed_lr"
    if run in IMAGINED_RUNS:
        return "imagined_lr"
    if run in FISTS_FEET_RUNS:
        return "fists_feet"
    return "unknown"


def load_run(subject, run, path=DATA_DIR):
    fname = eegbci.load_data(subject, [run], path=path, update_path=False, verbose="ERROR")[0]
    raw = read_raw_edf(fname, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="warn", verbose="ERROR")
    return raw


def load_runs(subject, runs, path=DATA_DIR):
    return [(run, load_run(subject, run, path)) for run in runs]


def audit_run(subject, run, path=DATA_DIR):
    record = {"subject": subject, "run": run, "family": run_family(run)}
    try:
        raw = load_run(subject, run, path)
    except Exception as exc:
        record["error"] = str(exc)
        return record

    counts = {}
    for description in raw.annotations.description:
        counts[description] = counts.get(description, 0) + 1

    record.update(
        {
            "sfreq": raw.info["sfreq"],
            "n_channels": len(raw.ch_names),
            "duration_s": round(raw.n_times / raw.info["sfreq"], 2),
            "n_annotations": len(raw.annotations),
            "n_T0": counts.get("T0", 0),
            "n_T1": counts.get("T1", 0),
            "n_T2": counts.get("T2", 0),
            "other_labels": sorted(set(counts) - {"T0", "T1", "T2"}),
            "ch_names_hash": hash(tuple(raw.ch_names)),
            "error": None,
        }
    )
    return record


def audit_subjects(subjects, runs, path=DATA_DIR):
    return [audit_run(s, r, path) for s in subjects for r in runs]
