from src.config import DATA_DIR

import argparse
import logging

from mne.datasets import eegbci

EEGBCI_PHYSIONET_URL = "https://physionet.org/files/eegmmidb/1.0.0/"

BASELINE_RUNS = [1, 2]
EXECUTED_RUNS = [3, 7, 11]
IMAGINED_RUNS = [4, 8, 12]
RUNS = BASELINE_RUNS + EXECUTED_RUNS + IMAGINED_RUNS


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def download(subjects, runs=RUNS, path=DATA_DIR):
    failed = []
    for subject in subjects:
        try:
            eegbci.load_data(subject, runs, path=path,
                             base_url=EEGBCI_PHYSIONET_URL, update_path=False, verbose="ERROR")
            log.info("subject %03d ok", subject)
        except Exception as exc:
            failed.append((subject, str(exc)))
            log.warning("subject %03d failed: %s", subject, exc)
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=109)
    args = parser.parse_args()

    failed = download(range(args.start, args.end + 1))
    if failed:
        log.warning("%d subjects failed: %s", len(
            failed), [s for s, _ in failed])
    else:
        log.info("all subjects downloaded")
