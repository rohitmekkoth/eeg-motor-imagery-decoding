import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "mne_data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

for d in (DATA_DIR, ARTIFACTS_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)
