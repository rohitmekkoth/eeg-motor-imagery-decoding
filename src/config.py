from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mne_data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

for path in (DATA_DIR, CACHE_DIR, RESULTS_DIR, ARTIFACTS_DIR):
    path.mkdir(exist_ok=True)
