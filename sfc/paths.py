from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parents[1]

# Data folders
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)