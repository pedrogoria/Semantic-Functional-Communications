from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
