"""
run.py

Self-contained entrypoint that works in ANY environment:
- PyCharm Run button ✅
- PyCharm Console ✅
- Terminal ✅

No IDE configuration required.

Author: SFC Project
"""

import os
import sys

# ----------------------------------------------------------
# ✅ AUTOCONFIG: ensure project root is always in PYTHONPATH
# ----------------------------------------------------------

# Get absolute path of this file (run.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path if not already present
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ----------------------------------------------------------
# Now imports will ALWAYS work
# ----------------------------------------------------------

import yaml
import numpy as np

from experiments import fig1


def load_config(path):
    """
    Load YAML configuration.
    Path is relative to project root.
    """
    full_path = os.path.join(CURRENT_DIR, path)

    with open(full_path, "r") as f:
        return yaml.safe_load(f)


def dispatch_experiment(cfg):
    """
    Select which experiment to run.
    """
    if cfg["experiment"] == "fig1":
        return fig1.run(cfg)

    raise ValueError(f"Unknown experiment: {cfg['experiment']}")


def save_results(cfg, data, header):
    """
    Save output to data/results/
    """
    out_dir = os.path.join(CURRENT_DIR, cfg["output"]["results_dir"])
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, cfg["output"]["filename"])

    np.savetxt(
        out_path,
        data,
        delimiter="\t",
        header=header,
        comments='',
        fmt="%.10e"
    )

    print(f"[INFO] Results saved to: {out_path}")


def main(config_path="configs/fig1.yaml"):
    """
    Main execution.
    Safe for interactive use.
    """
    cfg = load_config(config_path)

    data, header = dispatch_experiment(cfg)

    save_results(cfg, data, header)

    return data, header


if __name__ == "__main__":
    main()
