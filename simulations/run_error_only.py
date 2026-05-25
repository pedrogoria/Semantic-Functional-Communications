# simulations/run_error_only.py

"""
Thin wrapper for the error-only pipeline.
"""

import argparse
from sfc.pipelines.error_only_pipeline import run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/error_only_default.yaml")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main(config_path=None):
    args = parse_args() if config_path is None else None
    cfg_path = config_path or args.config
    save = False if (args and args.no_save) else True
    run(cfg_path, save=save)


if __name__ == "__main__":
    main()