# simulations/run_rbcp.py

"""
Thin wrapper for the RbCP pipeline.
"""

import argparse
from sfc.pipelines.rbcp_pipeline import run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/rbcp.yaml")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main(config_path=None):
    args = parse_args() if config_path is None else None
    cfg_path = config_path or args.config
    save = False if (args and args.no_save) else True
    run(cfg_path, save=save)


if __name__ == "__main__":
    main()
