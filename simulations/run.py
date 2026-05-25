# simulations/run.py

import argparse
from sfc.experiments import run_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["main", "error_only", "rbcp"])
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args.exp, args.config)


if __name__ == "__main__":
    main()