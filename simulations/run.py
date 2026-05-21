# simulations/run.py

import argparse
from sfc.experiments import run_experiment


def parse_args():
    """
    Parse command-line arguments.

    Allows selecting:
    - experiment type
    - configuration file
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp",
        required=True,
        choices=["main", "error_only", "rbcp"],
        help="Experiment to run"
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file (optional)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Run selected experiment with optional config
    run_experiment(args.exp, args.config)


if __name__ == "__main__":
    main()