"""Evaluate a completed frozen test or validation holdings exposures."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["test", "validation-exposures"])
    args = parser.parse_args()
    if args.stage == "test":
        from equity_research.evaluation.frozen import main as run
    else:
        from equity_research.evaluation.exposures import main as run
    run()


if __name__ == "__main__":
    main()
