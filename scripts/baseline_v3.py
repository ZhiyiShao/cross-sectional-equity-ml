"""V3 stage 0a baseline rebuild. Without --execute this is a configuration-only preflight.

Kept separate from train_models.py because that file's hash is part of the
completed V2 robustness run's input manifest and must not change.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly start the twelve frozen baseline fits")
    parser.add_argument("--resume", action="store_true", help="Resume verified completed tasks; never retry failed fits")
    args = parser.parse_args()
    from equity_research.models.baseline_v3 import main as run
    run(execute_requested=args.execute, resume=args.resume)


if __name__ == "__main__":
    main()
