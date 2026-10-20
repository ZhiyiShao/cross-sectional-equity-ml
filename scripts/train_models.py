"""Run fixed-budget development selection or the four frozen final fits."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "final-fit"])
    args = parser.parse_args()
    if args.stage == "select":
        from equity_research.models.selection import main as run
    else:
        from equity_research.models.final_fit import main as run
    run()


if __name__ == "__main__":
    main()
