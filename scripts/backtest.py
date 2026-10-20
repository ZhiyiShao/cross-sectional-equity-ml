"""Run existing predictions through the validation or frozen-test ledger."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("period", choices=["validation", "test"])
    args = parser.parse_args()
    if args.period == "validation":
        from equity_research.backtest.validation import main as run
    else:
        from equity_research.backtest.frozen import main as run
    run()


if __name__ == "__main__":
    main()
