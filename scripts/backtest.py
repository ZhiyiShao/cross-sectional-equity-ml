"""Run existing predictions through the validation or frozen-test ledger."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("period", choices=["validation", "test", "risk-comparison", "historical-v2", "robustness-validation"])
    args = parser.parse_args()
    if args.period == "robustness-validation":
        from equity_research.backtest.robustness_validation import main as run
    elif args.period == "historical-v2":
        from equity_research.backtest.replay_v2 import main as run
    elif args.period == "risk-comparison":
        from equity_research.backtest.risk_comparison import main as run
    elif args.period == "validation":
        from equity_research.backtest.validation import main as run
    else:
        from equity_research.backtest.frozen import main as run
    run()


if __name__ == "__main__":
    main()
