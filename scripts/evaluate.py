"""Evaluate a completed frozen test or validation holdings exposures."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["test", "validation-exposures", "test-exposures", "v2-preflight", "v2"])
    parser.add_argument("--release", help="Separately authorized audited V2 replay release JSON")
    args = parser.parse_args()
    if args.stage in ["v2-preflight", "v2"]:
        import json
        from equity_research.evaluation.v2 import preflight, run_evaluation
        if args.stage == "v2" and not args.release:
            parser.error("V2 evaluation sealed: --release with an audited authorization is required")
        print(json.dumps(preflight() if args.stage == "v2-preflight" else run_evaluation(args.release), indent=2))
        return
    if args.stage == "test":
        from equity_research.evaluation.frozen import main as run
    elif args.stage == "test-exposures":
        from equity_research.evaluation.test_exposures import main as run
    else:
        from equity_research.evaluation.exposures import main as run
    run()


if __name__ == "__main__":
    main()
