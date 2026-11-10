"""Run fixed-budget development selection or the four frozen final fits."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "final-fit", "context-v2", "ranking-v2", "final-fit-v2", "robustness-v2"])
    parser.add_argument("--execute", action="store_true", help="Explicitly start the frozen robustness candidate fits")
    parser.add_argument("--resume", action="store_true", help="Resume verified completed robustness tasks; never retry failed fits")
    args = parser.parse_args()
    if args.stage == "robustness-v2":
        from equity_research.models.parameter_robustness import main as run
        run(execute_requested=args.execute, resume=args.resume)
        return
    if args.execute or args.resume:
        parser.error("--execute/--resume only apply to robustness-v2")
    if args.stage == "final-fit-v2":
        from equity_research.models.final_fit_v2 import main as run
    elif args.stage == "ranking-v2":
        from equity_research.models.ranking_selection import main as run
    elif args.stage == "context-v2":
        from equity_research.models.context_selection import main as run
    elif args.stage == "select":
        from equity_research.models.selection import main as run
    else:
        from equity_research.models.final_fit import main as run
    run()


if __name__ == "__main__":
    main()
