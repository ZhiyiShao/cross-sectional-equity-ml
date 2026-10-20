"""Prepare licensed local data, one explicit stage at a time."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[
        "ingest", "universe", "audit", "returns", "features",
        "validate", "validation-events", "test-audit", "test-events",
    ])
    parser.add_argument("--chunksize", type=int, default=300_000)
    args = parser.parse_args()
    from equity_research.paths import ROOT
    source = ROOT / "data/interim/crsp_audit_v1"
    universe = ROOT / "data/processed/crsp_v1"
    if args.stage == "ingest":
        from equity_research.data.ingest import ingest
        ingest(source, args.chunksize)
    elif args.stage == "universe":
        from equity_research.data.universe import build
        build(source, universe)
    elif args.stage == "audit":
        from equity_research.data.audit import main as run
        run(source, universe)
    else:
        from importlib import import_module
        modules = {
            "returns": "returns", "features": "features", "validate": "validation",
            "validation-events": "validation_events", "test-audit": "event_audit",
            "test-events": "test_events",
        }
        import_module("equity_research.data." + modules[args.stage]).main()


if __name__ == "__main__":
    main()
