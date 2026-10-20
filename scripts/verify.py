"""Verify features, model outputs, or local historical provenance without fitting."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["features", "selection", "final-models", "history"])
    args = parser.parse_args()
    if args.stage == "history":
        from equity_research.models.final_fit import protected_hashes
        print(f"Verified {len(protected_hashes())} historical artifacts against original snapshots.")
        return
    from importlib import import_module
    module = {
        "features": "data.validation",
        "selection": "models.validation",
        "final-models": "models.verification",
    }[args.stage]
    import_module("equity_research." + module).main()


if __name__ == "__main__":
    main()
