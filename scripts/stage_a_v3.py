"""Stage A directional block. Default is a public-configuration-only check."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from equity_research.models.stage_a_v3 import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(main(**vars(parser.parse_args())), indent=2))
