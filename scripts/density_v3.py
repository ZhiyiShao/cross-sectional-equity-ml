"""Run 0b. Default: configuration only. --check-inputs: read-only, no fit ledger."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from equity_research.models.density_v3 import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute",action="store_true")
    actions.add_argument("--check-inputs",action="store_true")
    parser.add_argument("--resume",action="store_true")
    print(json.dumps(main(**vars(parser.parse_args())),indent=2))
