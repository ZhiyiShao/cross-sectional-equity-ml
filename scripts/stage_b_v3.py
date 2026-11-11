"""Stage B defaults to a public-configuration-only check."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from equity_research.models import stage_b_v3 as runner

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-labels",action="store_true")
    action.add_argument("--check-inputs",action="store_true")
    action.add_argument("--execute",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.prepare_labels:
        if args.resume:
            parser.error("--resume requires --execute")
        from build_daily_v3 import FIELDS
        cfg,_,_=runner.plan(runner.ROOT)
        result=runner.targets_v3.build(runner.ROOT,cfg,FIELDS)
    else:
        result=runner.main(execute=args.execute,resume=args.resume,check_inputs=args.check_inputs)
    print(json.dumps(result,indent=2))
