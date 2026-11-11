"""Compare the new vintage's 2025 parquet projection to the original CRSP vintage."""
import argparse
import hashlib
from pathlib import Path
import unseen_guard_v3 as guard
if __name__=="__main__":
    guard.seal()

import numpy as np
import pandas as pd

import final_v3 as final
from equity_research.models.preprocessing import require
from equity_research.models.robustness_ledger import atomic_json

KEY=["DlyCalDt","PERMNO"]
VALUES=["DlyRet","DlyRetx","vwretd","DlyRetMissFlg","DlyRetDurFlg","DlyDelFlg"]
FILTER=[("DlyCalDt",">=",pd.Timestamp("2025-01-01")),("DlyCalDt","<=",pd.Timestamp("2025-12-31"))]


def projected(path):
    return pd.read_parquet(path,columns=KEY+VALUES,filters=FILTER)


def overlap_digest(frame):
    ordered=frame[KEY+VALUES].sort_values(KEY)
    values=pd.util.hash_pandas_object(ordered,index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def compare(old,new):
    def validated(frame):
        require(frame.DlyCalDt.between("2025-01-01","2025-12-31").all(),"Overlap date breach")
        require(not frame.duplicated(KEY).any(),"Conflicting or duplicate vintage keys")
        return frame.set_index(KEY).sort_index()
    a,b=validated(old),validated(new)
    require(len(a)>0 and len(b)>0,"Empty overlap")
    shared=a.index.intersection(b.index)
    changes={}
    for name in VALUES:
        x,y=a.loc[shared,name],b.loc[shared,name]
        valid=x.notna()&y.notna()
        diff=valid & x.ne(y)
        record=dict(different_nonmissing=int(diff.sum()),missingness_changed=int((x.isna()!=y.isna()).sum()))
        if name in VALUES[:3]:
            record["max_abs_difference"]=float((x[valid]-y[valid]).abs().max()) if valid.any() else None
        changes[name]=record
    return dict(old_rows=len(a),new_rows=len(b),shared_keys=len(shared),old_only=len(a)-len(shared),
                new_only=len(b)-len(shared),columns=changes,
                next="Review overlap and record vintage handling before any 2026 observation is released.")


def main(new_path):
    guard.seal()
    final.locked()
    root=final.ROOT
    source=root/"data/interim/crsp_audit_v1"
    parts=final.b.base.read_json(source/"ingest.json")["parts"]
    out=root/"outputs/v3_unseen_2026_vintage"
    require(not out.exists(),"Overlap review already recorded")
    new=projected(new_path)
    result=compare(pd.concat([projected(source/p) for p in parts]),new)
    result["new_overlap_sha256"]=overlap_digest(new)
    out.mkdir()
    atomic_json(out/"overlap.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--new-parquet",type=Path)
    args=parser.parse_args()
    if args.new_parquet: main(args.new_parquet)
