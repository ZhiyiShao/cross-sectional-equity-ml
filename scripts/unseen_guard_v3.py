"""Local release seal and a per-stage, append-only Git history outside the outputs tree."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT=Path(__file__).resolve().parents[1]
SEAL="config/unseen_entry_v3.json"
TAG="refs/tags/v3-unseen-entry-v1"
PREFIX="refs/heads/audit/v3-unseen-2026-"
STAGES=("prediction","portfolio")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def git(root,*args,data=None):
    cmd=["git","-C",str(root),"-c","user.name=V3 evaluation recorder","-c","user.email=v3@localhost",*args]
    run=subprocess.run(cmd,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if run.returncode:
        raise ValueError("Git audit operation failed: "+run.stderr.decode(errors="replace").strip())
    return run.stdout


def seal(root=ROOT,*,released=True):
    path=root/SEAL
    if released:
        blob=git(root,"show",TAG+":"+SEAL)
        if blob!=path.read_bytes():
            raise ValueError("Entry seal differs from release tag")
    spec=json.loads(path.read_bytes())
    for name,expected in spec["sources"].items():
        if digest(root/name)!=expected: raise ValueError("Unsealed input: "+name)
    return digest(path)


@contextmanager
def exclusive(root,stage):
    common=git(root,"rev-parse","--path-format=absolute","--git-common-dir").decode().strip()
    with (Path(common)/("v3-unseen-"+stage+".lock")).open("a+b") as f:
        f.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield  # OS unlocks on close, including abnormal process exit.


class Journal:
    def __init__(self,root,stage):
        if stage not in STAGES: raise ValueError("Unknown prospective stage")
        self.root=root
        self.ref=PREFIX+stage
        self.head=git(root,"rev-parse","--verify",self.ref+"^{commit}").decode().strip()
        release=git(root,"rev-parse",TAG+"^{commit}").decode().strip()
        git(root,"merge-base","--is-ancestor",release,self.head)
        self.tree=git(root,"rev-parse",release+"^{tree}").decode().strip()
        self.record={} if self.head==release else json.loads(git(root,"show","-s","--format=%B",self.head))
        self.stage=stage
        self.witness="refs/tags/v3-unseen-2026-reserved-"+stage
        if self.record:
            git(root,"merge-base","--is-ancestor",self.witness,self.head)

    def event(self,status,**fields):
        record=dict(self.record,stage=self.stage,status=status,time=datetime.now(timezone.utc).isoformat(),**fields)
        commit=git(self.root,"commit-tree",self.tree,"-p",self.head,"-F","-",
                   data=json.dumps(record,sort_keys=True).encode()).decode().strip()
        if status=="reserved":
            # Both references change atomically. Recreating a removed branch at the
            # release commit cannot reuse an already issued reservation.
            commands=f"create {self.witness} {commit}\nupdate {self.ref} {commit} {self.head}\n"
            git(self.root,"update-ref","--create-reflog","--stdin",data=commands.encode())
        else:
            git(self.root,"update-ref","--create-reflog",self.ref,commit,self.head)
        self.head,self.record=commit,record
        return record


@contextmanager
def attempt(root,stage,manifest,resume=False):
    pin=seal(root)
    with exclusive(root,stage):
        journal=Journal(root,stage)
        old=journal.record
        identity=dict(manifest_sha256=digest(manifest),seal_sha256=pin)
        if old:
            if not resume or old.get("status")=="completed" or old.get("stage")!=stage:
                raise ValueError("Stage already reserved or completed")
            if any(old.get(k)!=v for k,v in identity.items()): raise ValueError("Resume identity mismatch")
        elif resume:
            raise ValueError("No reservation to resume")
        journal.event("resumed" if old else "reserved",attempt=old.get("attempt",0)+1,**identity)
        try:
            yield journal
        except BaseException as exc:
            frames=traceback.extract_tb(exc.__traceback__)
            location=frames[-1]
            # Keep paths/data outside Git; retain the exact local diagnostic by hash.
            folder=root/"outputs"/"v3_unseen_failures"/stage
            log=folder/(journal.head+".txt")
            try:
                folder.mkdir(parents=True,exist_ok=True)
                with log.open("x",encoding="utf8") as f:
                    traceback.print_exception(exc,file=f)
                diagnostic=digest(log)
            except OSError:
                diagnostic=None
            journal.event("failed",failed_at=journal.record["status"],error_type=type(exc).__name__,
                          location=f"{Path(location.filename).name}:{location.lineno}",diagnostic_sha256=diagnostic)
            raise


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Verify prospective code seals without reading observations or reserving a run.")
    parser.add_argument("--released",action="store_true",help="Also require the seal at the release tag and both audit refs")
    args=parser.parse_args()
    print(seal(released=args.released))
    if args.released:
        for stage in STAGES:
            print(stage,Journal(ROOT,stage).record.get("status","available"))
