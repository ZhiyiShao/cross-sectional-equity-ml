"""Verify old source against an explicit local snapshot, never substitute data."""
import hashlib
from pathlib import Path

from equity_research.paths import ROOT

SNAPSHOT = ROOT / "archive/github_cleanup_20261019/snapshot"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_matches(path, expected, *, root=ROOT, snapshot=SNAPSHOT):
    """Check live bytes first; permit historical source bytes for legacy manifests.

    Only old scripts, tests and root Markdown documents may use the snapshot.
    Configurations, data, fitted models and results must always match live bytes.
    New src/ paths never fall back to an older source version.
    """
    root = Path(root).resolve()
    path = Path(path).resolve()
    relative = path.relative_to(root)
    if path.is_file() and sha256(path) == expected:
        return True
    legacy_source = relative.parts[0] in {"scripts", "tests"}
    legacy_source |= len(relative.parts) == 1 and relative.suffix == ".md"
    if not legacy_source:
        return False
    old = Path(snapshot).resolve() / relative
    return old.is_file() and sha256(old) == expected
