"""Where all data lives, resolved like scripts/paths.sh:
$DATA_ROOT from the environment, then DATA_ROOT=... in configs/paths.local.env, then <repo>/data."""

import glob
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _data_root() -> str:
    if os.environ.get("DATA_ROOT"):
        return os.environ["DATA_ROOT"]
    local = REPO / "configs" / "paths.local.env"
    if local.is_file():
        for line in local.read_text().splitlines():
            key, _, val = line.strip().partition("=")
            if key in ("DATA_ROOT", "export DATA_ROOT") and val:
                return os.path.expandvars(os.path.expanduser(val.strip("\"'")))
    return str(REPO / "data")


DATA = _data_root()


def hf_snapshot(kind: str, repo_id: str) -> str | None:
    """Latest snapshot dir of a hub repo in $HF_HOME (default DATA/hf_cache); kind is 'models' or 'datasets'."""
    hf_home = os.environ.get("HF_HOME", f"{DATA}/hf_cache")
    dirs = sorted(glob.glob(f"{hf_home}/hub/{kind}--{repo_id.replace('/', '--')}/snapshots/*"))
    return dirs[-1] if dirs else None
