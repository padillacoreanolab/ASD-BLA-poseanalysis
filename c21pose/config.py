"""Paths and fixed analysis constants.

Paths come from ``config.toml`` at the repository root (copy ``config.example.toml`` and edit it), or from the file named by
the environment variable ``C21POSE_CONFIG``. Nothing else in the package hard-codes a machine path.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
META = REPO / "metadata"


def _load() -> dict:
    path = Path(os.environ.get("C21POSE_CONFIG", REPO / "config.toml"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: copy config.example.toml to config.toml and set the paths in it")
    with open(path, "rb") as f:
        return tomllib.load(f)


_C = _load()
_P = _C["paths"]
exp = lambda p: Path(os.path.expanduser(p)) if p else None

H5_ROOT = exp(_P["h5_root"])                 # SLEAP .analysis.h5 files; design_table.csv paths are relative to this
BORIS_ROOT = exp(_P["boris_root"])           # folder with the BORIS aggregated-event exports
VIDEO_ROOT = exp(_P.get("video_root", ""))   # 720p videos (only needed by 02_repair_poses.py --mode video)
OUT = exp(_P.get("out_dir", "")) or REPO / "outputs"
CACHE, TABLES, FIGS = OUT / "cache", OUT / "tables", OUT / "figures"
for d in (CACHE, TABLES, FIGS):
    d.mkdir(parents=True, exist_ok=True)
BORIS_SESSION_TABLE = exp(_P.get("boris_session_table", ""))   # optional: BORIS seconds per session (boris_long.csv)
REFERENCE_DIR = exp(_P.get("reference_dir", ""))               # optional: old project outputs_cm/, for verify_against_reference.py

BORIS_EXCLUDE = set(_C.get("boris", {}).get("exclude_sessions", []))   # sessions whose BORIS labels are blanked (see NOTES.md)

# ---- analysis constants (the values used for the Nov 2026 poster; change only with a reason)
WIN_S = 600.0            # analysis window: first 600 s of every session
CONTACT_CM = 2.0         # the 2 cm rule: a nose within 2 cm of a partner keypoint
CLOSE_CM = 10.3          # centers closer than 1.5 body lengths (6.85 cm x 1.5)
MOVE_CMS = 2.0           # moving = center speed above 2 cm/s
N_PER_SESSION = 1500     # frames sampled per session for the UMAP map
SEED = 0
FAMS = ["cagemate", "familiar", "novel"]
NODES = ["nose", "head", "rightear", "leftear", "neck", "center", "leftbody", "rightbody", "butt"]

# the 10 per-frame features of the behaviour map
MAP_FEATURES = ["dist_cc", "dist_nn", "s_nose_p_min", "p_nose_s_min", "s_nose_p_butt", "p_nose_s_butt",
                "s_nose_p_body", "p_nose_s_body", "s_speed", "p_speed"]
