"""Reading SLEAP analysis files and the low-level clean-up used before calibration.

Taken from dyadlib.py (BLA C21 project, Sep 2026) without changing any numbers:
  read_sleap    tracks -> (track, frame, node, xy); node names read from the file and matched by NAME
  swap_audit    frames where swapping the two identities explains the next frame better (> 0.5 body lengths cheaper)
  blank_around  NaN both tracks for +-5 frames around each flagged frame (brief tracker confusions at contact)
  fill_gaps     linear interpolation of missing runs <= 15 frames, per keypoint coordinate
"""
from __future__ import annotations

import h5py
import numpy as np

MAX_GAP, SMOOTH = 15, 5
ALIAS = {"tailbase": "butt", "tail_base": "butt", "trunk": "center", "centroid": "center",
         "left_ear": "leftear", "right_ear": "rightear"}


def read_sleap(path):
    """dict(xy=(track, frame, node, 2) float, node_names, track_names, fps or None)"""
    with h5py.File(path, "r") as f:
        tracks = f["tracks"][:]
        nn = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        tn = [n.decode() if isinstance(n, bytes) else str(n) for n in f["track_names"][:]] \
            if "track_names" in f else [f"track_{i}" for i in range(tracks.shape[0])]
        fps = None
        for k in ("fps", "video_fps"):
            if k in f.attrs:
                fps = float(f.attrs[k])
            elif k in f:
                fps = float(np.array(f[k]).ravel()[0])
    if tracks.ndim != 4 or tracks.shape[1] != 2:
        raise ValueError(f"{path}: unexpected tracks shape {tracks.shape}")
    xy = np.transpose(tracks, (0, 3, 2, 1)).astype(float)
    return dict(xy=xy, node_names=[n.strip().lower() for n in nn], track_names=tn, fps=fps)


def map_nodes(node_names, required=None):
    """node name -> index in the file; raises if a required node is missing"""
    from .config import NODES
    required = NODES if required is None else required
    idx = {ALIAS.get(n, n): i for i, n in enumerate(node_names)}
    missing = [n for n in required if n not in idx]
    if missing:
        raise KeyError(f"skeleton missing nodes {missing}; file has {node_names}")
    return {k: idx[k] for k in required}


def fill_gaps(xy, max_gap=MAX_GAP):
    """interpolate interior NaN runs of <= max_gap frames (axis 0 = frames)"""
    out = np.array(xy, float, copy=True)
    n = out.shape[0]
    t = np.arange(n)
    flat = out.reshape(n, -1)
    for j in range(flat.shape[1]):
        v = flat[:, j]
        bad = ~np.isfinite(v)
        if not bad.any() or bad.all():
            continue
        s = np.where(bad & ~np.r_[False, bad[:-1]])[0]
        e = np.where(bad & ~np.r_[bad[1:], False])[0]
        keep = np.zeros(n, bool)
        for a, b in zip(s, e):
            if b - a + 1 <= max_gap and a > 0 and b < n - 1:
                keep[a:b + 1] = True
        v[keep] = np.interp(t[keep], t[~bad], v[~bad])
    return flat.reshape(out.shape)


def swap_audit(xy, ni, thresh=0.5):
    """frame indices where the swapped identity assignment is cheaper by > thresh (xy in body-length units)"""
    keys = [ni["nose"], ni["center"]]
    A, B = xy[0][:, keys], xy[1][:, keys]
    same = np.linalg.norm(A[1:] - A[:-1], axis=-1).sum(-1) + np.linalg.norm(B[1:] - B[:-1], axis=-1).sum(-1)
    swap = np.linalg.norm(A[1:] - B[:-1], axis=-1).sum(-1) + np.linalg.norm(B[1:] - A[:-1], axis=-1).sum(-1)
    flag = np.isfinite(same) & np.isfinite(swap) & (same - swap > thresh)
    return np.where(flag)[0] + 1


def blank_around(xy, frames, radius=5):
    """NaN out both tracks for +-radius frames around each frame index"""
    out = np.array(xy, dtype=float, copy=True)
    n = out.shape[1]
    for f in frames:
        a, b = max(int(f) - radius, 0), min(int(f) + radius, n)
        out[:, a:b] = np.nan
    return out


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / n


def boxcar(a, k=SMOOTH):
    """centered k-frame mean along axis 0; NaN unless the full window is finite"""
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)
    ker = np.ones(k)
    f = lambda v: np.convolve(v, ker, "same")
    num = np.apply_along_axis(f, 0, x)
    den = np.apply_along_axis(f, 0, ok.astype(float))
    out = np.where(den >= k, num / np.maximum(den, 1e-12), np.nan)
    return np.where(ok, out, np.nan)
