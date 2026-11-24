"""Session records: raw SLEAP file -> calibrated, gap-filled skeletons of the subject (S) and partner (P), plus BORIS frame labels.

One record per session (dict):
  basename, subject, drug, fam, day, fps   design-table fields; fps = the video's true frame rate
  S, P        (n, 9, 2) float64, floor cm; keypoint order given by ni (the file's node order)
  ni          {node name: index}
  n           frames kept = round(WIN * fps), WIN = shortest session (600.58 s), so every session covers the same time
  both        both mice have a finite center keypoint (after gap filling)
  scored      the session has BORIS bouts for the treated mouse
  lab         BORIS frame labels {ag, face, body, selfgroom, allogroom}

Steps per session (as the Sep 2026 pipeline): read h5 -> swap audit on body-length-scaled coordinates -> blank +-5 frames around
each flagged frame (both mice) -> pixels to floor cm -> trim to n frames -> fill gaps <= 15 frames per keypoint.
The subject track (0 or 1) comes from the design table (identity rules: NOTES.md).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from . import boris, calibration as cal
from .config import BORIS_EXCLUDE, CACHE, H5_ROOT, META, WIN_S
from .sleapio import blank_around, fill_gaps, map_nodes, read_sleap, swap_audit


def design():
    """the 72-session design table (metadata/design_table.csv), with absolute h5 paths"""
    D = pd.read_csv(META / "design_table.csv")
    D["path"] = [str(H5_ROOT / p) for p in D.h5_relpath]
    return D


def common_window(D):
    """seconds covered by every session (the shortest one): 600.58 s for the 72 E-cohort sessions"""
    return float((D.n_frames / D.basename.map(cal.fps_for)).min())


def build_one(r, win, bouts):
    rec = read_sleap(r.path)
    ni = map_nodes(rec["node_names"])
    xy = rec["xy"]
    bl = np.nanmedian(np.linalg.norm(xy[:, :, ni["nose"]] - xy[:, :, ni["butt"]], axis=-1))
    xy = blank_around(xy, swap_audit(xy / bl, ni))
    xy = cal.to_cm(xy, r.basename)
    fps = cal.fps_for(r.basename)
    n = min(xy.shape[1], int(round(win * fps)))
    st = int(r.subject_track)
    S, P = fill_gaps(xy[st][:n]), fill_gaps(xy[1 - st][:n])          # trim first: a gap touching the window end stays missing
    c = ni["center"]
    both = np.isfinite(S[:, c]).all(1) & np.isfinite(P[:, c]).all(1)
    scored = len(bouts) > 0 and r.basename not in BORIS_EXCLUDE
    lab = boris.frame_labels(bouts if scored else bouts.iloc[:0], n, fps)
    return dict(basename=r.basename, subject=r.subject, drug=r.drug, fam=r.familiarity, day=int(r.day), fps=fps, n=n,
                S=S, P=P, ni=ni, both=both, scored=bool(scored), lab=lab)


def build_all():
    D = design()
    win = common_window(D)
    bouts, cov = boris.attach(D)
    return [build_one(r, win, bouts[r.basename]) for r in D.itertuples()], cov, win


def load(name="sessions_guarded"):
    """list of session records from outputs/cache/<name>.pkl (sessions_raw = before skeleton repair)"""
    return pickle.load(open(CACHE / f"{name}.pkl", "rb"))
