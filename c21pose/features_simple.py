"""Simple per-frame and per-session SLEAP features, 2 cm contact rule (from simple_features.py, 21-23 Oct 2026).

Per frame (calibrated cm, first WIN_S seconds; s_ = subject given C21 or saline, p_ = partner):
  dist_cc        center to center distance
  dist_nn        nose to nose distance
  s_nose_p_min   subject nose to the NEAREST of the partner's 9 keypoints (p_nose_s_min: the reverse)
  s_nose_p_butt  subject nose to the partner's tail base (anogenital distance; p_nose_s_butt: the reverse)
  s_speed        subject center speed, cm/s over a 0.2 s step (p_speed: partner)
  s_facing       angle between the subject's head direction (neck -> nose) and neck -> partner center; 0 = pointing at it
  p_facing       the same for the partner;  rel_heading = angle between the two body axes (tailbase -> neck)
  ok             both mice tracked and the 8 distance / speed values finite

Per session (16): mean_dist, frac_close (centers < 10.3 cm), close_bouts_per_min and close_bout_s (close runs >= 1 s),
frac_contact_subj / _part (nose within 2 cm of ANY keypoint of the other mouse), frac_ag_subj / _part (nose within 2 cm of the other's
tail base), subj_speed, part_speed, frac_moving_subj / _part (> 2 cm/s), subj_speed_close / _far, frac_oriented_ag_subj / _part
(facing < 60 deg and nose < 3 cm from the other's tail base).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLOSE_CM, CONTACT_CM, MOVE_CMS, WIN_S

FRAME_COLS = ["dist_cc", "dist_nn", "s_nose_p_min", "p_nose_s_min", "s_nose_p_butt", "p_nose_s_butt", "s_speed", "p_speed"]
ORIENT_COLS = ["s_facing", "p_facing", "rel_heading"]


def angle(u, v):
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9), -1, 1)))


def frame_table(x):
    S, P, ni, fps = x["S"], x["P"], x["ni"], x["fps"]
    nT = min(len(S), int(WIN_S * fps)); S, P = S[:nT], P[:nT]
    d = lambda a, b: np.linalg.norm(a - b, axis=-1)
    step = max(int(round(fps * 0.2)), 1)

    def speed(A):
        c = A[:, ni["center"]]; v = np.full(len(c), np.nan); v[step:] = d(c[step:], c[:-step]) / (step / fps); return v

    F = pd.DataFrame({
        "dist_cc": d(S[:, ni["center"]], P[:, ni["center"]]),
        "dist_nn": d(S[:, ni["nose"]], P[:, ni["nose"]]),
        "s_nose_p_min": np.min(np.linalg.norm(S[:, ni["nose"]][:, None] - P, axis=-1), axis=1),
        "p_nose_s_min": np.min(np.linalg.norm(P[:, ni["nose"]][:, None] - S, axis=-1), axis=1),
        "s_nose_p_butt": d(S[:, ni["nose"]], P[:, ni["butt"]]),
        "p_nose_s_butt": d(P[:, ni["nose"]], S[:, ni["butt"]]),
        "s_speed": speed(S), "p_speed": speed(P)})
    g = lambda A, k: A[:, ni[k]]
    F["s_facing"] = angle(g(S, "nose") - g(S, "neck"), g(P, "center") - g(S, "neck"))
    F["p_facing"] = angle(g(P, "nose") - g(P, "neck"), g(S, "center") - g(P, "neck"))
    F["rel_heading"] = angle(g(S, "neck") - g(S, "butt"), g(P, "neck") - g(P, "butt"))
    ok = np.asarray(x["both"])[:nT] & F[FRAME_COLS].notna().all(axis=1).to_numpy()
    return F, ok, fps


def runs(m):
    """lengths (frames) of True runs"""
    m = np.r_[False, m, False]; e = np.flatnonzero(np.diff(m.astype(int))); return e[1::2] - e[::2]


def session_features(F, ok, fps):
    f = F[ok]; close = (f.dist_cc < CLOSE_CM).to_numpy(); n = len(f)
    rl = runs(F.dist_cc.lt(CLOSE_CM).to_numpy() & ok); rl = rl[rl >= int(fps)]
    return dict(
        mean_dist=f.dist_cc.mean(), frac_close=close.mean(),
        close_bouts_per_min=len(rl) / (n / fps / 60), close_bout_s=rl.mean() / fps if len(rl) else 0.0,
        frac_contact_subj=(f.s_nose_p_min < CONTACT_CM).mean(), frac_contact_part=(f.p_nose_s_min < CONTACT_CM).mean(),
        frac_ag_subj=(f.s_nose_p_butt < CONTACT_CM).mean(), frac_ag_part=(f.p_nose_s_butt < CONTACT_CM).mean(),
        subj_speed=f.s_speed.mean(), part_speed=f.p_speed.mean(),
        frac_moving_subj=(f.s_speed > MOVE_CMS).mean(), frac_moving_part=(f.p_speed > MOVE_CMS).mean(),
        subj_speed_close=f.s_speed[close].mean() if close.any() else np.nan,
        subj_speed_far=f.s_speed[~close].mean() if (~close).any() else np.nan,
        frac_oriented_ag_subj=((f.s_facing < 60) & (f.s_nose_p_butt < 3)).mean(), frac_oriented_ag_part=((f.p_facing < 60) & (f.p_nose_s_butt < 3)).mean())
