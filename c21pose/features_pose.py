"""Posture + sniffing per-frame features and labels (from pose_features.py, 23 Oct 2026). Full definitions: docs/FEATURES.md.

Computed the same way for the subject (s_) and the partner (i_ = intruder: cagemate, familiar or novel mouse).
  28 per-frame features = 13 per mouse x 2 + 2 relational
     posture (9): length, width, elongation, bend, head_turn, speed, nose_speed, turn_rate, wall_dist
     sniffing (4): nose_face, nose_body, nose_ag (nose to the nearest keypoint of that region of the other mouse), facing
     relational (2): distance (center to center), rel_heading
  12 per-frame labels = 6 per mouse
     sniff_face / sniff_body / sniff_ag: nose within 2 cm of that region, the region being the nearest of the three (bout >= 0.1 s)
     rear, hunch, stretch: posture rules (bout >= 0.3 s; rearing <= 8 s)
Regions of the other mouse: face = nose, head, ears; body = neck, left side, right side, center; ag = tail base.
The behaviour map uses s_nose_body and i_nose_body (renamed s_nose_p_body / p_nose_s_body there).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONTACT_CM, MOVE_CMS, WIN_S

CAGE_X, CAGE_Y = 35.56, 14.22
FACE, BODY, AG = ["nose", "head", "leftear", "rightear"], ["neck", "leftbody", "rightbody", "center"], ["butt"]
POSTURE = ["length", "width", "elongation", "bend", "head_turn", "speed", "nose_speed", "turn_rate", "wall_dist"]
SNIFF = ["nose_face", "nose_body", "nose_ag", "facing"]
FEATS = [f"{m}_{f}" for m in ("s", "i") for f in POSTURE + SNIFF] + ["distance", "rel_heading"]
LABELS = [f"{m}_{f}" for m in ("s", "i") for f in ("sniff_face", "sniff_body", "sniff_ag", "rear", "hunch", "stretch")]
MIN_BOUT_S = {"sniff_face": 0.1, "sniff_body": 0.1, "sniff_ag": 0.1, "rear": 0.3, "hunch": 0.3, "stretch": 0.3}; MAX_REAR_S = 8.0


def angle(u, v):
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9), -1, 1)))


def runs(m):
    """(start, end) frame indices of True runs, end exclusive"""
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0])); return e[::2], e[1::2]


def frame_table(x):
    S, P, ni, fps = x["S"], x["P"], x["ni"], x["fps"]
    n = min(len(S), int(WIN_S * fps)); S, P = S[:n], P[:n]
    d = lambda a, b: np.linalg.norm(a - b, axis=-1); g = lambda A, k: A[:, ni[k]]
    step = max(int(round(fps * 0.2)), 1)

    def over_step(v):
        out = np.full(len(v), np.nan); out[step:] = v[step:] - v[:-step]; return out
    cols = {"distance": d(g(S, "center"), g(P, "center")), "rel_heading": angle(g(S, "neck") - g(S, "butt"), g(P, "neck") - g(P, "butt"))}
    for m, A, B in (("s", S, P), ("i", P, S)):
        nose, neck, butt, cen = g(A, "nose"), g(A, "neck"), g(A, "butt"), g(A, "center")
        spd = np.full(n, np.nan); spd[step:] = d(cen[step:], cen[:-step]) / (step / fps)
        nsp = np.full(n, np.nan); nsp[step:] = d(nose[step:], nose[:-step]) / (step / fps)
        walk = spd > MOVE_CMS
        L, W = d(nose, butt), d(g(A, "leftbody"), g(A, "rightbody"))
        body, head = neck - butt, nose - neck
        cross = body[:, 0] * head[:, 1] - body[:, 1] * head[:, 0]
        lr = g(A, "leftbody") - g(A, "rightbody"); left = np.stack([body[:, 1], -body[:, 0]], -1)
        sgn = 1.0 if np.nanmean((lr * left).sum(-1)) > 0 else -1.0          # image y points down: fix left/right per session
        hd = np.arctan2(body[:, 1], body[:, 0])
        cols.update({
            f"{m}_length": L / np.nanmedian(L[walk]), f"{m}_width": W / np.nanmedian(W[walk]),
            f"{m}_elongation": (L / np.nanmedian(L[walk])) / (W / np.nanmedian(W[walk])),
            f"{m}_bend": angle(neck - cen, butt - cen),
            f"{m}_head_turn": angle(body, head) * np.sign(-cross) * sgn,
            f"{m}_speed": spd, f"{m}_nose_speed": nsp,
            f"{m}_turn_rate": np.abs((np.degrees(over_step(hd)) + 180) % 360 - 180) / (step / fps),
            f"{m}_wall_dist": np.minimum.reduce([nose[:, 0], CAGE_X - nose[:, 0], nose[:, 1], CAGE_Y - nose[:, 1]])})
        for part, keys in (("face", FACE), ("body", BODY), ("ag", AG)):
            cols[f"{m}_nose_{part}"] = np.min(np.stack([d(nose, g(B, k)) for k in keys]), 0)
        nearest = B[np.arange(n), np.argmin(np.linalg.norm(nose[:, None] - B, axis=-1), 1)]
        cols[f"{m}_facing"] = angle(head, nearest - neck)
    F = pd.DataFrame(cols)[FEATS]
    ok = np.asarray(x["both"])[:n] & F.notna().all(axis=1).to_numpy()
    return F, ok, fps


def labels(F, ok, fps):
    Lb = {}
    for m in ("s", "i"):
        reg = F[[f"{m}_nose_face", f"{m}_nose_body", f"{m}_nose_ag"]].to_numpy(); nearest = reg.argmin(1); close = reg.min(1) < CONTACT_CM
        for j, part in enumerate(("face", "body", "ag")): Lb[f"{m}_sniff_{part}"] = ok & close & (nearest == j)
        Lb[f"{m}_rear"] = ok & (F[f"{m}_length"] < .65).to_numpy() & (F[f"{m}_bend"] > 140).to_numpy() & (F[f"{m}_speed"] < 3).to_numpy() & (F.distance > 8).to_numpy()
        Lb[f"{m}_hunch"] = ok & (F[f"{m}_length"] < .75).to_numpy() & (F[f"{m}_bend"] < 140).to_numpy()
        Lb[f"{m}_stretch"] = ok & (F[f"{m}_length"] > 1.1).to_numpy() & (F[f"{m}_speed"] < MOVE_CMS).to_numpy()
    for k, v in Lb.items():
        st, en = runs(v); kind = k[2:]; keep = np.zeros_like(v)
        for a, b in zip(st, en):
            dur = (b - a) / fps
            if dur >= MIN_BOUT_S[kind] and (kind != "rear" or dur <= MAX_REAR_S): keep[a:b] = True
        Lb[k] = keep
    return pd.DataFrame(Lb)[LABELS]


def session_features(F, Lb, ok, fps):
    mins = ok.sum() / fps / 60; out = {}
    for k in LABELS:
        out[f"{k}_frac"] = Lb[k][ok].mean(); out[f"{k}_per_min"] = len(runs(Lb[k].to_numpy())[0]) / mins
    out.update(distance_mean=F.distance[ok].mean(), s_speed_mean=F.s_speed[ok].mean(), i_speed_mean=F.i_speed[ok].mean())
    return out
