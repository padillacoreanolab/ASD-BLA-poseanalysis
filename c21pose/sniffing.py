"""SLEAP sniffing detection: the 2 cm rule and the final rule of the Nov 2026 poster.

2 cm rule (boris_lda_addon_features.targets, 26 Oct 2026). Per frame, mouse A sniffs mouse B when A's nose is within 2 cm of any of B's
keypoints; the target is the closest region of B:
    face = nose, head, ears      body = neck, left side, right side, center      anogenital = tail base
Each target mask: gaps < 0.2 s closed, then runs < 0.2 s dropped.

Final rule (sleap_new_vs_old.py / sniff_ag_ablation.py, 10 Nov 2026):
    anogenital  = the 2 cm rule, unchanged (a 0.5 s minimum bout removed 62% of anogenital bouts, most of them real)
    face/body   = nose within 2 cm, the closest target is face or body, AND
                  heading  angle between the subject's head direction (neck -> nose) and nose -> partner center < cutoff
                  lateral  partner center's distance off the subject's (5-frame smoothed) head axis < cutoff (cm)
                  axis     (optional) angle between the two neck - center axes >= cutoff
                  then gaps < 0.2 s closed and runs shorter than the minimum bout dropped.
    The cutoffs for each mouse are chosen on the OTHER 11 mice: the grid point with the best pooled frame F1 against BORIS face + body
    labels, with BORIS moved 0.5 s earlier (BORIS lags SLEAP contact by about 0.5 s). Grid: heading {45, 60, 75, 90, 120, 181} deg,
    axis {0, 45} deg, lateral {3, 5, 8, none} cm, min bout {0.2, 0.5} s. The rule was refined after seeing the drug results
    (post hoc): describe it as such.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

from .config import CONTACT_CM, NODES, WIN_S
from .sleapio import boxcar, unit

FACE, BODY, TAIL = ["nose", "head", "leftear", "rightear"], ["neck", "leftbody", "rightbody", "center"], ["butt"]
TGT = {"face": FACE, "body": BODY, "ag": TAIL}
T = list(TGT)
MIN_S, MERGE_S, BORIS_LAG_S = 0.2, 0.2, 0.5
HEADING, AXIS, LATERAL, MIN_BOUT = (45, 60, 75, 90, 120, 181), (0, 45), (3, 5, 8, 1e9), (.2, .5)
GRID_FB = list(itertools.product(HEADING, AXIS, LATERAL, MIN_BOUT))


def ix(name): return NODES.index(name)


def prepare_sessions(sess):
    """session records -> first 600 s, keypoints in NODES order, sorted by (subject, drug)"""
    out = []
    for x in sess:
        n = min(len(x["S"]), int(WIN_S * x["fps"])); idx = [x["ni"][k] for k in NODES]
        out.append(dict(basename=x["basename"], subject=x["subject"], drug=x["drug"], fam=x["fam"], fps=float(x["fps"]),
                        S=x["S"][:n][:, idx].astype(float), P=x["P"][:n][:, idx].astype(float), both=np.asarray(x["both"])[:n],
                        scored=bool(x["scored"]), lab={k: np.asarray(v)[:n].astype(bool) for k, v in x["lab"].items()}))
    return sorted(out, key=lambda s: (s["subject"], s["drug"]))


def nose_to_other(A, B, targets=None):
    """per-frame min distance (cm) from A's nose to B's keypoints (all 9, or the named ones)"""
    t = [ix(k) for k in (targets or NODES)]
    with warnings.catch_warnings():                       # untracked frames give NaN, as intended
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(np.linalg.norm(B[:, t] - A[:, [ix("nose")]], axis=-1), axis=1)


def runs(m):
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(m, int), 0])); return e[::2], e[1::2]


def close_gaps(m, k):
    m = m.copy(); st, en = runs(~m)
    for a, b in zip(st, en):
        if a > 0 and b < len(m) and b - a < k: m[a:b] = True
    return m


def drop_short(m, k):
    m = m.copy(); st, en = runs(m)
    for a, b in zip(st, en):
        if b - a < k: m[a:b] = False
    return m


def targets(A, B, fps):
    """the 2 cm rule: {face, body, ag: cleaned mask of A sniffing that region of B}"""
    d = np.stack([nose_to_other(A, B, TGT[t]) for t in TGT], 1)
    near = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1) < CONTACT_CM
    best = np.argmin(np.where(np.isfinite(d), d, np.inf), 1)
    return {t: drop_short(close_gaps(near & (best == j), int(MERGE_S * fps)), int(MIN_S * fps)) for j, t in enumerate(TGT)}


def ang(u, v):
    c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)); return np.degrees(np.arccos(np.clip(c, -1, 1)))


def frame_geometry(s):
    """per-frame quantities of the final rule for one session (subject sniffing partner)"""
    S, P, fps, n = s["S"], s["P"], s["fps"], len(s["S"])
    d = np.stack([nose_to_other(S, P, TGT[t]) for t in T], 1); dd = np.where(np.isfinite(d), d, np.inf)
    u = S[:, ix("nose")] - S[:, ix("neck")]; h = unit(boxcar(u)); left = np.stack([h[:, 1], -h[:, 0]], -1)
    tn = P[:, ix("center")] - S[:, ix("nose")]; y = {}; k = int(round(BORIS_LAG_S * fps))
    if s["scored"]:
        for t in T: q = np.zeros(n, bool); q[:n - k] = s["lab"][t][:n][k:]; y[t] = q          # BORIS moved 0.5 s earlier
    return dict(s=s, fps=fps, n=n, near=dd.min(1) < CONTACT_CM, best=dd.argmin(1), heading=ang(u, tn),
                axis=ang(S[:, ix("neck")] - S[:, ix("center")], P[:, ix("neck")] - P[:, ix("center")]),
                lat=np.abs((tn * left).sum(1)), y=y, old=targets(S, P, fps))


def mask(f, j, heading, axis, lateral, min_bout):
    with np.errstate(invalid="ignore"):
        g = f["near"] & (f["best"] == j) & (f["heading"] < heading) & (f["lat"] < lateral) & (f["axis"] >= axis if axis > 0 else True)
    return drop_short(close_gaps(g, int(.2 * f["fps"])), max(int(min_bout * f["fps"]), 1))


def facebody(f, g):
    return mask(f, T.index("face"), *g) | mask(f, T.index("body"), *g)


def _cnt(m, y): return np.array([(m & y).sum(), (m & ~y).sum(), (~m & y).sum()])
def _f1(c): return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def final_rule(F):
    """F = {basename: frame_geometry}. Face/body cutoffs held out by mouse. Returns ({basename: face/body mask}, per-mouse cutoffs)"""
    SC = {b: f for b, f in F.items() if f["s"]["scored"]}
    CC = {b: np.array([_cnt(facebody(f, g), f["y"]["face"] | f["y"]["body"]) for g in GRID_FB]) for b, f in SC.items()}
    FB, PAR = {}, []
    for m in sorted({f["s"]["subject"] for f in F.values()}):
        tot = sum(CC[b] for b, f in SC.items() if f["s"]["subject"] != m)
        g = GRID_FB[int(np.argmax([_f1(c) for c in tot]))]
        PAR.append(dict(held_out=m, heading_max=g[0], axis_min=g[1], lateral_max=g[2], min_bout=g[3]))
        for b, f in F.items():
            if f["s"]["subject"] == m: FB[b] = facebody(f, g)
    return FB, PAR


def bouts(m, fps):
    """(number of bouts, mean bout s, total s) of a mask"""
    st, en = runs(m); d = (np.asarray(en) - np.asarray(st)) / fps
    return len(d), (d.mean() if len(d) else np.nan), m.sum() / fps
