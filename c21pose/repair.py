"""Hybrid skeleton correction ("guarded" repair, 23 Oct 2026). From pose_repair.py + pose_repair_guarded.py.

SLEAP skeletons break at close contact: a keypoint lands on the other mouse, or the head-tail axis flips for a frame.
Detection (per session and mouse; reference = 'clean' frames, centers >= 15 cm apart, both tracked):
  crossed   keypoint closer to the OTHER mouse's center than to its own, and > 1.5 x its normal distance from its own center
  flip      body axis (tailbase -> neck) turns > 120 deg in one frame -> every keypoint except the center is flagged
  detached  a skeleton edge > 2 x its clean median length and its outer keypoint closer to the other mouse's center
(the 'shape', 'edge' and 'jump' rules exist but are NOT used: an image test showed they mostly flag real fast movements).
For each flagged keypoint there are up to three candidate positions:
  raw       SLEAP's position
  interp    linear interpolation between the nearest unflagged frames within 15 frames on each side
  template  the mouse's own clean body shape fitted to its good keypoints (similarity transform), if >= 5 good incl. center
A candidate is acceptable if, in the VIDEO frame, it sits on a dark (mouse) pixel, is closer to its own mouse's center, has a
normal distance to its own center and a normal edge length to its parent keypoint. Raw is kept if acceptable; otherwise the
acceptable candidate that best fits the anatomy; if none is acceptable, raw is kept.

Two ways to run (scripts/02_repair_poses.py):
  video   the full procedure; needs the 720p videos. 79% of flagged keypoints kept raw, 11% interp, 10% template.
  replay  re-applies the saved decisions (metadata/pose_repair_decisions.csv.gz: frame, keypoint, interp or template). The
          candidate positions depend only on the skeletons, not on the video, so replay gives the same skeletons exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLEAN_CM, MAXI, WIN = 15.0, 15, 600.0
RULES = ("crossed", "flip", "detached")
EDGES = [("head", "nose"), ("head", "leftear"), ("head", "rightear"), ("neck", "head"), ("center", "neck"),
         ("center", "leftbody"), ("center", "rightbody"), ("center", "butt")]                 # (inner, outer)
d2 = lambda a, b: np.linalg.norm(a - b, axis=-1)


def robust(v):
    v = v[np.isfinite(v)]
    return (np.median(v), 1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 50 else (np.nan, np.nan)


def body_frame(A, ni):
    """rotate each frame so the tailbase -> neck axis points +x, origin at the center"""
    ax = A[:, ni["neck"]] - A[:, ni["butt"]]; th = np.arctan2(ax[:, 1], ax[:, 0]); c, s = np.cos(-th), np.sin(-th)
    X = A - A[:, ni["center"]][:, None, :]
    return np.stack([X[..., 0] * c[:, None] - X[..., 1] * s[:, None], X[..., 0] * s[:, None] + X[..., 1] * c[:, None]], -1)


def umeyama(src, dst):
    """similarity transform mapping src -> dst (both n x 2)"""
    ms, md = src.mean(0), dst.mean(0); S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src)); E = np.eye(2)
    if np.linalg.det(U @ Vt) < 0: E[1, 1] = -1
    R = U @ E @ Vt; scale = (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src)); t = md - scale * R @ ms
    return lambda X: (scale * (R @ X.T)).T + t


def detect(A, B, ni, clean, rules=RULES):
    """(T, K) bool: keypoints of mouse A flagged by the rules (B = the other mouse)"""
    T, K = A.shape[:2]; c, cB = A[:, ni["center"]], B[:, ni["center"]]; F = np.zeros((T, K), bool)
    r, rB = d2(A, c[:, None]), d2(A, cB[:, None])
    for k in range(K):
        if k == ni["center"]: continue
        m, s = robust(r[clean, k])
        if np.isfinite(m):
            if "shape" in rules: F[:, k] |= np.abs(r[:, k] - m) > max(4 * s, .35 * m)
            if "crossed" in rules: F[:, k] |= (rB[:, k] < r[:, k]) & (r[:, k] > 1.5 * m)
    for a, b in EDGES:
        L = d2(A[:, ni[a]], A[:, ni[b]]); m, s = robust(L[clean])
        if np.isfinite(m) and "edge" in rules: F[:, ni[b]] |= np.abs(L - m) > max(4 * s, .5 * m)
        if np.isfinite(m) and "detached" in rules: F[:, ni[b]] |= (L > 2 * m) & (rB[:, ni[b]] < r[:, ni[b]])
    step = np.r_[np.nan, d2(c[1:], c[:-1])]
    for k in range(K):
        if "jump" in rules: F[:, k] |= (np.r_[np.nan, d2(A[1:, k], A[:-1, k])] > 2.0) & (step < 1.0)
    ax = A[:, ni["neck"]] - A[:, ni["butt"]]; th = np.arctan2(ax[:, 1], ax[:, 0]); dth = np.r_[0, np.abs(np.angle(np.exp(1j * np.diff(th))))]
    flip = (np.degrees(dth) > 120) if "flip" in rules else np.zeros(T, bool); F[flip] = True; F[flip, ni["center"]] = False
    F[~np.isfinite(A).all(-1)] = False
    return F


def interp_candidate(A, F, t, k):
    """keypoint k at frame t interpolated from the nearest unflagged, finite frames within MAXI on each side"""
    good = lambda u: 0 <= u < len(A) and not F[u, k] and np.isfinite(A[u, k]).all()
    a = next((u for u in range(t - 1, max(t - MAXI, -1), -1) if good(u)), None)
    b = next((u for u in range(t + 1, min(t + MAXI, len(A) - 1) + 1) if good(u)), None)
    if a is None or b is None: return None
    w = (t - a) / (b - a); return A[a, k] * (1 - w) + A[b, k] * w


def prepare(x):
    """flags and clean-frame references for both mice of one session"""
    S, P, ni = x["S"], x["P"], x["ni"]; n = min(len(S), int(WIN * x["fps"])); idx = np.arange(len(S)) < n
    ok = np.asarray(x["both"]) & np.isfinite(S).all((1, 2)) & np.isfinite(P).all((1, 2))
    clean = ok & idx & (d2(S[:, ni["center"]], P[:, ni["center"]]) >= CLEAN_CM)
    mice = {"S": (S, P), "P": (P, S)}; F = {m: detect(A, B, ni, clean) for m, (A, B) in mice.items()}
    parent = {ni[b]: ni[a] for a, b in EDGES}; ref = {}
    for m, (A, B) in mice.items():
        r = d2(A, A[:, ni["center"]][:, None]); ref[m] = np.array([robust(r[clean, k]) for k in range(A.shape[1])])
        ref[m + "_tmpl"] = np.nanmedian(body_frame(A[clean], ni), axis=0)
        ref[m + "_edge"] = {b: robust(d2(A[clean, b], A[clean, a]))[0] for b, a in parent.items()}
    need = sorted(set(np.flatnonzero((F["S"].any(1) | F["P"].any(1)) & ok & idx).tolist()))
    return mice, F, ref, parent, need


def template_candidate(A, Fm, ref_tmpl, t, k, ni):
    bad = Fm[t]; good = ~bad & np.isfinite(A[t]).all(1)
    if good.sum() >= 5 and good[ni["center"]]:
        return umeyama(ref_tmpl[good], A[t, good])(ref_tmpl[[k]])[0]
    return None


def run_video(x, video_path, to_px):
    """full guarded repair of one session using its video. to_px(p) maps a floor point (cm) to pixel (x, y)."""
    import cv2
    ni = x["ni"]; mice, F, ref, parent, need = prepare(x)
    out = {"S": x["S"].copy(), "P": x["P"].copy()}; rows = []
    if not need: return out, rows
    cap = cv2.VideoCapture(str(video_path)); cur = -1
    for t in need:
        if t != cur + 1: cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        okf, img = cap.read(); cur = t
        if not okf: continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); thr = .4 * np.median(g); mn = cv2.erode(g, np.ones((7, 7), np.uint8))
        for m, (A, B) in mice.items():
            bad = F[m][t]
            if not bad.any(): continue
            cO, cX = A[t, ni["center"]], B[t, ni["center"]]
            for k in np.flatnonzero(bad):
                med, sd = ref[m][k]; tol = max(4 * sd, .35 * med) * 1.5 if np.isfinite(med) else np.inf

                def judge(p):
                    if p is None or not np.isfinite(p).all(): return None
                    q = to_px(p); on = 0 <= q[0] < g.shape[1] and 0 <= q[1] < g.shape[0] and mn[int(q[1]), int(q[0])] < thr
                    own = d2(p, cO) < d2(p, cX); an = abs(d2(p, cO) - med) / tol if np.isfinite(tol) else 0
                    em = ref[m + "_edge"].get(k, np.nan); pa = out[m][t, parent[k]] if k in parent else None
                    eok = not (np.isfinite(em) and pa is not None and np.isfinite(pa).all() and d2(p, pa) > 2 * em)
                    return dict(on=bool(on), own=bool(own), anat=float(an), ok=bool(on and own and an <= 1 and eok))
                cand = {"raw": A[t, k], "interp": interp_candidate(A, F[m], t, k),
                        "template": template_candidate(A, F[m], ref[m + "_tmpl"], t, k, ni)}
                J = {c: judge(p) for c, p in cand.items()}
                if J["raw"] and J["raw"]["ok"]: choice = "raw"
                else:
                    opts = [c for c in ("interp", "template") if J[c] and J[c]["ok"]]
                    if len(opts) == 2 and cand["interp"] is not None and cand["template"] is not None:
                        choice = "interp" if J["interp"]["anat"] <= J["template"]["anat"] + .25 else "template"
                    else: choice = opts[0] if opts else "raw"
                out[m][t, k] = cand[choice]
                rows.append(dict(basename=x["basename"], mouse="subject" if m == "S" else "partner", frame=int(t), node=list(ni)[k], choice=choice))
    cap.release()
    return out, rows


def run_replay(x, decisions):
    """re-apply saved decisions (rows with choice interp / template) to one session"""
    ni = x["ni"]; out = {"S": x["S"].copy(), "P": x["P"].copy()}
    if decisions is None or not len(decisions): return out
    mice, F, ref, _, _ = prepare(x)
    for r in decisions.itertuples():
        m = "S" if r.mouse == "subject" else "P"; A = mice[m][0]; k = ni[r.node]; t = int(r.frame)
        p = interp_candidate(A, F[m], t, k) if r.choice == "interp" else template_candidate(A, F[m], ref[m + "_tmpl"], t, k, ni)
        if p is None: raise RuntimeError(f"replay: no {r.choice} candidate for {x['basename']} {r.mouse} frame {t} {r.node}")
        out[m][t, k] = p
    return out


def finish(x, out):
    """session record with repaired skeletons; 'both' now also requires every keypoint of both mice to be finite"""
    y = dict(x); y.update(S=out["S"], P=out["P"])
    y["both"] = np.asarray(x["both"]) & np.isfinite(y["S"]).all((1, 2)) & np.isfinite(y["P"]).all((1, 2))
    return y
