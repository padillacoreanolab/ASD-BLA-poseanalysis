"""The 10-feature behaviour map (umap10_cagemate.py + umap10_all_partners.py, 26-28 Oct 2026).

Frames   1500 random valid frames per session (numpy default_rng(0), re-seeded for each partner type, sessions in name order),
         from all 72 sessions: 36,000 per partner type.
Features the 10 MAP_FEATURES (config.py): 5 distances in each direction covering face, body and anogenital targets, 2 speeds.
         Transform: log(1 + x) (negatives clipped to 0), then z-scored with the mean / SD of the cagemate frames.
Map      UMAP(n_neighbors = 50, min_dist = 0.1, random_state = 0) fitted on the 36,000 CAGEMATE frames only.
Placing  familiar and novel frames: mean UMAP position of their 15 nearest cagemate frames in the z-scored 10-D space.
         (The same rule places every frame of a session for the transition analysis, see watershed.py.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAMS, MAP_FEATURES, N_PER_SESSION, SEED
from .features_pose import LABELS

PF_KEEP = ["basename", "fi", "s_nose_body", "i_nose_body", "distance", "s_speed", "i_speed", "scored",
           "b_ag", "b_face", "b_body", "b_selfgroom"] + LABELS


def prep(A):
    return np.log1p(np.asarray(A, float).clip(min=0))


def sample_frames(SF, PF, fams=FAMS, n_per=N_PER_SESSION, seed=SEED):
    """the map frames: per partner type, n_per valid frames per session; pose columns merged in by (basename, frame)"""
    SF = SF.copy(); SF["fi"] = SF.groupby("basename").cumcount()
    parts = []
    for fam in fams:
        F = SF[(SF.fam == fam) & (SF.ok > 0)]; rng = np.random.default_rng(seed)
        parts.append(pd.concat([g.iloc[rng.choice(len(g), size=min(n_per, len(g)), replace=False)] for _, g in F.groupby("basename")]))
    D = pd.concat(parts).reset_index(drop=True)
    D = D.merge(PF[PF_KEEP].rename(columns={"s_speed": "s_speed_pf"}), on=["basename", "fi"], how="left") \
         .rename(columns={"s_nose_body": "s_nose_p_body", "i_nose_body": "p_nose_s_body"})
    assert D.p_nose_s_body.notna().all(), "unmatched frames"
    for c in LABELS + ["b_ag", "b_face", "b_body", "b_selfgroom", "scored"]: D[c] = D[c].astype(bool)
    return D


def fit_map(D, seed=SEED):
    """UMAP on the cagemate frames -> (u1, u2) for those frames; returns embedding and the z-score parameters"""
    import umap
    X = prep(D[MAP_FEATURES]); MU, SD = X.mean(0), X.std(0)
    emb = umap.UMAP(n_neighbors=50, min_dist=0.1, random_state=seed).fit_transform((X - MU) / SD)
    return emb, MU, SD


def place(Zmap, Umap, Z, k=15, exclude_self=False):
    """positions of frames Z on the map: mean coordinates of their k nearest map frames (Zmap, Umap); returns (positions, neighbor idx)"""
    from sklearn.neighbors import NearestNeighbors
    NN = NearestNeighbors(n_neighbors=k + 1).fit(Zmap)
    idx = NN.kneighbors(Z, return_distance=False)
    idx = idx[:, 1:] if exclude_self else idx[:, :k]
    return Umap[idx].mean(1), idx
