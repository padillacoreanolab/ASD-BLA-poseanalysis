"""Watershed regions of the behaviour map (paper_panels_data_ws.py, 10 Nov 2026; the regions on the Nov 2026 poster).

Density  2-D histogram of the 36,000 cagemate map frames on a 200 x 200 grid (map extent +- 0.5), Gaussian smoothing sigma 3 bins.
Peaks    the 13 strongest local maxima (>= 2% of the top, >= 8 bins apart).
Regions  watershed on -density from those peaks, masked at 0.4% of the peak; grid cells outside the mask take the nearest
         region, so every frame gets one. Labels 0-12 (names: metadata/region_names_ws13.csv).
Every frame of a session (for transitions): the majority region of its 15 nearest cagemate map frames (z-scored 10-D space),
then a 9-frame rolling majority; frames with < 5 valid votes in the window are -1.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

B, SIGMA, NPEAK, MIN_DIST, THR_REL, MASK_REL = 200, 3.0, 13, 8, .02, .004


def fit(emb, npeak=NPEAK):
    lo, hi = emb.min(0) - .5, emb.max(0) + .5
    xe, ye = np.linspace(lo[0], hi[0], B + 1), np.linspace(lo[1], hi[1], B + 1)
    Hs = gaussian_filter(np.histogram2d(emb[:, 0], emb[:, 1], bins=[xe, ye])[0], SIGMA)
    pk = peak_local_max(Hs, min_distance=MIN_DIST, threshold_rel=THR_REL, exclude_border=False, num_peaks=npeak)
    mk = np.zeros_like(Hs, int); mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-Hs, mk, mask=Hs > Hs.max() * MASK_REL)
    _, (ii, jj) = distance_transform_edt(lab == 0, return_indices=True)
    WS = lab[ii, jj] - 1
    return dict(WS=WS, xe=xe, ye=ye, nreg=int(WS.max()) + 1, peaks=pk, density=Hs, inside=lab > 0)


def cell(W, E):
    """grid cell (i, j) of each map position"""
    xe, ye = W["xe"], W["ye"]
    return (np.clip(np.searchsorted(xe, E[:, 0]) - 1, 0, B - 1), np.clip(np.searchsorted(ye, E[:, 1]) - 1, 0, B - 1))


def region_of(W, E):
    return W["WS"][cell(W, E)]


def frame_sequence(Z, good, NN, REG, nreg):
    """region per frame of one session: 15-NN majority vote on the map frames, then 9-frame rolling majority (-1 = unknown)"""
    Z = Z.copy(); Z[~good] = 0
    idx = NN.kneighbors(Z, return_distance=False)
    vote = np.array([np.bincount(REG[i], minlength=nreg).argmax() for i in idx]); hot = np.eye(nreg)[vote]; hot[~good] = 0
    c = np.cumsum(np.vstack([np.zeros(nreg), hot]), 0); T = len(hot)
    w = c[np.clip(np.arange(T) + 5, 0, T)] - c[np.clip(np.arange(T) - 4, 0, T)]
    s = w.argmax(1).astype(np.int8); s[w.sum(1) < 5] = -1
    return s
