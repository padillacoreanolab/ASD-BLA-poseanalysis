"""Step 5 - 13 watershed regions on the map; region of every sampled frame and of every cagemate frame (for transitions).

Outputs
  cache/map_frames_regions.pkl   map_frames.pkl + region (0-12) for all 108,000 sampled frames
  cache/watershed.pkl            label grid, edges, peaks, per-region medians, representative frame, cagemate frame sequences
usage: python scripts/05_watershed_regions.py [--peaks 13]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import bmap, sessions, watershed as W      # noqa: E402
from c21pose.config import CACHE, MAP_FEATURES as F10    # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--peaks", type=int, default=W.NPEAK); a = ap.parse_args()
    tag = "" if a.peaks == W.NPEAK else f"_ws{a.peaks}"
    D = pd.read_pickle(CACHE / "map_frames.pkl"); M = pickle.load(open(CACHE / "map_model.pkl", "rb"))
    cm = (D.fam == "cagemate").to_numpy(); emb, Zm, MU, SD = M["U"], M["Z"], M["MU"], M["SD"]
    ws = W.fit(emb, a.peaks); NREG = ws["nreg"]; REG = W.region_of(ws, emb)
    print(f"watershed: {len(ws['peaks'])} peaks -> {NREG} regions; {100 * ws['inside'][W.cell(ws, emb)].mean():.1f}% of cagemate frames inside the mask")
    D["region"] = W.region_of(ws, D[["u1", "u2"]].to_numpy(float))

    C = D[cm].copy(); C["rw"] = REG; prof = C.groupby("rw")[F10].median(); prof["share"] = C.groupby("rw").size() / len(C)
    z = (C.dist_cc - C.dist_cc.mean()) / C.dist_cc.std(); prof["social_z"] = z.groupby(C.rw).mean(); order = list(prof.sort_values("social_z").index)
    Zd = (bmap.prep(C[F10]) - MU) / SD; rep = {}
    for r in prof.index:                                      # representative frame: closest to the region's median in z space
        k = np.flatnonzero((C.rw == r).to_numpy()); j = k[np.argmin(((Zd[k] - np.median(Zd[k], 0)) ** 2).sum(1))]; rep[int(r)] = (C.iloc[j].basename, int(C.iloc[j].fi))

    FPS = {x["basename"]: float(x["fps"]) for x in sessions.load("sessions_guarded")}
    SF = pd.read_pickle(CACHE / "simple_frames.pkl"); PF = pd.read_pickle(CACHE / "pose_frames.pkl")
    SF["s_nose_p_body"], SF["p_nose_s_body"] = PF.s_nose_body.to_numpy(), PF.i_nose_body.to_numpy()
    SF = SF[SF.fam == "cagemate"]; NN = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(Zm); seq = {}
    for b, F in SF.groupby("basename", sort=False):
        Z = (bmap.prep(F[F10]) - MU) / SD; good = (F.ok.to_numpy() > 0) & np.isfinite(Z).all(1)
        seq[b] = dict(seq=W.frame_sequence(Z, good, NN, REG, NREG), fps=FPS[b], subject=F.subject.iloc[0], drug=F.drug.iloc[0])
    pickle.dump(dict(WS=ws["WS"], xe=ws["xe"], ye=ws["ye"], nreg=NREG, prof=prof, order=order, rep=rep, seq=seq, F10=F10, MU=MU, SD=SD,
                     peaks=ws["peaks"]), open(CACHE / f"watershed{tag}.pkl", "wb"))
    D.to_pickle(CACHE / f"map_frames_regions{tag}.pkl")
    print(prof.round(2).to_string())
