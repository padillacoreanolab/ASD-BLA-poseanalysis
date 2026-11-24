"""Step 4 - fit the 10-feature UMAP behaviour map on cagemate frames and place the familiar / novel frames on it.

Outputs
  cache/map_frames.pkl   the 108,000 sampled frames (36,000 per partner type) with u1, u2 and every per-frame column used later
  cache/map_model.pkl    what is needed to place any new frame: MU, SD (z-score), Z and U of the 36,000 cagemate map frames
UMAP is seeded (random_state = 0) and single-threaded under that seed, so the map is reproducible with the pinned versions
(requirements.txt); other umap-learn / numba / pynndescent versions can give a different (rotated or reshaped) map.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import bmap                              # noqa: E402
from c21pose.config import CACHE, MAP_FEATURES        # noqa: E402

if __name__ == "__main__":
    SF = pd.read_pickle(CACHE / "simple_frames.pkl"); PF = pd.read_pickle(CACHE / "pose_frames.pkl")
    D = bmap.sample_frames(SF, PF).drop(columns=["s_speed_pf"])      # s_speed_pf equals s_speed (same 0.2 s center speed)
    cm = (D.fam == "cagemate").to_numpy()
    emb, MU, SD = bmap.fit_map(D[cm])
    Zm = (bmap.prep(D.loc[cm, MAP_FEATURES]) - MU) / SD
    pos, _ = bmap.place(Zm, emb, (bmap.prep(D.loc[~cm, MAP_FEATURES]) - MU) / SD)
    D["u1"], D["u2"] = np.nan, np.nan
    D.loc[cm, ["u1", "u2"]] = emb; D.loc[~cm, ["u1", "u2"]] = pos
    D.to_pickle(CACHE / "map_frames.pkl")
    pickle.dump(dict(features=MAP_FEATURES, MU=MU, SD=SD, Z=Zm, U=emb), open(CACHE / "map_model.pkl", "wb"))
    print(f"map fitted on {cm.sum():,} cagemate frames; {(~cm).sum():,} familiar / novel frames placed -> {CACHE / 'map_frames.pkl'}")
