"""Step 3 - per-frame and per-session pose features from the repaired skeletons.

Outputs
  cache/simple_frames.pkl          8 distance/speed features + 3 orientation angles + ok, every frame (first 600 s)
  cache/pose_frames.pkl            28 posture/sniffing features + 12 labels + BORIS frame labels (b_*), every frame
  tables/simple_features_session.csv, simple_features_contrasts.csv   16 session features (2 cm contact rule), C21 vs saline
  tables/pose_features_session.csv, pose_features_contrasts.csv       27 session features (label time fractions and rates)
The two frame tables have the same rows in the same order (asserted).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import features_pose as FP, features_simple as FS, sessions, stats   # noqa: E402
from c21pose.config import CACHE, TABLES                                           # noqa: E402

if __name__ == "__main__":
    sess = sessions.load("sessions_guarded")
    s_rows, s_frames, p_rows, p_frames = [], [], [], []
    for x in sess:
        meta = dict(basename=x["basename"], subject=x["subject"], drug=x["drug"], fam=x["fam"])
        F, ok, fps = FS.frame_table(x)
        s_rows.append(dict(**meta, **FS.session_features(F, ok, fps)))
        keep = F.copy(); keep["ok"] = ok
        s_frames.append(keep.astype("float32").assign(**meta, t=np.arange(len(F)) / fps))

        G, okp, _ = FP.frame_table(x); Lb = FP.labels(G, okp, fps); n = len(G)
        r = dict(**meta, scored=x["scored"], **FP.session_features(G, Lb, okp, fps))
        for k in ("ag", "face", "body"): r[f"boris_{k}_frac"] = np.asarray(x["lab"][k])[:n][okp].mean() if x["scored"] else np.nan
        p_rows.append(r)
        p_frames.append(pd.concat([G.astype("float32"), Lb], axis=1).assign(
            ok=okp, fi=np.arange(n), t=np.arange(n) / fps, scored=x["scored"],
            **{f"b_{k}": np.asarray(v)[:n].astype(bool) for k, v in x["lab"].items()}, **meta))

    SF = pd.concat(s_frames); PF = pd.concat(p_frames, ignore_index=True)
    assert len(SF) == len(PF) and (SF.basename.to_numpy() == PF.basename.to_numpy()).all(), "frame tables are not aligned"
    SF.to_pickle(CACHE / "simple_frames.pkl"); PF.to_pickle(CACHE / "pose_frames.pkl")

    A = pd.DataFrame(s_rows); A.to_csv(TABLES / "simple_features_session.csv", index=False)
    stats.paired_contrasts(A, [c for c in A.columns if c not in ("basename", "subject", "drug", "fam")]).to_csv(TABLES / "simple_features_contrasts.csv", index=False)
    B = pd.DataFrame(p_rows); B.to_csv(TABLES / "pose_features_session.csv", index=False)
    feats = [c for c in B.columns if c not in ("basename", "subject", "drug", "fam", "scored") and not c.startswith("boris_")]
    stats.paired_contrasts(B, feats).to_csv(TABLES / "pose_features_contrasts.csv", index=False)
    print(f"{len(sess)} sessions, {len(SF):,} frames ({SF.ok.mean():.1%} valid) -> {CACHE}")
