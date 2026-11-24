"""Step 2 - skeleton repair: sessions_raw.pkl -> sessions_guarded.pkl (the skeletons every later step uses).

usage:  python scripts/02_repair_poses.py                 # replay the saved decisions (no videos needed; exact)
        python scripts/02_repair_poses.py --mode video    # redo the decisions from the videos (paths.video_root)
In video mode the new decisions are written to outputs/tables/pose_repair_decisions.csv; to make them the reference for replay,
copy that file (gzipped, choice != raw rows) over metadata/pose_repair_decisions.csv.gz.
"""
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import calibration as cal, repair                  # noqa: E402
from c21pose.config import CACHE, META, TABLES, VIDEO_ROOT      # noqa: E402


def _video_job(x):
    vids = {v["id"]: v for v in json.load(open(META / "cage_calibration/videos.json"))}
    v = vids[cal.video_id(x["basename"])]; p = VIDEO_ROOT / v["cohort"] / v["file"]
    if not p.exists() or os.path.getsize(p) == 0:
        print("no video, keeping raw:", x["basename"], flush=True); return x["basename"], None, []
    out, rows = repair.run_video(x, p, lambda q: cal.to_px(q, x["basename"]))
    return x["basename"], out, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["replay", "video"], default="replay"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    sess = pickle.load(open(CACHE / "sessions_raw.pkl", "rb"))
    if a.mode == "video":
        with Pool(a.workers) as pool: res = pool.map(_video_job, sess, chunksize=1)
        done = {b: o for b, o, _ in res if o is not None}
        rows = pd.DataFrame([r for _, _, rr in res for r in rr]); rows.to_csv(TABLES / "pose_repair_decisions.csv", index=False)
        print("decisions:", (rows.choice.value_counts(normalize=True) * 100).round(1).to_dict())
        out = [repair.finish(x, done[x["basename"]]) if x["basename"] in done else repair.finish(x, {"S": x["S"], "P": x["P"]}) for x in sess]
    else:
        D = pd.read_csv(META / "pose_repair_decisions.csv.gz"); g = dict(tuple(D.groupby("basename")))
        out = [repair.finish(x, repair.run_replay(x, g.get(x["basename"]))) for x in sess]
        print(f"replayed {len(D)} keypoint corrections in {len(g)} sessions")
    pickle.dump(out, open(CACHE / "sessions_guarded.pkl", "wb"), protocol=4)
    print("->", CACHE / "sessions_guarded.pkl")
