"""Per-video cage-floor calibration (pixels -> cm) and true frame rates. From cagecal.py / dyadcal.py (Rig Design, Sep 2026).

The four floor corners of every video were clicked by hand on 720p frames (TL, TR, BR, BL) and saved in
metadata/cage_calibration/calibration.json. A perspective transform maps them to the 35.56 x 14.22 cm floor (14 x 5.6 in).
The cage length (14 in) was measured; the width (5.6 in) is an assumed value.
Keypoints above the floor (nose, ears) are slightly displaced by a floor-plane transform; distances between keypoints at a
similar height are affected much less. Scale check: cohort 1 is about 25.2 px/cm, cohorts 2-3 about 32.4 px/cm, and the
median nose-to-tailbase length is 6.85 cm in every cohort.
The transform is computed and applied in float32 with OpenCV (as cagecal.py did): this reproduces the Nov 2026 caches exactly.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd

from .config import META

CAL_DIR = META / "cage_calibration"
CAL = json.load(open(CAL_DIR / "calibration.json"))
MAP = pd.read_csv(CAL_DIR / "h5_to_video_map.csv").set_index("basename")


def video_id(basename):
    return MAP.loc[basename, "vid"]


def video_file(basename):
    return MAP.loc[basename, "video"]


def fps_for(basename):
    """frame rate of the matching video (three E1.1 day-1 videos are 24 fps, the rest about 30)"""
    return float(MAP.loc[basename, "video_fps"])


def homography(basename):
    """3 x 3 float32 transform, 720p pixels -> floor cm"""
    c = CAL[video_id(basename)]
    L, W = c["length_in"] * 2.54, c["width_in"] * 2.54
    src = np.array(c["pts"], np.float32)
    dst = np.array([[0, 0], [L, 0], [L, W], [0, W]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_cm(xy, basename):
    """(..., 2) pixel array -> floor cm as float64 (NaN stays NaN)"""
    flat = np.asarray(xy).reshape(-1, 2).astype(np.float32)
    out = np.full_like(flat, np.nan)
    ok = ~np.isnan(flat).any(1)
    out[ok] = cv2.perspectiveTransform(flat[ok][None], homography(basename))[0]
    return out.reshape(np.shape(xy)).astype(np.float64)


def to_px(p, basename):
    """one floor point (cm) -> pixel position (x, y), for checking a keypoint against the video frame"""
    Hinv = np.linalg.inv(homography(basename))
    return cv2.perspectiveTransform(np.asarray(p, np.float32).reshape(1, 1, 2), Hinv)[0, 0]
