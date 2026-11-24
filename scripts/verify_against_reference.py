"""Check this pipeline's outputs against the Nov 2026 project files (paths.reference_dir = the old outputs_cm/ folder).

Run after scripts 01-07. Every line should say OK. Last full check: 24 Nov 2026, all OK (macOS arm64, versions in requirements.txt).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import features_pose as FP, verify as V                 # noqa: E402
from c21pose.config import CACHE, REFERENCE_DIR as R, TABLES         # noqa: E402

ok_all = True


def report(name, good, detail=""):
    global ok_all
    ok_all &= bool(good); print(f"[{'OK' if good else 'DIFF'}] {name} {detail}")


if __name__ == "__main__":
    b = V.sessions("sessions_raw", "sessions_xy"); report("01 sessions (raw skeletons, BORIS labels)", not b, b[:3])
    b = V.sessions("sessions_guarded", "sessions_xy_guarded")
    report("02 repaired skeletons", all(x[1].startswith("max |coord") and x[2] < 1e-9 for x in b), b[:3])
    f = V.frames("simple_frames", "simple_frames", ["dist_cc", "dist_nn", "s_nose_p_min", "p_nose_s_min", "s_nose_p_butt", "p_nose_s_butt", "s_speed", "p_speed", "ok"])
    report("03 simple frame features", all(v == 0.0 for k, v in f.items() if k != "rows"))
    f = V.frames("pose_frames", "pose_frames", FP.FEATS + FP.LABELS)
    report("03 pose frame features + labels", all(v == "equal" or (isinstance(v, float) and v < 1e-6) for k, v in f.items() if k != "rows"))
    N = pd.read_pickle(CACHE / "map_frames.pkl"); A = pd.read_pickle(R / "cache/umap10_allfam_D.pkl")
    report("04 UMAP map coordinates", np.array_equal(N[["u1", "u2"]].to_numpy(float), A[["u1", "u2"]].to_numpy(float)))
    N = pd.read_pickle(CACHE / "map_frames_regions.pkl"); A = pd.read_pickle(R / "cache/umap_ws_allfam_D.pkl")
    p, q = pickle.load(open(CACHE / "watershed.pkl", "rb")), pickle.load(open(R / "cache/paper_panels_data_ws.pkl", "rb"))
    report("05 watershed regions + frame sequences", (N.region.to_numpy() == A.region.to_numpy()).all() and np.array_equal(p["WS"], q["WS"])
           and all(np.array_equal(p["seq"][k]["seq"], q["seq"][k]["seq"]) for k in q["seq"]))
    t = V.table("region_occupancy_tests.csv", "boris_corrected/paper_panels_ws_region_tests.csv", ["fam", "region"], ["SAL", "C21", "p", "q", "up"])
    report("06 region occupancy tests", t.pop("unmatched rows") == 0 and max(t.values()) < 1e-12, t)
    for a, c in (("transition_matrix", "paper_panels_ws_transition_matrix"), ("transition_C21_minus_SAL_p", "paper_panels_ws_transition_C21_minus_SAL_p")):
        x = pd.read_csv(TABLES / f"{a}.csv", index_col=0).to_numpy(); y = pd.read_csv(R / f"boris_corrected/{c}.csv", index_col=0).to_numpy()
        report(f"06 {a}", np.abs(x - y).max() < 1e-12)
    S = pd.read_csv(TABLES / "sniff_sessions.csv").merge(pd.read_csv(R / "boris_corrected/sleap_new_vs_old_sessions.csv"), on="session")
    report("07 sniffing seconds (2 cm rule, final face/body rule)", max(np.abs(S.rule2cm_ag_s - S.old_Anogenital).max(), np.abs(S.rule2cm_facebody_s - S.old_Combined).max(),
                                                                       np.abs(S.final_facebody_s - S.new_Combined).max()) < 1e-9)
    B = pd.read_csv(TABLES / "sniff_sessions.csv").merge(pd.read_csv(R / "boris_corrected/sniff_bouts_sessions.csv"), on="session")
    report("07 sniffing bouts", (B.final_all_bouts == B.sleap_all_bouts).all() and (B.final_facebody_bouts == B.sleap_fb_bouts).all())
    T = pd.read_csv(TABLES / "sniff_tests.csv"); F = pd.read_csv(R / "boris_corrected/sleap_final_tests.csv")
    mp = {("final", "Anogenital"): "final_ag_s", ("final", "Combined"): "final_facebody_s", ("boris", "Anogenital"): "boris_ag_s", ("boris", "Combined"): "boris_facebody_s"}
    F["measure"] = [mp[(v, c)] for v, c in zip(F.version, F.cls)]; x = F.merge(T, on=["measure", "fam"], suffixes=("_ref", ""))
    report("07 final sniffing tests (poster panel F)", len(x) == len(F) and np.abs(x.p - x.p_ref).max() < 1e-12)
    print("\nALL OK" if ok_all else "\nSOME CHECKS DIFFER")
    sys.exit(0 if ok_all else 1)
