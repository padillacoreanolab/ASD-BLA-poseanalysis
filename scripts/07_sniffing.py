"""Step 7 - SLEAP sniffing per session: the 2 cm rule and the final rule; C21 vs saline per partner type.

Outputs (tables/)
  sniff_sessions.csv        per session (first 600 s): seconds, bout count and mean bout length for
                              rule2cm_ag, rule2cm_face, rule2cm_body, rule2cm_facebody   (plain 2 cm rule)
                              final_ag (= rule2cm_ag), final_facebody, final_all (union of the two final masks)
                            plus BORIS seconds / bouts when paths.boris_session_table is set
  sniff_final_rule_cutoffs.csv   the face/body cutoffs used for each held-out mouse
  sniff_tests.csv           paired t over mice, per partner type and measure (uncorrected, as on the poster)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import sessions, sniffing as SN, stats as ST            # noqa: E402
from c21pose.config import BORIS_SESSION_TABLE, FAMS, TABLES          # noqa: E402

if __name__ == "__main__":
    SES = SN.prepare_sessions(sessions.load("sessions_guarded"))
    F = {s["basename"]: SN.frame_geometry(s) for s in SES}
    FB, PAR = SN.final_rule(F)
    pd.DataFrame(PAR).to_csv(TABLES / "sniff_final_rule_cutoffs.csv", index=False)
    rows = []
    for b, f in F.items():
        s, fps, o = f["s"], f["fps"], f["old"]
        r = dict(session=b, mouse=s["subject"], fam=s["fam"], drug=s["drug"], scored=s["scored"])
        masks = {"rule2cm_ag": o["ag"], "rule2cm_face": o["face"], "rule2cm_body": o["body"], "rule2cm_facebody": o["face"] | o["body"],
                 "final_ag": o["ag"], "final_facebody": FB[b], "final_all": o["ag"] | FB[b]}
        for k, m in masks.items():
            r[f"{k}_bouts"], r[f"{k}_meanbout"], r[f"{k}_s"] = SN.bouts(m, fps)
        rows.append(r)
    A = pd.DataFrame(rows)
    measures = [c for c in A.columns if c.endswith(("_s", "_bouts", "_meanbout"))]
    if BORIS_SESSION_TABLE and BORIS_SESSION_TABLE.exists():
        Bo = pd.read_csv(BORIS_SESSION_TABLE); cls = {"anogenital sniffing": "ag", "face sniffing": "facebody", "body sniffing": "facebody"}
        bw = Bo.assign(c=Bo.beh.map(cls)).dropna(subset=["c"]).groupby(["mouse", "fam", "drug", "c"])[["dur", "bouts"]].sum().unstack("c")
        for c in ("ag", "facebody"):
            A = A.merge(pd.DataFrame({f"boris_{c}_s": bw[("dur", c)], f"boris_{c}_bouts": bw[("bouts", c)]}).reset_index(), on=["mouse", "fam", "drug"], how="left")
        measures += ["boris_ag_s", "boris_facebody_s", "boris_ag_bouts", "boris_facebody_bouts"]
        for k in ("rule2cm_ag_s", "final_facebody_s", "rule2cm_facebody_s"):
            c = "boris_ag_s" if "ag" in k else "boris_facebody_s"; m = A[c].notna()
            print(f"{k} vs {c}: Spearman rho = {stats.spearmanr(A.loc[m, k], A.loc[m, c])[0]:.2f} (n = {m.sum()} sessions)")
    A.to_csv(TABLES / "sniff_sessions.csv", index=False)
    E = pd.DataFrame([dict(measure=k, fam=fam, **ST.paired_row(ST.wide(A, k, fam))) for k in measures for fam in FAMS])
    E.to_csv(TABLES / "sniff_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(E[E.measure.isin(["final_ag_s", "final_facebody_s", "final_all_bouts"])].round(3).to_string(index=False))
