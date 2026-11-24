"""Step 6 - C21 vs saline on the watershed regions, transitions, and what each region contains (ws_stats.py + region_profiles.py).

Occupancy   per session, share of its sampled frames in each region; mean per mouse; paired t over 12 mice; BH across the 13
            regions, separately for cagemate, familiar and novel.
Transitions cagemate only. Per session: region bouts >= 0.2 s from the per-frame sequence, consecutive repeats merged; pooled
            row-normalised transition matrix; C21 - saline difference tested by swapping drug labels within mouse (5,000 draws,
            seed 1, two-sided).
Profiles    per region (cagemate frames): distances, speeds, % approaching / leaving (> 3 cm/s along the line between centers,
            +-0.2 s), % facing (< 45 deg), % with a SLEAP sniff label, BORIS rates on scored sessions, rearing / hunching.
Outputs (tables/): region_occupancy_tests.csv, transition_matrix.csv, transition_C21_minus_SAL.csv, transition_C21_minus_SAL_p.csv,
                   region_profiles.csv
"""
import argparse
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import sessions                       # noqa: E402
from c21pose.config import CACHE, FAMS, META, TABLES  # noqa: E402


def occupancy_tests(D, NR):
    occ = D.groupby(["basename", "subject", "drug", "fam"]).region.value_counts(normalize=True).unstack(fill_value=0) \
           .reindex(columns=range(NR), fill_value=0).groupby(["subject", "fam", "drug"]).mean()
    RT = []
    for fam in FAMS:
        w = occ.xs(fam, level="fam"); a, b = w.xs("SAL", level="drug"), w.xs("C21", level="drug"); res = []
        for r in range(NR):
            d = b[r] - a[r]
            res.append(dict(fam=fam, region=r, SAL=a[r].mean(), C21=b[r].mean(), p=stats.ttest_rel(b[r], a[r]).pvalue if d.std() > 0 else np.nan, up=int((d > 0).sum())))
        R = pd.DataFrame(res); R["q"] = multipletests(R.p.fillna(1), method="fdr_bh")[1]; RT.append(R)
    return pd.concat(RT)


def region_bouts(SEQ, min_s=.2):
    out = []
    for b, s in SEQ.items():
        q = s["seq"]; i = 0; seq = []
        while i < len(q):
            j = i
            while j < len(q) and q[j] == q[i]: j += 1
            if q[i] >= 0 and (j - i) / s["fps"] >= min_s: seq.append(int(q[i]))
            i = j
        comp = [seq[0]] if seq else []
        for r in seq[1:]:
            if r != comp[-1]: comp.append(r)
        out.append(dict(subject=s["subject"], drug=s["drug"], bouts=np.array(comp)))
    return out


def transitions(BOUTS, NR, n_perm=5000, seed=1):
    def counts(bl):
        M = np.zeros((NR, NR))
        for x in bl:
            for a, b in zip(x[:-1], x[1:]): M[a, b] += 1
        return M
    rowp = lambda M: M / np.maximum(M.sum(1, keepdims=True), 1)
    Pm = rowp(counts([x["bouts"] for x in BOUTS])); mice = sorted({x["subject"] for x in BOUTS}); byk = {(x["subject"], x["drug"]): x["bouts"] for x in BOUTS}
    def pdiff(flip): return rowp(counts([byk[(m, "SAL" if f else "C21")] for m, f in zip(mice, flip)])) - rowp(counts([byk[(m, "C21" if f else "SAL")] for m, f in zip(mice, flip)]))
    dobs = pdiff(np.zeros(len(mice), bool)); rng = np.random.default_rng(seed)
    nd = np.array([pdiff(rng.random(len(mice)) < .5) for _ in range(n_perm)]); pdv = (np.abs(nd) >= np.abs(dobs)[None] - 1e-12).mean(0)
    return Pm, dobs, pdv


def profiles(D):
    D = D[D.fam == "cagemate"].copy(); SESS = {x["basename"]: x for x in sessions.load("sessions_guarded")}
    sa, pa = np.full(len(D), np.nan), np.full(len(D), np.nan)
    for b, g in D.groupby("basename"):
        x = SESS[b]; c = x["ni"]["center"]; S, P = x["S"][:, c].astype(float), x["P"][:, c].astype(float); k = max(int(round(.2 * x["fps"])), 1)
        fi = g.fi.to_numpy(); ok = (fi - k >= 0) & (fi + k < len(S)); i = np.flatnonzero(ok); f = fi[ok]
        u = P[f] - S[f]; u /= np.linalg.norm(u, axis=1, keepdims=True); vS = (S[f + k] - S[f - k]) * x["fps"] / (2 * k); vP = (P[f + k] - P[f - k]) * x["fps"] / (2 * k)
        pos = D.index.get_indexer(g.index[i]); sa[pos] = (vS * u).sum(1); pa[pos] = (vP * -u).sum(1)
    D["subj_approach"], D["part_approach"] = sa, pa
    for w, p in (("s", "s"), ("i", "p")): D[f"{p}_sniff_any"] = D[[f"{w}_sniff_face", f"{w}_sniff_body", f"{w}_sniff_ag"]].astype(float).max(axis=1)
    sc = D.scored.astype(bool); G = D.groupby("region"); bG = D[sc].groupby("region")
    pct = lambda col, f: G[col].apply(lambda v: f(v).mean() * 100)
    R = pd.DataFrame({"share_%": G.size() / len(D) * 100, "dist_cm": G.dist_cc.median(), "subj_nose_cm": G.s_nose_p_min.median(), "part_nose_cm": G.p_nose_s_min.median(),
                      "subj_speed": G.s_speed.median(), "part_speed": G.p_speed.median(),
                      "subj_approaching_%": pct("subj_approach", lambda v: v > 3), "subj_leaving_%": pct("subj_approach", lambda v: v < -3),
                      "part_approaching_%": pct("part_approach", lambda v: v > 3), "part_leaving_%": pct("part_approach", lambda v: v < -3),
                      "subj_facing_%": pct("s_facing", lambda v: v < 45), "part_facing_%": pct("p_facing", lambda v: v < 45),
                      "subj_sniffs_%": G.s_sniff_any.mean() * 100, "part_sniffs_%": G.p_sniff_any.mean() * 100,
                      "BORIS_ag_%": bG.b_ag.mean() * 100, "BORIS_face_%": bG.b_face.mean() * 100, "BORIS_body_%": bG.b_body.mean() * 100, "BORIS_selfgroom_%": bG.b_selfgroom.mean() * 100,
                      "subj_rear_%": G.s_rear.mean() * 100, "subj_hunch_%": G.s_hunch.mean() * 100})
    return R.sort_values("dist_cm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--peaks", type=int, default=13); a = ap.parse_args()
    tag = "" if a.peaks == 13 else f"_ws{a.peaks}"
    P = pickle.load(open(CACHE / f"watershed{tag}.pkl", "rb")); NR = P["nreg"]; D = pd.read_pickle(CACHE / f"map_frames_regions{tag}.pkl")
    names = pd.read_csv(META / "region_names_ws13.csv").set_index("region") if not tag else None
    RT = occupancy_tests(D, NR)
    if names is not None: RT = RT.join(names, on="region")
    RT.to_csv(TABLES / f"region_occupancy_tests{tag}.csv", index=False)
    Pm, dobs, pdv = transitions(region_bouts(P["seq"]), NR)
    pd.DataFrame(Pm).to_csv(TABLES / f"transition_matrix{tag}.csv"); pd.DataFrame(dobs).to_csv(TABLES / f"transition_C21_minus_SAL{tag}.csv")
    pd.DataFrame(pdv).to_csv(TABLES / f"transition_C21_minus_SAL_p{tag}.csv")
    R = profiles(D)
    if names is not None: R = R.join(names)
    R.to_csv(TABLES / f"region_profiles{tag}.csv")
    pd.set_option("display.width", 220)
    print(RT[RT.p < .1].round(3).to_string(index=False))
    off = ~np.eye(NR, dtype=bool); common = (Pm >= .02) & off
    print(f"transitions: {int(((pdv < .05) & common).sum())} of {int(common.sum())} common transitions p < .05 (expected {.05 * common.sum():.1f})")
