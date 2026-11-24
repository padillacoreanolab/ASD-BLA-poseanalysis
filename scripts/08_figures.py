"""Step 8 - overview figures (outputs/figures/). Plain matplotlib; the poster versions were drawn with the lab style in the old project.

  map_regions.png      cagemate map density with the 13 watershed regions numbered; C21 - saline density difference (cagemate)
  region_occupancy.png C21 - saline change in time per region, each mouse a dot, one panel per partner type
  sniffing.png         final SLEAP rule: anogenital seconds, face/body seconds, sniffing bouts; one line per mouse
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
from scipy.ndimage import gaussian_filter       # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import stats as ST                                   # noqa: E402
from c21pose.config import CACHE, FAMS, FIGS, META, TABLES        # noqa: E402

SAL, C21, INK, MUTED, GRID = "#6362EE", "#A9A9F9", "#1F2A44", "#5B6B8C", "#D5D9E0"   # lab poster colours (saline, C21)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED, "text.color": INK, "savefig.dpi": 200})
pfmt = lambda p: "p < .001" if p < .001 else "p = " + f"{p:.3f}".lstrip("0")


def map_figure():
    P = pickle.load(open(CACHE / "watershed.pkl", "rb")); D = pd.read_pickle(CACHE / "map_frames_regions.pkl"); cm = D[D.fam == "cagemate"]
    WS, xe, ye = P["WS"], P["xe"], P["ye"]; ext = [xe[0], xe[-1], ye[0], ye[-1]]
    names = pd.read_csv(META / "region_names_ws13.csv").set_index("region").name
    H = lambda g: gaussian_filter(np.histogram2d(g.u1, g.u2, bins=[xe, ye])[0], 3.0)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(H(cm).T, origin="lower", extent=ext, cmap="Greys", aspect="auto")
    edge = lambda c, a: np.ma.masked_where(~find_boundaries(WS, mode="inner").T, np.ones_like(WS.T, float))
    ax[0].imshow(edge(INK, .9), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([INK]), aspect="auto", interpolation="nearest")
    for r in range(P["nreg"]):
        c = cm[cm.region == r][["u1", "u2"]].median()
        ax[0].text(c.u1, c.u2, str(r), ha="center", va="center", fontsize=11, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax[0].set_title("Cagemate frames: 10-feature UMAP, 13 watershed regions", loc="left")
    # per-session normalised density, mean over sessions, C21 - saline
    dens = {d: np.mean([H(g) / len(g) for _, g in cm[cm.drug == d].groupby("basename")], 0) for d in ("SAL", "C21")}
    diff = (dens["C21"] - dens["SAL"]).T; v = np.abs(diff).max()
    im = ax[1].imshow(diff, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax[1].imshow(edge(MUTED, .6), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([MUTED]), alpha=.6, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax[1], label="C21 minus saline (share of frames)"); ax[1].set_title("Cagemate: where C21 shifts time (red = more under C21)", loc="left")
    for a in ax: a.set_xticks([]); a.set_yticks([]); a.set_xlabel("UMAP 1"); a.set_ylabel("UMAP 2")
    items = [f"{r} {n}" for r, n in names.items()]
    fig.text(.01, .005, "   ".join(items[:7]) + "\n" + "   ".join(items[7:]), fontsize=8, color=MUTED)
    plt.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(FIGS / "map_regions.png"); plt.close(fig)


def occupancy_figure():
    D = pd.read_pickle(CACHE / "map_frames_regions.pkl"); R = pd.read_csv(TABLES / "region_occupancy_tests.csv")
    names = pd.read_csv(META / "region_names_ws13.csv").set_index("region").name; NR = len(names)
    occ = D.groupby(["basename", "subject", "drug", "fam"]).region.value_counts(normalize=True).unstack(fill_value=0) \
           .reindex(columns=range(NR), fill_value=0).groupby(["subject", "fam", "drug"]).mean()
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True, gridspec_kw=dict(wspace=.28))
    for a, fam in zip(ax, FAMS):
        w = occ.xs(fam, level="fam"); d = (w.xs("C21", level="drug") - w.xs("SAL", level="drug")) * 100
        a.axvline(0, color=MUTED, lw=.8)
        for r in range(NR):
            a.scatter(d[r], np.full(len(d), r) + np.random.default_rng(r).uniform(-.15, .15, len(d)), s=16, color=SAL, alpha=.6, zorder=2)
            a.plot([d[r].mean()] * 2, [r - .3, r + .3], color=INK, lw=2.5, zorder=3)
            p = R[(R.fam == fam) & (R.region == r)].p.iloc[0]
            a.text(1.02, r, pfmt(p), transform=a.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, fontweight="bold" if p < .05 else "normal", clip_on=False)
        a.set_yticks(range(NR)); a.set_yticklabels([f"{r} {names[r]}" for r in range(NR)], fontsize=9); a.invert_yaxis()
        a.set_title(fam.capitalize(), loc="left"); a.set_xlabel("C21 minus saline (% of session), dot = mouse, bar = mean")
    fig.suptitle("Time in each region: paired t over 12 mice, uncorrected p (BH q in region_occupancy_tests.csv)", x=.01, ha="left")
    fig.subplots_adjust(left=.17, right=.95, top=.86, bottom=.1); fig.savefig(FIGS / "region_occupancy.png"); plt.close(fig)


def sniffing_figure():
    A = pd.read_csv(TABLES / "sniff_sessions.csv")
    M = [("final_ag_s", "Anogenital sniffing (s)"), ("final_facebody_s", "Face/body sniffing (s)"), ("final_all_bouts", "Sniffing bouts (n)")]
    fig, ax = plt.subplots(len(M), 3, figsize=(11, 10), sharey="row")
    for i, (k, lab) in enumerate(M):
        for j, fam in enumerate(FAMS):
            a = ax[i, j]; w = ST.wide(A, k, fam); r = ST.paired_row(w)
            for _, x in w.iterrows(): a.plot([0, 1], [x.SAL, x.C21], color=GRID, lw=1, zorder=1)
            a.scatter(np.zeros(len(w)), w.SAL, color=SAL, s=28, zorder=2); a.scatter(np.ones(len(w)), w.C21, color=C21, s=28, zorder=2, edgecolor=SAL, lw=.6)
            a.plot([-.15, .15], [w.SAL.mean()] * 2, color=INK, lw=2.5); a.plot([.85, 1.15], [w.C21.mean()] * 2, color=INK, lw=2.5)
            a.set_xticks([0, 1]); a.set_xticklabels(["Saline", "C21"]); a.set_xlim(-.5, 1.5)
            a.text(.5, .97, f"{pfmt(r['p'])}, {r['up']}/{r['n']} up", transform=a.transAxes, ha="center", va="top", fontsize=9.5)
            if i == 0: a.set_title(fam.capitalize())
            if j == 0: a.set_ylabel(lab)
    fig.suptitle("Final SLEAP sniffing rule, first 600 s; paired t, uncorrected", x=.01, ha="left")
    plt.tight_layout(); fig.savefig(FIGS / "sniffing.png"); plt.close(fig)


if __name__ == "__main__":
    map_figure(); occupancy_figure(); sniffing_figure(); print("figures ->", FIGS)
