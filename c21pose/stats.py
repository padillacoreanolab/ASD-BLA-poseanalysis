"""Paired C21 vs saline tests. The unit is the MOUSE: each of the 12 mice gives one saline and one C21 value per partner type."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import FAMS


def paired_contrasts(D, feats, mouse="subject", fam="fam", drug="drug", groups=FAMS):
    """per partner type and feature: saline and C21 means over mice, C21 - saline, dz, paired t p, mice up; BH within each partner type"""
    res = []
    for grp in groups:
        w = D[D[fam] == grp].groupby([mouse, drug])[feats].mean().unstack(drug)
        for f in feats:
            a, b = w[(f, "C21")], w[(f, "SAL")]; m = a.notna() & b.notna(); dd = a[m] - b[m]
            res.append(dict(group=grp, feature=f, n=int(m.sum()), SAL=b[m].mean(), C21=a[m].mean(), diff=dd.mean(),
                            dz=dd.mean() / dd.std(ddof=1), p=stats.ttest_rel(a[m], b[m]).pvalue, n_up=int((dd > 0).sum())))
    R = pd.DataFrame(res)
    R["q"] = R.groupby("group").p.transform(lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
    return R


def wide(D, value, fam, mouse="mouse", fam_col="fam", drug="drug"):
    """mice x [SAL, C21] for one partner type, mice missing either session dropped"""
    return D[D[fam_col] == fam].pivot_table(index=mouse, columns=drug, values=value)[["SAL", "C21"]].dropna()


def paired_row(w):
    d = w.C21 - w.SAL
    return dict(n=len(w), SAL=w.SAL.mean(), C21=w.C21.mean(), p=stats.ttest_rel(w.C21, w.SAL).pvalue,
                dz=d.mean() / d.std(ddof=1), up=int((d > 0).sum()))
