"""Checks of this pipeline's outputs against the Nov 2026 project files (config: paths.reference_dir = old outputs_cm/)."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .config import CACHE, REFERENCE_DIR, TABLES


def _ref(rel):
    if REFERENCE_DIR is None:
        raise RuntimeError("set paths.reference_dir in config.toml")
    return REFERENCE_DIR / rel


def sessions(new_name, ref_name):
    """exact comparison of two session caches; returns list of problems (empty = identical)"""
    ref = {x["basename"]: x for x in pickle.load(open(_ref(f"cache/{ref_name}.pkl"), "rb"))}
    new = pickle.load(open(CACHE / f"{new_name}.pkl", "rb"))
    bad, mx = [], 0.0
    for x in new:
        r = ref[x["basename"]]
        for k in ("subject", "drug", "fam", "day", "n", "scored"):
            if x[k] != r[k]:
                bad.append((x["basename"], k, x[k], r[k]))
        for k in ("S", "P"):
            a, b = x[k], r[k]
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                bad.append((x["basename"], k, "missing-value pattern"))
            mx = max(mx, float(np.nanmax(np.abs(a - b))))
        if not np.array_equal(x["both"], np.asarray(r["both"])):
            bad.append((x["basename"], "both", int((x["both"] != r["both"]).sum())))
        for k in r["lab"]:
            if not np.array_equal(x["lab"][k], np.asarray(r["lab"][k])):
                bad.append((x["basename"], "lab", k))
    if mx > 0:
        bad.append(("all", "max |coordinate difference| cm", mx))
    return bad


def frames(new_name, ref_name, cols, atol=0.0):
    """compare frame tables column by column (float32 values); returns {col: max abs diff or 'missing pattern'}"""
    a = pd.read_pickle(CACHE / f"{new_name}.pkl"); b = pd.read_pickle(_ref(f"cache/{ref_name}.pkl"))
    out = {"rows": (len(a), len(b))}
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fc":
            ok = np.array_equal(np.isnan(x), np.isnan(y))
            out[c] = float(np.nanmax(np.abs(x - y))) if ok else "missing pattern differs"
        else:
            out[c] = "equal" if np.array_equal(x, y) else f"{int((x != y).sum())} differ"
    return out


def table(new_rel, ref_rel, keys, cols, rtol=1e-9):
    """compare two CSV tables on key columns; returns max abs difference per column"""
    a = pd.read_csv(TABLES / new_rel); b = pd.read_csv(_ref(ref_rel))
    m = a.merge(b, on=keys, suffixes=("", "_ref"), how="outer", indicator=True)
    out = {"unmatched rows": int((m._merge != "both").sum())}
    for c in cols:
        out[c] = float(np.nanmax(np.abs(m[c] - m[c + "_ref"]))) if len(m) else np.nan
    return out
