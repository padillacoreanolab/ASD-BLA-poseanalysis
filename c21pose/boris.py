"""BORIS hand scoring -> per-frame labels for the treated mouse. From borisio.py + dyadstudy.normalise (Sep 2026).

Each BORIS aggregated-event export row is one bout (Subject, Behavior, Start/Stop in s of the media file). Exports are matched
to the 72 SLEAP sessions by the normalised media name, then by ID tokens, then by the BORIS file NAME (some exports carry stale
media names). Only the treated mouse's bouts are kept (Subject "11.1" for E11.1).
Frame labels: frames round(start * fps) to round(stop * fps) (end exclusive), at the video's true fps.
BORIS labels are used ONLY to describe map regions and to choose the face/body sniffing cutoffs; never as a drug outcome here.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from .config import BORIS_ROOT

# sub-folders of BORIS_ROOT searched for exports (missing ones are skipped)
DIRS = ["ecohort_boris", "boris", "BORIS E7.1_DAY 1", "BORIS E7.1_DAY 2", "BORIS E8.1 DAY 1", "BORIS E8.1_DAY 2", "BORIS E8.2 DAY1",
        "BORIS E8.2 DAY2", "BORIS E9.1_DAY 2", "BORIS E9.1_DAY1", "BORIS E11 (1).1_DAY2", "BORIS E11.1 DAY 1", "boris 5.1 day 2",
        "boris 5.1_day 1", "boris e6.1_day 1", "boris e6.1_day2", "boris E1.1_Day2", "boris e1.1_day1", "boris e2.1_day1",
        "boris e2.1_day2", "day 1 - 10.1", "day 1-7.2"]
BEHAVIORS = {"anogenital sniffing": "ag", "face sniffing": "face", "body sniffing": "body", "selfgrooming": "selfgroom",
             "allogrooming": "allogroom"}


def normalise(stem):
    """SLEAP/video stem -> upper-case token string (drops the '.000_<copy>' shard echo, 'copy'/'clean', '(2)')"""
    s = re.sub(r"\.\d{3}_.*$", "", stem)
    s = re.sub(r"\.(analysis|predictions|slp|mp4|avi|h5)", "", s, flags=re.I)
    s = re.sub(r"\b(copy|clean|correcter|corrected)\b|\(\d+\)", " ", s, flags=re.I)
    s = s.upper().replace("_", " ").replace("-", " ")
    s = re.sub(r"\bD(\d)(?=[A-Z])", r"D\1 ", s)
    return re.sub(r"\s+", " ", s).strip()


def _toks(n):
    t = [x for x in normalise(n).split() if re.search(r"\d", x) and not re.fullmatch(r"D\d", x)]
    return frozenset(re.sub(r"^D\d(?=[A-Z])", "", x) for x in t)


def _pair_raw(x):
    """older BORIS format (START/STOP rows) -> one row per bout"""
    if not {"Time", "Behavior", "Status", "Subject"} <= set(x.columns):
        return None
    rows = []
    media = x["Media file name"].iloc[0] if "Media file name" in x.columns else ""
    for (sub, beh), g in x.sort_values("Time").groupby(["Subject", "Behavior"]):
        t = g[g.Status.isin(["START", "STOP"])]
        st = t[t.Status == "START"].Time.to_numpy(); sp = t[t.Status == "STOP"].Time.to_numpy()
        for a, b in zip(st, sp):
            rows.append({"Subject": sub, "Behavior": beh, "Behavior type": "STATE", "Start (s)": a, "Stop (s)": b,
                         "Duration (s)": b - a, "Media file name": media, "FPS (frame/s)": np.nan})
    return pd.DataFrame(rows) if rows else None


def load_events(root=BORIS_ROOT):
    """all non-empty exports -> one DataFrame of bouts with a normalised media key"""
    rows = []
    for d in DIRS:
        for f in sorted(glob.glob(os.path.join(root, d, "**", "*.csv"), recursive=True)):
            if os.path.getsize(f) == 0:
                continue
            x = pd.read_csv(f)
            if "Start (s)" not in x.columns:
                x = _pair_raw(x)
                if x is None:
                    continue
            x = x[x["Behavior type"].isin(["STATE"])].copy()
            x["file"] = os.path.relpath(f, root)
            x["media"] = x["Media file name"].map(lambda p: os.path.basename(str(p).replace("\\", "/")))
            rows.append(x[["file", "media", "Subject", "Behavior", "Start (s)", "Stop (s)", "Duration (s)", "FPS (frame/s)"]])
    E = pd.concat(rows, ignore_index=True).rename(columns={"Start (s)": "start", "Stop (s)": "stop", "Duration (s)": "dur",
                                                          "FPS (frame/s)": "fps_boris"})
    E["Subject"] = E["Subject"].astype(str).str.lower().str.replace("e", "", n=1, regex=False)
    E["key"] = E.media.map(normalise)
    E["tok"] = E.media.map(_toks)
    return E


def _fname_toks(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    t = normalise(stem).split()
    ids = {re.sub(r"^E(?=\d)", "", x) for x in t if re.search(r"\d", x) and not re.fullmatch(r"D(AY)?\d", x)}
    day = None
    for x in t:
        m = re.fullmatch(r"D(?:AY)?(\d)", x)
        if m:
            day = int(m.group(1))
    return ids, day


def attach(design, E=None):
    """{basename: DataFrame of the treated mouse's bouts}, plus a coverage table (how each session was matched)"""
    E = load_events() if E is None else E
    out, cov = {}, []
    for r in design.itertuples():
        m = E[E.key == r.norm]
        how = "norm"
        if m.empty:
            m = E[E.tok == _toks(r.basename)]
            how = "tokens"
        if not m.empty:                       # a file named for the other day cannot belong to this session
            m = m[[(_fname_toks(f)[1] in (None, r.day)) for f in m.file]]
        if m.empty:                           # fall back on the BORIS file NAME
            want = {re.sub(r"^E(?=\d)", "", t) for t in _toks(r.basename)}
            ok = [f for f in E.file.unique() if _fname_toks(f)[0] >= want and _fname_toks(f)[1] in (None, r.day)]
            m = E[E.file.isin(ok)]
            how = "filename"
        subj = r.subject.lower().replace("e", "", 1)
        f = m[m.Subject == subj].drop_duplicates(["Behavior", "start", "stop"])
        out[r.basename] = f
        cov.append(dict(basename=r.basename, subject=r.subject, drug=r.drug, familiarity=r.familiarity, matched=not m.empty,
                        how=how if not m.empty else "", files=";".join(sorted(m.file.unique())), n_bouts=len(f)))
    return out, pd.DataFrame(cov)


def frame_labels(bouts, n, fps):
    """{ag, face, body, selfgroom, allogroom: bool array of n frames}"""
    lab = {}
    for beh, k in BEHAVIORS.items():
        m = np.zeros(n, bool)
        for a, b in bouts[bouts.Behavior.str.lower() == beh][["start", "stop"]].to_numpy():
            m[int(round(a * fps)):int(round(b * fps))] = True
        lab[k] = m
    return lab
