"""Step 1 - raw SLEAP files -> calibrated session records (outputs/cache/sessions_raw.pkl).

Reads the 72 .analysis.h5 files listed in metadata/design_table.csv, removes brief identity swaps, converts to floor cm, fills
gaps <= 15 frames, trims every session to the common 600.58 s, and attaches the BORIS frame labels of the treated mouse.
Also writes outputs/tables/boris_coverage.csv (which BORIS export matched each session, and how).
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from c21pose import sessions                      # noqa: E402
from c21pose.config import CACHE, TABLES          # noqa: E402

if __name__ == "__main__":
    S, cov, win = sessions.build_all()
    pickle.dump(S, open(CACHE / "sessions_raw.pkl", "wb"), protocol=4)
    cov.to_csv(TABLES / "boris_coverage.csv", index=False)
    print(f"{len(S)} sessions, common window {win:.2f} s, {sum(x['scored'] for x in S)} with BORIS labels -> {CACHE / 'sessions_raw.pkl'}")
