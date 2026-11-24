# Notes to self

Things the code doesn't explain. Last updated 24 Nov 2026.

## Where things live (my laptop)

| What | Path |
|---|---|
| SLEAP h5 (cleaned) | `~/ecohortanalysis/cleaned_h5files/`, `~/ecohortanalysis/h5 files cleaned predictions ecohort/` |
| BORIS exports | `~/Downloads/Research/BORIS Behavior Scoring/` (mostly `ecohort_boris/`) |
| Videos (720p) | Dropbox `Padilla-Coreano Lab/2026/ASD Mice/<cohort>/...`. Online-only until "Make available offline" is set in Finder, and **not on this laptop as of 24 Nov 2026** |
| Old project (all exploratory scripts, decks, poster) | `~/Downloads/Research/BLA C21 Social Analysis (code & results)/` |
| Calibration click tool | `~/Downloads/Research/Rig Design/cage_calibration/` (`server.py` + `index.html`, port 8765) |
| Python env used | `/opt/miniconda3/envs/cai5735` (it is also my course env, which is why the lock file has extra packages) |

## Identity rules (design table)

- The file name is `SUBJECT_AGENT`: the first ID is the treated subject.
- Partner prefix: F / FM / FF / FAMILIAR = familiar; N / NM / NF / NOVEL = novel. The same cage = cagemate.
- In raw inference h5 files, `track_0` is the subject. Hand-proofread copies (`clean` / `copy` / `correcter`) carry their own track names, and those override track_0. `subject_track` in the design table already has the result.
- 38 sessions are `assumed:track_0` (column `identity_source`).

## Data problems to remember

- **Frame rates.** The three E1.1 day-1 videos are 24 fps, not 30. Always use `calibration.fps_for`.
- **Window.** The common window is 600.58 s (the shortest session). Session records keep round(600.58 × fps) frames, and the feature code uses the first int(600 × fps).
- **Cage width.** The 5.6 in width is assumed, never measured. The length (14 in) is real.
- **BORIS wrong-video exports.** Two exports scored the wrong video. `E8.1_vs_FM3.2_Day2.csv` names the day-1 video; `E9.1_VS_FM2.2_DAY2 (1).csv` scored the novel day-2 video. On 9 Nov I decided to count both by file name (72/72 in `boris_600s.csv`). The frame labels in this repo come from `boris.attach`, which matches by video name first. These labels only describe regions and set the face/body cutoffs; they are never a drug outcome.
- **E2.1_N2_D1.** This session got a BORIS export on 8 Nov, after the pose caches were built (21 Oct). To reproduce the poster, it is listed in `[boris] exclude_sessions`. Removing it from that list adds its labels. That can change the face/body cutoff selection and the BORIS columns of the region profiles; I haven't run it that way. Features, the map and the regions don't use BORIS, so they don't change.
- **Old figures.** BORIS plots made before 27 Oct use a stale table (`subject_sum_check.csv`), so don't reuse them.

## How the old scripts map to this repo

The old pipeline had no script that built `sessions_xy.pkl` (it was a one-off). `01_build_sessions.py` reconstructs it bit for bit. The match needs three details:

- the cm conversion runs in **float32 through OpenCV**, as `cagecal.py` did;
- each session is trimmed to the window **before** gap filling;
- BORIS frames run from `round(start × fps)` to `round(stop × fps)`.

| Old script (project folder) | Now |
|---|---|
| dyadlib.py (read_sleap, swap_audit, blank_around, fill_gaps) | `c21pose/sleapio.py` |
| dyadcal.py, cagecal.py | `c21pose/calibration.py` |
| borisio.py, dyadstudy.normalise | `c21pose/boris.py` |
| (lost one-off) → sessions_xy.pkl | `scripts/01_build_sessions.py` |
| pose_repair.py, pose_repair_guarded.py | `c21pose/repair.py`, `scripts/02_repair_poses.py` |
| simple_features.py | `c21pose/features_simple.py` |
| pose_features.py | `c21pose/features_pose.py` |
| cagemate_umap.py (frame draw), umap10_cagemate.py, umap10_all_partners.py (part A) | `c21pose/bmap.py`, `scripts/04_fit_map.py` |
| paper_panels_data_ws.py | `c21pose/watershed.py`, `scripts/05_watershed_regions.py` |
| ws_stats.py, region_profiles.py, region_defs.py (ws names) | `scripts/06_region_stats.py`, `metadata/region_names_ws13.csv` |
| boris_lda_addon_features.targets, sleap_new_vs_old.py, sniff_ag_ablation.py, sniff_bouts.py | `c21pose/sniffing.py`, `scripts/07_sniffing.py` |

**Dropped on purpose:**

- the k-means (k = 8) regions of the same map; the poster uses watershed;
- `pose_umap.test_regions` / `pose_names` (rule-based region names);
- the 21-feature map;
- the HMMs, PCA/LDA and Daiyao-feature analyses.

All of these are still in the old project.

## Decisions (and why)

- **Map = 10 pose features only (11 Nov).** Adding angle, orientation, body length or IoU (21 features) diluted every region effect at every k, because those features don't change with C21. The map describes behaviour; the drug is tested on measures directly (sniffing, PCA, LDA, angle, laterality).
- **Final sniffing rule (10 Nov).**
  - A 0.5 s minimum bout removed the anogenital C21 effect (62% of anogenital bouts are < 0.5 s), so anogenital stays on the plain 2 cm rule.
  - The face/body rule was chosen after seeing p values, so call it **refined post hoc**.
  - Its session-level ρ with BORIS face + body seconds is 0.05 (the plain 2 cm rule gives 0.27). It catches C21 changes, but don't call it a validated face/body detector.
- **BORIS lags SLEAP contact by about 0.5 s**, so the cutoff search moves BORIS 0.5 s earlier.
- **Stats.**
  - The poster panels B/C use a two-way RM ANOVA (drug × partner; `poster_rm_anova.py`, not ported).
  - Occupancy fails sphericity (GG ε ≈ .19), so region occupancy stays paired t plus a permutation omnibus.
  - Cagemate is the main focus. Familiar and novel are reported but secondary.

## Results this repo reproduces (paired t, 12 mice)

- **Final anogenital sniffing:** cagemate 10.7 → 16.3 s, p = .046; novel 12.2 → 18.0 s, p = .030.
- **Final face/body sniffing:** familiar 13.5 → 20.5 s, p = .002.
- **Sniffing bouts (all):** cagemate p = .013, novel p = .021. C21 raises the bout count, not the bout length.
- **Map regions, cagemate:**
  - 10 "Subject follows and sniffs anogenital": p = .004, q = .058 (the only one near BH);
  - 12 "Fast movement near partner": p = .020;
  - 11 "Apart, subject still" is down: p = .039.

## Open items

- A second BORIS scorer re-scoring about 10 sessions, to get face/body reliability (kit in the old project: `boris_interrater.py`).
- If the videos come back: run `02_repair_poses.py --mode video` once and diff the new decisions against `metadata/pose_repair_decisions.csv.gz`.
- Not yet ported: angle / laterality panels (`poster_angle_laterality.py`), PCA / LDA on session features (`pca_focused.py`, `lda_pose.py`), the RM ANOVA. These are the next candidates if the lab wants the whole poster from this repo.
- Not yet done: a pre-commit check that `config.toml` and `outputs/` never get committed (`.gitignore` covers them).
