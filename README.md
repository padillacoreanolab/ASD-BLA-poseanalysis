# BLA C21 dyad pose pipeline

This pipeline takes SLEAP pose tracking of freely moving mouse pairs and produces three things:

- per-frame pose features;
- a 2 cm nose-contact sniffing rule;
- a 10-feature UMAP behaviour map, split into 13 watershed regions.

Each part ends with a saline vs C21 comparison. This is the code behind the BLA C21 poster (Nov 2026).

**Experiment.** There are 12 E-cohort 16p11.2 duplication mice with a DREADD in the basolateral amygdala (BLA). Each mouse was given saline on one day and C21 on the other, so every mouse is its own control (crossover). On each day it was tested with three partners: its cagemate, a familiar mouse and a novel mouse. That makes 72 sessions in total. Pose was tracked with SLEAP: 9 keypoints per mouse, 720p video, 24 or 30 fps. Every analysis uses the first 600 s of each session. Every test is paired over the 12 mice.

## Quick start

```bash
conda env create -f environment.yml && conda activate c21pose
cp config.example.toml config.toml        # then edit the paths in config.toml
./run_all.sh                              # about 2–3 min
```

Outputs go to `outputs/`, which git ignores:

- `cache/`: intermediate pickles;
- `tables/`: CSV results;
- `figures/`: overview plots.

If `reference_dir` in `config.toml` points at the old project's `outputs_cm/` folder, `run_all.sh` ends by checking every step against the Nov 2026 results.

## Inputs you need

| Input | Where | In the repo? |
|---|---|---|
| SLEAP `.analysis.h5` files (72) | `paths.h5_root` (see `metadata/design_table.csv`, column `h5_relpath`) | no (raw data) |
| BORIS exports | `paths.boris_root` | no (raw data) |
| Session design: subject, partner, drug, day, which track is the subject | `metadata/design_table.csv` | yes |
| Cage floor corners per video, true fps | `metadata/cage_calibration/` | yes |
| Skeleton-repair decisions | `metadata/pose_repair_decisions.csv.gz` | yes |
| Region names | `metadata/region_names_ws13.csv` | yes |
| Sex per mouse | `metadata/subjects.csv` | yes |
| Videos | `paths.video_root` | no; needed only for `02_repair_poses.py --mode video` |

## Pipeline

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `01_build_sessions.py` | read h5 → remove brief identity swaps → pixels to floor cm (per-video homography) → fill gaps ≤ 15 frames → trim to 600.58 s → attach BORIS frame labels | `cache/sessions_raw.pkl` |
| 2 | `02_repair_poses.py` | fix keypoints that jumped onto the other mouse, and head–tail flips. Default: replays the saved decisions (no videos). `--mode video` redoes them from the videos | `cache/sessions_guarded.pkl` |
| 3 | `03_frame_features.py` | per-frame distances, speeds, angles, posture and sniff labels; session summaries | `cache/simple_frames.pkl`, `cache/pose_frames.pkl`, `tables/*_features_session.csv` |
| 4 | `04_fit_map.py` | sample 1,500 frames per session; UMAP of the 10 map features fitted on the cagemate frames; familiar and novel frames placed by their 15 nearest map frames | `cache/map_frames.pkl`, `cache/map_model.pkl` |
| 5 | `05_watershed_regions.py` | density of the map → 13 watershed regions; region of every sampled frame and of every cagemate frame | `cache/map_frames_regions.pkl`, `cache/watershed.pkl` |
| 6 | `06_region_stats.py` | time per region, C21 vs saline (paired t, BH over 13 regions, per partner); cagemate transitions (within-mouse permutation); region profiles | `tables/region_occupancy_tests.csv`, `tables/transition_*.csv`, `tables/region_profiles.csv` |
| 7 | `07_sniffing.py` | 2 cm rule and final rule: seconds, bouts and mean bout length per session; C21 vs saline | `tables/sniff_sessions.csv`, `tables/sniff_tests.csv` |
| 8 | `08_figures.py` | overview figures | `figures/*.png` |
| — | `verify_against_reference.py` | compares every step with the Nov 2026 files | prints OK / DIFF |

The code lives in the `c21pose/` package. The `scripts/` are thin wrappers around it.

| Module | Contents |
|---|---|
| `config.py` | paths and constants: 600 s window, 2 cm contact, 1,500 frames per session, the 10 map features |
| `sleapio.py`, `calibration.py`, `boris.py`, `sessions.py` | raw data → session records |
| `repair.py` | skeleton repair |
| `features_simple.py`, `features_pose.py` | per-frame and per-session features |
| `sniffing.py` | 2 cm rule and final sniffing rule |
| `bmap.py`, `watershed.py` | behaviour map and regions |
| `stats.py` | paired tests |
| `verify.py` | comparison helpers |

Every feature is defined in [docs/FEATURES.md](docs/FEATURES.md).

## The model in one paragraph

Each frame is described by 10 numbers. Five distances run from each mouse's nose to the other mouse's nose, nearest keypoint, body and tail base, plus the distance between centers; the other two are each mouse's speed. These are log-transformed and z-scored. UMAP (`n_neighbors=50`, `min_dist=0.1`, `random_state=0`) embeds 36,000 cagemate frames in 2-D. The frame density on a 200 × 200 grid (Gaussian, σ = 3 bins) is split by watershed from its 13 strongest peaks. Familiar and novel frames take the mean map position of their 15 nearest cagemate frames. A mouse's time in a region is the share of its sampled frames that fall there, averaged over its sessions. Saline and C21 are compared with a paired t test over the 12 mice.

Sniffing uses the 2 cm rule: a nose within 2 cm of the partner, with the nearest region giving the target. On the poster, anogenital sniffing uses the 2 cm rule as is. Face/body sniffing adds heading and lateral-offset limits, with cutoffs chosen on the other 11 mice (details in docs/FEATURES.md).

## Reproducibility

- **Checked 24 Nov 2026.** Run from the raw h5 and BORIS files, the pipeline reproduces the Nov 2026 caches and poster tables exactly. That covers the session skeletons, frame features, UMAP coordinates, watershed regions, occupancy p and q values, transition permutation p values, the sniffing seconds and bouts, the held-out cutoffs and the final paired tests. The one exception is the repaired skeletons, which differ by at most 1e-14 cm.
- **Environment.** Python 3.13.15 with the pinned packages in `requirements.txt`. `requirements-lock.txt` is a full `pip freeze` of the environment used; it includes packages this repo does not need.
- **UMAP.** The map depends on the exact `umap-learn`, `numba`, `pynndescent` and `numpy` versions, and on the platform (checked on macOS arm64). With other versions the map can come out rotated or reshaped. If that happens, the region numbers no longer match `metadata/region_names_ws13.csv`, and the names must be reassigned from `tables/region_profiles.csv`.
- **Excluded BORIS session.** `E2.1_N2_D1` is in `[boris] exclude_sessions` in the config, to match the poster run. Its BORIS export was added after that run (see NOTES.md).
- **Randomness.** All of it is seeded. Frame sampling uses `default_rng(0)` per partner type, UMAP uses `random_state=0`, and the transition permutations use `default_rng(1)` with 5,000 draws.

## Caveats for anyone using the results

- n = 12 mice. Most p values on the poster are uncorrected; BH q values are in the tables. In the region tests, only one region comes close to surviving BH: cagemate region 10, q = .058.
- The final face/body rule was refined after the drug results were seen, so describe it as post hoc. Its session-level agreement with BORIS face + body seconds is poor (Spearman ρ = 0.05; the plain 2 cm rule gives 0.27). Anogenital agreement is good (ρ = 0.82).
- The map describes behaviour. Drug effects are tested on per-mouse measures, not inferred from the picture.
