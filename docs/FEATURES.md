# Feature definitions

All distances are in cm on the cage floor (per-video calibration), all speeds in cm/s, at each video's true frame rate (24 or 30 fps). Every value is computed on the first 600 s of a session. `s_` is the **subject** (the mouse given saline or C21) and `p_` / `i_` is the **partner** (cagemate, familiar or novel mouse).

Keypoints (9 per mouse): nose, head, left ear, right ear, neck, center, left side (`leftbody`), right side (`rightbody`), tail base (`butt`).

Regions of the other mouse, used by every sniffing measure:

| Region | Keypoints |
|---|---|
| face | nose, head, left ear, right ear |
| body | neck, left side, right side, center |
| anogenital | tail base |

## The 10 map features (`config.MAP_FEATURES`)

These are the only inputs of the UMAP behaviour map. They are transformed as log(1 + x), with negative values clipped to 0, then z-scored with the cagemate frames' mean and SD.

| Feature | Definition |
|---|---|
| `dist_cc` | subject center to partner center |
| `dist_nn` | subject nose to partner nose (face contact) |
| `s_nose_p_min` | subject nose to the nearest of the partner's 9 keypoints |
| `p_nose_s_min` | partner nose to the nearest of the subject's 9 keypoints |
| `s_nose_p_butt` | subject nose to partner tail base (anogenital) |
| `p_nose_s_butt` | partner nose to subject tail base |
| `s_nose_p_body` | subject nose to the nearest of the partner's neck, sides and center |
| `p_nose_s_body` | partner nose to the nearest of the subject's neck, sides and center |
| `s_speed` | subject center speed: displacement over a 0.2 s step divided by the step |
| `p_speed` | partner center speed, same definition |

Angle, orientation, body length and box overlap were tried as extra map features and diluted every region effect: a 21-feature map gave weaker results at every k. They stay out of the map and are tested directly.

## The 2 cm rule (`sniffing.targets`)

Mouse A sniffs mouse B on a frame when A's nose is within **2 cm** of any of B's keypoints. The target is whichever region of B is closest. In each target mask, gaps shorter than 0.2 s are closed, then runs shorter than 0.2 s are dropped.

## Final SLEAP sniffing rule (`sniffing.final_rule`; the Nov 2026 poster)

- **Anogenital:** the 2 cm rule, unchanged.
- **Face/body:** the 2 cm rule (target face or body), plus all of these:
  - **Heading:** the angle between the subject's neck → nose and nose → partner center is below a cutoff.
  - **Lateral offset:** the partner center's distance from the subject's head axis is below a cutoff. The head axis is smoothed over 5 frames.
  - **Axis angle (optional):** the angle between the two neck → center axes is at least a minimum.
  - **Minimum bout length.**
- **How the cutoffs are chosen:** each mouse's cutoffs are picked on the other 11 mice, as the grid point with the best frame-level F1 against BORIS face + body. BORIS is shifted 0.5 s earlier for this, because BORIS lags SLEAP contact. The cutoffs per mouse are in `tables/sniff_final_rule_cutoffs.csv`.
- **Grid:**
  - heading: 45, 60, 75, 90, 120, 181°
  - axis: 0 or 45°
  - lateral: 3, 5, 8 cm, or no limit
  - minimum bout: 0.2 or 0.5 s
- **"All sniffing"** is the union of the two final masks. A bout is one unbroken run.

## Simple session features (`features_simple.session_features`; 2 cm contact)

| Feature | Definition |
|---|---|
| `mean_dist` | mean `dist_cc` |
| `frac_close` | share of frames with centers < 10.3 cm (1.5 body lengths) |
| `close_bouts_per_min`, `close_bout_s` | close runs of at least 1 s: rate and mean length |
| `frac_contact_subj` / `_part` | that mouse's nose within 2 cm of any keypoint of the other |
| `frac_ag_subj` / `_part` | that mouse's nose within 2 cm of the other's tail base |
| `subj_speed`, `part_speed` | mean center speed |
| `frac_moving_subj` / `_part` | speed > 2 cm/s |
| `subj_speed_close` / `_far` | subject speed when the centers are closer / farther than 10.3 cm |
| `frac_oriented_ag_subj` / `_part` | facing < 60° and nose < 3 cm from the other's tail base |

## Pose features (`features_pose`)

There are 28 per-frame features and 12 labels, computed the same way for each mouse.

- `length`: nose to tail base, divided by the mouse's median while walking.
- `width`: side to side, divided the same way.
- `elongation`: length / width.
- `bend`: the angle at the center between neck and tail base (180° = straight).
- `head_turn`: signed angle between the body axis and the head axis (+ = turned left).
- `speed`, `nose_speed`: over 0.2 s.
- `turn_rate`: change in body heading, in °/s.
- `wall_dist`: nose to the nearest cage wall.
- `nose_face`, `nose_body`, `nose_ag`: nose to the nearest keypoint of that region of the other mouse.
- `facing`: the angle between neck → nose and neck → the other mouse's nearest keypoint.
- Relational: `distance` (center to center) and `rel_heading` (angle between the two body axes).

Labels:

- `sniff_face`, `sniff_body`, `sniff_ag`: nose < 2 cm from that region, and that region is the nearest (bouts ≥ 0.1 s).
- `rear`: length < 0.65, bend > 140°, speed < 3, centers > 8 cm, lasting 0.3–8 s.
- `hunch`: length < 0.75 and bend < 140°.
- `stretch`: length > 1.1 and speed < 2 cm/s.
- `hunch` and `stretch` need bouts ≥ 0.3 s.

## Angle conventions (easy to get backwards)

| Angle | 0° means | Where |
|---|---|---|
| `s_facing`, `p_facing` (simple frames) | pointing straight at the other mouse's center (measured from the neck) | map profiles: facing = < 45° |
| `*_facing` (pose features) | pointing at the other mouse's nearest keypoint | pose features |
| head-centroid angle (`paperfeat`, old project) | angle **at the head** between the neck and the partner's centroid: **180° = facing**, small = facing away | poster panel 5 only, not in this repo |
