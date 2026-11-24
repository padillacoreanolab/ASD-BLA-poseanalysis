#!/usr/bin/env bash
# Run the whole pipeline (about 2-3 minutes on an M-series Mac). Usage: ./run_all.sh [python]
set -euo pipefail
PY="${1:-python}"
cd "$(dirname "$0")"
for s in 01_build_sessions 02_repair_poses 03_frame_features 04_fit_map 05_watershed_regions 06_region_stats 07_sniffing 08_figures; do
  echo "== $s"; "$PY" "scripts/$s.py"
done
if grep -qE '^reference_dir = ".+"' config.toml; then echo "== verify"; "$PY" scripts/verify_against_reference.py; fi
