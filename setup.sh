#!/usr/bin/env bash
# One-shot project setup: fetches large assets that aren't checked into git
# and does any other local setup needed before a first build. Safe to re-run.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAJ_DIR="$DIR/source_assets/mujoco_trajectories"
TRAJ_FOLDER_URL="https://drive.google.com/drive/folders/1u40p4fdvgRfVgqDOMejafBekzL-plcjZ"

fetch_trajectories() {
  for f in lift_ph_low_dim.hdf5 square_ph_low_dim.hdf5 tool_hang_ph_low_dim.hdf5; do
    if [ ! -f "$TRAJ_DIR/$f" ]; then
      echo "==> Fetching robomimic trajectory datasets into $TRAJ_DIR"
      python3 -m pip show gdown >/dev/null 2>&1 || python3 -m pip install --quiet gdown
      mkdir -p "$TRAJ_DIR"
      python3 -m gdown --folder "$TRAJ_FOLDER_URL" -O "$TRAJ_DIR"
      return
    fi
  done
  echo "==> Trajectory datasets already present, skipping"
}

fetch_trajectories

echo "==> Setup complete"
