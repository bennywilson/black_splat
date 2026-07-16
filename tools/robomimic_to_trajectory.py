#!/usr/bin/env python3
"""Converts a robomimic low_dim demo file into the engine's trajectory JSON
schema (see `src/trajectory.rs`'s `TrajectoryClip`), so `trajectory::cache`
can retarget/play it back through a loaded MuJoCo scene.

This is a one-off, offline step -- the Rust runtime never reads HDF5 itself
(see the module docs on `src/trajectory.rs` for why). Run it once per demo
you want to use, commit or otherwise ship the resulting JSON, and point the
engine at that file.

Usage:
    python3 tools/robomimic_to_trajectory.py lift_ph_low_dim.hdf5 --demo demo_0 -o traj.json
    python3 tools/robomimic_to_trajectory.py lift_ph_low_dim.hdf5 --all -o out_dir/

Requires: h5py (`pip install h5py`).

Joint naming: robomimic's low_dim obs give the Panda arm's 7 joint angles
(`robot0_joint_pos`) and the 2 gripper finger positions (`robot0_gripper_qpos`)
directly -- no need to decode the raw simulator `states` blob. What's *not*
guaranteed is that these line up name-for-name with your MJCF's own joint
names; mujoco_menagerie's franka_emika_panda/panda.xml uses "joint1".."joint7"
and "finger_joint1"/"finger_joint2", which is the default below, but if your
XML uses different names, override with --joint-names. A wrong name here
isn't fatal -- `TrajectoryClip::retarget` just drops any joint that doesn't
match by name in the target model, so the arm would silently sit still
instead of erroring. Verify a few frames visually before trusting a full
conversion.
"""
import argparse
import json
import os
import sys

try:
    import h5py
    import numpy as np
except ImportError:
    sys.exit("This script needs h5py and numpy: pip install h5py numpy")

# Order matches robot0_joint_pos's 7 columns, then robot0_gripper_qpos's 2.
DEFAULT_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
]

DEFAULT_CONTROL_FREQ = 20.0  # robosuite's default, used if env_args lacks it.


def demo_control_freq(f):
    """Best-effort dt source: robomimic stores the recording env's kwargs as
    a JSON string on data.attrs["env_args"]; control_freq lives under
    env_kwargs there for robosuite-backed datasets. Falls back to
    DEFAULT_CONTROL_FREQ if the attr is missing or shaped unexpectedly (older
    dataset versions, non-robosuite envs)."""
    raw = f["data"].attrs.get("env_args")
    if not raw:
        return DEFAULT_CONTROL_FREQ
    try:
        env_args = json.loads(raw)
        return float(env_args["env_kwargs"]["control_freq"])
    except (KeyError, ValueError, TypeError):
        return DEFAULT_CONTROL_FREQ


def convert_demo(f, demo_key, joint_names):
    demo = f["data"][demo_key]
    joint_pos = demo["obs"]["robot0_joint_pos"][:]        # (T, 7)
    gripper_pos = demo["obs"]["robot0_gripper_qpos"][:]   # (T, 2)
    if joint_pos.shape[1] + gripper_pos.shape[1] != len(joint_names):
        sys.exit(
            f"{demo_key}: obs give {joint_pos.shape[1] + gripper_pos.shape[1]} "
            f"joint values but {len(joint_names)} names were provided "
            f"(--joint-names); pass one name per column."
        )
    frames = [row.tolist() for row in np.concatenate([joint_pos, gripper_pos], axis=1)]
    return {
        "joints": [{"name": n, "dofs": 1} for n in joint_names],
        "dt": 1.0 / demo_control_freq(f),
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a robomimic *_low_dim.hdf5 file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", help="Convert a single demo (e.g. demo_0)")
    group.add_argument("--all", action="store_true", help="Convert every demo in the file")
    parser.add_argument("-o", "--output", required=True,
                         help="Output JSON path (--demo) or directory (--all)")
    parser.add_argument("--joint-names", nargs="+", default=DEFAULT_JOINT_NAMES,
                         help="Target MJCF joint names, in the same order as "
                              "robot0_joint_pos's 7 columns then robot0_gripper_qpos's 2 "
                              f"(default: {' '.join(DEFAULT_JOINT_NAMES)})")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        if args.demo:
            clip = convert_demo(f, args.demo, args.joint_names)
            with open(args.output, "w") as out:
                json.dump(clip, out)
            print(f"Wrote {args.output} ({len(clip['frames'])} frames)")
        else:
            demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            os.makedirs(args.output, exist_ok=True)
            for key in demo_keys:
                clip = convert_demo(f, key, args.joint_names)
                out_path = os.path.join(args.output, f"{key}.json")
                with open(out_path, "w") as out:
                    json.dump(clip, out)
                print(f"Wrote {out_path} ({len(clip['frames'])} frames)")


if __name__ == "__main__":
    main()
