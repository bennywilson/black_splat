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

Source of truth: each demo carries its own `states` array -- MuJoCo's raw
`[time, qpos, qvel]` per frame -- and its own compiled MJCF on
`attrs["model_file"]`. Reading `states` (rather than the `obs` dict) is what
gives access to every free-jointed body in the scene, not just the robot: a
robosuite task like ToolHang has the arm's own joints *and* a free joint per
manipulated object (`stand_joint0`, `frame_joint0`, `tool_joint0`), all
packed into the same flat qpos vector. This script walks each demo's own
`model_file` worldbody in document order -- which is also MuJoCo's own qpos
layout order, since a body's joints are compiled before its children -- to
find every joint's qpos offset, then slices `states` at those offsets rather
than assuming a fixed column layout (frame counts and object counts differ
per task).

The robot's 9 joints (7 arm + 2 gripper fingers) are renamed to the target
MJCF's own joint names via `--joint-names`/`ROBOT_SOURCE_NAMES`, since
robosuite's `robot0_...`/`gripper0_right_...` names won't generally match a
target model (mujoco_menagerie's franka_emika_panda/panda.xml uses
"joint1".."joint7"/"finger_joint1"/"finger_joint2", the default below).
Every *other* free joint found in the model -- i.e. a manipulated object --
is carried through under its own source name and full 7-dof (xyz + wxyz
quat) free-joint qpos, unrenamed: to play an object's motion back, give it a
body with a freejoint of the same name in the target scene's MJCF. As with
the robot joints, `TrajectoryClip::retarget` silently drops anything that
doesn't match by name -- verify a few frames visually before trusting a full
conversion.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

try:
    import h5py
except ImportError:
    sys.exit("This script needs h5py: pip install h5py")

# Order matches robot0_joint_pos's 7 columns, then robot0_gripper_qpos's 2 --
# these are the fixed names robosuite's Panda + parallel-jaw gripper use in
# every task's embedded model_file.
ROBOT_SOURCE_NAMES = [
    "robot0_joint1", "robot0_joint2", "robot0_joint3", "robot0_joint4",
    "robot0_joint5", "robot0_joint6", "robot0_joint7",
    "gripper0_right_finger_joint1", "gripper0_right_finger_joint2",
]

# robosuite's gripper records the two fingers as literal mirror images (e.g.
# finger_joint1 in [0, 0.04] while finger_joint2 in [-0.04, 0], each tracking
# the gripper's open/close amount with an opposite sign). A target MJCF built
# the usual way (mujoco_menagerie's panda.xml included) instead mirrors the
# *body* -- a 180 degree quat on the right finger -- and expects both joints
# to share the same [0, 0.04] convention, tied together via an equality
# constraint. Copying the source's raw sign into that layout drives the right
# finger the wrong way (closing further instead of opening), so it needs
# negating here to match the target's convention.
ROBOT_SOURCE_SIGNS = [1, 1, 1, 1, 1, 1, 1, 1, -1]

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


def model_joint_layout(model_xml):
    """Walks a demo's embedded MJCF worldbody depth-first (a body's own
    joints, then its children -- matching how MuJoCo itself assigns qpos
    addresses during compilation) and returns `(name, qpos_offset, dofs)`
    for every joint. `dofs` is 7 for a freejoint, 1 for a hinge/slide;
    robosuite/robomimic models don't use ball joints, so that case isn't
    handled here."""
    root = ET.fromstring(model_xml)
    world = root.find("worldbody")
    joints = []
    offset = 0

    def walk(body):
        nonlocal offset
        for child in body:
            if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
                joints.append((child.get("name"), offset, 7))
                offset += 7
            elif child.tag == "joint" and child.get("type", "hinge") in ("hinge", "slide"):
                joints.append((child.get("name"), offset, 1))
                offset += 1
        for child in body:
            if child.tag == "body":
                walk(child)

    for b in world:
        if b.tag == "body":
            walk(b)
    return joints


def robot_base_pose(model_xml):
    """Finds `robot0_base`'s own `(pos, quat)` in the demo's embedded MJCF --
    its fixed mount point in *this demo's* world frame (e.g. robosuite mounts
    the arm on a pedestal, so this is typically offset a good half-meter up
    and sideways from world origin). `pos`/`quat` default to identity if the
    attributes are absent, matching MJCF's own defaults."""
    root = ET.fromstring(model_xml)
    world = root.find("worldbody")
    for b in world:
        if b.tag == "body" and b.get("name") == "robot0_base":
            pos = [float(v) for v in b.get("pos", "0 0 0").split()]
            quat = [float(v) for v in b.get("quat", "1 0 0 0").split()]
            return pos, quat
    sys.exit("expected a 'robot0_base' body in this demo's model_file")


def quat_conj(q):
    w, x, y, z = q
    return [w, -x, -y, -z]


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_rotate_inv(q, v):
    """Rotates `v` by `q`'s inverse (`q` is unit, so inverse == conjugate)."""
    qc = quat_conj(q)
    qw, qx, qy, qz = qc
    vw = [0.0] + list(v)
    rw, rx, ry, rz = quat_mul(quat_mul(qc, vw), q)
    return [rx, ry, rz]


def convert_demo(f, demo_key, joint_names):
    if len(joint_names) != len(ROBOT_SOURCE_NAMES):
        sys.exit(
            f"--joint-names needs {len(ROBOT_SOURCE_NAMES)} names (one per "
            f"robot joint: {', '.join(ROBOT_SOURCE_NAMES)}), got {len(joint_names)}"
        )

    demo = f["data"][demo_key]
    states = demo["states"][:]  # (T, 1 + nq + nv): time, qpos, qvel.
    model_xml = demo.attrs["model_file"]
    layout = {name: (offset, dofs) for name, offset, dofs in model_joint_layout(model_xml)}
    base_pos, base_quat = robot_base_pose(model_xml)
    base_quat_conj = quat_conj(base_quat)

    tracks = []  # (output_name, qpos_offset, dofs)
    signs = {}  # qpos_offset -> sign, for the robot's own joints only
    for src, out, sign in zip(ROBOT_SOURCE_NAMES, joint_names, ROBOT_SOURCE_SIGNS):
        if src not in layout:
            sys.exit(f"{demo_key}: expected robot joint '{src}' not found in this demo's model_file")
        offset, dofs = layout[src]
        tracks.append((out, offset, dofs))
        signs[offset] = sign

    # Any other free joint in the model is a manipulated object -- carried
    # through under its own name/full qpos width, unrenamed (see module docs).
    # Its qpos is recorded as an absolute pose in *this demo's* world frame,
    # which generally doesn't line up with the target model's world frame
    # (e.g. robosuite mounts the arm on a pedestal well above and off to the
    # side of world origin, but a standalone target panda.xml puts its base
    # at the origin). Re-expressing it relative to robot0_base's fixed mount
    # pose fixes that: the target scene's arm base sits at its own origin, so
    # a robot0_base-relative object pose lines up with it the same way it did
    # in the source scene, regardless of where either scene's world origin is.
    object_tracks = []  # (name, qpos_offset) -- always 7 dof.
    robot_names = set(ROBOT_SOURCE_NAMES)
    for name, offset, dofs in model_joint_layout(model_xml):
        if name not in robot_names and dofs == 7:
            object_tracks.append((name, offset))
    tracks.extend((name, offset, 7) for name, offset in object_tracks)
    object_offsets = {offset for _, offset in object_tracks}

    frames = []
    for row in states:
        qpos = row[1:]  # drop the leading time column
        frame = []
        for _, offset, dofs in tracks:
            if offset in object_offsets:
                pos = qpos[offset:offset + 3].tolist()
                quat = qpos[offset + 3:offset + 7].tolist()
                rel_pos = quat_rotate_inv(base_quat, [p - b for p, b in zip(pos, base_pos)])
                rel_quat = quat_mul(base_quat_conj, quat)
                frame.extend(rel_pos)
                frame.extend(rel_quat)
            else:
                sign = signs.get(offset, 1)
                frame.extend((sign * v for v in qpos[offset:offset + dofs].tolist()))
        frames.append(frame)

    return {
        "joints": [{"name": n, "dofs": d} for n, _, d in tracks],
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
                         help="Target MJCF names for the robot's 7 arm joints + 2 gripper "
                              f"fingers, in that order (default: {' '.join(DEFAULT_JOINT_NAMES)}). "
                              "Manipulated objects' free joints are always carried through "
                              "under their own source name -- see the module docstring.")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        if args.demo:
            clip = convert_demo(f, args.demo, args.joint_names)
            with open(args.output, "w") as out:
                json.dump(clip, out)
            print(f"Wrote {args.output} ({len(clip['frames'])} frames, joints: "
                  f"{', '.join(j['name'] for j in clip['joints'])})")
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
