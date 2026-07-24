//! Robot-arm trajectory playback: loads a JSON clip recorded against some
//! source model's joint layout, retargets it onto whatever MuJoCo model is
//! currently loaded (matching by joint name, dropping anything that doesn't
//! line up), and caches the retargeted result to disk so the remap only runs
//! once per (source clip, target model) pairing. See
//! `MujocoScene::joint_tracks`/`apply_trajectory_frame` in `mujoco.rs` for
//! the playback half.
//!
//! The JSON schema here (`TrajectoryClip`) is the interchange format an
//! offline converter is expected to produce -- e.g. from Robomimic's HDF5
//! demos. The runtime never reads HDF5/npz itself: those formats need a
//! native-only reader with no wasm story, whereas this module (bar its disk
//! cache) has none of that baggage.
//!
//! Retargeting here is *only* joint reordering/filtering by name -- it does
//! not account for differing link lengths or kinematics. Two models with the
//! same joint names but different arm geometry will still produce
//! correct-looking-but-wrong motion; that's a modeling problem this module
//! doesn't attempt to solve.

use serde::{Deserialize, Serialize};

/// One joint's slot in a [`TrajectoryClip`]/[`RetargetedClip`] frame: its
/// name and how many consecutive `qpos` values it occupies (1 for a
/// hinge/slide, 4 for a ball, 7 for a free joint -- matches each MuJoCo
/// joint type's own `qpos` width).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct JointTrack {
    pub name: String,
    pub dofs: usize,
}

/// A trajectory as recorded against some source model: a fixed joint layout
/// (order matches how each frame's values are packed) plus one flattened
/// `qpos` vector per frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrajectoryClip {
    pub joints: Vec<JointTrack>,
    /// Seconds between frames.
    pub dt: f32,
    /// One entry per frame; each is `sum(joints[i].dofs)` values long,
    /// packed in `joints` order.
    pub frames: Vec<Vec<f64>>,
}

impl TrajectoryClip {
    /// Loads a clip from a JSON file through the engine's asset pipeline
    /// (native: disk; wasm: fetch), see [`crate::assets::load_string`].
    pub async fn load(path: &str) -> anyhow::Result<Self> {
        let text = crate::assets::load_string(path).await?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Per-joint `(start, dofs)` offsets into a frame's flat `qpos` vector.
    fn joint_offsets(&self) -> Vec<(usize, usize)> {
        let mut offsets = Vec::with_capacity(self.joints.len());
        let mut cursor = 0;
        for jt in &self.joints {
            offsets.push((cursor, jt.dofs));
            cursor += jt.dofs;
        }
        offsets
    }

    /// Remaps this clip onto `target_joints` (the currently loaded model's
    /// joint layout, see `MujocoScene::joint_tracks`): keeps only joints
    /// present in both, by name, in the *target's* order -- dropping any
    /// source joint the target doesn't have, and leaving any target joint
    /// the source doesn't animate untouched at playback time. A joint
    /// present in both but with mismatched `dofs` (i.e. a different joint
    /// type) is dropped too, rather than copying raw values across
    /// incompatible joint types and silently corrupting the pose.
    pub fn retarget(&self, target_joints: &[JointTrack]) -> RetargetedClip {
        let offsets = self.joint_offsets();
        let mut joints = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut matched = vec![false; self.joints.len()];
        for tj in target_joints {
            let Some(i) = self.joints.iter().position(|sj| sj.name == tj.name) else {
                continue;
            };
            if self.joints[i].dofs != tj.dofs {
                continue;
            }
            matched[i] = true;
            ranges.push(offsets[i]);
            joints.push(tj.clone());
        }

        // Free-joint (7-dof) source tracks with no matching joint in the
        // target model -- typically a manipulated object whose body hasn't
        // been added to the target MJCF yet. Kept alongside the matched set
        // (not written into qpos, since there's no joint to write) so
        // playback can still visualize their recorded motion as a
        // placeholder -- see `MujocoScene::draw_unmatched_objects`.
        let mut unmatched = Vec::new();
        let mut unmatched_ranges: Vec<(usize, usize)> = Vec::new();
        for (i, sj) in self.joints.iter().enumerate() {
            if sj.dofs == 7 && !matched[i] {
                unmatched.push(sj.clone());
                unmatched_ranges.push(offsets[i]);
            }
        }

        let pack = |ranges: &[(usize, usize)]| -> Vec<Vec<f64>> {
            self.frames
                .iter()
                .map(|f| {
                    ranges
                        .iter()
                        .flat_map(|&(start, dofs)| f[start..start + dofs].iter().copied())
                        .collect()
                })
                .collect()
        };
        let frames = pack(&ranges);
        let unmatched_frames = pack(&unmatched_ranges);
        RetargetedClip { dt: self.dt, joints, frames, unmatched, unmatched_frames }
    }
}

/// A [`TrajectoryClip`] already remapped onto a specific model's joint
/// layout -- every frame's values line up 1:1 with `joints`, so playback
/// (`MujocoScene::apply_trajectory_frame`) can write them straight into the
/// sim with no further name lookups.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RetargetedClip {
    pub dt: f32,
    pub joints: Vec<JointTrack>,
    pub frames: Vec<Vec<f64>>,
    /// Free-joint source tracks that had no matching joint in the target
    /// model at retarget time (e.g. a manipulated object not yet modeled in
    /// the target MJCF) -- one frame per entry in `frames`, packed the same
    /// way. `#[serde(default)]` so a disk cache written before this field
    /// existed still deserializes (as empty) instead of forcing a recompute.
    #[serde(default)]
    pub unmatched: Vec<JointTrack>,
    #[serde(default)]
    pub unmatched_frames: Vec<Vec<f64>>,
}

impl RetargetedClip {
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

// ---- Disk cache -----------------------------------------------------------
//
// Retargeting is deterministic given (source clip, target joint layout), so
// the result is cached to disk keyed by a hash of both -- a second load of
// the same pairing (e.g. relaunching the editor) skips straight to the
// cached JSON instead of recomputing. Native only: the web build has no
// writable filesystem (see `resource_library.rs` for the same native/wasm
// split on a similar cache).
#[cfg(not(target_arch = "wasm32"))]
pub mod cache {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const CACHE_DIR: &str = "resources/trajectory_cache";

    /// Identifies a (source clip, target model) pairing for the cache --
    /// hashes the source file's own contents (not just its path, so an
    /// edited/replaced clip doesn't hit a stale cache) together with the
    /// target's joint layout (name + dofs, order-sensitive since retargeted
    /// output order follows it).
    fn cache_key(clip_path: &str, source_text: &str, target_joints: &[JointTrack]) -> String {
        let mut hasher = DefaultHasher::new();
        clip_path.hash(&mut hasher);
        source_text.hash(&mut hasher);
        for jt in target_joints {
            jt.name.hash(&mut hasher);
            jt.dofs.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    }

    fn cache_path(key: &str) -> std::path::PathBuf {
        std::path::Path::new(CACHE_DIR).join(format!("{key}.json"))
    }

    /// Loads `clip_path`, retargeting it onto `target_joints` -- reusing a
    /// cached result from a previous run if one matches, and writing a fresh
    /// one otherwise. This is the entry point callers should use instead of
    /// calling `TrajectoryClip::load` + `retarget` directly.
    pub async fn load_retargeted(
        clip_path: &str,
        target_joints: &[JointTrack],
    ) -> anyhow::Result<RetargetedClip> {
        let source_text = crate::assets::load_string(clip_path).await?;
        let key = cache_key(clip_path, &source_text, target_joints);
        let cache_file = cache_path(&key);
        if let Ok(cached) = std::fs::read_to_string(&cache_file) {
            if let Ok(clip) = serde_json::from_str::<RetargetedClip>(&cached) {
                return Ok(clip);
            }
        }

        let source: TrajectoryClip = serde_json::from_str(&source_text)?;
        let retargeted = source.retarget(target_joints);

        if let Ok(json) = serde_json::to_string(&retargeted) {
            let _ = std::fs::create_dir_all(CACHE_DIR);
            let _ = std::fs::write(&cache_file, json);
        }
        Ok(retargeted)
    }
}

#[cfg(target_arch = "wasm32")]
pub mod cache {
    use super::*;

    /// Wasm has nowhere to cache to (see the native module above), so this
    /// retargets on every load. Retargeting is an O(joints x frames) remap
    /// with no I/O -- next to the fetch it costs nothing; the native cache
    /// exists to skip repeated *process launches* re-doing the work, which
    /// a page load can't benefit from anyway.
    pub async fn load_retargeted(
        clip_path: &str,
        target_joints: &[JointTrack],
    ) -> anyhow::Result<RetargetedClip> {
        Ok(TrajectoryClip::load(clip_path).await?.retarget(target_joints))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jt(name: &str, dofs: usize) -> JointTrack {
        JointTrack { name: name.to_string(), dofs }
    }

    #[test]
    fn retarget_reorders_and_drops_unmatched() {
        let clip = TrajectoryClip {
            joints: vec![jt("a", 1), jt("b", 1), jt("free_base", 7), jt("c", 1)],
            dt: 1.0 / 30.0,
            frames: vec![vec![1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]],
        };
        // Target only has b, c (reordered), and a joint the clip never
        // mentions at all ("d") -- and free_base with a mismatched dof
        // count that should be dropped rather than misread.
        let target = vec![jt("c", 1), jt("b", 1), jt("d", 1), jt("free_base", 4)];
        let retargeted = clip.retarget(&target);
        assert_eq!(
            retargeted.joints.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(),
            vec!["c", "b"]
        );
        assert_eq!(retargeted.frames[0], vec![3.0, 2.0]);
        // free_base is 7-dof in the source but the target's version is a
        // mismatched 4-dof, so it's dropped from the matched set above --
        // it should surface as unmatched instead of vanishing entirely.
        assert_eq!(
            retargeted.unmatched.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(),
            vec!["free_base"]
        );
        assert_eq!(retargeted.unmatched_frames[0], vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
