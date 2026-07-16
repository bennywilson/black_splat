//! MJCF loading and wireframe rendering for MuJoCo scenes.
//!
//! MuJoCo owns parsing and forward kinematics; the engine never computes a
//! pose itself -- it only reads geom poses each frame and draws what MuJoCo
//! reports (see `draw_mj_geom`). Models are just wireframed for now; loading
//! real meshes for MuJoCo geoms is future work.
//!
//! Two very different MuJoCo builds back this depending on target:
//!   - native: `mujoco-rs`, an FFI binding to the C library, driven directly
//!     by [`MujocoScene`].
//!   - wasm32: `mujoco-rs` can't target `wasm32-unknown-unknown` (it's FFI to
//!     a native binary). Instead the page loads DeepMind's official
//!     `@mujoco/mujoco` wasm build as a second, independent wasm module, and
//!     steps it in its own requestAnimationFrame loop in JS. Each frame it
//!     copies the geom arrays into this module's `wasm_bridge` via
//!     wasm-bindgen; [`MujocoScene::tick_and_draw`] just reads whatever the
//!     bridge last received and draws it. Two separate wasm linear memories
//!     means this copy is unavoidable -- there's no way to share pointers
//!     across the boundary. `examples/mujoco_test/index.html` is the JS half
//!     of this handoff.

use crate::{config::Config, renderer::Renderer, utils::*};

#[cfg(not(target_arch = "wasm32"))]
use mujoco_rs::prelude::*;

// Raw mjtGeom values (stable across the native crate and the JS bindings --
// both wrap the same C enum), so geom kind travels as a plain u32 across the
// wasm-bindgen boundary instead of needing a shared Rust type.
const MJ_GEOM_PLANE: u32 = 0;
const MJ_GEOM_SPHERE: u32 = 2;
const MJ_GEOM_CAPSULE: u32 = 3;
const MJ_GEOM_CYLINDER: u32 = 5;
const MJ_GEOM_BOX: u32 = 6;
const MJ_GEOM_MESH: u32 = 7;

/// A loaded MuJoCo scene: owns and steps the sim (native) or reflects the
/// sibling wasm module's latest frame (wasm32), and draws every geom as a
/// wireframe via the engine's line pass.
pub struct MujocoScene {
    #[cfg(not(target_arch = "wasm32"))]
    mj_data: MjData<Box<MjModel>>,
    #[cfg(not(target_arch = "wasm32"))]
    sim_time_accum: f32,
    // mesh name -> resolved absolute .obj path, for `mesh_geoms`. Built by
    // `collect_mesh_paths` -- mujoco-rs's own `mesh_pathadr` only exposes the
    // MJCF's raw, un-resolved `file` attribute (e.g. "link0/link0.obj", not
    // joined with `meshdir` or the declaring file's directory), so it can't
    // be opened as-is.
    #[cfg(not(target_arch = "wasm32"))]
    mesh_paths: std::collections::HashMap<String, String>,
    /// Whether `tick_and_draw` advances the sim each frame. Paused scenes
    /// still draw their current pose.
    playing: bool,
    /// Multiplies `game_config.delta_time` before it's accumulated into sim
    /// steps -- 1.0 is realtime, 0.5 is half-speed, etc.
    speed: f32,
}

impl MujocoScene {
    /// Loads an MJCF file through the engine's asset pipeline (native: reads
    /// from disk; wasm: fetches from `/rust_assets/`, see
    /// [`crate::assets::load_string`]) and parses it.
    pub async fn load(file_path: &str) -> anyhow::Result<Self> {
        let xml = crate::assets::load_string(file_path).await?;
        Self::from_xml(file_path, &xml)
    }

    /// Parses an MJCF file already fetched as `xml`, using `path` to resolve
    /// relative `<include>` references on native. `MjModel` has no notion of
    /// a base directory when parsing from a string (it stages the text into
    /// a synthetic VFS entry with no real path), so an `<include file="...">`
    /// would otherwise be looked up relative to the process's CWD instead of
    /// the MJCF's own directory. Loading straight from `path` lets MuJoCo
    /// resolve those relative to the real file. Wasm has no filesystem to
    /// resolve against, so it always parses `xml`.
    ///
    /// Note this does *not* help a `<compiler meshdir="...">` attribute: a
    /// relative `meshdir` replaces the model directory rather than being
    /// joined with it, and is resolved against the process's CWD regardless
    /// of how `path` is given (relative or absolute) -- MJCF authors need to
    /// either drop `meshdir` and inline the subdirectory into each
    /// `<mesh file="...">`, or pass an absolute `meshdir`.
    pub fn from_xml(path: &str, xml: &str) -> anyhow::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = xml;
            Self::from_xml_path(path)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
            Self::from_xml_str(xml)
        }
    }

    /// Loads and parses an MJCF file directly from disk (native only) --
    /// preserves the file's directory so relative `<include>` references
    /// resolve correctly (see [`from_xml`](Self::from_xml); `meshdir` is a
    /// separate concern, noted there).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_xml_path(path: &str) -> anyhow::Result<Self> {
        let model = MjModel::from_xml(path)
            .map_err(|e| anyhow::anyhow!("failed to parse MJCF: {e}"))?;
        Ok(Self {
            mj_data: MjData::new(Box::new(model)),
            sim_time_accum: 0.0,
            mesh_paths: collect_mesh_paths(path),
            playing: true,
            speed: 1.0,
        })
    }

    /// Parses an already-loaded MJCF string (e.g. an `include_str!`'d scene)
    /// with no base directory -- relative `<include>`/`meshdir` references
    /// won't resolve on native (see [`from_xml`](Self::from_xml)).
    pub fn from_xml_str(xml: &str) -> anyhow::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model = MjModel::from_xml_string(xml)
                .map_err(|e| anyhow::anyhow!("failed to parse MJCF: {e}"))?;
            Ok(Self {
                mj_data: MjData::new(Box::new(model)),
                sim_time_accum: 0.0,
                mesh_paths: std::collections::HashMap::new(),
                playing: true,
                speed: 1.0,
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bridge::set_model_xml(xml.to_string());
            wasm_bridge::set_playback(true, 1.0);
            Ok(Self { playing: true, speed: 1.0 })
        }
    }

    /// Whether the sim is currently advancing each frame.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Pauses/resumes sim stepping (drawing continues either way).
    pub fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::set_playback(self.playing, self.speed);
    }

    /// Sim-time multiplier applied to `game_config.delta_time` (1.0 = realtime).
    pub fn speed(&self) -> f32 {
        self.speed
    }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::set_playback(self.playing, self.speed);
    }

    /// Advances the sim by exactly one fixed timestep, bypassing `playing`
    /// -- backs a "Step" button.
    pub fn step_once(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.mj_data.step();
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::request_single_step();
    }

    /// Resets the sim to the MJCF's initial state (`qpos0`/keyframe 0).
    pub fn reset(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.mj_data.reset();
            self.sim_time_accum = 0.0;
        }
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::request_reset();
    }

    /// Steps the simulation (native only; wasm steps in JS, see the module
    /// docs) and draws every geom as a wireframe line via the renderer's
    /// debug line pass.
    pub fn tick_and_draw(&mut self, renderer: &mut Renderer, game_config: &Config) {
        self.tick_and_draw_at(renderer, game_config, CG_VEC3_ZERO, CG_QUAT_IDENT);
    }

    /// Like [`tick_and_draw`](Self::tick_and_draw), but places the whole
    /// scene by a rigid transform on top of the MJCF's own local coordinates
    /// -- lets a caller (e.g. an editor actor) position/orient a MuJoCo scene
    /// without editing the XML.
    pub fn tick_and_draw_at(
        &mut self,
        renderer: &mut Renderer,
        game_config: &Config,
        origin: CgVec3,
        rotation: CgQuat,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Fixed-timestep sim stepping decoupled from render framerate;
            // capped so a debugger pause / hitch can't spiral into a
            // catch-up storm. Paused scenes skip stepping but still draw
            // their current pose below.
            if self.playing {
                let dt = self.mj_data.model().opt().timestep as f32;
                self.sim_time_accum += game_config.delta_time * self.speed;
                let mut steps = 0;
                while self.sim_time_accum >= dt && steps < 8 {
                    self.mj_data.step();
                    self.sim_time_accum -= dt;
                    steps += 1;
                }
            }

            let ngeom = self.mj_data.model().ffi().ngeom as usize;
            for i in 0..ngeom {
                let gtype = self.mj_data.model().geom_type()[i] as u32;
                let size = self.mj_data.model().geom_size()[i];
                let rgba = self.mj_data.model().geom_rgba()[i];
                let xpos = self.mj_data.geom_xpos()[i];
                let xmat = self.mj_data.geom_xmat()[i];
                draw_mj_geom(
                    renderer,
                    game_config,
                    gtype,
                    [size[0] as f32, size[1] as f32, size[2] as f32],
                    rgba,
                    [xpos[0] as f32, xpos[1] as f32, xpos[2] as f32],
                    std::array::from_fn(|k| xmat[k] as f32),
                    origin,
                    rotation,
                );
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bridge::with_geoms(|geoms| {
                for g in geoms {
                    draw_mj_geom(
                        renderer, game_config, g.kind, g.size, g.rgba, g.xpos, g.xmat, origin,
                        rotation,
                    );
                }
            });
        }
    }

    /// Registers interest in a named geom so [`geom_world_pos`](Self::geom_world_pos)
    /// can resolve it. Native resolves names directly and doesn't need this;
    /// on wasm, JS must look the name up against its own MuJoCo model and
    /// report the id back (see `index.html`), so the position isn't
    /// available until that round-trip completes.
    pub fn watch_geom(&self, _name: &str) {
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::watch_geom(_name);
    }

    /// Registers interest in a named joint so
    /// [`apply_joint_qvel`](Self::apply_joint_qvel) can resolve it. See
    /// [`watch_geom`](Self::watch_geom) for why this is needed on wasm.
    pub fn watch_joint(&self, _name: &str) {
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::watch_joint(_name);
    }

    /// Current world-space position of a named geom, in engine (Y-up)
    /// coordinates. On wasm this only resolves after [`watch_geom`](Self::watch_geom)
    /// was called and JS has reported the id back.
    pub fn geom_world_pos(&self, name: &str) -> Option<CgVec3> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let id = self.mj_data.model().name_to_id(MjtObj::mjOBJ_GEOM, name)?;
            let p = self.mj_data.geom_xpos()[id];
            Some(mj_vec3([p[0] as f32, p[1] as f32, p[2] as f32]))
        }
        #[cfg(target_arch = "wasm32")]
        {
            let id = wasm_bridge::named_geom_id(name)?;
            wasm_bridge::with_geoms(|geoms| geoms.get(id).map(|g| mj_vec3(g.xpos)))
        }
    }

    /// Every mesh-type geom this frame, resolved to a source .obj path (via
    /// `mesh_paths`, built at load time by re-walking the MJCF -- see
    /// [`collect_mesh_paths`]) plus the world transform and MJCF-resolved
    /// `<material>` rgba to draw it with. Callers load/cache a real
    /// triangle-mesh `Model` per unique path (see
    /// `crate::assets::AssetManager::load_model`, which dispatches to
    /// `Model::from_obj_path` for `.obj`) and draw one instance per entry --
    /// `draw_mj_geom` skips `MJ_GEOM_MESH` since it can't do this itself (it
    /// only has a `Renderer`, not an `AssetManager`). Native only: mesh geoms
    /// aren't supported over the wasm bridge yet, so this is always empty
    /// there. `origin`/`rotation` place the whole scene the same way
    /// `tick_and_draw_at` does.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mesh_geoms(&self, origin: CgVec3, rotation: CgQuat) -> Vec<MeshGeomInstance> {
        let model = self.mj_data.model();
        let ngeom = model.ffi().ngeom as usize;
        let mut out = Vec::new();
        for i in 0..ngeom {
            if model.geom_type()[i] as u32 != MJ_GEOM_MESH {
                continue;
            }
            let mesh_id = model.geom_dataid()[i];
            if mesh_id < 0 {
                continue;
            }
            let Some(mesh_name) = model.id_to_name(MjtObj::mjOBJ_MESH, mesh_id as usize) else {
                continue;
            };
            let Some(mesh_path) = self.mesh_paths.get(mesh_name) else {
                continue;
            };
            // A geom with a <material> takes its color from the material's
            // own rgba, not geom_rgba (which stays at its unset default --
            // effectively white -- whenever a material is assigned; MuJoCo
            // never copies mat_rgba into it). geom_rgba only applies to
            // geoms colored directly via their own `rgba` attribute, with no
            // material reference (matid < 0).
            let mat_id = model.geom_matid()[i];
            let rgba = if mat_id >= 0 {
                model.mat_rgba()[mat_id as usize]
            } else {
                model.geom_rgba()[i]
            };
            let xpos = self.mj_data.geom_xpos()[i];
            let xmat = self.mj_data.geom_xmat()[i];
            let (xpos_eff, xmat_eff) =
                undo_mesh_asset_transform(model, mesh_id as usize, xpos, xmat);
            let (ex, ey, ez) = mj_basis(xmat_eff);
            let (ex, ey, ez) = (rotation * ex, rotation * ey, rotation * ez);
            out.push(MeshGeomInstance {
                mesh_path: mesh_path.clone(),
                rgba,
                position: origin + rotation * mj_vec3(xpos_eff),
                rotation: quat_from_basis(ex, ey, ez),
            });
        }
        out
    }

    #[cfg(target_arch = "wasm32")]
    pub fn mesh_geoms(&self, _origin: CgVec3, _rotation: CgQuat) -> Vec<MeshGeomInstance> {
        Vec::new()
    }

    /// Adds `delta` to a joint's first degree-of-freedom velocity (e.g. a
    /// hinge's qvel) -- native mutates the sim directly; wasm calls back into
    /// JS (`window.__bsMujocoApplyQvel`, see `index.html`), since the sim
    /// itself lives in the sibling wasm module there. Requires
    /// [`watch_joint`](Self::watch_joint) on wasm.
    pub fn apply_joint_qvel(&mut self, joint_name: &str, delta: f64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(joint) = self.mj_data.joint(joint_name) {
                joint.view_mut(&mut self.mj_data).qvel[0] += delta;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(dof) = wasm_bridge::named_joint_dof(joint_name) {
                wasm_bridge::apply_qvel(dof, delta);
            }
        }
    }

    /// This model's joints in MuJoCo's own order: name + `qpos` width (1 for
    /// a hinge/slide, 4 for a ball, 7 for a free joint). This is the layout
    /// [`crate::trajectory::TrajectoryClip::retarget`] remaps a trajectory
    /// onto before [`apply_trajectory_frame`](Self::apply_trajectory_frame)
    /// can play it back. Wasm has no qpos access over the bridge yet, so
    /// this is always empty there.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn joint_tracks(&self) -> Vec<crate::trajectory::JointTrack> {
        let model = self.mj_data.model();
        let njnt = model.ffi().njnt as usize;
        let jnt_type = model.jnt_type();
        (0..njnt)
            .filter_map(|i| {
                let name = model.id_to_name(MjtObj::mjOBJ_JOINT, i)?.to_string();
                let dofs = match jnt_type[i] {
                    MjtJoint::mjJNT_FREE => 7,
                    MjtJoint::mjJNT_BALL => 4,
                    MjtJoint::mjJNT_SLIDE | MjtJoint::mjJNT_HINGE => 1,
                };
                Some(crate::trajectory::JointTrack { name, dofs })
            })
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn joint_tracks(&self) -> Vec<crate::trajectory::JointTrack> {
        Vec::new()
    }

    /// Writes one frame of a [`crate::trajectory::RetargetedClip`] straight
    /// into the sim's `qpos` (per joint, by name -- the clip was already
    /// remapped onto this model's own `joint_tracks` by
    /// [`crate::trajectory::TrajectoryClip::retarget`], so no further
    /// lookups/conversion happens here) and re-runs forward kinematics so
    /// drawn geom poses reflect it immediately, without stepping physics.
    /// Out-of-range `frame_idx` is a no-op. Native only, see
    /// [`joint_tracks`](Self::joint_tracks).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn apply_trajectory_frame(&mut self, clip: &crate::trajectory::RetargetedClip, frame_idx: usize) {
        let Some(frame) = clip.frames.get(frame_idx) else {
            return;
        };
        let mut offset = 0;
        for jt in &clip.joints {
            if let Some(joint) = self.mj_data.joint(&jt.name) {
                joint.view_mut(&mut self.mj_data).qpos.copy_from_slice(&frame[offset..offset + jt.dofs]);
            }
            offset += jt.dofs;
        }
        self.mj_data.forward();
    }

    #[cfg(target_arch = "wasm32")]
    pub fn apply_trajectory_frame(&mut self, _clip: &crate::trajectory::RetargetedClip, _frame_idx: usize) {}
}

/// One mesh-type geom's world placement for this frame, plus which .obj to
/// draw there -- see [`MujocoScene::mesh_geoms`].
pub struct MeshGeomInstance {
    pub mesh_path: String,
    pub rgba: [f32; 4],
    pub position: CgVec3,
    pub rotation: CgQuat,
}

/// Re-walks the MJCF (and any `<include>`d files, recursively) starting from
/// `xml_path` to build a mesh name -> resolved absolute .obj path map.
///
/// mujoco-rs's `MjModel::mesh_pathadr`/`paths` only exposes the MJCF's raw,
/// un-resolved `<mesh file="...">` attribute (e.g. "link0/link0.obj"), not
/// joined with `<compiler meshdir="...">` or the declaring file's own
/// directory -- so it can't be opened as a real path. This does that join
/// ourselves, mirroring MJCF's own resolution rules: `meshdir` (if set) is
/// relative to the file that declares it; a mesh's `file` is relative to
/// `meshdir`; an omitted `<mesh name="...">` defaults to the file's stem
/// (MuJoCo's own default-naming rule), which is how `mesh_geoms` looks
/// entries up (via `MjModel::id_to_name`).
#[cfg(not(target_arch = "wasm32"))]
fn collect_mesh_paths(xml_path: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let mut visited = std::collections::HashSet::new();
    collect_mesh_paths_into(xml_path, &mut map, &mut visited);
    map
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_mesh_paths_into(
    xml_path: &str,
    map: &mut std::collections::HashMap<String, String>,
    visited: &mut std::collections::HashSet<std::path::PathBuf>,
) {
    let path = std::path::Path::new(xml_path);
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if !visited.insert(canon) {
        return; // Already walked (or a cyclic <include>).
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let Ok(doc) = roxmltree::Document::parse(&text) else {
        return;
    };
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut meshdir = dir.to_path_buf();

    for node in doc.descendants().filter(|n| n.is_element()) {
        match node.tag_name().name() {
            "compiler" => {
                if let Some(md) = node.attribute("meshdir") {
                    meshdir = dir.join(md);
                }
            }
            "mesh" => {
                if let Some(file) = node.attribute("file") {
                    let name = node.attribute("name").map(str::to_string).unwrap_or_else(|| {
                        std::path::Path::new(file)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    });
                    map.insert(name, meshdir.join(file).to_string_lossy().into_owned());
                }
            }
            "include" => {
                if let Some(inc_file) = node.attribute("file") {
                    let inc_path = dir.join(inc_file);
                    collect_mesh_paths_into(&inc_path.to_string_lossy(), map, visited);
                }
            }
            _ => {}
        }
    }
}

/// Re-expresses a mesh geom's world transform so it applies to the *source
/// asset's* vertices rather than MuJoCo's own processed copy.
///
/// MuJoCo doesn't keep mesh vertices as authored: at compile time it recenters
/// each mesh on its center of mass and rotates it onto its principal axes of
/// inertia, recording what it did in `mesh_pos`/`mesh_quat` ("translation /
/// rotation applied to asset vertices"). `geom_xpos`/`geom_xmat` then place
/// *that* processed mesh. We render the .obj as authored (see
/// [`MujocoScene::mesh_geoms`]), so applying the geom transform raw would
/// offset every piece by its own centroid and spin it into its own inertia
/// frame. Undoing the asset transform first puts the authored vertices where
/// MuJoCo would have drawn them.
///
/// With `P` = `mesh_pos`, `Q` = `mesh_quat`, MuJoCo's processed vertices are
/// `v_processed = Qᵀ(v_asset - P)`, so substituting into
/// `xpos + R·v_processed` gives the transform this returns:
/// `R_eff = R·Qᵀ`, `pos_eff = xpos - R_eff·P`.
///
/// `mesh_scale` is not undone here: it's a non-uniform scale, which a rigid
/// (position + rotation) instance transform can't carry. Scaled `<mesh>`
/// assets will render at their authored size.
#[cfg(not(target_arch = "wasm32"))]
fn undo_mesh_asset_transform(
    model: &MjModel,
    mesh_id: usize,
    xpos: [f64; 3],
    xmat: [f64; 9],
) -> ([f32; 3], [f32; 9]) {
    use cgmath::Matrix as _; // `transpose`

    // cgmath is column-major; xmat is row-major (column j of R is the image
    // of local axis j, see `mj_basis`).
    let r = cgmath::Matrix3::new(
        xmat[0] as f32, xmat[3] as f32, xmat[6] as f32,
        xmat[1] as f32, xmat[4] as f32, xmat[7] as f32,
        xmat[2] as f32, xmat[5] as f32, xmat[8] as f32,
    );
    let q = model.mesh_quat()[mesh_id];
    // MuJoCo quats are [w, x, y, z]; cgmath's `new` takes (scalar, x, y, z).
    let q = CgQuat::new(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32);
    let r_eff = r * cgmath::Matrix3::from(q).transpose();

    let p = model.mesh_pos()[mesh_id];
    let p = CgVec3::new(p[0] as f32, p[1] as f32, p[2] as f32);
    let pos_eff = CgVec3::new(xpos[0] as f32, xpos[1] as f32, xpos[2] as f32) - r_eff * p;

    // Back to row-major for `mj_basis` (Matrix3's x/y/z fields are columns).
    let xmat_eff = [
        r_eff.x.x, r_eff.y.x, r_eff.z.x,
        r_eff.x.y, r_eff.y.y, r_eff.z.y,
        r_eff.x.z, r_eff.y.z, r_eff.z.z,
    ];
    (pos_eff.into(), xmat_eff)
}

/// Builds a rotation from a geom's world-space basis vectors (already
/// engine-space, see [`mj_basis`]) -- `ex`/`ey`/`ez` are orthonormal, so this
/// is a well-defined proper rotation.
#[cfg(not(target_arch = "wasm32"))]
fn quat_from_basis(ex: CgVec3, ey: CgVec3, ez: CgVec3) -> CgQuat {
    cgmath::Matrix3::from_cols(ex, ey, ez).into()
}

// MuJoCo is Z-up right-handed; the engine is Y-up right-handed. This maps
// (x, y, z)_mujoco -> (x, z, -y)_engine, a proper rotation (det = 1) so it's
// safe to apply to both positions and a geom's local basis vectors.
fn mj_vec3(v: [f32; 3]) -> CgVec3 {
    CgVec3::new(v[0], v[2], -v[1])
}

fn mj_basis(xmat: [f32; 9]) -> (CgVec3, CgVec3, CgVec3) {
    // xmat is row-major 3x3; column j is local axis j in world space.
    let ex = mj_vec3([xmat[0], xmat[3], xmat[6]]);
    let ey = mj_vec3([xmat[1], xmat[4], xmat[7]]);
    let ez = mj_vec3([xmat[2], xmat[5], xmat[8]]);
    (ex, ey, ez)
}

fn draw_wire_circle(
    renderer: &mut Renderer,
    game_config: &Config,
    center: CgVec3,
    u: CgVec3,
    v: CgVec3,
    radius: f32,
    color: CgVec4,
) {
    const SEGMENTS: usize = 20;
    let mut prev = center + u * radius;
    for i in 1..=SEGMENTS {
        let t = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
        let p = center + u * (radius * t.cos()) + v * (radius * t.sin());
        renderer.add_line(&prev, &p, &color, 0.02, 0.001, game_config);
        prev = p;
    }
}

fn draw_wire_box(
    renderer: &mut Renderer,
    game_config: &Config,
    center: CgVec3,
    ex: CgVec3,
    ey: CgVec3,
    ez: CgVec3,
    half: CgVec3,
    color: CgVec4,
) {
    let corner =
        |sx: f32, sy: f32, sz: f32| center + ex * (half.x * sx) + ey * (half.y * sy) + ez * (half.z * sz);
    let c = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(-1.0, 1.0, -1.0),
        corner(-1.0, -1.0, 1.0),
        corner(1.0, -1.0, 1.0),
        corner(1.0, 1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
    ];
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (a, b) in EDGES {
        renderer.add_line(&c[a], &c[b], &color, 0.02, 0.001, game_config);
    }
}

fn draw_mj_geom(
    renderer: &mut Renderer,
    game_config: &Config,
    kind: u32,
    size: [f32; 3],
    rgba: [f32; 4],
    xpos: [f32; 3],
    xmat: [f32; 9],
    origin: CgVec3,
    rotation: CgQuat,
) {
    let color = CgVec4::new(rgba[0], rgba[1], rgba[2], 1.0);
    let center = origin + rotation * mj_vec3(xpos);
    let (ex, ey, ez) = mj_basis(xmat);
    let (ex, ey, ez) = (rotation * ex, rotation * ey, rotation * ez);

    match kind {
        MJ_GEOM_PLANE => {
            let hx = if size[0] > 0.0 { size[0] } else { 3.0 };
            let hy = if size[1] > 0.0 { size[1] } else { 3.0 };
            let corners = [
                center + ex * hx + ey * hy,
                center - ex * hx + ey * hy,
                center - ex * hx - ey * hy,
                center + ex * hx - ey * hy,
            ];
            for i in 0..4 {
                renderer.add_line(&corners[i], &corners[(i + 1) % 4], &color, 0.02, 0.001, game_config);
            }
            const DIVS: i32 = 6;
            for i in -DIVS..=DIVS {
                let t = i as f32 / DIVS as f32;
                let a = center + ex * (hx * t) + ey * hy;
                let b = center + ex * (hx * t) - ey * hy;
                renderer.add_line(&a, &b, &color, 0.008, 0.001, game_config);
                let a = center + ey * (hy * t) + ex * hx;
                let b = center + ey * (hy * t) - ex * hx;
                renderer.add_line(&a, &b, &color, 0.008, 0.001, game_config);
            }
        }
        MJ_GEOM_SPHERE => {
            let r = size[0];
            draw_wire_circle(renderer, game_config, center, ex, ey, r, color);
            draw_wire_circle(renderer, game_config, center, ex, ez, r, color);
            draw_wire_circle(renderer, game_config, center, ey, ez, r, color);
        }
        MJ_GEOM_CAPSULE | MJ_GEOM_CYLINDER => {
            let r = size[0];
            let half_len = size[1];
            let top = center + ez * half_len;
            let bottom = center - ez * half_len;
            draw_wire_circle(renderer, game_config, top, ex, ey, r, color);
            draw_wire_circle(renderer, game_config, bottom, ex, ey, r, color);
            const SIDES: usize = 8;
            for i in 0..SIDES {
                let t = (i as f32 / SIDES as f32) * std::f32::consts::TAU;
                let offset = ex * (r * t.cos()) + ey * (r * t.sin());
                renderer.add_line(&(top + offset), &(bottom + offset), &color, 0.02, 0.001, game_config);
            }
        }
        MJ_GEOM_BOX => {
            let half = CgVec3::new(size[0], size[1], size[2]);
            draw_wire_box(renderer, game_config, center, ex, ey, ez, half, color);
        }
        // Drawn as a real triangle mesh by the caller (see
        // `MujocoScene::mesh_geoms`), not wireframed here.
        MJ_GEOM_MESH => {}
        _ => {
            // No wireframe for this geom type yet (ellipsoid, mesh, hfield,
            // ...); draw a small axis cross so it's still visible.
            let s = 0.1;
            renderer.add_line(&(center - ex * s), &(center + ex * s), &color, 0.02, 0.001, game_config);
            renderer.add_line(&(center - ey * s), &(center + ey * s), &color, 0.02, 0.001, game_config);
            renderer.add_line(&(center - ez * s), &(center + ez * s), &color, 0.02, 0.001, game_config);
        }
    }
}

// Bridge to the sibling `@mujoco/mujoco` wasm module that a game's
// index.html loads and steps in its own requestAnimationFrame loop. It calls
// back into these #[wasm_bindgen] exports once per frame with the current
// geom arrays (a straight copy -- the two wasm modules have separate linear
// memories, so there's no cheaper option), and `MujocoScene` reads whatever
// it last sent when drawing or resolving a named geom/joint.
#[cfg(target_arch = "wasm32")]
mod wasm_bridge {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use wasm_bindgen::prelude::*;

    #[derive(Clone, Copy)]
    pub struct GeomFrame {
        pub kind: u32,
        pub size: [f32; 3],
        pub rgba: [f32; 4],
        pub xpos: [f32; 3],
        pub xmat: [f32; 9],
    }

    thread_local! {
        static MODEL_XML: RefCell<Option<String>> = const { RefCell::new(None) };
        static GEOMS: RefCell<Vec<GeomFrame>> = const { RefCell::new(Vec::new()) };
        // Name -> id/dof, filled in once JS resolves it against its own
        // parsed model (see mj_bridge_report_geom_id/mj_bridge_report_joint_dof).
        static WATCHED_GEOMS: RefCell<HashMap<String, Option<usize>>> = RefCell::new(HashMap::new());
        static WATCHED_JOINTS: RefCell<HashMap<String, Option<u32>>> = RefCell::new(HashMap::new());
        // (playing, speed) -- JS's frame() loop polls this each frame to
        // gate/scale its own stepping, since the sim itself runs in JS.
        static PLAYBACK: RefCell<(bool, f32)> = const { RefCell::new((true, 1.0)) };
        // One-shot flags JS polls-and-clears each frame.
        static SINGLE_STEP_REQUESTED: RefCell<bool> = const { RefCell::new(false) };
        static RESET_REQUESTED: RefCell<bool> = const { RefCell::new(false) };
    }

    pub(super) fn set_model_xml(xml: String) {
        MODEL_XML.with(|m| *m.borrow_mut() = Some(xml));
    }

    pub(super) fn set_playback(playing: bool, speed: f32) {
        PLAYBACK.with(|p| *p.borrow_mut() = (playing, speed));
    }

    pub(super) fn request_single_step() {
        SINGLE_STEP_REQUESTED.with(|f| *f.borrow_mut() = true);
    }

    pub(super) fn request_reset() {
        RESET_REQUESTED.with(|f| *f.borrow_mut() = true);
    }

    pub(super) fn with_geoms<R>(f: impl FnOnce(&[GeomFrame]) -> R) -> R {
        GEOMS.with(|g| f(&g.borrow()))
    }

    pub(super) fn watch_geom(name: &str) {
        WATCHED_GEOMS.with(|m| {
            m.borrow_mut().entry(name.to_string()).or_insert(None);
        });
    }

    pub(super) fn named_geom_id(name: &str) -> Option<usize> {
        WATCHED_GEOMS.with(|m| m.borrow().get(name).copied().flatten())
    }

    pub(super) fn watch_joint(name: &str) {
        WATCHED_JOINTS.with(|m| {
            m.borrow_mut().entry(name.to_string()).or_insert(None);
        });
    }

    pub(super) fn named_joint_dof(name: &str) -> Option<u32> {
        WATCHED_JOINTS.with(|m| m.borrow().get(name).copied().flatten())
    }

    pub(super) fn apply_qvel(dof_index: u32, delta: f64) {
        js_apply_qvel(dof_index, delta);
    }

    /// JS polls this (the model text is only known once `MujocoScene::load`'s
    /// async asset fetch resolves) before loading its own MuJoCo wasm module
    /// -- returns "" until a scene has been loaded.
    #[wasm_bindgen]
    pub fn mj_bridge_model_xml() -> String {
        MODEL_XML.with(|m| m.borrow().clone().unwrap_or_default())
    }

    /// (playing, speed) for JS's frame() loop to gate/scale its own
    /// stepping -- `speed` is meaningless while `playing` is false.
    #[wasm_bindgen]
    pub fn mj_bridge_playing() -> bool {
        PLAYBACK.with(|p| p.borrow().0)
    }

    #[wasm_bindgen]
    pub fn mj_bridge_speed() -> f32 {
        PLAYBACK.with(|p| p.borrow().1)
    }

    /// JS polls this once per frame and, if true, steps exactly once
    /// regardless of `mj_bridge_playing` -- backs a "Step" button.
    #[wasm_bindgen]
    pub fn mj_bridge_take_single_step_requested() -> bool {
        SINGLE_STEP_REQUESTED.with(|f| f.replace(false))
    }

    /// JS polls this once per frame and, if true, resets the sim to its
    /// initial state (`mj_resetData` equivalent) -- backs a "Reset" button.
    #[wasm_bindgen]
    pub fn mj_bridge_take_reset_requested() -> bool {
        RESET_REQUESTED.with(|f| f.replace(false))
    }

    /// Names a game asked to `watch_geom`/`watch_joint`, so JS can resolve
    /// each one against its own parsed model and report the id/dof back.
    #[wasm_bindgen]
    pub fn mj_bridge_watched_geom_names() -> Vec<String> {
        WATCHED_GEOMS.with(|m| m.borrow().keys().cloned().collect())
    }

    #[wasm_bindgen]
    pub fn mj_bridge_watched_joint_names() -> Vec<String> {
        WATCHED_JOINTS.with(|m| m.borrow().keys().cloned().collect())
    }

    #[wasm_bindgen]
    pub fn mj_bridge_report_geom_id(name: String, id: u32) {
        WATCHED_GEOMS.with(|m| {
            m.borrow_mut().insert(name, Some(id as usize));
        });
    }

    #[wasm_bindgen]
    pub fn mj_bridge_report_joint_dof(name: String, dof_index: u32) {
        WATCHED_JOINTS.with(|m| {
            m.borrow_mut().insert(name, Some(dof_index));
        });
    }

    /// `sizes`/`xpos` are ngeom*3, `rgba` is ngeom*4, `xmat` is ngeom*9 --
    /// same flat row-major layout mujoco-rs uses natively.
    #[wasm_bindgen]
    pub fn mj_bridge_set_geoms(
        types: &[u32],
        sizes: &[f32],
        rgba: &[f32],
        xpos: &[f32],
        xmat: &[f32],
    ) {
        let ngeom = types.len();
        if sizes.len() != ngeom * 3
            || rgba.len() != ngeom * 4
            || xpos.len() != ngeom * 3
            || xmat.len() != ngeom * 9
        {
            return; // Malformed frame (shouldn't happen); skip it.
        }
        let frames: Vec<GeomFrame> = (0..ngeom)
            .map(|i| GeomFrame {
                kind: types[i],
                size: [sizes[i * 3], sizes[i * 3 + 1], sizes[i * 3 + 2]],
                rgba: [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]],
                xpos: [xpos[i * 3], xpos[i * 3 + 1], xpos[i * 3 + 2]],
                xmat: std::array::from_fn(|k| xmat[i * 9 + k]),
            })
            .collect();
        GEOMS.with(|g| *g.borrow_mut() = frames);
    }

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = window, js_name = "__bsMujocoApplyQvel")]
        fn js_apply_qvel(dof_index: u32, delta: f64);
    }
}
