//! MJCF loading and rendering for MuJoCo scenes.
//!
//! MuJoCo owns parsing and forward kinematics; the engine never computes a
//! pose itself -- it only reads geom poses each frame and draws what MuJoCo
//! reports. Primitive geoms are wireframed here (see `draw_mj_geom`);
//! mesh geoms are drawn as real triangle meshes by the caller, from the .obj
//! the MJCF names (see [`MujocoScene::mesh_geoms`], which `draw_mj_geom`
//! can't do itself -- it has a `Renderer` but no `AssetManager`).
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
//!
//! The split runs deeper than stepping. Native MuJoCo opens an MJCF's
//! `<include>`s and meshes off the filesystem itself; the wasm build has no
//! filesystem, so [`MujocoScene::load_bundle`] gathers every file the model
//! needs up front and the JS half stages them into MuJoCo's in-memory FS
//! before loading. Trajectory playback splits the same way: native writes
//! `qpos` through mujoco-rs, while wasm reads the joint layout JS reported at
//! load ([`MujocoScene::joint_tracks`]) and sends resolved qpos addresses
//! back for JS to write ([`MujocoScene::apply_trajectory_frame`]).

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

/// Synthetic path a [`MujocoScene::from_xml_str`] model is staged under on
/// wasm -- it has no real one, and MuJoCo is only ever asked to load by path.
#[cfg(target_arch = "wasm32")]
const INLINE_MJCF_PATH: &str = "inline_model.xml";

/// An MJCF plus every file it needs to compile, as gathered by
/// [`MujocoScene::load_bundle`] and consumed by
/// [`MujocoScene::from_bundle`].
pub struct MjcfBundle {
    /// Path of the entry MJCF -- what MuJoCo is asked to load.
    pub root: String,
    /// `(path, bytes)` for the entry MJCF, every `<include>` reachable from
    /// it, and every asset any of them reference. Paths are as written in the
    /// MJCF graph (already joined against the declaring file's directory), so
    /// staging them verbatim reproduces the layout MuJoCo expects to resolve
    /// against.
    ///
    /// Empty on native: MuJoCo reads the real filesystem itself, so `root`
    /// alone is enough.
    pub files: Vec<(String, Vec<u8>)>,
    /// mesh name -> .obj path, for `MujocoScene::mesh_geoms`. Populated only
    /// on wasm, which has to walk the MJCF to gather `files` anyway and so
    /// gets this for free; native's `from_xml_path` builds its own with
    /// [`collect_mesh_paths`] instead, since it can be called without a
    /// bundle at all.
    pub mesh_paths: std::collections::HashMap<String, String>,
}

/// One MJCF file's outbound file references, already joined against the
/// declaring file's own directory (and `meshdir`/`texturedir` where those
/// apply), so each is ready to open or fetch as-is. See [`scan_mjcf`].
struct MjcfRefs {
    /// Other MJCF files pulled in by `<include>`.
    includes: Vec<std::path::PathBuf>,
    /// `<mesh>` assets as `(name, path)`, where `name` is how `mesh_geoms`
    /// looks an entry up.
    meshes: Vec<(String, std::path::PathBuf)>,
    /// Other files MuJoCo opens while compiling: textures, heightfields,
    /// skins. Only wasm cares -- these are staged, never resolved by name.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    other_assets: Vec<std::path::PathBuf>,
}

/// Scans one MJCF file's text for the files it references, mirroring MJCF's
/// own resolution rules: `<compiler assetdir/meshdir/texturedir>` are taken
/// relative to the declaring file's directory; a `<mesh>`/`<hfield>`/`<skin>`
/// `file` is relative to `meshdir`, a `<texture>` `file` to `texturedir`, and
/// an `<include>` `file` to the directory itself. `assetdir` sets the default
/// for both `meshdir` and `texturedir`; an explicit either overrides it.
///
/// A `<mesh>` with no `name` takes the file's stem, which is MuJoCo's own
/// default-naming rule and what `mesh_geoms` relies on when it looks a mesh
/// up via `MjModel::id_to_name`.
///
/// Document order is load-bearing here exactly as it is for MuJoCo:
/// `<compiler>` precedes `<asset>` in any well-formed MJCF, so the asset
/// dirs are known by the time the assets referencing them are seen.
fn scan_mjcf(dir: &std::path::Path, text: &str) -> MjcfRefs {
    let mut refs =
        MjcfRefs { includes: Vec::new(), meshes: Vec::new(), other_assets: Vec::new() };
    let Ok(doc) = roxmltree::Document::parse(text) else {
        return refs;
    };
    let mut meshdir = dir.to_path_buf();
    let mut texturedir = dir.to_path_buf();

    for node in doc.descendants().filter(|n| n.is_element()) {
        match node.tag_name().name() {
            "compiler" => {
                if let Some(ad) = node.attribute("assetdir") {
                    meshdir = dir.join(ad);
                    texturedir = dir.join(ad);
                }
                if let Some(md) = node.attribute("meshdir") {
                    meshdir = dir.join(md);
                }
                if let Some(td) = node.attribute("texturedir") {
                    texturedir = dir.join(td);
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
                    refs.meshes.push((name, meshdir.join(file)));
                }
            }
            // Both resolve against meshdir, same as <mesh>.
            "hfield" | "skin" => {
                if let Some(file) = node.attribute("file") {
                    refs.other_assets.push(meshdir.join(file));
                }
            }
            "texture" => {
                if let Some(file) = node.attribute("file") {
                    refs.other_assets.push(texturedir.join(file));
                }
            }
            "include" => {
                if let Some(file) = node.attribute("file") {
                    refs.includes.push(dir.join(file));
                }
            }
            _ => {}
        }
    }
    refs
}

/// Picks the entry MJCF out of a folder that holds several: the one no other
/// file `<include>`s.
///
/// A mujoco_menagerie model ships both a `scene.xml` (floor, lights, skybox)
/// and the bare robot it includes (`panda.xml`). Both parse, but only the
/// outermost is meant to be loaded -- open the fragment and you get an arm
/// floating in a void. Rather than hardcode menagerie's `scene.xml` naming,
/// this reads the actual include graph: whatever nothing else pulls in is a
/// root.
///
/// `files` is `(path, text)` for every candidate `.xml`, with paths relative
/// to a common base -- include edges are resolved against each file's own
/// directory, so they only compare equal to the paths here if both share a
/// base.
///
/// Ties (a folder with two independent scenes) break toward a `scene`-named
/// file and then alphabetically, so the pick is at least deterministic.
/// `None` if `files` is empty, or if every candidate is included by another
/// (an include cycle -- which MuJoCo would reject anyway).
pub fn find_root_mjcf(files: &[(String, String)]) -> Option<String> {
    // Keyed by PathBuf, not String: Path compares and hashes by component, so
    // an edge joined into "dir\panda.xml" still matches a "dir/panda.xml"
    // key. Only wasm calls this today, where separators are always '/', but
    // that's a thin thing to rely on.
    let mut included: std::collections::HashSet<std::path::PathBuf> =
        std::collections::HashSet::new();
    for (path, text) in files {
        let dir = std::path::Path::new(path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""));
        included.extend(scan_mjcf(dir, text).includes);
    }

    let mut roots: Vec<&String> = files
        .iter()
        .map(|(path, _)| path)
        .filter(|path| !included.contains(std::path::Path::new(path.as_str())))
        .collect();
    roots.sort();
    roots
        .iter()
        .find(|path| {
            std::path::Path::new(path)
                .file_name()
                .is_some_and(|f| f.to_string_lossy().to_ascii_lowercase().starts_with("scene"))
        })
        .or_else(|| roots.first())
        .map(|path| path.to_string())
}

/// Walks `root`'s `<include>` graph and fetches every file the model
/// references, for staging into MuJoCo's in-memory FS -- see
/// [`MujocoScene::load_bundle`], which documents why wasm needs this.
///
/// Any file failing to fetch is an error rather than a skip: MuJoCo would
/// fail to compile the model anyway, and it reports a missing file far less
/// legibly than the fetch does.
#[cfg(target_arch = "wasm32")]
async fn load_bundle_wasm(root: &str) -> anyhow::Result<MjcfBundle> {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut mesh_paths = std::collections::HashMap::new();
    let mut assets: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut queue = vec![root.to_string()];

    while let Some(xml_path) = queue.pop() {
        if !seen.insert(xml_path.clone()) {
            continue; // Already walked, or a cyclic <include>.
        }
        let text = crate::assets::load_string(&xml_path)
            .await
            .map_err(|e| anyhow::anyhow!("MJCF {xml_path}: {e}"))?;
        let dir = std::path::Path::new(&xml_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""))
            .to_path_buf();
        let refs = scan_mjcf(&dir, &text);
        files.push((xml_path, text.into_bytes()));

        for (name, path) in refs.meshes {
            let path = path.to_string_lossy().into_owned();
            assets.push(path.clone());
            mesh_paths.insert(name, path);
        }
        for path in refs.other_assets {
            assets.push(path.to_string_lossy().into_owned());
        }
        for path in refs.includes {
            queue.push(path.to_string_lossy().into_owned());
        }
    }

    // Two geoms commonly share one <mesh>, and two MJCFs can declare the same
    // asset -- fetch each distinct file once.
    assets.sort();
    assets.dedup();
    for path in assets {
        let bytes = crate::assets::load_binary(&path)
            .await
            .map_err(|e| anyhow::anyhow!("MJCF asset {path}: {e}"))?;
        files.push((path, bytes));
    }
    Ok(MjcfBundle { root: root.to_string(), files, mesh_paths })
}

/// A loaded MuJoCo scene: owns and steps the sim (native) or reflects the
/// sibling wasm module's latest frame (wasm32), and draws every geom as a
/// wireframe via the engine's line pass.
pub struct MujocoScene {
    #[cfg(not(target_arch = "wasm32"))]
    mj_data: MjData<Box<MjModel>>,
    #[cfg(not(target_arch = "wasm32"))]
    sim_time_accum: f32,
    // mesh name -> resolved .obj path, for `mesh_geoms`. Native builds it with
    // `collect_mesh_paths` -- mujoco-rs's own `mesh_pathadr` only exposes the
    // MJCF's raw, un-resolved `file` attribute (e.g. "link0/link0.obj", not
    // joined with `meshdir` or the declaring file's directory), so it can't
    // be opened as-is. Wasm builds the same map from the bundle it just
    // staged, since it has the MJCF text in hand anyway.
    mesh_paths: std::collections::HashMap<String, String>,
    /// Whether `tick_and_draw` advances the sim each frame. Paused scenes
    /// still draw their current pose.
    playing: bool,
    /// Multiplies `game_config.delta_time` before it's accumulated into sim
    /// steps -- 1.0 is realtime, 0.5 is half-speed, etc.
    speed: f32,
    /// Suppresses physics stepping in `tick_and_draw`, leaving `qpos` under
    /// whoever is writing it -- see [`set_kinematic`](Self::set_kinematic).
    kinematic: bool,
    /// Whether `tick_and_draw` wireframes primitive geoms. Mesh geoms are
    /// unaffected -- the caller draws those itself from [`mesh_geoms`](Self::mesh_geoms).
    wireframe: bool,
}

impl MujocoScene {
    /// Loads an MJCF file and everything it references, then parses it.
    /// Convenience wrapper over [`load_bundle`](Self::load_bundle) +
    /// [`from_bundle`](Self::from_bundle) for callers that don't need to run
    /// the fetch separately from the parse.
    pub async fn load(file_path: &str) -> anyhow::Result<Self> {
        Self::from_bundle(Self::load_bundle(file_path).await?)
    }

    /// Gathers everything MuJoCo needs to compile the model at `path` -- an
    /// async fetch, kept separate from the (synchronous) parse in
    /// [`from_bundle`](Self::from_bundle) so callers can run it off the frame
    /// loop. Cheap on native and expensive on wasm, for the same underlying
    /// reason:
    ///
    /// Native MuJoCo opens `path` and every `<include>`/`<mesh>`/`<texture>`
    /// it references straight from the real filesystem, resolving each
    /// relative to the declaring file -- so there's nothing to gather up
    /// front and the returned bundle carries no bytes at all.
    ///
    /// The wasm build has no filesystem to resolve against. It gets an
    /// emscripten in-memory FS instead, which the JS half stages this
    /// bundle's files into before asking MuJoCo to load `root` by path (see
    /// `examples/splat/index.html`). That only works if we hand it *every*
    /// file up front, so this walks the MJCF's `<include>` graph and asset
    /// references and fetches each one -- meshes included, which for a
    /// mujoco_menagerie arm is tens of megabytes.
    ///
    /// Parsing the MJCF from a string instead would sidestep the staging but
    /// not the problem: MuJoCo stages string-loaded text into a synthetic VFS
    /// entry with no real path, so every relative reference in it is looked
    /// up against the process CWD and fails. That's what `<include>` did
    /// before this existed.
    pub async fn load_bundle(path: &str) -> anyhow::Result<MjcfBundle> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(MjcfBundle {
                root: path.to_string(),
                files: Vec::new(),
                mesh_paths: std::collections::HashMap::new(),
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            load_bundle_wasm(path).await
        }
    }

    /// Parses a bundle from [`load_bundle`](Self::load_bundle) into a live
    /// scene: native loads `root` from disk directly; wasm hands the files to
    /// the JS bridge, which stages them into MuJoCo's in-memory FS and loads
    /// `root` from there.
    pub fn from_bundle(bundle: MjcfBundle) -> anyhow::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::from_xml_path(&bundle.root)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let MjcfBundle { root, files, mesh_paths } = bundle;
            wasm_bridge::set_model_bundle(root, files);
            // The bridge is one global sim, so a new scene has to push its
            // whole starting state -- leaving `kinematic` set from whatever
            // clip the *previous* scene had bound would freeze this one.
            wasm_bridge::set_playback(true, 1.0);
            wasm_bridge::set_kinematic(false);
            Ok(Self { playing: true, speed: 1.0, kinematic: false, wireframe: true, mesh_paths })
        }
    }

    /// Loads and parses an MJCF file directly from disk (native only) --
    /// preserves the file's directory, so MuJoCo resolves relative
    /// `<include>`/`<mesh>` references against it (see
    /// [`load_bundle`](Self::load_bundle) for why wasm can't do this).
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
            kinematic: false,
            wireframe: true,
        })
    }

    /// Parses an already-loaded MJCF string (e.g. an `include_str!`'d scene).
    /// A bare string has no directory to resolve against, so this only suits
    /// self-contained models -- any relative `<include>`/`<mesh>` reference in
    /// `xml` will fail to open on either target. Use
    /// [`load`](Self::load)/[`load_bundle`](Self::load_bundle) for a model
    /// that references other files.
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
                kinematic: false,
                wireframe: true,
            })
        }
        #[cfg(target_arch = "wasm32")]
        {
            // The bridge only knows how to stage-and-load-by-path (see
            // `load_bundle`), so a lone string becomes a one-file bundle
            // under a synthetic name. No mesh_paths: a string model can't
            // reference a mesh file that would resolve anyway.
            wasm_bridge::set_model_bundle(
                INLINE_MJCF_PATH.to_string(),
                vec![(INLINE_MJCF_PATH.to_string(), xml.as_bytes().to_vec())],
            );
            wasm_bridge::set_playback(true, 1.0);
            Ok(Self {
                playing: true,
                speed: 1.0,
                kinematic: false,
                wireframe: true,
                mesh_paths: std::collections::HashMap::new(),
            })
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

    pub fn is_kinematic(&self) -> bool {
        self.kinematic
    }

    pub fn draws_wireframe(&self) -> bool {
        self.wireframe
    }

    /// Shows/hides the wireframe drawn for primitive geoms (the sim keeps
    /// stepping either way). Mesh geoms are drawn by the caller from
    /// [`mesh_geoms`](Self::mesh_geoms) and stay visible regardless.
    pub fn set_wireframe(&mut self, wireframe: bool) {
        self.wireframe = wireframe;
    }

    /// Stops [`tick_and_draw`](Self::tick_and_draw) from stepping physics,
    /// so whatever last wrote `qpos` stays put.
    ///
    /// Set this while replaying a recorded trajectory
    /// ([`apply_trajectory_frame`](Self::apply_trajectory_frame)). Most MJCFs
    /// -- mujoco_menagerie's `panda.xml` included -- drive their joints with
    /// position servos, which would spend every step hauling the arm from the
    /// pose the clip just wrote back toward `ctrl`, so the replay would fight
    /// the controller instead of showing the recorded motion. A recorded demo
    /// is already a full `qpos` history: there's nothing left for the solver
    /// to work out, and stepping can only corrupt it.
    ///
    /// The tradeoff is that nothing else in the sim moves either -- no gravity,
    /// no contacts, so objects the arm "grasps" won't be carried along. Driving
    /// the actuators instead (write `ctrl`, keep stepping) is the physical
    /// alternative, at the cost of the arm lagging the demo.
    pub fn set_kinematic(&mut self, kinematic: bool) {
        self.kinematic = kinematic;
        // Wasm steps in JS, so this has to travel there to have any effect --
        // `tick_and_draw_at`'s own stepping guard is native-only.
        #[cfg(target_arch = "wasm32")]
        wasm_bridge::set_kinematic(kinematic);
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
            // catch-up storm. Paused and kinematic scenes skip stepping but
            // still draw their current pose below.
            if self.playing && !self.kinematic {
                let dt = self.mj_data.model().opt().timestep as f32;
                self.sim_time_accum += game_config.delta_time * self.speed;
                let mut steps = 0;
                while self.sim_time_accum >= dt && steps < 8 {
                    self.mj_data.step();
                    self.sim_time_accum -= dt;
                    steps += 1;
                }
            }

            if self.wireframe {
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
        }
        #[cfg(target_arch = "wasm32")]
        {
            if self.wireframe {
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
    /// `Model::from_obj_bytes` for `.obj`) and draw one instance per entry --
    /// `draw_mj_geom` skips `MJ_GEOM_MESH` since it can't do this itself (it
    /// only has a `Renderer`, not an `AssetManager`). `origin`/`rotation`
    /// place the whole scene the same way `tick_and_draw_at` does.
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

    /// Wasm's half of [`mesh_geoms`](Self::mesh_geoms): the same result, from
    /// the same inputs, just gathered differently. The per-frame world
    /// transform comes from the geom snapshot JS last pushed, and everything
    /// static (which mesh, what colour, the mesh asset transform) from the
    /// table JS reported at load -- see `wasm_bridge::MeshGeomInfo` for why
    /// those can't be read on this side.
    #[cfg(target_arch = "wasm32")]
    pub fn mesh_geoms(&self, origin: CgVec3, rotation: CgQuat) -> Vec<MeshGeomInstance> {
        wasm_bridge::with_mesh_geoms(|mesh_geoms| {
            wasm_bridge::with_geoms(|geoms| {
                let mut out = Vec::new();
                // Driven by geom index ascending rather than by iterating the
                // table: that matches native's 0..ngeom order, and callers
                // index-align their own state against this vec frame to frame
                // (see the splat editor's `mesh_geom_actors`), so a
                // HashMap's iteration order won't do.
                for (i, geom) in geoms.iter().enumerate() {
                    let Some(info) = mesh_geoms.get(&i) else {
                        continue; // Not a mesh geom.
                    };
                    let Some(mesh_path) = self.mesh_paths.get(&info.mesh_name) else {
                        continue;
                    };
                    let (xpos_eff, xmat_eff) = undo_mesh_asset_transform_raw(
                        info.mesh_pos,
                        info.mesh_quat,
                        geom.xpos,
                        geom.xmat,
                    );
                    let (ex, ey, ez) = mj_basis(xmat_eff);
                    let (ex, ey, ez) = (rotation * ex, rotation * ey, rotation * ez);
                    out.push(MeshGeomInstance {
                        mesh_path: mesh_path.clone(),
                        rgba: info.rgba,
                        position: origin + rotation * mj_vec3(xpos_eff),
                        rotation: quat_from_basis(ex, ey, ez),
                    });
                }
                out
            })
        })
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
    /// can play it back. Unnamed joints are skipped -- a clip addresses
    /// joints by name, so one that can't be named can't be retargeted onto.
    ///
    /// On wasm this reads back what JS reported at load
    /// (`mj_bridge_set_joint_track`), so it's empty until a model finishes
    /// loading there -- callers re-dispatch once `scene` exists.
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
        // Keyed by joint index and iterated in key order, so this comes back
        // in the model's own joint order like native's `0..njnt` -- retarget
        // output order follows this, and the trajectory cache keys on it.
        wasm_bridge::with_joint_tracks(|joints| {
            joints
                .values()
                .map(|j| crate::trajectory::JointTrack { name: j.name.clone(), dofs: j.dofs })
                .collect()
        })
    }

    /// Writes one frame of a [`crate::trajectory::RetargetedClip`] straight
    /// into the sim's `qpos` (per joint, by name -- the clip was already
    /// remapped onto this model's own `joint_tracks` by
    /// [`crate::trajectory::TrajectoryClip::retarget`], so no further
    /// lookups/conversion happens here) and re-runs forward kinematics so
    /// drawn geom poses reflect it immediately, without stepping physics.
    /// Out-of-range `frame_idx` is a no-op.
    ///
    /// Pair this with [`set_kinematic`](Self::set_kinematic) -- otherwise
    /// stepping fights the pose this just wrote.
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

    /// Wasm's half of the above. The sim lives in the sibling MuJoCo module,
    /// so this resolves the frame to flat (qpos address, value) pairs and
    /// hands those to JS to write -- addresses rather than names, because JS
    /// would otherwise re-resolve every joint by name every frame. JS runs
    /// forward kinematics once the whole frame has landed.
    #[cfg(target_arch = "wasm32")]
    pub fn apply_trajectory_frame(&mut self, clip: &crate::trajectory::RetargetedClip, frame_idx: usize) {
        let Some(frame) = clip.frames.get(frame_idx) else {
            return;
        };
        let (adrs, values) = wasm_bridge::with_joint_tracks(|joints| {
            let by_name: std::collections::HashMap<&str, &wasm_bridge::JointInfo> =
                joints.values().map(|j| (j.name.as_str(), j)).collect();
            let mut adrs: Vec<u32> = Vec::new();
            let mut values: Vec<f64> = Vec::new();
            let mut offset = 0;
            for jt in &clip.joints {
                // The clip was retargeted onto this model's own joint_tracks,
                // so a miss here means the model was swapped out from under a
                // clip that's already bound -- skip rather than write another
                // joint's qpos.
                if let (Some(info), Some(vals)) =
                    (by_name.get(jt.name.as_str()), frame.get(offset..offset + jt.dofs))
                {
                    if info.dofs == jt.dofs {
                        adrs.extend((0..jt.dofs).map(|k| info.qpos_adr + k as u32));
                        values.extend_from_slice(vals);
                    }
                }
                offset += jt.dofs;
            }
            (adrs, values)
        });
        if !adrs.is_empty() {
            wasm_bridge::set_qpos(&adrs, &values);
        }
    }
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
/// `xml_path` to build a mesh name -> resolved .obj path map, applying
/// MJCF's own resolution rules via [`scan_mjcf`].
///
/// mujoco-rs's `MjModel::mesh_pathadr`/`paths` only exposes the MJCF's raw,
/// un-resolved `<mesh file="...">` attribute (e.g. "link0/link0.obj"), not
/// joined with `<compiler meshdir="...">` or the declaring file's own
/// directory -- so it can't be opened as a real path. This does that join
/// ourselves.
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
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let refs = scan_mjcf(dir, &text);

    for (name, mesh_path) in refs.meshes {
        map.insert(name, mesh_path.to_string_lossy().into_owned());
    }
    for inc_path in refs.includes {
        collect_mesh_paths_into(&inc_path.to_string_lossy(), map, visited);
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
    let p = model.mesh_pos()[mesh_id];
    let q = model.mesh_quat()[mesh_id];
    undo_mesh_asset_transform_raw(
        std::array::from_fn(|k| p[k] as f32),
        std::array::from_fn(|k| q[k] as f32),
        std::array::from_fn(|k| xpos[k] as f32),
        std::array::from_fn(|k| xmat[k] as f32),
    )
}

/// The math behind [`undo_mesh_asset_transform`], over plain values rather
/// than an `MjModel` -- wasm has no model to read `mesh_pos`/`mesh_quat`
/// from, only what JS reported over the bridge (see
/// `wasm_bridge::set_mesh_table`).
fn undo_mesh_asset_transform_raw(
    mesh_pos: [f32; 3],
    mesh_quat: [f32; 4],
    xpos: [f32; 3],
    xmat: [f32; 9],
) -> ([f32; 3], [f32; 9]) {
    use cgmath::Matrix as _; // `transpose`

    // cgmath is column-major; xmat is row-major (column j of R is the image
    // of local axis j, see `mj_basis`).
    let r = cgmath::Matrix3::new(
        xmat[0], xmat[3], xmat[6],
        xmat[1], xmat[4], xmat[7],
        xmat[2], xmat[5], xmat[8],
    );
    // MuJoCo quats are [w, x, y, z]; cgmath's `new` takes (scalar, x, y, z).
    let q = CgQuat::new(mesh_quat[0], mesh_quat[1], mesh_quat[2], mesh_quat[3]);
    let r_eff = r * cgmath::Matrix3::from(q).transpose();

    let p = CgVec3::new(mesh_pos[0], mesh_pos[1], mesh_pos[2]);
    let pos_eff = CgVec3::new(xpos[0], xpos[1], xpos[2]) - r_eff * p;

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
    use std::collections::{BTreeMap, HashMap};
    use wasm_bindgen::prelude::*;

    #[derive(Clone, Copy)]
    pub struct GeomFrame {
        pub kind: u32,
        pub size: [f32; 3],
        pub rgba: [f32; 4],
        pub xpos: [f32; 3],
        pub xmat: [f32; 9],
    }

    /// One named joint's qpos layout, reported by JS once per model load.
    /// `qpos_adr` is the joint's first index into `data.qpos`; `dofs` is how
    /// many entries from there belong to it (see `MujocoScene::joint_tracks`).
    #[derive(Clone)]
    pub struct JointInfo {
        pub name: String,
        pub dofs: usize,
        pub qpos_adr: u32,
    }

    /// The parts of a mesh geom that don't change frame to frame, reported
    /// once per model load (see `mj_bridge_set_mesh_geom`). Everything else
    /// `MujocoScene::mesh_geoms` needs -- the world transform -- is already
    /// in that geom's per-frame `GeomFrame`.
    #[derive(Clone)]
    pub struct MeshGeomInfo {
        /// Which `<mesh>` asset this geom draws, by MJCF name -- the key
        /// `MujocoScene::mesh_paths` resolves to a loadable .obj path.
        pub mesh_name: String,
        /// The geom's colour with its `<material>` already resolved by JS,
        /// since only JS can see `mat_rgba` (see `mj_bridge_set_mesh_geom`).
        pub rgba: [f32; 4],
        /// `mesh_pos`/`mesh_quat` for this geom's mesh asset, for
        /// `undo_mesh_asset_transform_raw`.
        pub mesh_pos: [f32; 3],
        pub mesh_quat: [f32; 4],
    }

    thread_local! {
        // The current model as a path MuJoCo should load plus the files to
        // stage into its FS first (see `MujocoScene::load_bundle`). Bytes are
        // handed over one at a time and dropped as they go -- see
        // `mj_bridge_take_model_file_bytes`.
        static MODEL_ROOT: RefCell<Option<String>> = const { RefCell::new(None) };
        static MODEL_FILES: RefCell<Vec<(String, Vec<u8>)>> = const { RefCell::new(Vec::new()) };
        // Bumped on every staged model. JS reloads when this changes, rather
        // than diffing the model text as it once did: the text is no longer
        // the whole model, and two loads of the same path can differ (an
        // edited MJCF, or a re-picked file).
        static MODEL_GENERATION: RefCell<u32> = const { RefCell::new(0) };
        static GEOMS: RefCell<Vec<GeomFrame>> = const { RefCell::new(Vec::new()) };
        // joint index -> its name and qpos layout, for named joints only.
        // Like MESH_GEOMS this is reported once per load, not per frame. A
        // BTreeMap rather than a HashMap because `joint_tracks` must come
        // back in the model's own joint order.
        static JOINT_TRACKS: RefCell<BTreeMap<usize, JointInfo>> = RefCell::new(BTreeMap::new());
        // Whether a clip owns qpos, in which case JS must not step over it.
        static KINEMATIC: RefCell<bool> = const { RefCell::new(false) };
        // geom index -> its static mesh info, for the mesh-type geoms only.
        // Reported once per load rather than per frame: none of it changes as
        // the sim runs, and it carries a string per entry.
        static MESH_GEOMS: RefCell<HashMap<usize, MeshGeomInfo>> = RefCell::new(HashMap::new());
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

    pub(super) fn set_model_bundle(root: String, files: Vec<(String, Vec<u8>)>) {
        MODEL_ROOT.with(|r| *r.borrow_mut() = Some(root));
        MODEL_FILES.with(|f| *f.borrow_mut() = files);
        // The old model's mesh/joint tables describe a model that no longer
        // exists -- stale qpos addresses would write into whatever joint now
        // occupies them.
        MESH_GEOMS.with(|m| m.borrow_mut().clear());
        JOINT_TRACKS.with(|j| j.borrow_mut().clear());
        MODEL_GENERATION.with(|g| {
            let mut generation = g.borrow_mut();
            *generation = generation.wrapping_add(1);
        });
    }

    pub(super) fn with_mesh_geoms<R>(f: impl FnOnce(&HashMap<usize, MeshGeomInfo>) -> R) -> R {
        MESH_GEOMS.with(|m| f(&m.borrow()))
    }

    pub(super) fn with_joint_tracks<R>(f: impl FnOnce(&BTreeMap<usize, JointInfo>) -> R) -> R {
        JOINT_TRACKS.with(|j| f(&j.borrow()))
    }

    pub(super) fn set_kinematic(kinematic: bool) {
        KINEMATIC.with(|k| *k.borrow_mut() = kinematic);
    }

    pub(super) fn set_qpos(adrs: &[u32], values: &[f64]) {
        js_set_qpos(adrs, values);
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

    /// Bumped every time a model is staged. JS polls this and reloads when it
    /// changes -- the model is only known once `MujocoScene::load_bundle`'s
    /// async fetch resolves, so there's nothing to load at boot.
    #[wasm_bindgen]
    pub fn mj_bridge_model_generation() -> u32 {
        MODEL_GENERATION.with(|g| *g.borrow())
    }

    /// Path to hand MuJoCo's `from_xml_path`, once every file from
    /// [`mj_bridge_model_file_paths`] has been staged into its FS at that
    /// same path. "" until a scene has been loaded.
    #[wasm_bindgen]
    pub fn mj_bridge_model_root() -> String {
        MODEL_ROOT.with(|r| r.borrow().clone().unwrap_or_default())
    }

    /// Every file JS must stage before loading [`mj_bridge_model_root`]: the
    /// MJCF itself, whatever it `<include>`s, and their assets.
    #[wasm_bindgen]
    pub fn mj_bridge_model_file_paths() -> Vec<String> {
        MODEL_FILES.with(|f| f.borrow().iter().map(|(path, _)| path.clone()).collect())
    }

    /// Hands JS one staged file's bytes and drops this side's copy. Staging a
    /// mujoco_menagerie arm moves tens of megabytes of mesh, and there's no
    /// reason to hold it in this module's linear memory and MuJoCo's at once
    /// -- the two are separate wasm instances with separate heaps.
    ///
    /// Returns empty for an unknown or already-taken path, which JS surfaces
    /// as MuJoCo failing to open the file.
    #[wasm_bindgen]
    pub fn mj_bridge_take_model_file_bytes(path: String) -> Vec<u8> {
        MODEL_FILES.with(|f| {
            let mut files = f.borrow_mut();
            match files.iter().position(|(p, _)| *p == path) {
                Some(i) => files.swap_remove(i).1,
                None => Vec::new(),
            }
        })
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

    /// True while a trajectory clip is driving `qpos` directly. JS must not
    /// step physics then -- see `MujocoScene::set_kinematic` for why stepping
    /// a replayed pose destroys it. Independent of `mj_bridge_playing`, which
    /// gates the *clip's* own advance on the Rust side.
    #[wasm_bindgen]
    pub fn mj_bridge_kinematic() -> bool {
        KINEMATIC.with(|k| *k.borrow())
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

    /// Reports one mesh geom's static description, once per model load.
    ///
    /// JS has to resolve two things this side can't see. `mesh_name` needs
    /// `geom_dataid` -> `model.mesh(id).name`, and `rgba` needs the geom's
    /// `<material>`: a geom with one takes the material's `mat_rgba`, not its
    /// own `geom_rgba` (which stays at its unset default whenever a material
    /// is assigned -- MuJoCo never copies one into the other). Both live in
    /// MuJoCo's linear memory, not ours.
    ///
    /// Non-mesh geoms are simply never reported; `mesh_geoms` skips any geom
    /// with no entry here.
    #[wasm_bindgen]
    pub fn mj_bridge_set_mesh_geom(
        geom_index: usize,
        mesh_name: String,
        rgba: &[f32],
        mesh_pos: &[f32],
        mesh_quat: &[f32],
    ) {
        if rgba.len() != 4 || mesh_pos.len() != 3 || mesh_quat.len() != 4 {
            return; // Malformed (shouldn't happen); leave the geom unreported.
        }
        MESH_GEOMS.with(|m| {
            m.borrow_mut().insert(
                geom_index,
                MeshGeomInfo {
                    mesh_name,
                    rgba: std::array::from_fn(|k| rgba[k]),
                    mesh_pos: std::array::from_fn(|k| mesh_pos[k]),
                    mesh_quat: std::array::from_fn(|k| mesh_quat[k]),
                },
            );
        });
    }

    /// Reports one named joint's qpos layout, once per model load -- this is
    /// what `MujocoScene::joint_tracks` hands to retargeting, so a model with
    /// none of these reported can't play a clip at all.
    ///
    /// `dofs` is the joint's qpos width (7 free / 4 ball / 1 hinge or slide),
    /// which JS derives from `jnt_type`; `qpos_adr` is `jnt_qposadr[i]`. Both
    /// live in MuJoCo's linear memory, not ours. Unnamed joints are simply
    /// never reported, matching native's `id_to_name` filter.
    #[wasm_bindgen]
    pub fn mj_bridge_set_joint_track(joint_index: usize, name: String, dofs: usize, qpos_adr: u32) {
        JOINT_TRACKS.with(|j| {
            j.borrow_mut().insert(joint_index, JointInfo { name, dofs, qpos_adr });
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

        /// Writes `values[i]` into `data.qpos[adrs[i]]` and runs forward
        /// kinematics -- one call per frame carrying the whole pose, since
        /// each one crosses into the other wasm module's memory.
        #[wasm_bindgen(js_namespace = window, js_name = "__bsMujocoSetQpos")]
        fn js_set_qpos(adrs: &[u32], values: &[f64]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercised against the real vendored mujoco_menagerie panda rather than a
    // fixture: the whole point of these rules is that a menagerie model
    // resolves unedited, so drift in the vendored copy should fail here.
    const PANDA_DIR: &str = "examples/splat/game_assets/mujoco/franka_emika_panda";

    fn panda_dir() -> &'static std::path::Path {
        std::path::Path::new(PANDA_DIR)
    }

    fn read(name: &str) -> String {
        std::fs::read_to_string(panda_dir().join(name)).unwrap()
    }

    #[test]
    fn include_resolves_against_the_declaring_files_directory() {
        let refs = scan_mjcf(panda_dir(), &read("scene.xml"));
        assert_eq!(refs.includes, vec![panda_dir().join("panda.xml")]);
    }

    #[test]
    fn every_panda_mesh_resolves_to_a_file_that_exists() {
        let refs = scan_mjcf(panda_dir(), &read("panda.xml"));
        assert_eq!(refs.meshes.len(), 56);
        for (name, path) in &refs.meshes {
            assert!(path.exists(), "mesh {name} -> {} does not exist", path.display());
        }
    }

    #[test]
    fn mesh_name_defaults_to_the_file_stem() {
        let refs = scan_mjcf(panda_dir(), &read("panda.xml"));
        let (_, path) = refs
            .meshes
            .iter()
            .find(|(name, _)| name == "link0_0")
            .expect("mesh named after its file stem");
        assert_eq!(path, &panda_dir().join("assets/link0_0.obj"));
    }

    #[test]
    fn meshdir_replaces_the_model_dir_and_texturedir_is_separate() {
        let xml = r#"<mujoco>
            <compiler meshdir="meshes" texturedir="tex"/>
            <asset>
                <mesh file="arm.obj"/>
                <texture file="skin.png"/>
            </asset>
        </mujoco>"#;
        let refs = scan_mjcf(std::path::Path::new("model"), xml);
        assert_eq!(refs.meshes[0].1, std::path::Path::new("model/meshes/arm.obj"));
        assert_eq!(refs.other_assets, vec![std::path::Path::new("model/tex/skin.png")]);
    }

    #[test]
    fn assetdir_backs_both_and_loses_to_an_explicit_dir() {
        let xml = r#"<mujoco>
            <compiler assetdir="all" meshdir="meshes"/>
            <asset>
                <mesh file="arm.obj"/>
                <texture file="skin.png"/>
            </asset>
        </mujoco>"#;
        let refs = scan_mjcf(std::path::Path::new("model"), xml);
        assert_eq!(refs.meshes[0].1, std::path::Path::new("model/meshes/arm.obj"));
        assert_eq!(refs.other_assets, vec![std::path::Path::new("model/all/skin.png")]);
    }

    #[test]
    fn root_mjcf_is_the_one_nothing_includes() {
        // Deliberately listed with the fragment first: the answer has to come
        // from the include graph, not from ordering or naming.
        let files = vec![
            (format!("{PANDA_DIR}/panda.xml"), read("panda.xml")),
            (format!("{PANDA_DIR}/scene.xml"), read("scene.xml")),
        ];
        assert_eq!(find_root_mjcf(&files).unwrap(), format!("{PANDA_DIR}/scene.xml"));
    }

    #[test]
    fn root_mjcf_of_an_unincluded_pair_prefers_the_scene() {
        let files = vec![
            ("m/arm.xml".to_string(), "<mujoco/>".to_string()),
            ("m/scene_b.xml".to_string(), "<mujoco/>".to_string()),
        ];
        assert_eq!(find_root_mjcf(&files).unwrap(), "m/scene_b.xml");
    }

    #[test]
    fn root_mjcf_is_none_when_everything_is_included() {
        let files = vec![
            ("m/a.xml".to_string(), r#"<mujoco><include file="b.xml"/></mujoco>"#.to_string()),
            ("m/b.xml".to_string(), r#"<mujoco><include file="a.xml"/></mujoco>"#.to_string()),
        ];
        assert_eq!(find_root_mjcf(&files), None);
    }
}
