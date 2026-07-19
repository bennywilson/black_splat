use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use serde::{Deserialize, Serialize};

use black_splat::{
    egui, assets::*, config::*, editor::{self, EditorChoice, GizmoSpace, TransformGizmo}, engine::*,
    fly_camera::*, game_object::*, input::*, mujoco::{MjcfBundle, MujocoScene}, renderer::*, resource::SceneLayer,
    touch_pads::*, trajectory::RetargetedClip, utils::*, log,
    passes::ambient::AmbientSettings,
    passes::deferred::ShadowSettings,
    passes::gaussian_splat::SplatParams,
    passes::postprocess::PostProcessSettings,
};

use crate::editor_config::{self, EditorConfig, GIZMO_ACTIONS};
use crate::resource_library::{self, MaterialFile};

// How far in front of the camera a newly added game object is dropped.
const ADD_OBJECT_DISTANCE: f32 = 5.0;

/// The renderer-side shadow settings an EditorConfig describes.
fn shadow_settings_from_config(config: &EditorConfig) -> ShadowSettings {
    ShadowSettings {
        resolution: config.shadow_resolution,
        depth_bias: config.shadow_depth_bias,
    }
}

// Default skylight hemisphere colors: cool sky above, warm bounce below.
// Used for the auto-added skylight in new scenes and Add > Light > Skylight.
const SKYLIGHT_TOP: CgVec3 = CgVec3::new(0.55, 0.65, 0.85);
const SKYLIGHT_BOTTOM: CgVec3 = CgVec3::new(0.28, 0.24, 0.2);

// Keyboard/mouse fly-camera movement and look come from the shared FlyCamera and
// the on-screen touch pads from the shared TouchPads (black_splat::fly_camera /
// ::touch_pads), whose defaults already match this viewer's feel.
const PARAM_RATE: f32 = 1.5;
// Smallest a splat's cloud scale / render scale may be dragged to in the
// Details panel: a zero scale collapses the cloud and it can't be recovered by
// dragging (0 * anything stays 0).  Matches the gizmo's own floor in spirit.
const SPLAT_MIN_SCALE: f32 = 0.001;

/// Draws a resource inspector's header row: the bold name with a `*` when it
/// has unsaved edits, plus a floppy-disk save button that's enabled only while
/// dirty.  Returns true iff the save button was clicked this frame -- the
/// caller performs the actual save (so it can take `&mut` on the resource,
/// which the `name` borrow here would otherwise block).
fn resource_header(ui: &mut egui::Ui, name: &str, dirty: bool) -> bool {
    let mut clicked = false;
    ui.horizontal(|ui| {
        let title = if dirty {
            format!("{name}  *")
        } else {
            name.to_string()
        };
        ui.label(egui::RichText::new(title).strong());
        if ui
            .add_enabled(dirty, egui::Button::new("💾").small())
            .on_hover_text("Save to disk")
            .clicked()
        {
            clicked = true;
        }
    });
    clicked
}

/// Saves a material, first deleting its previous file/localStorage entry if it
/// was renamed since the last save (so a rename doesn't orphan the old copy).
/// Updates `saved_name` on success.
fn save_material_file(mat: &mut MaterialResource) -> Result<(), String> {
    if let Some(old) = &mat.saved_name {
        if *old != mat.name {
            resource_library::delete_material(old);
        }
    }
    resource_library::save_material(&mat.name, &mat.desc)?;
    mat.saved_name = Some(mat.name.clone());
    Ok(())
}

/// Particle counterpart to [`save_material_file`].
fn save_particle_file(res: &mut ParticleResource) -> Result<(), String> {
    if let Some(old) = &res.saved_name {
        if *old != res.name {
            resource_library::delete_particle(old);
        }
    }
    resource_library::save_particle(&res.name, &res.params)?;
    res.saved_name = Some(res.name.clone());
    Ok(())
}

/// A kind of content-browser asset.  Splats/Models are read-only;
/// Materials/Particles are editable+saveable; Textures are imported.  Used as
/// the browser's type-filter key (see `browser_filters`) and to tag each
/// unified [`AssetEntry`], so it must be hashable.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BrowserCategory {
    Models,
    Materials,
    Particles,
    Textures,
    Splats,
}

const BROWSER_CATEGORIES: &[(BrowserCategory, &str)] = &[
    (BrowserCategory::Models, "Models"),
    (BrowserCategory::Materials, "Materials"),
    (BrowserCategory::Particles, "Particles"),
    (BrowserCategory::Textures, "Textures"),
    (BrowserCategory::Splats, "Splats"),
];

// Content-browser tile geometry (points): a square thumbnail with a name below.
const TILE_THUMB: f32 = 72.0;
const TILE_W: f32 = 92.0;
const TILE_H: f32 = TILE_THUMB + 24.0;
// Width (points) of the content browser's left-hand folder tree.
const FOLDER_TREE_W: f32 = 160.0;

/// What a browser tile draws in its thumbnail square.
enum Thumb {
    /// A loaded texture image (see Renderer::egui_texture_id).
    Image(egui::TextureId),
    /// A material's base color, as a swatch.
    Color([f32; 3]),
    /// A single glyph on a neutral plate (models, particles, splats).
    Glyph(&'static str),
}

/// Case-insensitive substring filter used by the content browser.  `filter` is
/// already lowercased by the caller.
fn name_matches(name: &str, filter: &str) -> bool {
    filter.is_empty() || name.to_lowercase().contains(filter)
}

/// Places a small floppy-disk save button in a tile's top-right corner (shown
/// for dirty materials/particles).  Returns its response.
fn tile_save_button(ui: &mut egui::Ui, tile: &egui::Response) -> egui::Response {
    let rect = egui::Rect::from_min_size(
        egui::pos2(tile.rect.right() - 20.0, tile.rect.top() + 2.0),
        egui::vec2(18.0, 18.0),
    );
    ui.put(rect, egui::Button::new("💾").small())
        .on_hover_text("Save to disk")
}

/// Truncates a label to `max` chars with an ellipsis, on a char boundary.
fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let kept: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{kept}…")
    }
}

/// Draws one content-browser tile (thumbnail + name) with a selection/hover
/// background and a `*` when dirty.  Returns the tile's click response.
fn browser_tile(
    ui: &mut egui::Ui,
    name: &str,
    thumb: &Thumb,
    selected: bool,
    dirty: bool,
) -> egui::Response {
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(TILE_W, TILE_H), egui::Sense::click());
    let painter = ui.painter_at(rect);
    let bg = if selected {
        ui.visuals().selection.bg_fill
    } else if resp.hovered() {
        ui.visuals().widgets.hovered.bg_fill
    } else {
        egui::Color32::TRANSPARENT
    };
    painter.rect_filled(rect, 4.0, bg);
    let thumb_rect = egui::Rect::from_min_size(
        egui::pos2(rect.center().x - TILE_THUMB / 2.0, rect.top() + 4.0),
        egui::vec2(TILE_THUMB, TILE_THUMB),
    );
    match thumb {
        Thumb::Image(id) => {
            painter.image(
                *id,
                thumb_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        Thumb::Color(rgb) => {
            let color = egui::Color32::from_rgb(
                (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
                (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            );
            painter.rect_filled(thumb_rect, 4.0, color);
        }
        Thumb::Glyph(g) => {
            painter.rect_filled(thumb_rect, 4.0, ui.visuals().extreme_bg_color);
            painter.text(
                thumb_rect.center(),
                egui::Align2::CENTER_CENTER,
                g,
                egui::FontId::proportional(30.0),
                ui.visuals().weak_text_color(),
            );
        }
    }
    let label = if dirty {
        format!("{}  *", elide(name, 11))
    } else {
        elide(name, 12)
    };
    painter.text(
        egui::pos2(rect.center().x, thumb_rect.bottom() + 3.0),
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(12.0),
        ui.visuals().text_color(),
    );
    resp.on_hover_text(name)
}

/// One entry in the unified content browser: any asset (model, material,
/// particle, texture, or scene splat) tagged with the folder it lives in, so
/// the browser can list every kind together and narrow by folder + type + name
/// like a file browser.
struct AssetEntry {
    kind: BrowserCategory,
    name: String,
    /// Virtual folder path ("/"-separated): the asset's on-disk directory, or a
    /// synthetic folder for path-less items (scene splats live under "Scene").
    folder: String,
    payload: AssetPayload,
}

/// Per-kind data a browser tile needs to draw itself and react to clicks.
enum AssetPayload {
    /// (relative path, loaded handle if any).  A `None` handle draws a tile
    /// that lazily loads the model when clicked.
    Model {
        path: String,
        loaded: Option<ModelHandle>,
    },
    Material {
        handle: MaterialHandle,
        dirty: bool,
        rgb: [f32; 3],
    },
    Particle {
        index: usize,
        dirty: bool,
    },
    Texture {
        handle: TextureHandle,
    },
    /// A scene splat cloud (display only in the browser).
    Splat,
}

/// The directory portion of a relative asset path (forward-slashed), or "" for
/// a bare filename.  `game_assets/models/Barrel/x.glb` -> `game_assets/models/Barrel`.
fn parent_dir(path: &str) -> String {
    match path.replace('\\', "/").rsplit_once('/') {
        Some((dir, _)) => dir.to_string(),
        None => String::new(),
    }
}

/// Whether an asset in `asset_folder` shows under the selected tree folder: the
/// root ("") shows everything, otherwise the folder itself and anything nested
/// beneath it (recursive, so a parent folder gathers all its descendants).
fn folder_contains(selected: &str, asset_folder: &str) -> bool {
    selected.is_empty()
        || asset_folder == selected
        || asset_folder.starts_with(&format!("{selected}/"))
}

/// A node in the content browser's folder tree, built from asset folder paths.
#[derive(Default)]
struct FolderNode {
    children: std::collections::BTreeMap<String, FolderNode>,
}

impl FolderNode {
    /// Adds a "/"-separated folder path, creating intermediate nodes so every
    /// ancestor folder appears in the tree even if nothing sits directly in it.
    fn insert(&mut self, path: &str) {
        let mut node = self;
        for comp in path.split('/').filter(|c| !c.is_empty()) {
            node = node.children.entry(comp.to_string()).or_default();
        }
    }
}

/// Draws the folder tree recursively.  Each folder is a selectable row that sets
/// `selected` to its full path; folders with children get a collapse triangle,
/// leaves are indented to line up beneath them.  `prefix` is `node`'s own path.
fn draw_folder_tree(ui: &mut egui::Ui, node: &FolderNode, prefix: &str, selected: &mut String) {
    for (name, child) in &node.children {
        let path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        let is_sel = *selected == path;
        let label = format!("📁 {name}");
        if child.children.is_empty() {
            ui.horizontal(|ui| {
                ui.add_space(18.0); // line up with the collapsible rows' labels
                if ui.selectable_label(is_sel, label).clicked() {
                    *selected = path.clone();
                }
            });
        } else {
            let id = ui.make_persistent_id(("cb_folder", &path));
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
                .show_header(ui, |ui| {
                    if ui.selectable_label(is_sel, label).clicked() {
                        *selected = path.clone();
                    }
                })
                .body(|ui| draw_folder_tree(ui, child, &path, selected));
        }
    }
}

/// A labelled dropdown for choosing a texture (or "(none)") from the library,
/// writing the chosen relative path into `slot`.  Returns true if the choice
/// changed.  `options` is (display name, relative path).
fn texture_combo(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    slot: &mut Option<String>,
    options: &[(String, String)],
) -> bool {
    let mut changed = false;
    let current = slot.clone();
    let selected_text = current
        .as_deref()
        .map(resource_display_name)
        .unwrap_or_else(|| "(none)".to_string());
    ui.label(label);
    egui::ComboBox::from_id_salt(id)
        .selected_text(selected_text)
        .show_ui(ui, |ui| {
            if ui.selectable_label(slot.is_none(), "(none)").clicked() && slot.is_some() {
                *slot = None;
                changed = true;
            }
            for (name, path) in options {
                let is_selected = current.as_deref() == Some(path.as_str());
                if ui.selectable_label(is_selected, name).clicked() && !is_selected {
                    *slot = Some(path.clone());
                    changed = true;
                }
            }
        });
    changed
}

/// Draws the "Editor | Game" mode switch and applies clicks to `editor_mode`.
/// Always shown (in both modes) so it's the way back from game mode -- important
/// on touch, where there's no keyboard shortcut.
fn draw_mode_switch(ui: &mut egui::Ui, editor_mode: &mut bool) {
    if ui.selectable_label(*editor_mode, "Editor").clicked() {
        *editor_mode = true;
    }
    if ui.selectable_label(!*editor_mode, "Game").clicked() {
        *editor_mode = false;
    }
}

/// Whether a finger is this browser's *primary* pointer.  Finger-friendly GUI
/// sizing should only kick in there; desktop browsers keep egui's defaults so
/// the GUI matches the native desktop build.  `pointer: coarse` rather than
/// `maxTouchPoints > 0` because the latter is true for any touch-capable
/// laptop or touch monitor, whose users still drive the GUI with a trackpad.
#[cfg(target_arch = "wasm32")]
fn is_touch_device() -> bool {
    web_sys::window().is_some_and(|w| {
        w.match_media("(pointer: coarse)")
            .ok()
            .flatten()
            .is_some_and(|m| m.matches())
    })
}

/// Saves `json` through the browser's File System Access API
/// (`window.showSaveFilePicker`) -- a real save dialog, writing wherever the
/// user points it.  Chrome/Edge support it; Firefox/Safari don't.
///
/// Returns `Err(())` only when the API is unavailable, so the caller can fall
/// back to a download.  A user cancel resolves `Ok` -- the dialog was shown and
/// answered, nothing left to do.
///
/// Called through `js_sys::Reflect` because web-sys gates its typed bindings
/// for this API behind the unstable-APIs cfg flag.
#[cfg(target_arch = "wasm32")]
async fn save_with_fs_access_api(json: &str) -> Result<(), ()> {
    use wasm_bindgen::{JsCast, JsValue};

    let window = web_sys::window().ok_or(())?;
    let picker = js_sys::Reflect::get(&window, &JsValue::from_str("showSaveFilePicker"))
        .map_err(|_| ())?;
    let picker: js_sys::Function = picker.dyn_into().map_err(|_| ())?;

    // One await step: call `method` on `target` and wait out the promise.
    async fn call_async(
        target: &JsValue,
        method: &js_sys::Function,
        arg: Option<&JsValue>,
    ) -> Result<JsValue, JsValue> {
        let promise = match arg {
            Some(arg) => method.call1(target, arg)?,
            None => method.call0(target)?,
        };
        wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise)).await
    }
    let method_of = |target: &JsValue, name: &str| -> Option<js_sys::Function> {
        js_sys::Reflect::get(target, &JsValue::from_str(name))
            .ok()?
            .dyn_into()
            .ok()
    };

    // showSaveFilePicker({ suggestedName: "scene.json" }) -> FileSystemFileHandle.
    // A rejected promise here is the user cancelling: report success (handled).
    let options = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &options,
        &JsValue::from_str("suggestedName"),
        &JsValue::from_str("scene.json"),
    );
    let window: JsValue = window.into();
    let Ok(handle) = call_async(&window, &picker, Some(&options.into())).await else {
        return Ok(());
    };

    // handle.createWritable() -> stream; stream.write(text); stream.close().
    let Some(create_writable) = method_of(&handle, "createWritable") else {
        return Ok(());
    };
    let Ok(stream) = call_async(&handle, &create_writable, None).await else {
        return Ok(());
    };
    if let Some(write) = method_of(&stream, "write") {
        let _ = call_async(&stream, &write, Some(&JsValue::from_str(json))).await;
    }
    if let Some(close) = method_of(&stream, "close") {
        let _ = call_async(&stream, &close, None).await;
    }
    Ok(())
}

// "game_assets/models/barrel.glb" -> "barrel" for resource lists.
fn resource_display_name(path: &str) -> String {
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    file.rsplit_once('.').map_or(file, |(stem, _)| stem).to_string()
}

/// Which tab of the right-hand editor panel is showing.  (Resources is a
/// separate bottom panel.)
#[derive(Clone, Copy, PartialEq, Eq)]
enum EditorTab {
    Scene,
    Details,
    /// Inspector for the resource highlighted in the bottom Resources panel
    /// (materials and particle definitions are edited here).
    Resource,
    Settings,
}

/// The resource highlighted in the bottom Resources panel, shown in the
/// right panel's Resource inspector tab.
#[derive(Clone, Copy, PartialEq)]
enum ResourceSelection {
    Model(ModelHandle),
    Material(MaterialHandle),
    /// Index into SplatGame::particle_resources.
    Particle(usize),
}

/// An editable particle definition in the resource library (seeded from the
/// built-in presets or loaded from `resources/particles/`).  Scene emitters
/// reference one by name.
struct ParticleResource {
    name: String,
    params: ParticleParams,
    /// True when edited since its last save; drives the `*` marker and save
    /// icon in the content browser.
    dirty: bool,
    /// The name this resource currently exists under on disk / in localStorage,
    /// or None if never saved.  Lets a rename-then-save delete the stale file
    /// rather than orphaning it (see [`save_particle_file`]).
    saved_name: Option<String>,
}

/// An editable material in the resource library (seeded from built-in defaults
/// or loaded from `resources/materials/`).  Actors reference one by handle;
/// the owned `desc` is what gets written back to disk on save (the renderer
/// keeps only the built GPU material, not its description).
struct MaterialResource {
    name: String,
    desc: MaterialDesc,
    handle: MaterialHandle,
    /// True when edited since its last save (see [`ParticleResource::dirty`]).
    dirty: bool,
    /// Name last saved under (see [`ParticleResource::saved_name`]).
    saved_name: Option<String>,
}

/// A texture in the library: an image file under `game_assets/textures/` (or a
/// preset's `game_assets/fx/`), already loaded into the renderer.  Materials
/// reference one by its relative `path`.  Textures have no editable state, so
/// nothing to save -- the file on disk is the resource.
struct TextureResource {
    /// Short display name (file stem), for dropdowns and the browser.
    name: String,
    /// Relative path the material stores and the renderer loads from.
    path: String,
    handle: TextureHandle,
}

/// A discovered model in the editor's catalog.  Discovery is decoupled from
/// loading: every model under game_assets/models/ (and imports) is catalogued
/// so it shows in the browser, but its geometry isn't loaded until it's
/// actually used -- selected in the browser, or referenced by an opened scene.
/// Whether a catalog entry is currently loaded (and its handle) is owned by the
/// renderer's AssetManager, keyed by `path`.
struct ModelResource {
    /// Short display name (file stem), for dropdowns and the browser.
    name: String,
    /// Relative path the model loads from (also the AssetManager key).
    path: String,
}

/// The currently selected scene object.  The three lists (actors, lights,
/// particle systems) are kept separate, so a selection names both the list and
/// the index into it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Selection {
    Actor(usize),
    Light(usize),
    Particle(usize),
    Splat(usize),
    MujocoActor(usize),
    // The scene-wide post-process settings.  A singleton (no index): one per
    // scene for now, though it is shaped to become a per-volume list later.
    PostProcess,
}

// ---- Undo/redo ---------------------------------------------------------
// Command-based: every mutation the editor makes (transform, property edit,
// create, delete) pushes an `UndoAction` describing how to reverse and
// reapply it.  Actions are keyed by `ObjectRef` (a stable id/handle/name),
// never by `Selection`'s raw Vec index -- indices shift as other objects are
// inserted/removed elsewhere in the same list, which would silently corrupt
// any older pending entry if it stored a plain `usize`.

/// Stable reference to a scene object, independent of its current position in
/// `scene_actors`/etc.  Resolved back to a `Selection` (i.e. a live index) at
/// undo/redo time via `SplatGame::resolve_ref`.
#[derive(Clone, PartialEq)]
enum ObjectRef {
    Actor(u32),
    Light(u32),
    Particle(ParticleHandle),
    // By name: splats have no other stable id, and can't be deleted/recreated
    // in the same synchronous frame as other list edits, so name collisions
    // aren't a practical concern here.
    Splat(String),
    // Same by-name identity as Splat: a MujocoActor's live sim can't survive
    // a delete/recreate cycle either (see MujocoSceneActor), so it's reloaded
    // fresh from its xml_path just like a splat reloads its .ply.
    MujocoActor(String),
    PostProcess,
}

/// A full clone of one scene object's editable state: what an undo/redo entry
/// restores, and what a pending edit is diffed against once its gesture ends.
#[derive(Clone)]
enum ObjectSnapshot {
    Actor(Actor),
    Light(Light),
    Particle(SceneParticle),
    Splat(SceneSplat),
    MujocoActor(MujocoActorSnapshot),
    PostProcess(PostProcessSettings),
}

/// Everything about a `MujocoSceneActor` except its live sim (see
/// `MujocoSceneActor::scene`) -- what an undo/redo entry restores.  Restoring
/// re-triggers a reload from `xml_path` rather than trying to resurrect the
/// simulation state itself.
#[derive(Clone)]
struct MujocoActorSnapshot {
    name: String,
    xml_path: String,
    trajectory_path: String,
    position: CgVec3,
    rotation: CgQuat,
    playing: bool,
    speed: f32,
    looping: bool,
    wireframe: bool,
}

/// One undoable editor action.  `Edit` covers both gizmo-transform drags and
/// Details-panel property edits (including rename) -- both are just "restore
/// this object's full state," so they share one variant.
enum UndoAction {
    Edit {
        obj: ObjectRef,
        before: ObjectSnapshot,
        after: ObjectSnapshot,
    },
    Create {
        obj: ObjectRef,
        snapshot: ObjectSnapshot,
        index_hint: usize,
    },
    Delete {
        obj: ObjectRef,
        snapshot: ObjectSnapshot,
        index_hint: usize,
    },
    /// Several edits committed as one gesture (a multi-selection gizmo drag);
    /// undone/redone together.
    Batch(Vec<UndoAction>),
}

const UNDO_STACK_MAX: usize = 100;

/// Two LIFO stacks.  Pushing a new action always clears `redo`
#[derive(Default)]
struct UndoStack {
    undo: Vec<UndoAction>,
    redo: Vec<UndoAction>,
}

impl UndoStack {
    fn push(&mut self, action: UndoAction) {
        self.undo.push(action);
        if self.undo.len() > UNDO_STACK_MAX {
            self.undo.remove(0);
        }
        self.redo.clear();
    }
}

/// What the "Add" menu asked to create this frame (applied after the egui pass).
#[derive(Clone, Copy)]
enum AddKind {
    Actor,
    Light(LightType),
    Particle(usize), // index into SplatGame::particle_resources
    Splat,           // opens the .ply picker
    MujocoActor,
}

/// A loaded splat cloud as a scene object: a display name plus its own render
/// params (shown in the Details panel when selected).  Index-aligned with the
/// renderer's splat clouds.
#[derive(Clone)]
struct SceneSplat {
    name: String,
    // Where the .ply came from (stored in saved scenes so they can reload
    // it): a real filesystem path on native, an IndexedDB cache key on web.
    // Empty only for scenes saved before this was tracked.
    path: String,
    params: SplatParams,
    transform: ActorTransform,
}

/// A MuJoCo physics scene as a scene object: an MJCF file rendered as
/// wireframe lines (see `black_splat::mujoco::MujocoScene`), placed by a
/// rigid transform on top of the MJCF's own local coordinates, with live
/// playback controls. `scene` is the live sim -- not serializable and not
/// snapshotted for undo (see `MujocoActorSnapshot`); `loaded_path` tracks
/// which path `scene` currently reflects so the per-frame reconcile in
/// `tick_mujoco_actors` knows when to kick off a (re)load.
struct MujocoSceneActor {
    name: String,
    xml_path: String,
    loaded_path: String,
    // The path an in-flight async load was dispatched for, if any -- lets
    // tick_mujoco_actors avoid spawning a new load every frame while one is
    // already in flight for the current xml_path.
    pending_path: Option<String>,
    // A recorded joint trajectory to replay over the sim instead of stepping
    // physics (see `MujocoScene::set_kinematic`). Empty means "just simulate".
    // These mirror the xml_path/loaded_path/pending_path reconcile above, but
    // can only be dispatched once `scene` exists -- retargeting needs the
    // loaded model's joint layout.
    trajectory_path: String,
    loaded_trajectory_path: String,
    pending_trajectory_path: Option<String>,
    clip: Option<RetargetedClip>,
    // Seconds into `clip`, advanced by delta_time * speed. Not snapshotted for
    // undo -- playback position is transient, like the sim state itself.
    clip_time: f32,
    looping: bool,
    position: CgVec3,
    rotation: CgQuat,
    playing: bool,
    speed: f32,
    // Whether primitive geoms draw as wireframe lines (mesh geoms always
    // render regardless -- see `MujocoScene::set_wireframe`).
    wireframe: bool,
    scene: Option<MujocoScene>,
    // Real triangle-mesh actors for the scene's mesh-type geoms (registered
    // into the renderer's own actor map, see `tick_mujoco_actors`) -- index-
    // aligned with `MujocoScene::mesh_geoms`'s output, which is stable across
    // frames since a loaded model's geom set never changes. Rebuilt whenever
    // `scene` is (re)loaded.
    mesh_geom_actors: Vec<Actor>,
    // mesh .obj path -> loaded Model, so each unique mesh is only loaded once
    // per scene load instead of every frame.
    mesh_model_cache: std::collections::HashMap<String, ModelHandle>,
    // Mesh paths a fetch has already been dispatched for, so a mesh isn't
    // re-requested every frame while its load is in flight (wasm only --
    // native loads meshes inline, see `tick_mujoco_actors`).
    #[cfg(target_arch = "wasm32")]
    requested_meshes: std::collections::HashSet<String>,
}

impl MujocoSceneActor {
    fn new(name: String) -> Self {
        MujocoSceneActor {
            name,
            xml_path: String::new(),
            loaded_path: String::new(),
            pending_path: None,
            trajectory_path: String::new(),
            loaded_trajectory_path: String::new(),
            pending_trajectory_path: None,
            clip: None,
            clip_time: 0.0,
            looping: true,
            position: CG_VEC3_ZERO,
            rotation: CG_QUAT_IDENT,
            playing: true,
            speed: 1.0,
            wireframe: true,
            scene: None,
            mesh_geom_actors: Vec::new(),
            mesh_model_cache: std::collections::HashMap::new(),
            #[cfg(target_arch = "wasm32")]
            requested_meshes: std::collections::HashSet::new(),
        }
    }
}

impl editor::EditorInspect for MujocoSceneActor {
    fn inspect_properties(&mut self, visitor: &mut dyn editor::PropertyVisitor) -> bool {
        let mut changed = false;
        changed |= visitor.edit_text("Name", &mut self.name);
        // XML Path is set only via the Details panel's Browse… button (see
        // the custom UI in EditorTab::Details), not typed here.
        changed |= visitor.edit_vec3("Position", &mut self.position);
        changed |= visitor.edit_rotation("Rotation", &mut self.rotation);
        changed |= visitor.edit_bool("Playing", &mut self.playing);
        changed |= visitor.edit_float("Speed", &mut self.speed);
        changed |= visitor.edit_bool("Looping", &mut self.looping);
        changed |= visitor.edit_bool("Wireframe", &mut self.wireframe);
        if let Some(scene) = &mut self.scene {
            scene.set_playing(self.playing);
            scene.set_speed(self.speed);
            scene.set_wireframe(self.wireframe);
        }
        changed
    }
}

/// Default render params for splat clouds
fn default_splat_params() -> SplatParams {
    SplatParams {
        falloff: 4.65,
        scale: 3.0,
        contrast: 1.0,
        max_sh_degree: 2.0,
        overall_scale: 1.0,
    }
}

// ---- Scene save/load (JSON) -------------------------------------------------
// A flexible, human-readable snapshot of the editable scene.
// Bump SCENE_FORMAT_VERSION on breaking changes.

const SCENE_FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SceneFile {
    version: u32,
    // When absent in a scene this falls back to the config's start camera
    #[serde(default)]
    camera: Option<CameraDto>,
    #[serde(default)]
    actors: Vec<ActorDto>,
    #[serde(default)]
    lights: Vec<LightDto>,
    #[serde(default)]
    particles: Vec<ParticleDto>,
    #[serde(default)]
    splats: Vec<SplatDto>,
    #[serde(default)]
    mujoco_actors: Vec<MujocoActorDto>,
    #[serde(default)]
    post_process: Option<PostProcessDto>,
    #[serde(default)]
    ambient: Option<AmbientDto>,
}

/// Serialized scene-wide post-process / tonemap settings (see PostProcessSettings).
#[derive(Serialize, Deserialize)]
struct PostProcessDto {
    tonemap_enabled: bool,
    exposure: f32,
    highlight_scale: f32,
    midtone_scale: f32,
    highlight_curve: f32,
    midtone_curve: f32,
    shadow_offset: f32,
    // Scenes saved before the upscale sharpen existed get the engine default
    // rather than 0.0, which would silently leave them unsharpened.
    #[serde(default = "default_sharpen_strength")]
    sharpen_strength: f32,
}

fn default_sharpen_strength() -> f32 {
    PostProcessSettings::default().sharpen_strength
}

impl PostProcessDto {
    fn from_settings(s: &PostProcessSettings) -> Self {
        Self {
            tonemap_enabled: s.tonemap_enabled,
            exposure: s.exposure,
            highlight_scale: s.highlight_scale,
            midtone_scale: s.midtone_scale,
            highlight_curve: s.highlight_curve,
            midtone_curve: s.midtone_curve,
            shadow_offset: s.shadow_offset,
            sharpen_strength: s.sharpen_strength,
        }
    }

    fn to_settings(&self) -> PostProcessSettings {
        let mut s = PostProcessSettings {
            tonemap_enabled: self.tonemap_enabled,
            exposure: self.exposure,
            highlight_scale: self.highlight_scale,
            midtone_scale: self.midtone_scale,
            highlight_curve: self.highlight_curve,
            midtone_curve: self.midtone_curve,
            shadow_offset: self.shadow_offset,
            sharpen_strength: self.sharpen_strength,
        };
        // A hand-edited file could carry a non-invertible curve; snap it back.
        s.enforce_invertible();
        s
    }
}

/// Serialized screen-space AO / diffuse-GI toggles (see AmbientSettings).
#[derive(Serialize, Deserialize)]
struct AmbientDto {
    ssao_enabled: bool,
    ssgi_enabled: bool,
    // Scenes saved before this existed get the engine default (1.0) rather
    // than 0.0, which would silently zero out their GI.
    #[serde(default = "default_gi_intensity")]
    gi_intensity: f32,
    // Scenes saved before sample/blur controls existed get the engine
    // defaults rather than 0, which would zero out AO/GI or disable the
    // denoise blur entirely.
    #[serde(default = "default_ao_samples")]
    ao_samples: u32,
    #[serde(default = "default_gi_samples")]
    gi_samples: u32,
    #[serde(default = "default_denoise_radius")]
    denoise_radius: u32,
    #[serde(default = "default_denoise_strength")]
    denoise_strength: f32,
    #[serde(default = "default_denoise_iterations")]
    denoise_iterations: u32,
}

fn default_gi_intensity() -> f32 {
    AmbientSettings::default().gi_intensity
}

fn default_ao_samples() -> u32 {
    AmbientSettings::default().ao_samples
}

fn default_gi_samples() -> u32 {
    AmbientSettings::default().gi_samples
}

fn default_denoise_radius() -> u32 {
    AmbientSettings::default().denoise_radius
}

fn default_denoise_strength() -> f32 {
    AmbientSettings::default().denoise_strength
}

fn default_denoise_iterations() -> u32 {
    AmbientSettings::default().denoise_iterations
}

impl AmbientDto {
    fn from_settings(s: &AmbientSettings) -> Self {
        Self {
            ssao_enabled: s.ssao_enabled,
            ssgi_enabled: s.ssgi_enabled,
            gi_intensity: s.gi_intensity,
            ao_samples: s.ao_samples,
            gi_samples: s.gi_samples,
            denoise_radius: s.denoise_radius,
            denoise_strength: s.denoise_strength,
            denoise_iterations: s.denoise_iterations,
        }
    }

    fn to_settings(&self) -> AmbientSettings {
        AmbientSettings {
            ssao_enabled: self.ssao_enabled,
            ssgi_enabled: self.ssgi_enabled,
            gi_intensity: self.gi_intensity,
            ao_samples: self.ao_samples,
            gi_samples: self.gi_samples,
            denoise_radius: self.denoise_radius,
            denoise_strength: self.denoise_strength,
            denoise_iterations: self.denoise_iterations,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct CameraDto {
    position: [f32; 3],
    rotation: [f32; 3],
}

#[derive(Serialize, Deserialize)]
struct ActorDto {
    name: String,
    position: [f32; 3],
    rotation: [f32; 4], // quaternion x, y, z, w
    scale: [f32; 3],
    #[serde(default)]
    model: Option<String>, // resource display name, or none for an empty actor
    #[serde(default)]
    material: Option<String>, // material library name, or none for the model's default
    #[serde(default)]
    layer: u32, // SceneLayer choice index
    #[serde(default)]
    shadow_catcher: bool,
}

#[derive(Serialize, Deserialize)]
struct LightDto {
    name: String,
    position: [f32; 3],
    rotation: [f32; 4],
    #[serde(rename = "type", default)]
    light_type: u32, // LightType choice index
    color: [f32; 3],
    // Skylight bottom-hemisphere color (`color` is the top).
    #[serde(default = "default_light_color2")]
    color2: [f32; 3],
    intensity: f32,
    #[serde(default = "default_light_range")]
    range: f32,
    #[serde(default = "default_light_spot_angle")]
    spot_angle: f32,
    casts_shadow: bool,
    // Directional-light cascade controls.
    #[serde(default = "default_light_shadow_cascades")]
    shadow_cascades: u32,
    #[serde(default = "default_light_shadow_distance")]
    shadow_distance: f32,
    #[serde(default = "default_light_shadow_density")]
    shadow_density: f32,
}

// Serde defaults for lights in pre-lighting scene files (match Light::new).
fn default_light_color2() -> [f32; 3] {
    [0.25, 0.22, 0.2]
}
fn default_light_range() -> f32 {
    10.0
}
fn default_light_spot_angle() -> f32 {
    30.0
}
fn default_light_shadow_cascades() -> u32 {
    3
}
fn default_light_shadow_distance() -> f32 {
    75.0
}
fn default_light_shadow_density() -> f32 {
    1.0
}

#[derive(Serialize, Deserialize)]
struct ParticleDto {
    name: String,
    preset: String, // PARTICLE_PRESETS name
    position: [f32; 3],
    scale: [f32; 3],
}

#[derive(Clone, Serialize, Deserialize)]
struct SplatDto {
    name: String,
    #[serde(default)]
    path: String,
    params: SplatParamsDto,
    #[serde(default)]
    position: [f32; 3],
    #[serde(default = "ident_quat")]
    rotation: [f32; 4],
    #[serde(default = "ones3")]
    scale: [f32; 3],
}

fn ident_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}
fn ones3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

#[derive(Clone, Serialize, Deserialize)]
struct SplatParamsDto {
    falloff: f32,
    scale: f32,
    contrast: f32,
    overall_scale: f32,
    max_sh_degree: f32,
}

impl SplatParamsDto {
    fn from_params(p: &SplatParams) -> Self {
        SplatParamsDto {
            falloff: p.falloff,
            scale: p.scale,
            contrast: p.contrast,
            overall_scale: p.overall_scale,
            max_sh_degree: p.max_sh_degree,
        }
    }

    fn to_params(&self) -> SplatParams {
        SplatParams {
            falloff: self.falloff,
            scale: self.scale,
            contrast: self.contrast,
            max_sh_degree: self.max_sh_degree,
            overall_scale: self.overall_scale,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct MujocoActorDto {
    name: String,
    #[serde(default)]
    xml_path: String,
    #[serde(default)]
    trajectory_path: String,
    #[serde(default)]
    position: [f32; 3],
    #[serde(default = "ident_quat")]
    rotation: [f32; 4],
    #[serde(default = "default_true")]
    playing: bool,
    #[serde(default = "default_speed_one")]
    speed: f32,
    #[serde(default = "default_true")]
    looping: bool,
    #[serde(default = "default_true")]
    wireframe: bool,
}

fn default_true() -> bool {
    true
}
fn default_speed_one() -> f32 {
    1.0
}

/// The scene the editor opens when the user hasn't saved a startup scene of
/// their own: the church cloud plus the default skylight, camera left at the
/// config's start pose.
fn default_startup_scene() -> SceneFile {
    SceneFile {
        version: SCENE_FORMAT_VERSION,
        camera: None,
        actors: Vec::new(),
        lights: vec![LightDto {
            name: "Skylight".to_string(),
            position: [0.0, 5.0, 0.0],
            rotation: ident_quat(),
            light_type: LightType::Skylight.choice_index() as u32,
            color: [SKYLIGHT_TOP.x, SKYLIGHT_TOP.y, SKYLIGHT_TOP.z],
            color2: [SKYLIGHT_BOTTOM.x, SKYLIGHT_BOTTOM.y, SKYLIGHT_BOTTOM.z],
            intensity: 1.0,
            range: default_light_range(),
            spot_angle: default_light_spot_angle(),
            casts_shadow: false,
            shadow_cascades: default_light_shadow_cascades(),
            shadow_distance: default_light_shadow_distance(),
            shadow_density: default_light_shadow_density(),
        }],
        particles: Vec::new(),
        splats: vec![SplatDto {
            name: "church".to_string(),
            path: "game_assets/splats/church.ply".to_string(),
            params: SplatParamsDto::from_params(&default_splat_params()),
            position: [0.0; 3],
            rotation: ident_quat(),
            scale: ones3(),
        }],
        mujoco_actors: Vec::new(),
        post_process: None,
        ambient: None,
    }
}

// Plain-array <-> cgmath conversions for the DTOs above.
fn vec3_arr(v: CgVec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}
fn arr_vec3(a: [f32; 3]) -> CgVec3 {
    CgVec3::new(a[0], a[1], a[2])
}
fn quat_arr(q: CgQuat) -> [f32; 4] {
    [q.v.x, q.v.y, q.v.z, q.s]
}
fn arr_quat(a: [f32; 4]) -> CgQuat {
    // cgmath's Quaternion::new is (w, xi, yj, zk); the array is (x, y, z, w).
    CgQuat::new(a[3], a[0], a[1], a[2])
}

/// Built-in particle-system presets the Add menu offers.  Each preset's texture
/// is preloaded in initialize_world so instances can be spawned synchronously
/// from the frame tick (see Renderer::spawn_particle_actor).
const PARTICLE_PRESETS: &[&str] = &["Fire", "Smoke", "Embers"];

/// The built-in materials seeded into a fresh library (constant-only PBR
/// looks).  Written to resources/materials/ the first time the editor runs, and
/// editable/saveable thereafter.
fn default_materials() -> Vec<MaterialFile> {
    vec![
        MaterialFile {
            name: "Matte".to_string(),
            desc: MaterialDesc::default(),
        },
        MaterialFile {
            name: "Plastic".to_string(),
            desc: MaterialDesc {
                mr_constant: CgVec4::new(0.0, 0.35, 0.0, 0.0),
                ..Default::default()
            },
        },
        MaterialFile {
            name: "Chrome".to_string(),
            desc: MaterialDesc {
                mr_constant: CgVec4::new(1.0, 0.12, 0.0, 0.0),
                ..Default::default()
            },
        },
    ]
}

/// ParticleParams for a named preset (see PARTICLE_PRESETS).  Shares one emitter
/// shape; only the texture, blend mode and colors differ.
fn preset_particle_params(preset: &str) -> ParticleParams {
    let (texture_file, blend_mode, start_color, end_color) = match preset {
        "Smoke" => (
            "game_assets/fx/smoke_t.png",
            ParticleBlendMode::AlphaBlend,
            CgVec4::new(0.55, 0.55, 0.55, 0.7),
            CgVec4::new(0.3, 0.3, 0.3, 0.0),
        ),
        "Embers" => (
            "game_assets/fx/ember_t.png",
            ParticleBlendMode::Additive,
            CgVec4::new(1.0, 0.65, 0.2, 1.0),
            CgVec4::new(1.0, 0.15, 0.0, 0.0),
        ),
        // "Fire" and any unknown preset.
        _ => (
            "game_assets/fx/fire_t.png",
            ParticleBlendMode::Additive,
            CgVec4::new(1.0, 0.8, 0.4, 1.0),
            CgVec4::new(1.0, 0.3, 0.05, 0.0),
        ),
    };
    ParticleParams {
        texture_file: texture_file.to_string(),
        blend_mode,
        min_burst_count: 0,
        max_burst_count: 0,
        min_particle_life: 0.6,
        max_particle_life: 1.4,
        _min_actor_life: 0.0,
        _max_actor_life: 0.0,
        min_start_spawn_rate: 0.02,
        max_start_spawn_rate: 0.05,
        min_start_pos: CgVec3::new(-0.1, 0.0, -0.1),
        max_start_pos: CgVec3::new(0.1, 0.0, 0.1),
        min_start_scale: CgVec3::new(0.15, 0.15, 0.15),
        max_start_scale: CgVec3::new(0.3, 0.3, 0.3),
        min_end_scale: CgVec3::new(0.4, 0.4, 0.4),
        max_end_scale: CgVec3::new(0.9, 0.9, 0.9),
        min_start_velocity: CgVec3::new(-0.3, 1.0, -0.3),
        max_start_velocity: CgVec3::new(0.3, 2.0, 0.3),
        min_start_rotation_rate: -60.0,
        max_start_rotation_rate: 60.0,
        min_start_acceleration: CgVec3::new(0.0, 0.5, 0.0),
        max_start_acceleration: CgVec3::new(0.0, 1.0, 0.0),
        min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
        max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
        start_color_0: start_color,
        start_color_1: start_color,
        end_color_0: end_color,
        _end_color1: end_color,
    }
}

/// Fills an "Add" menu with the object choices, recording the pick in `add`.
/// Shared by the menu bar's Add menu and the Scene tab's Add button.
/// `particle_names` are the particle library's entries (see
/// SplatGame::particle_resources), indexed by AddKind::Particle.
fn add_menu_ui(ui: &mut egui::Ui, add: &mut Option<AddKind>, particle_names: &[String]) {
    // Buttons auto-close the menu chain on click.
    if ui.button("Actor").clicked() {
        *add = Some(AddKind::Actor);
    }
    ui.menu_button("Light", |ui| {
        if ui.button("Directional").clicked() {
            *add = Some(AddKind::Light(LightType::Directional));
        }
        if ui.button("Point").clicked() {
            *add = Some(AddKind::Light(LightType::Point));
        }
        if ui.button("Spot").clicked() {
            *add = Some(AddKind::Light(LightType::Spot));
        }
        if ui.button("Skylight").clicked() {
            *add = Some(AddKind::Light(LightType::Skylight));
        }
    });
    ui.menu_button("Particle System", |ui| {
        for (i, preset) in particle_names.iter().enumerate() {
            if ui.button(preset.as_str()).clicked() {
                *add = Some(AddKind::Particle(i));
            }
        }
    });
    ui.separator();
    if ui.button("Gaussian Splat…").clicked() {
        *add = Some(AddKind::Splat);
    }
    if ui.button("Mujoco Actor").clicked() {
        *add = Some(AddKind::MujocoActor);
    }
}

/// Editable rows for a particle definition (the Resource inspector).  Returns
/// true if anything changed this frame.  `textures` are the preloaded particle
/// textures offered by the texture dropdown; a texture change only affects
/// future spawns (live emitters keep the texture they were built with).
fn draw_particle_params_ui(
    ui: &mut egui::Ui,
    params: &mut ParticleParams,
    textures: &[(String, TextureHandle)],
) -> bool {
    let mut changed = false;

    fn drag(ui: &mut egui::Ui, value: &mut f32) -> bool {
        ui.add(egui::DragValue::new(value).speed(0.02).max_decimals(3))
            .changed()
    }
    // A "min / max" pair of drag values under one label.
    fn minmax_f32(ui: &mut egui::Ui, label: &str, min: &mut f32, max: &mut f32) -> bool {
        let mut changed = false;
        ui.label(label);
        ui.horizontal(|ui| {
            changed |= drag(ui, min);
            ui.label("to");
            changed |= drag(ui, max);
        });
        changed
    }
    fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut CgVec3) -> bool {
        let mut changed = false;
        ui.label(label);
        ui.horizontal(|ui| {
            changed |= drag(ui, &mut v.x);
            changed |= drag(ui, &mut v.y);
            changed |= drag(ui, &mut v.z);
        });
        changed
    }
    fn color_row(ui: &mut egui::Ui, label: &str, color: &mut CgVec4) -> bool {
        ui.label(label);
        let mut rgba = [color.x, color.y, color.z, color.w];
        if ui.color_edit_button_rgba_unmultiplied(&mut rgba).changed() {
            *color = CgVec4::new(rgba[0], rgba[1], rgba[2], rgba[3]);
            true
        } else {
            false
        }
    }

    // "smoke_t" from "game_assets/fx/smoke_t.png" for the dropdown rows.
    let texture_label = |path: &str| resource_display_name(path);
    ui.label("Texture");
    egui::ComboBox::from_id_salt("particle_texture")
        .selected_text(texture_label(&params.texture_file))
        .show_ui(ui, |ui| {
            for (path, _) in textures {
                changed |= ui
                    .selectable_value(&mut params.texture_file, path.clone(), texture_label(path))
                    .changed();
            }
        });

    ui.label("Blend");
    egui::ComboBox::from_id_salt("particle_blend")
        .selected_text(match params.blend_mode {
            ParticleBlendMode::Additive => "Additive",
            ParticleBlendMode::AlphaBlend => "Alpha Blend",
        })
        .show_ui(ui, |ui| {
            changed |= ui
                .selectable_value(&mut params.blend_mode, ParticleBlendMode::Additive, "Additive")
                .changed();
            changed |= ui
                .selectable_value(
                    &mut params.blend_mode,
                    ParticleBlendMode::AlphaBlend,
                    "Alpha Blend",
                )
                .changed();
        });

    changed |= minmax_f32(
        ui,
        "Spawn Rate (s)",
        &mut params.min_start_spawn_rate,
        &mut params.max_start_spawn_rate,
    );
    changed |= minmax_f32(
        ui,
        "Particle Life (s)",
        &mut params.min_particle_life,
        &mut params.max_particle_life,
    );
    {
        // Burst counts are u32; edited through f32 drags.
        let (mut burst_min, mut burst_max) =
            (params.min_burst_count as f32, params.max_burst_count as f32);
        if minmax_f32(ui, "Burst Count", &mut burst_min, &mut burst_max) {
            params.min_burst_count = burst_min.max(0.0) as u32;
            params.max_burst_count = burst_max.max(0.0) as u32;
            changed = true;
        }
    }

    changed |= vec3_row(ui, "Spawn Pos Min", &mut params.min_start_pos);
    changed |= vec3_row(ui, "Spawn Pos Max", &mut params.max_start_pos);
    changed |= vec3_row(ui, "Start Scale Min", &mut params.min_start_scale);
    changed |= vec3_row(ui, "Start Scale Max", &mut params.max_start_scale);
    changed |= vec3_row(ui, "End Scale Min", &mut params.min_end_scale);
    changed |= vec3_row(ui, "End Scale Max", &mut params.max_end_scale);
    changed |= vec3_row(ui, "Velocity Min", &mut params.min_start_velocity);
    changed |= vec3_row(ui, "Velocity Max", &mut params.max_start_velocity);
    changed |= vec3_row(ui, "Acceleration Min", &mut params.min_start_acceleration);
    changed |= vec3_row(ui, "Acceleration Max", &mut params.max_start_acceleration);
    changed |= minmax_f32(
        ui,
        "Rotation Rate",
        &mut params.min_start_rotation_rate,
        &mut params.max_start_rotation_rate,
    );

    changed |= color_row(ui, "Start Color A", &mut params.start_color_0);
    changed |= color_row(ui, "Start Color B", &mut params.start_color_1);
    changed |= color_row(ui, "End Color", &mut params.end_color_0);

    changed
}

/// A particle system placed in the scene.  The live emitter lives in the
/// renderer (keyed by `handle`); its name and transform are mirrored here so the
/// outliner, Details panel and gizmo can edit it -- transform edits are pushed
/// back with Renderer::update_particle_transform.
#[derive(Clone)]
struct SceneParticle {
    name: String,
    handle: ParticleHandle,
    // Which preset (PARTICLE_PRESETS) this was spawned from, so it can be
    // recreated on scene load.
    preset: String,
    position: CgVec3,
    scale: CgVec3,
}

impl editor::EditorInspect for SceneParticle {
    fn inspect_properties(&mut self, visitor: &mut dyn editor::PropertyVisitor) -> bool {
        let mut changed = false;
        changed |= visitor.edit_text("Name", &mut self.name);
        changed |= visitor.edit_vec3("Position", &mut self.position);
        changed |= visitor.edit_vec3("Scale", &mut self.scale);
        changed
    }
}

impl editor::EditorInspect for SceneSplat {
    fn inspect_properties(&mut self, visitor: &mut dyn editor::PropertyVisitor) -> bool {
        let mut changed = false;
        // Transform (also draggable via the viewport gizmo).
        changed |= visitor.edit_vec3("Position", &mut self.transform.position);
        changed |= visitor.edit_rotation("Rotation", &mut self.transform.rotation);
        // Uniform scale only (non-uniform cloud scale is only approximate).
        // Every scale edit is floored above zero: a zero scale collapses the
        // cloud's world matrix and makes it vanish with no easy way back.  The
        // gizmo/hotkey paths already guard this; the Details drags (edit_float
        // allows 0) were the one gap that let the scale drop to 0.
        let mut uniform = self.transform.scale.x;
        if visitor.edit_float("Scale", &mut uniform) {
            uniform = uniform.max(SPLAT_MIN_SCALE);
            self.transform.scale = CgVec3::new(uniform, uniform, uniform);
            changed = true;
        }
        // Render params.
        changed |= visitor.edit_float("Falloff", &mut self.params.falloff);
        if visitor.edit_float("Splat Scale", &mut self.params.scale) {
            self.params.scale = self.params.scale.max(SPLAT_MIN_SCALE);
            changed = true;
        }
        changed |= visitor.edit_float("Contrast", &mut self.params.contrast);
        if visitor.edit_float("Overall Scale", &mut self.params.overall_scale) {
            self.params.overall_scale = self.params.overall_scale.max(SPLAT_MIN_SCALE);
            changed = true;
        }
        changed |= visitor.edit_float("SH Degree", &mut self.params.max_sh_degree);
        changed
    }
}

/// New selection after deleting the item at `deleted`, keeping indices in the
/// same list valid (shift later ones down; clear if the deleted item was it).
fn selection_after_delete(selected: Option<Selection>, deleted: Selection) -> Option<Selection> {
    fn shift(sel: usize, del: usize) -> Option<usize> {
        match sel.cmp(&del) {
            std::cmp::Ordering::Equal => None,
            std::cmp::Ordering::Greater => Some(sel - 1),
            std::cmp::Ordering::Less => Some(sel),
        }
    }
    match (selected, deleted) {
        (Some(Selection::Actor(s)), Selection::Actor(d)) => shift(s, d).map(Selection::Actor),
        (Some(Selection::Light(s)), Selection::Light(d)) => shift(s, d).map(Selection::Light),
        (Some(Selection::Particle(s)), Selection::Particle(d)) => {
            shift(s, d).map(Selection::Particle)
        }
        (other, _) => other,
    }
}

/// Draws one outliner list section (header + rows) for objects of one kind.
/// `add_ui` renders the section's add control (a "+" button/menu) beside the
/// header.  Rows support click-to-select, double-click-to-rename and a delete
/// button; picks are reported through the `*_out` accumulators (applied after
/// the pass).
#[allow(clippy::too_many_arguments)]
fn draw_outliner_section(
    ui: &mut egui::Ui,
    header: &str,
    make_sel: fn(usize) -> Selection,
    names: &[String],
    selected: Option<Selection>,
    multi: &[Selection],
    name_edit: &mut Option<Selection>,
    name_edit_buffer: &mut String,
    name_edit_focus: &mut bool,
    select_out: &mut Option<(Selection, bool)>,
    rename_out: &mut Vec<(usize, String)>,
    delete_out: &mut Option<Selection>,
    add_ui: impl FnOnce(&mut egui::Ui),
) {
    ui.add_space(8.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(header).strong());
        add_ui(ui);
    });
    if names.is_empty() {
        ui.label("(none)");
    }
    for (i, name) in names.iter().enumerate() {
        let this = make_sel(i);
        ui.horizontal(|ui| {
            if *name_edit == Some(this) {
                // Inline rename: save on Enter/blur, cancel on Escape.
                let edit_resp = ui.text_edit_singleline(name_edit_buffer);
                if *name_edit_focus {
                    edit_resp.request_focus();
                    *name_edit_focus = false;
                }
                let finish = edit_resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter));
                let cancel = ui.input(|i| i.key_pressed(egui::Key::Escape));
                if finish || cancel {
                    let new_name = name_edit_buffer.trim();
                    if finish && !new_name.is_empty() {
                        rename_out.push((i, new_name.to_string()));
                    }
                    *name_edit = None;
                }
            } else {
                let highlighted = selected == Some(this) || multi.contains(&this);
                let label_resp = ui.selectable_label(highlighted, name.as_str());
                if label_resp.clicked() {
                    // Ctrl+click joins/leaves the multi-selection.
                    let additive =
                        ui.input(|i| i.modifiers.ctrl || i.modifiers.command);
                    *select_out = Some((this, additive));
                }
                if label_resp.double_clicked() {
                    *name_edit = Some(this);
                    *name_edit_focus = true;
                    *name_edit_buffer = name.clone();
                }
            }
            if ui
                .small_button(
                    egui::RichText::new("✕").color(egui::Color32::from_rgb(235, 80, 80)),
                )
                .clicked()
            {
                *delete_out = Some(this);
                *name_edit = None; // Cancel any edit.
            }
        });
    }
}

// "6185394" -> "6,185,394" for status messages.
fn group_digits(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Where the async file pick currently is.  `Open`/`Reading` block a second
/// picker from being opened; `Reading` also drives a "Loading..." status line,
/// since a large file's read can take a while (especially in the browser).
enum PickerState {
    Idle,
    Open,
    Reading(String),
}

// Status line colors: red for errors/warnings the user must see, white for
// progress.
const STATUS_RED: CgVec4 = CgVec4::new(1.0, 0.25, 0.2, 1.0);
const STATUS_WHITE: CgVec4 = CgVec4::new(1.0, 1.0, 1.0, 1.0);

pub struct SplatGame {
    game_objects: Vec<GameObject>,
    game_camera: Camera,

    scene_actors: Vec<Actor>,
    scene_lights: Vec<Light>,
    scene_particles: Vec<SceneParticle>,
    // Preloaded particle-preset textures, as (texture path, handle), so presets
    // can be spawned synchronously from the tick (see PARTICLE_PRESETS).
    particle_textures: Vec<(String, TextureHandle)>,
    // Scene-wide post-process / tonemap settings -- the singleton "Post Process"
    // scene object.  Pushed to the renderer each frame
    scene_post_process: PostProcessSettings,
    // Screen-space AO / diffuse-GI toggles, edited alongside Post Process in
    // the same panel. Not a selectable scene object (no undo/redo) -- it's a
    // renderer-wide switch, not per-scene-object state.
    scene_ambient: AmbientSettings,

    // Undo/redo history
    undo_stack: UndoStack,
    // "Before" snapshot(s) for a transform/property gesture in progress,
    // captured on its first changed frame and committed to `undo_stack` once
    // the gesture ends (gizmo release / a frame with no further change).  A
    // single-selection gizmo drag or a Details edit uses a 1-element Vec; a
    // multi-selection gizmo drag uses one entry per dragged object (committed
    // as an `UndoAction::Batch`).
    pending_gizmo_edit: Option<Vec<(Selection, ObjectRef, ObjectSnapshot)>>,
    pending_property_edit: Option<(Selection, ObjectRef, ObjectSnapshot)>,
    // Splat names currently being (re)loaded because of an undo/redo (rather
    // than a user Add/Load), so `drain_pending_splats` knows not to push
    // another Create action for them when they land.
    undo_pending_splat_loads: std::collections::HashSet<String>,
    // Scene-tab selection (actor / light / particle), if any.
    selected: Option<Selection>,
    // Ctrl+click multi-selection, in click order (`selected` is its last
    // entry).
    multi_selected: Vec<Selection>,
    // Viewport context menu (right-click without dragging), at screen pos.
    context_menu: Option<egui::Pos2>,
    // Mouse travel while the right button is held, to tell a right-click
    // (context menu) from a right-drag (camera look).
    rmb_drag_accum: f32,
    // Whether the fly camera's look engaged at any point during the current
    // right-button hold; a release without it is a right-click.
    rmb_look_engaged: bool,
    // Object awaiting delete confirmation (the Scene tab's ✕ button).
    confirm_delete: Option<Selection>,
    // File > New Scene awaiting its confirmation modal (it wipes unsaved work).
    confirm_new_scene: bool,
    // Whether a user-saved startup scene exists (drives the Settings tab UI;
    // cached so native doesn't stat the config file every frame).
    has_custom_startup: bool,
    // Resource highlighted in the bottom Resources panel: shown in the right
    // panel's Resource inspector tab, and (for models) offered to the Details
    // panel's Model row as a one-click assignment.
    selected_resource: Option<ResourceSelection>,
    // The editable particle-definition library (see ParticleResource).  Scene
    // emitters reference an entry by name; edits push to live emitters.
    particle_resources: Vec<ParticleResource>,
    // The editable material library (see MaterialResource).  Populated in
    // initialize_world from resources/materials/ (or seeded defaults); the
    // renderer holds the built GPU materials, this holds their descriptions
    // so edits can be saved back to disk.
    material_library: Vec<MaterialResource>,
    // The texture library (see TextureResource): images under
    // game_assets/textures/ plus preset fx textures, scanned at startup and
    // grown by the "import texture" button.  Materials pick from these.
    texture_resources: Vec<TextureResource>,
    // The model library (see ModelResource): models under game_assets/models/,
    // scanned at startup. Actors pick from these.
    model_resources: Vec<ModelResource>,
    // "Import texture" plumbing: the picked image's (file name, bytes) land
    // here for a later tick to copy into game_assets/textures/ and load
    // (mirrors picked_ply).  Native only; wasm can't write project assets.
    picked_texture: Arc<Mutex<Option<(String, Vec<u8>)>>>,
    // "Browse…" plumbing for a MujocoSceneActor's XML Path field: (actor
    // name, resolved xml_path) lands here for a later tick to assign. The
    // path is a real filesystem path on native, or an IndexedDB cache key on
    // web (see `open_mujoco_xml_picker`).
    picked_mujoco_xml: Arc<Mutex<Option<(String, String)>>>,
    // The same, for a MujocoSceneActor's Trajectory field (see
    // `open_mujoco_trajectory_picker`).
    picked_mujoco_trajectory: Arc<Mutex<Option<(String, String)>>>,
    // Object currently being renamed (double-click in the Scene tab).
    name_edit: Option<Selection>,
    // The rename field's working text.  Persists across frames -- re-deriving
    // it from the actor each frame would wipe every keystroke and commit the
    // unchanged name.  Seeded from the actor's name when a rename begins.
    name_edit_buffer: String,
    // Set when a rename begins so the text field grabs keyboard focus on its
    // first frame; cleared once focused (requesting every frame would block
    // click-away-to-save).
    name_edit_focus: bool,
    // Viewport translate/rotate/scale gizmo for the selected actor.
    gizmo: TransformGizmo,
    // Persisted editor preferences (currently the gizmo hotkeys).  Loaded from
    // disk at startup; re-saved whenever a binding changes.
    editor_config: EditorConfig,
    // Keybindings window (opened from the right-hand Settings tab).
    show_settings: bool,
    // Which gizmo action (index into GIZMO_ACTIONS) is listening for its new
    // key, if the user clicked a binding in the keybindings window.
    rebinding: Option<usize>,
    // Keybindings "reset to defaults" awaiting the confirmation modal.
    confirm_reset: bool,
    // Which tab the right-hand editor panel shows; None keeps the panel
    // collapsed to just its tab strip.
    active_tab: Option<EditorTab>,
    // Bottom resources panel: shown/hidden by its "Resources" tab, height set
    // by dragging the grab strip along its top edge.
    resources_open: bool,
    resources_height: f32,
    // Content browser: the type-filter set (empty = show every kind), the
    // selected folder in the tree ("" = root / all folders), and the name
    // search text.  Assets of all kinds show together, narrowed by these.
    browser_filters: std::collections::HashSet<BrowserCategory>,
    browser_folder: String,
    browser_filter: String,
    // Content browser: material being renamed via its right-click menu, and
    // the working text.  Persists while the menu is open (like name_edit_*);
    // cleared when the menu closes so a re-open re-seeds from the current name.
    material_rename: Option<(MaterialHandle, String)>,
    // Monotonic counter so added objects get unique default names.
    next_object_num: u32,
    // Splat clouds that loaded, each carrying its own render params, aligned with
    // the renderer's splat indices.  Selecting one (in the outliner or by picking
    // empty space) shows its params in Details; `active_splat` is the one being
    // shown (only one renders at a time).
    scene_splats: Vec<SceneSplat>,
    active_splat: usize,
    // MuJoCo physics actors (see MujocoSceneActor): each owns its own live
    // sim, loaded/reloaded from xml_path by tick_mujoco_actors.
    scene_mujoco_actors: Vec<MujocoSceneActor>,
    // MJCF fetches that finished since the last tick: (actor name, requested
    // path, fetch result). MujocoScene itself is only ever built on the main
    // thread (mujoco-rs FFI state isn't Send), so the background task only
    // gathers the model's files; tick_mujoco_actors does the parse.
    pending_mujoco_loads: Arc<Mutex<Vec<(String, String, anyhow::Result<MjcfBundle>)>>>,
    // Trajectory clips that finished loading + retargeting since the last
    // tick: (actor name, requested path, result). Unlike the MJCF fetch above
    // this does do its real work off-thread -- retargeting only touches plain
    // data (see `trajectory::cache::load_retargeted`), no FFI state.
    pending_mujoco_trajectories: Arc<Mutex<Vec<(String, String, anyhow::Result<RetargetedClip>)>>>,
    // MuJoCo mesh geom .obj bytes that finished loading: (actor name, mesh
    // path, bytes). Web only -- the renderer's own load_model suspends on
    // IndexedDB, which the frame tick can't do, so the fetch is spawned and a
    // later tick turns the bytes into a Model (see `tick_mujoco_actors`).
    #[cfg(target_arch = "wasm32")]
    pending_mujoco_meshes: Arc<Mutex<Vec<(String, String, Vec<u8>)>>>,
    // "Load .ply" plumbing.  The file dialog must run asynchronously (a browser
    // file input can't block the frame loop), so it drops its result here and
    // tick_frame picks it up: (file name, source path if one exists -- native
    // only, browsers don't expose one -- and the bytes).  `picker_state` keeps a
    // held key / double tap from stacking dialogs and reports read progress.
    picked_ply: Arc<Mutex<Option<(String, Option<String>, Vec<u8>)>>>,
    // "Import model" plumbing (glb): the picked file's name + bytes land here
    // for a later tick to register (so it shows in the model picker) and
    // persist -- to game_assets/models on native, IndexedDB on web.
    picked_model: Arc<Mutex<Option<(String, Vec<u8>)>>>,
    // Lazily-loaded model bytes awaiting GPU upload on the render thread (web):
    // (path, bytes, select-when-loaded).  A background task fetches from
    // IndexedDB/network; a later tick uploads via load_model_from_bytes.  Feeds
    // both browser lazy-select and scene models that weren't loaded yet.
    #[allow(dead_code)] // web-only: native loads models synchronously.
    pending_model_uploads: Arc<Mutex<Vec<(String, Vec<u8>, bool)>>>,
    // Actors from a just-loaded scene whose model wasn't resolved yet (web,
    // while its bytes are still fetching): (actor index, model display name).
    // Reassigned when the matching model finishes uploading.
    pending_actor_models: Vec<(usize, String)>,
    picker_state: Arc<Mutex<PickerState>>,
    // "Load Scene" plumbing: the picked JSON file's bytes land here for a later
    // tick to parse (the file dialog runs async, like the .ply picker).
    picked_scene: Arc<Mutex<Option<Vec<u8>>>>,
    // Settings > "Choose start-up scene…" plumbing: the picked file's bytes are
    // validated and persisted as the startup scene on a later tick.
    picked_startup: Arc<Mutex<Option<Vec<u8>>>>,
    // Splat clouds a scene load requested: their .ply bytes are fetched on a
    // background task (native thread / wasm fetch) and applied here in a later
    // tick, since cloud creation must happen on the render thread.
    pending_splats: Arc<Mutex<Vec<(SplatDto, Vec<u8>)>>>,
    // Transient bottom-center message (load errors, clamp warnings) and its
    // remaining time on screen.
    status: Option<(String, CgVec4, f32)>,
    // On-screen move/look thumb-pads for touch.  Set to reveal on the first
    // touch so desktop mouse users never see them.
    touch_pads: TouchPads,
    // Shared keyboard/mouse fly-camera controller (WASD + arrow/mouse look).
    // Its defaults match this viewer; touch is handled separately below.
    fly_camera: FlyCamera,
    // Editor vs game mode.  Editor: the full menu bar + debug overlay are always
    // shown.  Game: only the small mode switch remains, for an unobstructed view.
    editor_mode: bool,
}

impl SplatGame {
    /// Opens the platform file dialog (native window / browser file input) on a
    /// background task; the picked file's name + bytes land in `picked_ply` for
    /// a later tick to upload.  No-op while a pick or read is already underway.
    fn open_ply_picker(&self) {
        {
            let mut state = self.picker_state.lock().unwrap();
            if !matches!(*state, PickerState::Idle) {
                return;
            }
            *state = PickerState::Open;
        }
        let picked = self.picked_ply.clone();
        let picker_state = self.picker_state.clone();

        let dialog = rfd::AsyncFileDialog::new();
        // Native pickers filter by extension reliably; browser `accept` handling
        // of unknown types like .ply is flaky (iOS Safari can grey out valid
        // files), so on wasm accept everything and let the parser validate.
        #[cfg(not(target_arch = "wasm32"))]
        let dialog = dialog.add_filter("Gaussian splat", &["ply"]);

        let pick = async move {
            if let Some(file) = dialog.pick_file().await {
                let name = file.file_name();
                *picker_state.lock().unwrap() = PickerState::Reading(name.clone());
                let bytes = file.read().await;
                // Remember where the file lives when the platform can tell us
                // (native), so saved scenes can reload the cloud by path. The
                // web build has no filesystem path, so cache the bytes into
                // IndexedDB under a synthetic key instead -- `load_binary`
                // checks there first, so saved scenes can reload it the same
                // way MuJoCo trajectory picks do (see
                // `open_mujoco_trajectory_picker`).
                #[cfg(not(target_arch = "wasm32"))]
                let path = Some(file.path().to_string_lossy().to_string());
                #[cfg(target_arch = "wasm32")]
                let path = {
                    let key = format!("splats/{name}");
                    black_splat::idb::put(&key, &bytes).await;
                    Some(key)
                };
                *picked.lock().unwrap() = Some((name, path, bytes));
            }
            *picker_state.lock().unwrap() = PickerState::Idle;
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(pick);
        // rfd's Windows/Linux dialogs run fine off the main thread; block a
        // throwaway thread on the future rather than stalling the frame loop.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(pick));
    }

    /// Opens the file dialog to import an image (png/jpg) as a texture
    /// resource.  The picked file's name + bytes land in `picked_texture` for a
    /// later tick to copy into game_assets/textures/ and load.  Native only:
    /// the web build can't write into the project's assets folder.
    fn open_texture_picker(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let picked = self.picked_texture.clone();
            let dialog = rfd::AsyncFileDialog::new().add_filter("Image", &["png", "jpg", "jpeg"]);
            let pick = async move {
                if let Some(file) = dialog.pick_file().await {
                    let name = file.file_name();
                    let bytes = file.read().await;
                    *picked.lock().unwrap() = Some((name, bytes));
                }
            };
            std::thread::spawn(move || pollster::block_on(pick));
        }
    }

    /// Opens the file dialog to pick a MuJoCo scene XML for a
    /// MujocoSceneActor's XML Path field. Works on both platforms: native
    /// resolves to the file's real path (read straight off disk by
    /// `assets::load_string`); web has no real path, so the bytes are cached
    /// into IndexedDB under a synthetic key that `load_string` also checks.
    /// Either way, the resulting path/key lands in `picked_mujoco_xml` for a
    /// later tick to assign.
    fn open_mujoco_xml_picker(&self, actor_name: String) {
        let picked = self.picked_mujoco_xml.clone();

        // Native hands back the .xml's real path and stops there: MuJoCo
        // opens the model's <include>s and meshes off disk relative to it.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dialog = rfd::AsyncFileDialog::new().add_filter("MuJoCo XML", &["xml"]);
            let pick = async move {
                if let Some(file) = dialog.pick_file().await {
                    let path = file.path().to_string_lossy().into_owned();
                    *picked.lock().unwrap() = Some((actor_name, path));
                }
            };
            std::thread::spawn(move || pollster::block_on(pick));
        }

        // The web build has no filesystem behind the picked file, so it
        // imports the model's whole folder instead (see
        // `import_mujoco_model_folder`). Picking a lone .xml can't work: every
        // real model references siblings, and they'd all dangle.
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            match import_mujoco_model_folder().await {
                Ok(Some(root)) => *picked.lock().unwrap() = Some((actor_name, root)),
                // Dialog dismissed, or an empty folder.
                Ok(None) => {}
                Err(e) => log!("MuJoCo model import failed: {e:?}"),
            }
        });
    }

    /// Opens the file dialog to pick a trajectory clip JSON for a
    /// MujocoSceneActor's Trajectory field, landing the result in
    /// `picked_mujoco_trajectory`. Same native/web path-vs-IndexedDB-key
    /// split as [`open_mujoco_xml_picker`](Self::open_mujoco_xml_picker).
    fn open_mujoco_trajectory_picker(&self, actor_name: String) {
        let picked = self.picked_mujoco_trajectory.clone();
        let dialog = rfd::AsyncFileDialog::new();
        #[cfg(not(target_arch = "wasm32"))]
        let dialog = dialog.add_filter("Trajectory clip", &["json"]);
        let pick = async move {
            if let Some(file) = dialog.pick_file().await {
                #[cfg(not(target_arch = "wasm32"))]
                let path = file.path().to_string_lossy().into_owned();
                #[cfg(target_arch = "wasm32")]
                let path = {
                    let key = format!("mujoco_trajectories/{}", file.file_name());
                    let bytes = file.read().await;
                    black_splat::idb::put(&key, &bytes).await;
                    key
                };
                *picked.lock().unwrap() = Some((actor_name, path));
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(pick);
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(pick));
    }

    /// Opens the file dialog to import a glb model.  The picked file's name +
    /// bytes land in `picked_model` for a later tick to register and persist
    /// (game_assets/models on native, IndexedDB on web).  Works on both
    /// platforms, unlike texture import (see `open_ply_picker` for the pattern).
    fn open_model_picker(&self) {
        let picked = self.picked_model.clone();
        let dialog = rfd::AsyncFileDialog::new();
        #[cfg(not(target_arch = "wasm32"))]
        let dialog = dialog.add_filter("Model", &["glb", "gltf"]);
        let pick = async move {
            if let Some(file) = dialog.pick_file().await {
                let name = file.file_name();
                let bytes = file.read().await;
                *picked.lock().unwrap() = Some((name, bytes));
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(pick);
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(pick));
    }

    /// The catalog path of the model whose display name is `name`, if any.
    fn catalog_path_for(&self, name: &str) -> Option<String> {
        self.model_resources
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.path.clone())
    }

    /// Spawn point a few units ahead of the camera for newly added objects.
    fn spawn_point(&self) -> CgVec3 {
        let (_view, view_dir, _right) = self.game_camera.calculate_view_matrix();
        self.game_camera.get_position() + view_dir * ADD_OBJECT_DISTANCE
    }

    /// Selects a freshly added object, staying on whatever tab is open -- adding
    /// from the Scene tab keeps you there, with the new object shown selected in
    /// the outliner (and its gizmo/icon in the viewport).
    fn select_after_add(&mut self, sel: Selection) {
        self.selected = Some(sel);
    }

    /// Display name of a selected object (None if the index went stale).
    fn selection_name(&self, sel: Selection) -> Option<String> {
        match sel {
            Selection::Actor(i) => self.scene_actors.get(i).map(|a| a.get_name().to_string()),
            Selection::Light(i) => self.scene_lights.get(i).map(|l| l.get_name().to_string()),
            Selection::Particle(i) => self.scene_particles.get(i).map(|p| p.name.clone()),
            Selection::Splat(i) => self.scene_splats.get(i).map(|s| s.name.clone()),
            Selection::MujocoActor(i) => {
                self.scene_mujoco_actors.get(i).map(|m| m.name.clone())
            }
            Selection::PostProcess => Some("Post Process".to_string()),
        }
    }

    /// World position of a selected object of any kind (None if stale).
    fn selection_position(&self, sel: Selection) -> Option<CgVec3> {
        match sel {
            Selection::Actor(i) => self.scene_actors.get(i).map(|a| a.get_position()),
            Selection::Light(i) => self.scene_lights.get(i).map(|l| l.get_position()),
            Selection::Particle(i) => self.scene_particles.get(i).map(|p| p.position),
            Selection::Splat(i) => self.scene_splats.get(i).map(|s| s.transform.position),
            Selection::MujocoActor(i) => self.scene_mujoco_actors.get(i).map(|m| m.position),
            // The post-process singleton has no world position.
            Selection::PostProcess => None,
        }
    }

    /// World-space orientation of a selected object of any kind, for the
    /// gizmo's Local space (identity for kinds with no rotation of their own,
    /// e.g. particles; None if stale or the kind has no transform at all).
    fn selection_rotation(&self, sel: Selection) -> Option<CgQuat> {
        match sel {
            Selection::Actor(i) => self.scene_actors.get(i).map(|a| a.get_rotation()),
            Selection::Light(i) => self.scene_lights.get(i).map(|l| l.get_rotation()),
            Selection::Particle(i) => self.scene_particles.get(i).map(|_| CG_QUAT_IDENT),
            Selection::Splat(i) => self.scene_splats.get(i).map(|s| s.transform.rotation),
            Selection::MujocoActor(i) => self.scene_mujoco_actors.get(i).map(|m| m.rotation),
            Selection::PostProcess => None,
        }
    }

    /// Moves a selected object of any kind to `pos`, pushing the change to the
    /// renderer's copy.
    fn set_selection_position(&mut self, sel: Selection, pos: &CgVec3, renderer: &mut Renderer) {
        match sel {
            Selection::Actor(i) => {
                if let Some(actor) = self.scene_actors.get_mut(i) {
                    actor.set_position(pos);
                    renderer.add_or_update_actor(actor);
                }
            }
            Selection::Light(i) => {
                if let Some(light) = self.scene_lights.get_mut(i) {
                    light.set_position(pos);
                    renderer.add_or_update_light(light);
                }
            }
            Selection::Particle(i) => {
                if let Some(particle) = self.scene_particles.get_mut(i) {
                    particle.position = *pos;
                    renderer.update_particle_transform(&particle.handle, pos, &None);
                }
            }
            Selection::Splat(i) => {
                if let Some(splat) = self.scene_splats.get_mut(i) {
                    splat.transform.position = *pos;
                    if i == self.active_splat {
                        renderer.set_gaussian_splat_transform(&splat.transform);
                    }
                }
            }
            Selection::MujocoActor(i) => {
                if let Some(m) = self.scene_mujoco_actors.get_mut(i) {
                    m.position = *pos;
                }
            }
            Selection::PostProcess => {}
        }
    }

    /// Applies one frame of a multi-selection gizmo drag to a selected object.
    /// `new_pos` is the object's position already orbited/scaled about the
    /// pivot; the rotation/scale deltas compose onto the object's own where
    /// the kind has them (particles don't rotate; splats scale uniformly).
    fn apply_pivot_delta(
        &mut self,
        sel: Selection,
        new_pos: CgVec3,
        delta_rot: CgQuat,
        scale_mult: CgVec3,
        renderer: &mut Renderer,
    ) {
        fn scaled(v: CgVec3, m: CgVec3) -> CgVec3 {
            CgVec3::new(
                (v.x * m.x).max(0.001),
                (v.y * m.y).max(0.001),
                (v.z * m.z).max(0.001),
            )
        }
        match sel {
            Selection::Actor(i) => {
                if let Some(actor) = self.scene_actors.get_mut(i) {
                    actor.set_position(&new_pos);
                    actor.set_rotation(&(delta_rot * actor.get_rotation()).normalize());
                    actor.set_scale(&scaled(actor.get_scale(), scale_mult));
                    renderer.add_or_update_actor(actor);
                }
            }
            Selection::Light(i) => {
                if let Some(light) = self.scene_lights.get_mut(i) {
                    light.set_position(&new_pos);
                    light.set_rotation(&(delta_rot * light.get_rotation()).normalize());
                    renderer.add_or_update_light(light);
                }
            }
            Selection::Particle(i) => {
                if let Some(particle) = self.scene_particles.get_mut(i) {
                    particle.position = new_pos;
                    particle.scale = scaled(particle.scale, scale_mult);
                    renderer.update_particle_transform(
                        &particle.handle,
                        &new_pos,
                        &Some(particle.scale),
                    );
                }
            }
            Selection::Splat(i) => {
                if let Some(splat) = self.scene_splats.get_mut(i) {
                    splat.transform.position = new_pos;
                    splat.transform.rotation =
                        (delta_rot * splat.transform.rotation).normalize();
                    // Uniform scale only (non-uniform cloud scale is only
                    // approximate).
                    let uniform = (splat.transform.scale.x * scale_mult.x).max(0.001);
                    splat.transform.scale = CgVec3::new(uniform, uniform, uniform);
                    if i == self.active_splat {
                        renderer.set_gaussian_splat_transform(&splat.transform);
                    }
                }
            }
            Selection::MujocoActor(i) => {
                if let Some(m) = self.scene_mujoco_actors.get_mut(i) {
                    m.position = new_pos;
                    m.rotation = (delta_rot * m.rotation).normalize();
                }
            }
            Selection::PostProcess => {}
        }
    }

    /// World position and an approximate radius of the selected object, used to
    /// frame it (the [F] hotkey).  None if nothing is selected.
    fn selected_focus(&self) -> Option<(CgVec3, f32)> {
        fn radius_of(scale: CgVec3) -> f32 {
            scale.x.abs().max(scale.y.abs()).max(scale.z.abs()).max(0.5)
        }
        match self.selected? {
            Selection::Actor(i) => self
                .scene_actors
                .get(i)
                .map(|a| (a.get_position(), radius_of(a.get_scale()))),
            Selection::Light(i) => self.scene_lights.get(i).map(|l| (l.get_position(), 0.6)),
            Selection::Particle(i) => self
                .scene_particles
                .get(i)
                .map(|p| (p.position, radius_of(p.scale))),
            // Frame a splat cloud at its transform origin (its true extent isn't
            // known here, so use a generous default radius).
            Selection::Splat(i) => self
                .scene_splats
                .get(i)
                .map(|s| (s.transform.position, 4.0)),
            Selection::MujocoActor(i) => self
                .scene_mujoco_actors
                .get(i)
                .map(|m| (m.position, 1.5)),
            Selection::PostProcess => None,
        }
    }

    /// Frames the selected object: dollies the camera along its current view
    /// direction so the object sits centered at a comfortable distance, keeping
    /// the current orientation.  No-op if nothing is selected.
    fn frame_selected(&mut self) {
        let Some((target, radius)) = self.selected_focus() else {
            return;
        };
        let (_view, view_dir, _right) = self.game_camera.calculate_view_matrix();
        let dir = view_dir.normalize();
        // Pull back far enough for the object's radius to fit, with some margin.
        let distance = (radius * 3.5 + 1.0).max(2.0);
        self.game_camera.set_position(&(target - dir * distance));
    }

    /// Adds a new Actor into the scene ahead of the camera and selects it
    fn add_actor(&mut self, renderer: &mut Renderer) {
        let mut actor = Actor::new();
        actor.set_name(&format!("Actor {}", self.next_object_num));
        actor.set_position(&self.spawn_point());
        renderer.add_or_update_actor(&actor);

        self.next_object_num += 1;
        self.scene_actors.push(actor);
        let index = self.scene_actors.len() - 1;
        self.select_after_add(Selection::Actor(index));
        if let Some((obj, snapshot)) = self.snapshot_object(Selection::Actor(index)) {
            self.undo_stack.push(UndoAction::Create {
                obj,
                snapshot,
                index_hint: index,
            });
        }
    }

    /// Adds a new light of the given type into the scene ahead of the camera
    /// and mirrors it into the renderer's light map.
    fn add_light(&mut self, light_type: LightType, renderer: &mut Renderer) {
        let mut light = Light::new();
        light.set_light_type(light_type);
        light.set_position(&self.spawn_point());
        if light_type == LightType::Skylight {
            // Hemisphere defaults: cool sky above, warm bounce below.
            light.set_color(SKYLIGHT_TOP);
            light.set_color2(SKYLIGHT_BOTTOM);
        }
        renderer.add_or_update_light(&light);
        self.scene_lights.push(light);
        let index = self.scene_lights.len() - 1;
        self.select_after_add(Selection::Light(index));
        if let Some((obj, snapshot)) = self.snapshot_object(Selection::Light(index)) {
            self.undo_stack.push(UndoAction::Create {
                obj,
                snapshot,
                index_hint: index,
            });
        }
    }

    /// Adds a new (initially unloaded) MuJoCo actor ahead of the camera and
    /// selects it. Its Details panel's XML Path field is where the user
    /// points it at an MJCF file -- tick_mujoco_actors loads it once set.
    fn add_mujoco_actor(&mut self) {
        self.next_object_num += 1;
        let mut actor = MujocoSceneActor::new(format!("Mujoco Actor {}", self.next_object_num));
        actor.position = self.spawn_point();
        self.scene_mujoco_actors.push(actor);
        let index = self.scene_mujoco_actors.len() - 1;
        self.select_after_add(Selection::MujocoActor(index));
        if let Some((obj, snapshot)) = self.snapshot_object(Selection::MujocoActor(index)) {
            self.undo_stack.push(UndoAction::Create {
                obj,
                snapshot,
                index_hint: index,
            });
        }
    }

    /// The skylight every new scene starts with (deletable like any light).
    fn add_default_skylight(&mut self, renderer: &mut Renderer) {
        let mut light = Light::new();
        light.set_name("Skylight");
        light.set_light_type(LightType::Skylight);
        light.set_position(&CgVec3::new(0.0, 5.0, 0.0));
        light.set_color(SKYLIGHT_TOP);
        light.set_color2(SKYLIGHT_BOTTOM);
        renderer.add_or_update_light(&light);
        self.scene_lights.push(light);
    }

    /// Spawns a particle system from a named particle resource at the given
    /// transform, recording it as a scene object.  Uses the preloaded texture, so
    /// no async is needed.  Returns false if the resource is unknown or its
    /// texture wasn't preloaded.
    fn spawn_particle(
        &mut self,
        preset_name: &str,
        name: String,
        position: CgVec3,
        scale: CgVec3,
        renderer: &mut Renderer,
    ) -> bool {
        let Some(params) = self
            .particle_resources
            .iter()
            .find(|r| r.name == preset_name)
            .map(|r| r.params.clone())
        else {
            self.status = Some((
                format!("Unknown particle resource: {preset_name}"),
                STATUS_RED,
                5.0,
            ));
            return false;
        };
        let Some((_, texture)) = self
            .particle_textures
            .iter()
            .find(|(path, _)| *path == params.texture_file)
        else {
            self.status = Some((
                format!("Particle texture not loaded: {}", params.texture_file),
                STATUS_RED,
                5.0,
            ));
            return false;
        };
        let texture = *texture;
        let mut transform = ActorTransform::from_position(position);
        transform.scale = scale;
        let handle = renderer.spawn_particle_actor(&transform, &params, &texture, true);
        self.scene_particles.push(SceneParticle {
            name,
            handle,
            preset: preset_name.to_string(),
            position,
            scale,
        });
        true
    }

    /// Spawns a particle system from resource `preset` (an index into
    /// particle_resources) ahead of the camera and selects it (Add menu).
    fn add_particle(&mut self, preset: usize, renderer: &mut Renderer) {
        let Some(preset_name) = self.particle_resources.get(preset).map(|r| r.name.clone())
        else {
            return;
        };
        self.next_object_num += 1;
        let name = format!("{preset_name} {}", self.next_object_num);
        let spawn_pos = self.spawn_point();
        if self.spawn_particle(&preset_name, name, spawn_pos, CG_VEC3_ONE, renderer) {
            let index = self.scene_particles.len() - 1;
            self.select_after_add(Selection::Particle(index));
            if let Some((obj, snapshot)) = self.snapshot_object(Selection::Particle(index)) {
                self.undo_stack.push(UndoAction::Create {
                    obj,
                    snapshot,
                    index_hint: index,
                });
            }
        }
    }

    /// Removes the selected object from its list and the renderer, keeping the
    /// Scene-tab selection valid.
    fn delete_selected(&mut self, sel: Selection, renderer: &mut Renderer) {
        // Snapshot before removal so Ctrl+Z can restore it.
        let before = self.snapshot_object(sel);
        let removed_index = match sel {
            Selection::Actor(i) => {
                if i >= self.scene_actors.len() {
                    None
                } else {
                    let actor = self.scene_actors.remove(i);
                    renderer.remove_actor(&actor);
                    Some(i)
                }
            }
            Selection::Light(i) => {
                if i >= self.scene_lights.len() {
                    None
                } else {
                    let light = self.scene_lights.remove(i);
                    renderer.remove_light(&light);
                    Some(i)
                }
            }
            Selection::Particle(i) => {
                if i >= self.scene_particles.len() {
                    None
                } else {
                    let particle = self.scene_particles.remove(i);
                    renderer.remove_particle_actor(&particle.handle);
                    Some(i)
                }
            }
            Selection::Splat(i) => {
                if i >= self.scene_splats.len() {
                    None
                } else {
                    self.scene_splats.remove(i);
                    renderer.remove_gaussian_splat(i);
                    // Mirror the active-cloud renumbering GaussianSplatPass::
                    // remove_model applied on the renderer side.
                    if self.scene_splats.is_empty() {
                        self.active_splat = 0;
                    } else if i < self.active_splat {
                        self.active_splat -= 1;
                    } else if i == self.active_splat {
                        self.active_splat = self.active_splat.min(self.scene_splats.len() - 1);
                        renderer.set_active_gaussian_splat(self.active_splat);
                        if let Some(active) = self.scene_splats.get(self.active_splat) {
                            renderer.set_gaussian_splat_params(&active.params);
                            renderer.set_gaussian_splat_transform(&active.transform);
                        }
                    }
                    Some(i)
                }
            }
            Selection::MujocoActor(i) => {
                if i >= self.scene_mujoco_actors.len() {
                    None
                } else {
                    let actor = self.scene_mujoco_actors.remove(i);
                    for mesh_actor in &actor.mesh_geom_actors {
                        renderer.remove_actor(mesh_actor);
                    }
                    Some(i)
                }
            }
            // The post-process singleton is always present; it can't be deleted.
            Selection::PostProcess => None,
        };
        let Some(index) = removed_index else {
            return;
        };
        if let Some((obj, snapshot)) = before {
            self.undo_stack.push(UndoAction::Delete {
                obj,
                snapshot,
                index_hint: index,
            });
        }
        self.selected = selection_after_delete(self.selected, sel);
        // Deleting shifts indices; drop the multi-selection rather than remap.
        self.multi_selected.clear();
        self.context_menu = None;
    }

    /// Captures a stable reference + full clone of the given selection's
    /// current state, for the undo stack.  `None` if the index is stale.
    fn snapshot_object(&self, sel: Selection) -> Option<(ObjectRef, ObjectSnapshot)> {
        match sel {
            Selection::Actor(i) => self
                .scene_actors
                .get(i)
                .map(|a| (ObjectRef::Actor(a.id), ObjectSnapshot::Actor(a.clone()))),
            Selection::Light(i) => self
                .scene_lights
                .get(i)
                .map(|l| (ObjectRef::Light(l.id), ObjectSnapshot::Light(l.clone()))),
            Selection::Particle(i) => self.scene_particles.get(i).map(|p| {
                (
                    ObjectRef::Particle(p.handle.clone()),
                    ObjectSnapshot::Particle(p.clone()),
                )
            }),
            Selection::Splat(i) => self
                .scene_splats
                .get(i)
                .map(|s| (ObjectRef::Splat(s.name.clone()), ObjectSnapshot::Splat(s.clone()))),
            Selection::MujocoActor(i) => self.scene_mujoco_actors.get(i).map(|m| {
                (
                    ObjectRef::MujocoActor(m.name.clone()),
                    ObjectSnapshot::MujocoActor(MujocoActorSnapshot {
                        name: m.name.clone(),
                        xml_path: m.xml_path.clone(),
                        trajectory_path: m.trajectory_path.clone(),
                        position: m.position,
                        rotation: m.rotation,
                        playing: m.playing,
                        speed: m.speed,
                        looping: m.looping,
                        wireframe: m.wireframe,
                    }),
                )
            }),
            Selection::PostProcess => Some((
                ObjectRef::PostProcess,
                ObjectSnapshot::PostProcess(self.scene_post_process),
            )),
        }
    }

    /// Finds the current `Selection` (i.e. live Vec index) a stable
    /// `ObjectRef` refers to.  `None` if that object no longer exists.
    fn resolve_ref(&self, obj: &ObjectRef) -> Option<Selection> {
        match obj {
            ObjectRef::Actor(id) => self
                .scene_actors
                .iter()
                .position(|a| a.id == *id)
                .map(Selection::Actor),
            ObjectRef::Light(id) => self
                .scene_lights
                .iter()
                .position(|l| l.id == *id)
                .map(Selection::Light),
            ObjectRef::Particle(h) => self
                .scene_particles
                .iter()
                .position(|p| &p.handle == h)
                .map(Selection::Particle),
            ObjectRef::Splat(name) => self
                .scene_splats
                .iter()
                .position(|s| &s.name == name)
                .map(Selection::Splat),
            ObjectRef::MujocoActor(name) => self
                .scene_mujoco_actors
                .iter()
                .position(|m| &m.name == name)
                .map(Selection::MujocoActor),
            ObjectRef::PostProcess => Some(Selection::PostProcess),
        }
    }

    /// Overwrites `obj` with `snapshot`'s data and pushes the change to 
    /// the renderer.
    fn restore_snapshot(
        &mut self,
        obj: &ObjectRef,
        snapshot: &ObjectSnapshot,
        renderer: &mut Renderer,
    ) {
        let Some(sel) = self.resolve_ref(obj) else {
            return;
        };
        match (sel, snapshot) {
            (Selection::Actor(i), ObjectSnapshot::Actor(a)) => {
                if let Some(slot) = self.scene_actors.get_mut(i) {
                    *slot = a.clone();
                    renderer.add_or_update_actor(slot);
                }
            }
            (Selection::Light(i), ObjectSnapshot::Light(l)) => {
                if let Some(slot) = self.scene_lights.get_mut(i) {
                    *slot = l.clone();
                    renderer.add_or_update_light(slot);
                }
            }
            (Selection::Particle(i), ObjectSnapshot::Particle(p)) => {
                if let Some(slot) = self.scene_particles.get_mut(i) {
                    // Keep the live handle -- the snapshot's is only an
                    // ObjectRef identity, not a different live emitter.
                    let handle = slot.handle.clone();
                    *slot = p.clone();
                    slot.handle = handle;
                    renderer.update_particle_transform(
                        &slot.handle,
                        &slot.position,
                        &Some(slot.scale),
                    );
                }
            }
            (Selection::Splat(i), ObjectSnapshot::Splat(s)) => {
                if let Some(slot) = self.scene_splats.get_mut(i) {
                    *slot = s.clone();
                    if i == self.active_splat {
                        renderer.set_gaussian_splat_params(&slot.params);
                        renderer.set_gaussian_splat_transform(&slot.transform);
                    }
                }
            }
            (Selection::MujocoActor(i), ObjectSnapshot::MujocoActor(m)) => {
                if let Some(slot) = self.scene_mujoco_actors.get_mut(i) {
                    slot.name = m.name.clone();
                    slot.xml_path = m.xml_path.clone();
                    slot.trajectory_path = m.trajectory_path.clone();
                    slot.position = m.position;
                    slot.rotation = m.rotation;
                    slot.playing = m.playing;
                    slot.speed = m.speed;
                    slot.looping = m.looping;
                    slot.wireframe = m.wireframe;
                    // scene/loaded_path and clip/loaded_trajectory_path are
                    // left alone -- tick_mujoco_actors reconciles a changed
                    // xml_path/trajectory_path against the live sim.
                    if let Some(scene) = &mut slot.scene {
                        scene.set_playing(slot.playing);
                        scene.set_speed(slot.speed);
                        scene.set_wireframe(slot.wireframe);
                    }
                }
            }
            (Selection::PostProcess, ObjectSnapshot::PostProcess(pp)) => {
                self.scene_post_process = *pp;
                renderer.set_post_process_settings(&self.scene_post_process);
            }
            // Kind mismatch shouldn't happen: an ObjectRef always resolves to
            // the same kind of Selection its own snapshot variant carries.
            _ => {}
        }
        self.selected = Some(sel);
    }

    /// Removes the object `obj` currently refers to from its list and the
    /// renderer.  Used by undo-of-create and redo-of-delete; splats route
    /// through the same active-cloud renumbering as `delete_selected`.
    fn remove_by_ref(&mut self, obj: &ObjectRef, renderer: &mut Renderer) {
        let Some(sel) = self.resolve_ref(obj) else {
            return;
        };
        match sel {
            Selection::Actor(i) => {
                let actor = self.scene_actors.remove(i);
                renderer.remove_actor(&actor);
            }
            Selection::Light(i) => {
                let light = self.scene_lights.remove(i);
                renderer.remove_light(&light);
            }
            Selection::Particle(i) => {
                let particle = self.scene_particles.remove(i);
                renderer.remove_particle_actor(&particle.handle);
            }
            Selection::Splat(i) => {
                self.scene_splats.remove(i);
                renderer.remove_gaussian_splat(i);
                if self.scene_splats.is_empty() {
                    self.active_splat = 0;
                } else if i < self.active_splat {
                    self.active_splat -= 1;
                } else if i == self.active_splat {
                    self.active_splat = self.active_splat.min(self.scene_splats.len() - 1);
                    renderer.set_active_gaussian_splat(self.active_splat);
                    if let Some(active) = self.scene_splats.get(self.active_splat) {
                        renderer.set_gaussian_splat_params(&active.params);
                        renderer.set_gaussian_splat_transform(&active.transform);
                    }
                }
            }
            Selection::MujocoActor(i) => {
                let actor = self.scene_mujoco_actors.remove(i);
                for mesh_actor in &actor.mesh_geom_actors {
                    renderer.remove_actor(mesh_actor);
                }
            }
            Selection::PostProcess => {}
        }
        self.selected = selection_after_delete(self.selected, sel);
        self.multi_selected.clear();
    }

    /// Recreates `snapshot` (undo-of-delete / redo-of-create) and selects it.
    /// Actor/Light/Particle recreate synchronously; a splat's GPU data isn't
    /// retained after removal, so it re-queues an async reload from its
    /// stored path (same path scene load uses) -- if that's empty (a
    /// browser-picked cloud with no re-readable source), recreation can't
    /// happen and this is a no-op, matching the same documented limitation
    /// scene save/load already has for such clouds.
    ///
    /// Returns `Some((old_handle, new_handle))` when recreating a particle
    /// issued a fresh renderer handle, so the caller can repoint the action
    /// currently being processed (not yet back on either stack, so
    /// `remap_particle_handle` -- called internally for every OTHER pending
    /// entry -- can't reach it).
    fn recreate_snapshot(
        &mut self,
        snapshot: &ObjectSnapshot,
        renderer: &mut Renderer,
    ) -> Option<(ParticleHandle, ParticleHandle)> {
        match snapshot {
            ObjectSnapshot::Actor(a) => {
                renderer.add_or_update_actor(a);
                self.scene_actors.push(a.clone());
                self.select_after_add(Selection::Actor(self.scene_actors.len() - 1));
                None
            }
            ObjectSnapshot::Light(l) => {
                renderer.add_or_update_light(l);
                self.scene_lights.push(l.clone());
                self.select_after_add(Selection::Light(self.scene_lights.len() - 1));
                None
            }
            ObjectSnapshot::Particle(p) => {
                // Re-spawns through the normal path (issues a NEW renderer
                // handle) rather than trying to resurrect the old one.
                if self.spawn_particle(&p.preset, p.name.clone(), p.position, p.scale, renderer) {
                    let new_index = self.scene_particles.len() - 1;
                    let new_handle = self.scene_particles[new_index].handle.clone();
                    self.select_after_add(Selection::Particle(new_index));
                    if new_handle != p.handle {
                        self.remap_particle_handle(&p.handle, &new_handle);
                        return Some((p.handle.clone(), new_handle));
                    }
                }
                None
            }
            ObjectSnapshot::Splat(s) => {
                if s.path.is_empty() {
                    self.status = Some((
                        format!("Can't restore '{}': no reloadable source", s.name),
                        STATUS_RED,
                        5.0,
                    ));
                    return None;
                }
                self.undo_pending_splat_loads.insert(s.name.clone());
                self.queue_splat_load(SplatDto {
                    name: s.name.clone(),
                    path: s.path.clone(),
                    params: SplatParamsDto::from_params(&s.params),
                    position: vec3_arr(s.transform.position),
                    rotation: quat_arr(s.transform.rotation),
                    scale: vec3_arr(s.transform.scale),
                });
                // Lands asynchronously via drain_pending_splats, which selects
                // it once the reload completes.
                None
            }
            ObjectSnapshot::MujocoActor(m) => {
                self.scene_mujoco_actors.push(MujocoSceneActor {
                    name: m.name.clone(),
                    xml_path: m.xml_path.clone(),
                    loaded_path: String::new(),
                    pending_path: None,
                    trajectory_path: m.trajectory_path.clone(),
                    loaded_trajectory_path: String::new(),
                    pending_trajectory_path: None,
                    clip: None,
                    clip_time: 0.0,
                    looping: m.looping,
                    position: m.position,
                    rotation: m.rotation,
                    playing: m.playing,
                    speed: m.speed,
                    wireframe: m.wireframe,
                    scene: None,
                    mesh_geom_actors: Vec::new(),
                    mesh_model_cache: std::collections::HashMap::new(),
                    #[cfg(target_arch = "wasm32")]
                    requested_meshes: std::collections::HashSet::new(),
                });
                // scene is None / loaded_path empty, so tick_mujoco_actors
                // reloads it from xml_path on the next tick.
                self.select_after_add(Selection::MujocoActor(self.scene_mujoco_actors.len() - 1));
                None
            }
            // The singleton is never deleted/recreated, only edited.
            ObjectSnapshot::PostProcess(_) => None,
        }
    }

    /// After a particle is respawned with a NEW renderer handle, repoints any
    /// OTHER pending stack entry that still references the OLD handle --
    /// otherwise it would resolve to nothing the next time it's applied.
    /// (The action currently being processed isn't in either stack yet; its
    /// caller patches that one separately using this function's return --
    /// see `recreate_snapshot`.)
    fn remap_particle_handle(&mut self, old: &ParticleHandle, new: &ParticleHandle) {
        fn remap(action: &mut UndoAction, old: &ParticleHandle, new: &ParticleHandle) {
            match action {
                UndoAction::Edit { obj, .. }
                | UndoAction::Create { obj, .. }
                | UndoAction::Delete { obj, .. } => {
                    if let ObjectRef::Particle(h) = obj {
                        if h == old {
                            *h = new.clone();
                        }
                    }
                }
                UndoAction::Batch(actions) => {
                    for a in actions {
                        remap(a, old, new);
                    }
                }
            }
        }
        for action in self
            .undo_stack
            .undo
            .iter_mut()
            .chain(self.undo_stack.redo.iter_mut())
        {
            remap(action, old, new);
        }
    }

    /// Applies the "undo" side of `action` and returns it (patched if a
    /// particle recreate inside it issued a new renderer handle) for the
    /// caller to push onto the redo stack.
    fn apply_undo(&mut self, action: UndoAction, renderer: &mut Renderer) -> UndoAction {
        match action {
            UndoAction::Edit { obj, before, after } => {
                self.restore_snapshot(&obj, &before, renderer);
                UndoAction::Edit { obj, before, after }
            }
            UndoAction::Create {
                obj,
                snapshot,
                index_hint,
            } => {
                self.remove_by_ref(&obj, renderer);
                UndoAction::Create {
                    obj,
                    snapshot,
                    index_hint,
                }
            }
            UndoAction::Delete {
                obj,
                snapshot,
                index_hint,
            } => {
                let obj = match self.recreate_snapshot(&snapshot, renderer) {
                    Some((old, new)) if matches!(&obj, ObjectRef::Particle(h) if *h == old) => {
                        ObjectRef::Particle(new)
                    }
                    _ => obj,
                };
                UndoAction::Delete {
                    obj,
                    snapshot,
                    index_hint,
                }
            }
            UndoAction::Batch(actions) => UndoAction::Batch(
                actions
                    .into_iter()
                    .map(|a| self.apply_undo(a, renderer))
                    .collect(),
            ),
        }
    }

    /// Applies the "redo" side of `action` and returns it (patched if a
    /// particle recreate inside it issued a new renderer handle) for the
    /// caller to push back onto the undo stack.
    fn apply_redo(&mut self, action: UndoAction, renderer: &mut Renderer) -> UndoAction {
        match action {
            UndoAction::Edit { obj, before, after } => {
                self.restore_snapshot(&obj, &after, renderer);
                UndoAction::Edit { obj, before, after }
            }
            UndoAction::Create {
                obj,
                snapshot,
                index_hint,
            } => {
                let obj = match self.recreate_snapshot(&snapshot, renderer) {
                    Some((old, new)) if matches!(&obj, ObjectRef::Particle(h) if *h == old) => {
                        ObjectRef::Particle(new)
                    }
                    _ => obj,
                };
                UndoAction::Create {
                    obj,
                    snapshot,
                    index_hint,
                }
            }
            UndoAction::Delete {
                obj,
                snapshot,
                index_hint,
            } => {
                self.remove_by_ref(&obj, renderer);
                UndoAction::Delete {
                    obj,
                    snapshot,
                    index_hint,
                }
            }
            UndoAction::Batch(actions) => UndoAction::Batch(
                actions
                    .into_iter()
                    .map(|a| self.apply_redo(a, renderer))
                    .collect(),
            ),
        }
    }

    /// Ctrl+Z: reverses the most recent action, if any.
    fn undo(&mut self, renderer: &mut Renderer) {
        let Some(action) = self.undo_stack.undo.pop() else {
            return;
        };
        let action = self.apply_undo(action, renderer);
        self.undo_stack.redo.push(action);
    }

    /// Ctrl+Shift+Z / Ctrl+Y: reapplies the most recently undone action, if any.
    fn redo(&mut self, renderer: &mut Renderer) {
        let Some(action) = self.undo_stack.redo.pop() else {
            return;
        };
        let action = self.apply_redo(action, renderer);
        self.undo_stack.undo.push(action);
    }

    /// Makes splat `i` the rendered cloud and pushes its params to the renderer.
    /// Selecting or cycling splats routes through here (only one renders).
    fn activate_splat(&mut self, i: usize, renderer: &mut Renderer) {
        if let Some(splat) = self.scene_splats.get(i) {
            self.active_splat = i;
            renderer.set_active_gaussian_splat(i);
            renderer.set_gaussian_splat_params(&splat.params);
            renderer.set_gaussian_splat_transform(&splat.transform);
        }
    }

    /// Reconciles each MujocoSceneActor's live sim against its `xml_path`
    /// field (kicking off an async reload when they diverge -- after a
    /// Details-panel edit, undo/redo, or scene load) and steps + wireframes
    /// every loaded one. Called once per frame from tick_frame_internal.
    fn tick_mujoco_actors(&mut self, renderer: &mut Renderer, game_config: &Config) {
        // Mesh .obj bytes that arrived since the last tick, turned into Models
        // here on the render thread. Dropped on the floor if the actor is gone
        // or has since reloaded (a stale mesh belongs to a model that no
        // longer exists).
        #[cfg(target_arch = "wasm32")]
        {
            let fetched: Vec<(String, String, Vec<u8>)> =
                std::mem::take(&mut *self.pending_mujoco_meshes.lock().unwrap());
            for (name, path, bytes) in fetched {
                if !self
                    .scene_mujoco_actors
                    .iter()
                    .any(|a| a.name == name && a.requested_meshes.contains(&path))
                {
                    continue;
                }
                let handle = renderer.load_model_from_bytes(&path, &bytes, false);
                if let Some(actor) = self.scene_mujoco_actors.iter_mut().find(|a| a.name == name) {
                    actor.mesh_model_cache.insert(path, handle);
                }
            }
        }

        let finished: Vec<(String, String, anyhow::Result<MjcfBundle>)> =
            std::mem::take(&mut *self.pending_mujoco_loads.lock().unwrap());
        for (name, path, result) in finished {
            if let Some(actor) = self.scene_mujoco_actors.iter_mut().find(|a| a.name == name) {
                if actor.pending_path.as_deref() == Some(path.as_str()) {
                    actor.pending_path = None;
                }
                // Only apply if xml_path still matches what was requested --
                // a further edit in the meantime gets its own load below.
                if actor.xml_path == path {
                    match result.and_then(MujocoScene::from_bundle) {
                        Ok(mut scene) => {
                            scene.set_playing(actor.playing);
                            scene.set_speed(actor.speed);
                            scene.set_wireframe(actor.wireframe);
                            actor.scene = Some(scene);
                            actor.loaded_path = path;
                            // A clip is retargeted onto one specific model's
                            // joint layout, so it can't carry over to a
                            // different sim -- drop it and let the reconcile
                            // below re-retarget against the new model.
                            actor.clip = None;
                            actor.loaded_trajectory_path = String::new();
                            actor.clip_time = 0.0;
                            // The old sim's mesh geoms (if any) no longer
                            // correspond to anything -- drop them from the
                            // renderer; the loop below rebuilds fresh ones
                            // from the new scene.
                            for mesh_actor in actor.mesh_geom_actors.drain(..) {
                                renderer.remove_actor(&mesh_actor);
                            }
                            actor.mesh_model_cache.clear();
                            // Else the new model's meshes would look already
                            // requested and never be fetched.
                            #[cfg(target_arch = "wasm32")]
                            actor.requested_meshes.clear();
                        }
                        Err(e) => {
                            log!("Mujoco actor '{name}' failed to load '{path}': {e}");
                            // Mark handled so this path isn't retried every
                            // frame; changing xml_path again will retry.
                            actor.loaded_path = path;
                        }
                    }
                }
            }
        }

        for actor in &mut self.scene_mujoco_actors {
            if actor.xml_path.is_empty()
                || actor.xml_path == actor.loaded_path
                || actor.pending_path.as_deref() == Some(actor.xml_path.as_str())
            {
                continue;
            }
            actor.pending_path = Some(actor.xml_path.clone());
            let name = actor.name.clone();
            let path = actor.xml_path.clone();
            let pending = self.pending_mujoco_loads.clone();
            let task = async move {
                let result = MujocoScene::load_bundle(&path).await;
                pending.lock().unwrap().push((name, path, result));
            };
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(task);
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::spawn(move || pollster::block_on(task));
        }

        // Trajectory clips that finished loading. Same "still wanted?" guard as
        // the MJCF drain above: an edit while the load was in flight wins.
        let finished_clips: Vec<(String, String, anyhow::Result<RetargetedClip>)> =
            std::mem::take(&mut *self.pending_mujoco_trajectories.lock().unwrap());
        for (name, path, result) in finished_clips {
            if let Some(actor) = self.scene_mujoco_actors.iter_mut().find(|a| a.name == name) {
                if actor.pending_trajectory_path.as_deref() == Some(path.as_str()) {
                    actor.pending_trajectory_path = None;
                }
                if actor.trajectory_path != path {
                    continue;
                }
                actor.loaded_trajectory_path = path.clone();
                match result {
                    Ok(clip) => {
                        if clip.joints.is_empty() {
                            // Retargeting matched nothing, so playback would
                            // silently leave the arm limp -- the usual cause is
                            // a clip recorded against a different robot's joint
                            // names. Say so rather than looking like a hang.
                            log!(
                                "Mujoco actor '{name}': trajectory '{path}' shares no joint names \
                                 with the loaded model -- not playing it."
                            );
                            actor.clip = None;
                        } else {
                            log!(
                                "Mujoco actor '{name}': playing '{path}' -- {} joints, {} frames.",
                                clip.joints.len(),
                                clip.frame_count()
                            );
                            actor.clip = Some(clip);
                            actor.clip_time = 0.0;
                        }
                    }
                    Err(e) => {
                        log!("Mujoco actor '{name}' failed to load trajectory '{path}': {e}");
                        actor.clip = None;
                    }
                }
                // A clip takes the sim off physics; clearing one hands it back.
                if let Some(scene) = &mut actor.scene {
                    scene.set_kinematic(actor.clip.is_some());
                }
            }
        }

        // Dispatch trajectory loads. Unlike the MJCF reconcile this also waits
        // on `scene` -- retargeting needs the loaded model's joint layout.
        for actor in &mut self.scene_mujoco_actors {
            if actor.trajectory_path == actor.loaded_trajectory_path
                || actor.pending_trajectory_path.as_deref() == Some(actor.trajectory_path.as_str())
            {
                continue;
            }
            // Cleared field: drop the clip and resume simulating.
            if actor.trajectory_path.is_empty() {
                actor.clip = None;
                actor.clip_time = 0.0;
                actor.loaded_trajectory_path = String::new();
                if let Some(scene) = &mut actor.scene {
                    scene.set_kinematic(false);
                }
                continue;
            }
            let Some(scene) = &actor.scene else {
                continue; // Retry once the sim finishes loading.
            };
            let target_joints = scene.joint_tracks();
            if target_joints.is_empty() {
                // On wasm `scene` exists as soon as the files are staged, but
                // the joint layout only arrives once the JS side has actually
                // parsed the model a few frames later. Retargeting against an
                // empty layout matches nothing and would drop the clip for
                // good, since the dispatch guard above won't fire twice for
                // the same path. Wait for the layout instead.
                continue;
            }
            actor.pending_trajectory_path = Some(actor.trajectory_path.clone());
            let name = actor.name.clone();
            let path = actor.trajectory_path.clone();
            let pending = self.pending_mujoco_trajectories.clone();
            let task = async move {
                let result =
                    black_splat::trajectory::cache::load_retargeted(&path, &target_joints).await;
                pending.lock().unwrap().push((name, path, result));
            };
            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_futures::spawn_local(task);
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::spawn(move || pollster::block_on(task));
        }

        for actor in &mut self.scene_mujoco_actors {
            // Drive the pose from the clip before drawing, so the wireframe and
            // mesh geoms below both read the frame we just wrote. The sim won't
            // step out from under it -- a bound clip sets `kinematic`. Also
            // remembers the frame index so the unmatched-object placeholder
            // draw below (a separate call, since it doesn't touch the sim)
            // can read the same frame.
            let mut clip_frame_idx = None;
            if let (Some(scene), Some(clip)) = (&mut actor.scene, &actor.clip) {
                let total = clip.frame_count() as f32 * clip.dt;
                if total > 0.0 {
                    if actor.playing {
                        actor.clip_time += game_config.delta_time * actor.speed;
                    }
                    actor.clip_time = if actor.looping {
                        actor.clip_time.rem_euclid(total)
                    } else {
                        actor.clip_time.clamp(0.0, total - clip.dt)
                    };
                    let frame = (actor.clip_time / clip.dt) as usize;
                    let frame = frame.min(clip.frame_count() - 1);
                    scene.apply_trajectory_frame(clip, frame);
                    clip_frame_idx = Some(frame);
                }
            }

            if let Some(scene) = &mut actor.scene {
                scene.tick_and_draw_at(renderer, game_config, actor.position, actor.rotation);

                if let (Some(clip), Some(frame)) = (&actor.clip, clip_frame_idx) {
                    scene.draw_unmatched_objects(
                        renderer,
                        game_config,
                        clip,
                        frame,
                        actor.position,
                        actor.rotation,
                    );
                }

                let mesh_geoms = scene.mesh_geoms(actor.position, actor.rotation);

                // Native reads the .obj straight off disk, so the load can
                // just happen here.
                #[cfg(not(target_arch = "wasm32"))]
                for geom in &mesh_geoms {
                    if !actor.mesh_model_cache.contains_key(&geom.mesh_path) {
                        let handle = pollster::block_on(renderer.load_model(&geom.mesh_path, false));
                        actor.mesh_model_cache.insert(geom.mesh_path.clone(), handle);
                    }
                }

                // Wasm can't: the .obj lives in IndexedDB and reading it
                // suspends, which the frame tick can't do. Dispatch the fetch
                // and let a later tick register the model (see the drain at
                // the top of this function).
                #[cfg(target_arch = "wasm32")]
                for geom in &mesh_geoms {
                    if actor.mesh_model_cache.contains_key(&geom.mesh_path)
                        || !actor.requested_meshes.insert(geom.mesh_path.clone())
                    {
                        continue;
                    }
                    let name = actor.name.clone();
                    let path = geom.mesh_path.clone();
                    let pending = self.pending_mujoco_meshes.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        match black_splat::assets::load_binary(&path).await {
                            Ok(bytes) => pending.lock().unwrap().push((name, path, bytes)),
                            Err(e) => log!("MuJoCo mesh {path} failed to load: {e}"),
                        }
                    });
                }

                // One Actor per mesh geom, index-aligned with `mesh_geoms`
                // (stable across frames -- see the field comment on
                // `mesh_geom_actors`). Held back until every mesh has a model:
                // the alignment is by index, so a partial build would pair
                // actors with the wrong geoms.
                if actor.mesh_geom_actors.is_empty()
                    && !mesh_geoms.is_empty()
                    && mesh_geoms.iter().all(|g| actor.mesh_model_cache.contains_key(&g.mesh_path))
                {
                    for geom in &mesh_geoms {
                        let handle = actor.mesh_model_cache[&geom.mesh_path];
                        let mut mesh_actor = Actor::new();
                        mesh_actor.set_model(&handle);
                        mesh_actor.set_layer(&SceneLayer::World, &None);
                        mesh_actor.set_exclude_from_env_capture(true);
                        // Named by its (metallic, roughness) pair so geoms
                        // sharing a MJCF <material> share one Material too --
                        // create_material returns the existing handle for a
                        // name it's already seen.
                        let material_name =
                            format!("mj_mr_{:.3}_{:.3}", geom.metallic, geom.roughness);
                        let material = renderer.create_material(
                            &material_name,
                            &MaterialDesc {
                                mr_constant: CgVec4::new(geom.metallic, geom.roughness, 0.0, 0.0),
                                ..Default::default()
                            },
                        );
                        mesh_actor.set_material(&material);
                        actor.mesh_geom_actors.push(mesh_actor);
                    }
                }

                for (mesh_actor, geom) in actor.mesh_geom_actors.iter_mut().zip(mesh_geoms.iter()) {
                    mesh_actor.set_position(&geom.position);
                    mesh_actor.set_rotation(&geom.rotation);
                    mesh_actor.set_color(geom.rgba.into());
                    renderer.add_or_update_actor(mesh_actor);
                }
            }
        }
    }

    /// Draws editor billboard icons for every light over the 3D view, and
    /// returns the index of a light whose icon was clicked this frame (for
    /// selection).  Point lights show a tinted disc; directional/spot lights add
    /// a short direction arrow.
    fn draw_light_icons(&self, ctx: &egui::Context, config: &Config) -> Option<usize> {
        if self.scene_lights.is_empty() {
            return None;
        }
        let (view, _, _) = self.game_camera.calculate_view_matrix();
        let proj = cgmath::perspective(
            cgmath::Deg(config.fov),
            config.window_width as f32 / config.window_height as f32,
            0.1,
            10000.0,
        );
        let view_proj = proj * view;
        let screen = ctx.content_rect();
        let project = |world: CgVec3| -> Option<egui::Pos2> {
            let clip = view_proj * CgVec4::new(world.x, world.y, world.z, 1.0);
            if clip.w < 0.01 {
                return None; // Behind the camera.
            }
            Some(egui::pos2(
                screen.left() + (clip.x / clip.w + 1.0) * 0.5 * screen.width(),
                screen.top() + (1.0 - clip.y / clip.w) * 0.5 * screen.height(),
            ))
        };

        // Under the panels/menus but over the 3D view (like the gizmo).
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("light_icons"),
        ));
        let (pointer, pressed) =
            ctx.input(|i| (i.pointer.interact_pos(), i.pointer.primary_pressed()));
        let over_ui = ctx.egui_wants_pointer_input();
        let highlight = egui::Color32::from_rgb(255, 220, 60);

        let mut clicked = None;
        for (i, light) in self.scene_lights.iter().enumerate() {
            let position = light.get_position();
            let Some(center) = project(position) else {
                continue;
            };
            let is_selected = self.selected == Some(Selection::Light(i));
            let c = light.get_color();
            let tint = egui::Color32::from_rgb(
                (c.x.clamp(0.0, 1.0) * 255.0) as u8,
                (c.y.clamp(0.0, 1.0) * 255.0) as u8,
                (c.z.clamp(0.0, 1.0) * 255.0) as u8,
            );
            let radius = 7.0;

            // Direction arrow for directional/spot lights, camera-scaled so it
            // stays a constant on-screen size.  Point lights and skylights are
            // directionless.
            if matches!(
                light.get_light_type(),
                LightType::Directional | LightType::Spot
            ) {
                let dist = (position - self.game_camera.get_position())
                    .magnitude()
                    .max(0.01);
                let forward = light.get_rotation() * CgVec3::new(0.0, 0.0, 1.0);
                if let Some(tip) = project(position + forward * dist * 0.18) {
                    painter.arrow(center, tip - center, egui::Stroke::new(2.0, tint));
                }
            }

            painter.circle_filled(center, radius, tint);
            painter.circle_stroke(
                center,
                radius,
                egui::Stroke::new(
                    if is_selected { 2.5 } else { 1.5 },
                    if is_selected {
                        highlight
                    } else {
                        egui::Color32::from_gray(235)
                    },
                ),
            );
            painter.text(
                center + egui::vec2(0.0, radius + 2.0),
                egui::Align2::CENTER_TOP,
                light.get_name(),
                egui::FontId::proportional(12.0),
                egui::Color32::from_gray(230),
            );

            if pressed && !over_ui {
                if let Some(p) = pointer {
                    if p.distance(center) < radius + 4.0 {
                        clicked = Some(i);
                    }
                }
            }
        }
        clicked
    }

    /// Picks the front-most actor or particle whose on-screen footprint contains
    /// `pointer` (viewport click-to-select).  Lights are picked via their icons
    /// and splats as a fallback -- both handled by the caller -- so this only
    /// covers the placed 3D objects.  None if the click missed them all.
    fn pick_object(
        &self,
        ctx: &egui::Context,
        config: &Config,
        pointer: egui::Pos2,
    ) -> Option<Selection> {
        let (view, _dir, right) = self.game_camera.calculate_view_matrix();
        let proj = cgmath::perspective(
            cgmath::Deg(config.fov),
            config.window_width as f32 / config.window_height as f32,
            0.1,
            10000.0,
        );
        let view_proj = proj * view;
        let screen = ctx.content_rect();
        let cam_pos = self.game_camera.get_position();
        let project = |world: CgVec3| -> Option<egui::Pos2> {
            let clip = view_proj * CgVec4::new(world.x, world.y, world.z, 1.0);
            if clip.w < 0.01 {
                return None; // Behind the camera.
            }
            Some(egui::pos2(
                screen.left() + (clip.x / clip.w + 1.0) * 0.5 * screen.width(),
                screen.top() + (1.0 - clip.y / clip.w) * 0.5 * screen.height(),
            ))
        };
        // Minimum tap radius in points, so small/distant objects stay pickable
        // (also finger-friendly on touch).
        const MIN_PICK_PX: f32 = 16.0;
        // On-screen radius of a world sphere of `radius` at `center`: project a
        // point one radius to the side and measure the pixel gap.
        let screen_radius = |center: CgVec3, c_screen: egui::Pos2, radius: f32| -> f32 {
            project(center + right * radius)
                .map_or(MIN_PICK_PX, |edge| edge.distance(c_screen))
                .max(MIN_PICK_PX)
        };
        fn radius_of(scale: CgVec3) -> f32 {
            scale.x.abs().max(scale.y.abs()).max(scale.z.abs()).max(0.5)
        }

        // Actors and particles as (selection, center, radius); the front-most
        // (nearest camera) whose disc contains the pointer wins.
        let candidates = self
            .scene_actors
            .iter()
            .enumerate()
            .map(|(i, a)| (Selection::Actor(i), a.get_position(), radius_of(a.get_scale())))
            .chain(
                self.scene_particles
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (Selection::Particle(i), p.position, radius_of(p.scale))),
            )
            .chain(
                self.scene_mujoco_actors
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (Selection::MujocoActor(i), m.position, 1.0)),
            );
        let mut best: Option<(Selection, f32)> = None;
        for (sel, center, radius) in candidates {
            let Some(c_screen) = project(center) else {
                continue;
            };
            if pointer.distance(c_screen) <= screen_radius(center, c_screen, radius) {
                let dist = (center - cam_pos).magnitude();
                if best.map_or(true, |(_, d)| dist < d) {
                    best = Some((sel, dist));
                }
            }
        }
        best.map(|(sel, _)| sel)
    }

    /// Snapshots the editable scene into the serializable form (see SceneFile).
    /// `model_resources` maps model handles to the resource names stored in the
    /// file (resolved back on load).
    fn build_scene_file(&self, model_resources: &[(String, ModelHandle)]) -> SceneFile {
        let model_name = |handle: ModelHandle| -> Option<String> {
            model_resources
                .iter()
                .find(|(_, h)| *h == handle)
                .map(|(name, _)| name.clone())
        };
        let material_name = |handle: MaterialHandle| -> Option<String> {
            self.material_library
                .iter()
                .find(|m| m.handle == handle)
                .map(|m| m.name.clone())
        };
        SceneFile {
            version: SCENE_FORMAT_VERSION,
            camera: Some(CameraDto {
                position: vec3_arr(self.game_camera.get_position()),
                rotation: vec3_arr(self.game_camera.get_rotation()),
            }),
            actors: self
                .scene_actors
                .iter()
                .map(|a| ActorDto {
                    name: a.get_name().to_string(),
                    position: vec3_arr(a.get_position()),
                    rotation: quat_arr(a.get_rotation()),
                    scale: vec3_arr(a.get_scale()),
                    model: model_name(a.get_model()),
                    material: material_name(a.get_material()),
                    layer: a.get_layer().0.choice_index() as u32,
                    shadow_catcher: a.is_shadow_catcher(),
                })
                .collect(),
            lights: self
                .scene_lights
                .iter()
                .map(|l| LightDto {
                    name: l.get_name().to_string(),
                    position: vec3_arr(l.get_position()),
                    rotation: quat_arr(l.get_rotation()),
                    light_type: l.get_light_type().choice_index() as u32,
                    color: vec3_arr(l.get_color()),
                    color2: vec3_arr(l.get_color2()),
                    intensity: l.get_intensity(),
                    range: l.get_range(),
                    spot_angle: l.get_spot_angle(),
                    casts_shadow: l.casts_shadow(),
                    shadow_cascades: l.shadow_cascades(),
                    shadow_distance: l.shadow_distance(),
                    shadow_density: l.shadow_density(),
                })
                .collect(),
            particles: self
                .scene_particles
                .iter()
                .map(|p| ParticleDto {
                    name: p.name.clone(),
                    preset: p.preset.clone(),
                    position: vec3_arr(p.position),
                    scale: vec3_arr(p.scale),
                })
                .collect(),
            splats: self
                .scene_splats
                .iter()
                .map(|s| SplatDto {
                    name: s.name.clone(),
                    path: s.path.clone(),
                    params: SplatParamsDto::from_params(&s.params),
                    position: vec3_arr(s.transform.position),
                    rotation: quat_arr(s.transform.rotation),
                    scale: vec3_arr(s.transform.scale),
                })
                .collect(),
            mujoco_actors: self
                .scene_mujoco_actors
                .iter()
                .map(|m| MujocoActorDto {
                    name: m.name.clone(),
                    xml_path: m.xml_path.clone(),
                    trajectory_path: m.trajectory_path.clone(),
                    position: vec3_arr(m.position),
                    rotation: quat_arr(m.rotation),
                    playing: m.playing,
                    speed: m.speed,
                    looping: m.looping,
                    wireframe: m.wireframe,
                })
                .collect(),
            post_process: Some(PostProcessDto::from_settings(&self.scene_post_process)),
            ambient: Some(AmbientDto::from_settings(&self.scene_ambient)),
        }
    }

    /// Serializes the scene to pretty JSON and writes it to a file the user
    /// picks.  Native shows the OS save dialog.  On the web, browsers with the
    /// File System Access API (Chrome/Edge) get a real save dialog too; the rest
    /// (Firefox/Safari) fall back to a download.  Either way everything happens
    /// locally -- the file never leaves the machine.
    fn save_scene(&self, model_resources: &[(String, ModelHandle)]) {
        let scene = self.build_scene_file(model_resources);
        let json = match serde_json::to_string_pretty(&scene) {
            Ok(json) => json,
            Err(_) => return,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let task = async move {
                let dialog = rfd::AsyncFileDialog::new()
                    .set_file_name("scene.json")
                    .add_filter("Scene", &["json"]);
                if let Some(file) = dialog.save_file().await {
                    let _ = std::fs::write(file.path(), json.as_bytes());
                }
            };
            std::thread::spawn(move || pollster::block_on(task));
        }
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move {
            match save_with_fs_access_api(&json).await {
                // Saved through the real dialog, or the user cancelled it --
                // either way, done.
                Ok(()) => {}
                // No File System Access API in this browser: fall back to the
                // classic download (lands in the Downloads folder).
                Err(()) => {
                    let dialog = rfd::AsyncFileDialog::new().set_file_name("scene.json");
                    if let Some(file) = dialog.save_file().await {
                        let _ = file.write(json.as_bytes()).await;
                    }
                }
            }
        });
    }

    /// Opens the async file dialog to pick a scene .json; its bytes land in
    /// `destination` for a later tick to handle.  Used by both File > Load
    /// Scene (`picked_scene`) and the startup-scene setting (`picked_startup`).
    fn open_scene_picker_into(destination: Arc<Mutex<Option<Vec<u8>>>>) {
        let task = async move {
            let dialog = rfd::AsyncFileDialog::new();
            #[cfg(not(target_arch = "wasm32"))]
            let dialog = dialog.add_filter("Scene", &["json"]);
            if let Some(file) = dialog.pick_file().await {
                let bytes = file.read().await;
                *destination.lock().unwrap() = Some(bytes);
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(task);
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(task));
    }

    /// Empties the scene: actors, lights, particles and splat clouds all removed
    /// from the lists and the renderer (File > New Scene, and the first step of
    /// a scene load).
    fn clear_scene(&mut self, renderer: &mut Renderer) {
        for actor in self.scene_actors.drain(..) {
            renderer.remove_actor(&actor);
        }
        for particle in self.scene_particles.drain(..) {
            renderer.remove_particle_actor(&particle.handle);
        }
        self.scene_lights.clear();
        renderer.clear_lights();
        self.scene_splats.clear();
        renderer.clear_gaussian_splats();
        self.active_splat = 0;
        self.pending_splats.lock().unwrap().clear();
        for actor in self.scene_mujoco_actors.drain(..) {
            for mesh_actor in &actor.mesh_geom_actors {
                renderer.remove_actor(mesh_actor);
            }
        }
        self.selected = None;
        self.multi_selected.clear();
        self.context_menu = None;
        self.confirm_delete = None;
        self.name_edit = None;
        // A fresh scene starts from the default tonemap; a loaded scene overrides
        // it in apply_scene_objects.
        self.scene_post_process = PostProcessSettings::default();
        renderer.set_post_process_settings(&self.scene_post_process);
        self.scene_ambient = AmbientSettings::default();
        renderer.set_ambient_settings(&self.scene_ambient);
    }

    /// Rebuilds the editable scene objects (actors / lights / particles +
    /// camera) from a parsed scene.  Splat clouds are NOT loaded here -- their
    /// .ply reads are async; see `queue_splat_load` / `initialize_world`.
    /// Assumes the scene was already cleared.
    fn apply_scene_objects(
        &mut self,
        scene: &SceneFile,
        model_resources: &[(String, ModelHandle)],
        renderer: &mut Renderer,
    ) {
        // Fresh set of "waiting for its model to finish loading" actors (web
        // lazy fetch); populated below and drained as uploads arrive.
        self.pending_actor_models.clear();
        for dto in &scene.actors {
            let mut actor = Actor::new();
            actor.set_name(&dto.name);
            actor.set_position(&arr_vec3(dto.position));
            actor.set_rotation(&arr_quat(dto.rotation));
            actor.set_scale(&arr_vec3(dto.scale));
            if let Some(model) = &dto.model {
                match model_resources.iter().find(|(n, _)| n == model) {
                    Some((_, handle)) => actor.set_model(handle),
                    // Not loaded yet: leave the model unset and remember to
                    // assign it when its bytes arrive (pending_model_uploads).
                    None => self
                        .pending_actor_models
                        .push((self.scene_actors.len(), model.clone())),
                }
            }
            if let Some(material) = &dto.material {
                if let Some(res) = self.material_library.iter().find(|m| &m.name == material) {
                    actor.set_material(&res.handle);
                }
            }
            actor.set_layer(&SceneLayer::from_choice_index(dto.layer as usize), &None);
            actor.set_shadow_catcher(dto.shadow_catcher);
            renderer.add_or_update_actor(&actor);
            self.scene_actors.push(actor);
        }

        for dto in &scene.lights {
            let mut light = Light::new();
            light.set_name(&dto.name);
            light.set_position(&arr_vec3(dto.position));
            light.set_rotation(&arr_quat(dto.rotation));
            light.set_light_type(LightType::from_choice_index(dto.light_type as usize));
            light.set_color(arr_vec3(dto.color));
            light.set_color2(arr_vec3(dto.color2));
            light.set_intensity(dto.intensity);
            light.set_range(dto.range);
            light.set_spot_angle(dto.spot_angle);
            light.set_casts_shadow(dto.casts_shadow);
            light.set_shadow_cascades(dto.shadow_cascades);
            light.set_shadow_distance(dto.shadow_distance);
            light.set_shadow_density(dto.shadow_density);
            renderer.add_or_update_light(&light);
            self.scene_lights.push(light);
        }

        for dto in &scene.particles {
            self.spawn_particle(
                &dto.preset,
                dto.name.clone(),
                arr_vec3(dto.position),
                arr_vec3(dto.scale),
                renderer,
            );
        }

        if let Some(camera) = &scene.camera {
            self.game_camera.set_position(&arr_vec3(camera.position));
            self.game_camera.set_rotation(&arr_vec3(camera.rotation));
            renderer.set_camera(&self.game_camera);
        }

        for dto in &scene.mujoco_actors {
            self.scene_mujoco_actors.push(MujocoSceneActor {
                name: dto.name.clone(),
                xml_path: dto.xml_path.clone(),
                loaded_path: String::new(),
                pending_path: None,
                trajectory_path: dto.trajectory_path.clone(),
                loaded_trajectory_path: String::new(),
                pending_trajectory_path: None,
                clip: None,
                clip_time: 0.0,
                looping: dto.looping,
                position: arr_vec3(dto.position),
                rotation: arr_quat(dto.rotation),
                playing: dto.playing,
                speed: dto.speed,
                wireframe: dto.wireframe,
                scene: None,
                mesh_geom_actors: Vec::new(),
                mesh_model_cache: std::collections::HashMap::new(),
                #[cfg(target_arch = "wasm32")]
                requested_meshes: std::collections::HashSet::new(),
            });
        }

        // Scene-wide post-process (absent in scenes saved before it existed:
        // keep the running settings so those scenes look unchanged).
        if let Some(pp) = &scene.post_process {
            self.scene_post_process = pp.to_settings();
        }
        renderer.set_post_process_settings(&self.scene_post_process);
        // Same story for the ambient AO/GI toggles: absent in older scenes,
        // keep whatever's currently running.
        if let Some(amb) = &scene.ambient {
            self.scene_ambient = amb.to_settings();
        }
        renderer.set_ambient_settings(&self.scene_ambient);
    }

    /// The world transform a splat DTO describes (scale collapsed to uniform).
    fn splat_dto_transform(dto: &SplatDto) -> ActorTransform {
        let u = dto.scale[0];
        ActorTransform::new(
            arr_vec3(dto.position),
            arr_quat(dto.rotation),
            CgVec3::new(u, u, u),
        )
    }

    /// Fetches a scene splat's .ply bytes on a background task (native thread /
    /// wasm fetch) and queues them for `drain_pending_splats` to upload.
    fn queue_splat_load(&self, dto: SplatDto) {
        let pending = self.pending_splats.clone();
        let task = async move {
            match load_binary(&dto.path).await {
                Ok(bytes) => pending.lock().unwrap().push((dto, bytes)),
                Err(e) => log!("Couldn't read splat '{}': {e}", dto.path),
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(task);
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || pollster::block_on(task));
    }

    /// Uploads any splat .ply bytes that arrived from `queue_splat_load`.  The
    /// first cloud to arrive becomes the rendered one.
    fn drain_pending_splats(&mut self, renderer: &mut Renderer) {
        loop {
            // Take one entry per iteration (not holding the lock across the
            // GPU upload).
            let entry = self.pending_splats.lock().unwrap().pop();
            let Some((dto, bytes)) = entry else {
                return;
            };
            let params = dto.params.to_params();
            match renderer.load_gaussian_splat_from_bytes(&bytes, &dto.name, &params) {
                Ok(_info) => {
                    self.scene_splats.push(SceneSplat {
                        name: dto.name.clone(),
                        path: dto.path.clone(),
                        params,
                        transform: Self::splat_dto_transform(&dto),
                    });
                    let index = self.scene_splats.len() - 1;
                    if self.undo_pending_splat_loads.remove(&dto.name) {
                        // Queued by undo/redo (see recreate_snapshot), not a
                        // user Add/Load: select/activate it so the effect is
                        // visible, but don't push another Create action --
                        // that would duplicate/loop the history.
                        self.select_after_add(Selection::Splat(index));
                        self.activate_splat(index, renderer);
                    } else {
                        if self.scene_splats.len() == 1 {
                            self.activate_splat(0, renderer);
                        }
                        if let Some((obj, snapshot)) =
                            self.snapshot_object(Selection::Splat(index))
                        {
                            self.undo_stack.push(UndoAction::Create {
                                obj,
                                snapshot,
                                index_hint: index,
                            });
                        }
                    }
                }
                Err(reason) => {
                    self.status = Some((
                        format!("Couldn't load splat {}: {reason}", dto.name),
                        STATUS_RED,
                        10.0,
                    ));
                }
            }
        }
    }

    /// Replaces the whole scene with the one parsed from `bytes`: objects and
    /// camera immediately, splat clouds via background .ply loads (they appear
    /// as their files arrive).  Returns a short summary on success.
    fn load_scene_from_bytes(
        &mut self,
        bytes: &[u8],
        _model_resources: &[(String, ModelHandle)],
        renderer: &mut Renderer,
    ) -> Result<String, String> {
        let scene: SceneFile = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;

        self.clear_scene(renderer);

        // Eager-load the models this scene references (a scene's own models are
        // needed, so this isn't wasteful).  Native loads them synchronously so
        // handles resolve before apply; web fetches in the background and the
        // meshes pop in as they arrive (pending_model_uploads/actor_models).
        for name in scene.actors.iter().filter_map(|a| a.model.clone()).collect::<Vec<_>>() {
            let Some(path) = self.catalog_path_for(&name) else {
                continue;
            };
            if renderer.get_model_resources().iter().any(|(p, _)| *p == path) {
                continue; // Already loaded.
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                pollster::block_on(renderer.load_model(&path, false));
            }
            #[cfg(target_arch = "wasm32")]
            {
                let pending = self.pending_model_uploads.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(bytes) = load_binary(&path).await {
                        pending.lock().unwrap().push((path, bytes, false));
                    }
                });
            }
        }

        // Rebuild the loaded (name -> handle) list now that native scene models
        // are loaded, so apply_scene_objects resolves them.
        let model_resources: Vec<(String, ModelHandle)> = renderer
            .get_model_resources()
            .into_iter()
            .map(|(path, handle)| (resource_display_name(&path), handle))
            .collect();
        self.apply_scene_objects(&scene, &model_resources, renderer);

        let mut skipped_splats = 0;
        for dto in &scene.splats {
            if dto.path.is_empty() {
                // Saved before splat paths were tracked: no re-readable source.
                skipped_splats += 1;
                continue;
            }
            self.queue_splat_load(dto.clone());
        }

        let mut summary = format!(
            "{} actors, {} lights, {} particles, {} splats",
            self.scene_actors.len(),
            self.scene_lights.len(),
            self.scene_particles.len(),
            scene.splats.len() - skipped_splats,
        );
        if skipped_splats > 0 {
            summary.push_str(&format!(" ({skipped_splats} splats had no path; re-load their .ply manually)"));
        }
        Ok(summary)
    }
}

/// Imports a MuJoCo model's whole folder into IndexedDB, returning the key of
/// its entry MJCF -- ready to drop straight into a `MujocoSceneActor`'s
/// `xml_path`, since `MujocoScene::load_bundle`'s walk asks IndexedDB for
/// exactly the keys written here. `Ok(None)` if the visitor picked nothing.
///
/// rfd has no folder picker on web (its wasm backend only wraps a plain
/// `<input type="file">`), so this drives the element itself to get at
/// `webkitdirectory` -- the one broadly-supported way a page can read a
/// directory without asking the visitor to zip it first.
///
/// Nothing is uploaded: the files go from the visitor's disk straight into
/// this origin's IndexedDB, and the model never leaves the device.
#[cfg(target_arch = "wasm32")]
async fn import_mujoco_model_folder() -> anyhow::Result<Option<String>> {
    use wasm_bindgen::{closure::Closure, JsCast, JsValue};

    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow::anyhow!("no document"))?;
    let input: web_sys::HtmlInputElement = document
        .create_element("input")
        .map_err(|e| anyhow::anyhow!("create <input>: {e:?}"))?
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("created element wasn't an <input>"))?;
    input.set_type("file");
    input.set_webkitdirectory(true);

    // Resolve once the visitor confirms. A dismissed dialog fires no event in
    // most browsers, so this can simply never resolve -- the task then parks
    // forever holding just the input, which is why nothing else waits on it.
    let input_for_cb = input.clone();
    let chosen = js_sys::Promise::new(&mut move |resolve, _reject| {
        let cb = Closure::once(Box::new(move || {
            let _ = resolve.call0(&JsValue::NULL);
        }) as Box<dyn FnOnce()>);
        input_for_cb.set_onchange(Some(cb.as_ref().unchecked_ref()));
        cb.forget();
    });
    input.click();
    wasm_bindgen_futures::JsFuture::from(chosen)
        .await
        .map_err(|e| anyhow::anyhow!("folder picker: {e:?}"))?;

    let Some(files) = input.files() else {
        return Ok(None);
    };

    // Every .xml's text, kept aside so the include graph can name the entry
    // point once the whole folder is in (see `mujoco::find_root_mjcf`).
    let mut mjcfs: Vec<(String, String)> = Vec::new();
    for i in 0..files.length() {
        let Some(file) = files.get(i) else { continue };
        // webkitRelativePath ("panda/assets/link0_0.obj") is what makes this
        // worth doing -- it's the model's own layout, which every <include>
        // and <mesh file="..."> in the MJCF is written against. web-sys
        // doesn't bind it (it's non-standard), hence the Reflect call.
        let relative = js_sys::Reflect::get(&file, &JsValue::from_str("webkitRelativePath"))
            .ok()
            .and_then(|v| v.as_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| file.name());
        let key = format!("mujoco_scenes/{relative}");

        let buffer = wasm_bindgen_futures::JsFuture::from(file.array_buffer())
            .await
            .map_err(|e| anyhow::anyhow!("read {relative}: {e:?}"))?;
        let bytes = js_sys::Uint8Array::new(&buffer).to_vec();

        if relative.to_ascii_lowercase().ends_with(".xml") {
            if let Ok(text) = String::from_utf8(bytes.clone()) {
                mjcfs.push((key.clone(), text));
            }
        }
        black_splat::idb::put(&key, &bytes).await;
    }

    if mjcfs.is_empty() {
        anyhow::bail!("no MJCF (.xml) file in the picked folder");
    }
    let root = black_splat::mujoco::find_root_mjcf(&mjcfs)
        .ok_or_else(|| anyhow::anyhow!("every .xml in the folder is <include>d by another"))?;
    log!("imported MuJoCo model: {} file(s), root {root}", files.length());
    Ok(Some(root))
}

impl GameEngine for SplatGame {
    fn new(game_config: &Config) -> Self {
        log!("SplatGame::new()");
        let mut game_camera = Camera::new();
        game_camera.set_position(&game_config.start_position);
        game_camera.set_rotation(&game_config.start_rotation);
        Self {
            game_objects: Vec::new(),
            game_camera,
            scene_splats: Vec::new(),
            active_splat: 0,
            scene_mujoco_actors: Vec::new(),
            pending_mujoco_loads: Arc::new(Mutex::new(Vec::new())),
            pending_mujoco_trajectories: Arc::new(Mutex::new(Vec::new())),
            #[cfg(target_arch = "wasm32")]
            pending_mujoco_meshes: Arc::new(Mutex::new(Vec::new())),
            picked_mujoco_trajectory: Arc::new(Mutex::new(None)),
            scene_actors: Vec::new(),
            scene_lights: Vec::new(),
            scene_particles: Vec::new(),
            scene_post_process: PostProcessSettings::default(),
            scene_ambient: AmbientSettings::default(),
            undo_stack: UndoStack::default(),
            pending_gizmo_edit: None,
            pending_property_edit: None,
            undo_pending_splat_loads: std::collections::HashSet::new(),
            particle_textures: Vec::new(),
            selected: None,
            multi_selected: Vec::new(),
            context_menu: None,
            rmb_drag_accum: 0.0,
            rmb_look_engaged: false,
            confirm_delete: None,
            confirm_new_scene: false,
            has_custom_startup: editor_config::load_startup_scene().is_some(),
            selected_resource: None,
            // Seed the particle library with the built-in presets; if the user
            // has a saved library on disk, initialize_world replaces this.
            particle_resources: PARTICLE_PRESETS
                .iter()
                .map(|preset| ParticleResource {
                    name: preset.to_string(),
                    params: preset_particle_params(preset),
                    dirty: false,
                    // Set once initialize_world seeds/loads the library.
                    saved_name: None,
                })
                .collect(),
            // Filled in initialize_world (needs the renderer to build handles).
            material_library: Vec::new(),
            texture_resources: Vec::new(),
            model_resources: Vec::new(),
            picked_texture: Arc::new(Mutex::new(None)),
            picked_mujoco_xml: Arc::new(Mutex::new(None)),
            name_edit: None,
            name_edit_buffer: String::new(),
            name_edit_focus: false,
            gizmo: TransformGizmo::default(),
            editor_config: EditorConfig::load(),
            show_settings: false,
            rebinding: None,
            confirm_reset: false,
            active_tab: None,
            resources_open: false,
            browser_filters: std::collections::HashSet::new(),
            browser_folder: String::new(),
            browser_filter: String::new(),
            material_rename: None,
            resources_height: 200.0,
            next_object_num: 1,
            picked_ply: Arc::new(Mutex::new(None)),
            picked_model: Arc::new(Mutex::new(None)),
            pending_model_uploads: Arc::new(Mutex::new(Vec::new())),
            pending_actor_models: Vec::new(),
            picker_state: Arc::new(Mutex::new(PickerState::Idle)),
            picked_scene: Arc::new(Mutex::new(None)),
            picked_startup: Arc::new(Mutex::new(None)),
            pending_splats: Arc::new(Mutex::new(Vec::new())),
            status: None,
            touch_pads: {
                // Desktop viewer: keep the pads hidden until a touch appears.
                let mut pads = TouchPads::default();
                pads.reveal_on_touch = true;
                pads
            },
            fly_camera: FlyCamera::default(),
            editor_mode: true,
        }
    }

    async fn initialize_world(
        &mut self,
        renderer: &mut Renderer<'_>,
        game_config: &mut Config,
    ) {
        log!("SplatGame::initialize_world()");
        game_config.clear_color = CgVec4::new(0.02, 0.02, 0.03, 1.0);
        renderer.set_tonemap_enabled(true);
        // This is the PBR/lighting showcase: world actors go through the
        // G-buffer + per-light passes (the other demos keep the forward path).
        renderer.set_deferred_world_enabled(true);
        // Persisted shadow quality (Settings tab > Shadows).
        renderer.set_shadow_settings(&shadow_settings_from_config(&self.editor_config));

        // Material library: load the user's saved materials from
        // resources/materials/, or seed the built-in defaults (and write them
        // out) the first time.  Each becomes a GPU material in the renderer
        // plus a MaterialResource here so edits can be saved back.  (Materials
        // are constant-only unless a texture is assigned: no texture means the
        // built-in white, so the metallic/roughness constants pass through.
        // Assigning one to an actor overrides its model's textures.)
        let mut material_files = resource_library::load_materials();
        if material_files.is_empty() {
            material_files = default_materials();
            for file in &material_files {
                if let Err(e) = resource_library::save_material(&file.name, &file.desc) {
                    log!("Couldn't seed material {}: {e}", file.name);
                }
            }
        }
        for file in material_files {
            let handle = renderer.load_material(&file.name, &file.desc).await;
            self.material_library.push(MaterialResource {
                // Whether seeded or loaded, it now exists on disk under this
                // name -- record it so a later rename can clean up the old file.
                saved_name: Some(file.name.clone()),
                name: file.name,
                desc: file.desc,
                handle,
                dirty: false,
            });
        }

        // Particle library: same pattern.  If the user has saved particle
        // definitions, they replace the preset seeding from `new`; otherwise
        // write the presets out as the initial library.
        let particle_files = resource_library::load_particles();
        if particle_files.is_empty() {
            for resource in &mut self.particle_resources {
                if let Err(e) = resource_library::save_particle(&resource.name, &resource.params) {
                    log!("Couldn't seed particle {}: {e}", resource.name);
                } else {
                    resource.saved_name = Some(resource.name.clone());
                }
            }
        } else {
            self.particle_resources = particle_files
                .into_iter()
                .map(|file| ParticleResource {
                    saved_name: Some(file.name.clone()),
                    name: file.name,
                    params: file.params,
                    dirty: false,
                })
                .collect();
        }

        // Preload each particle definition's texture so "Add > Particle System"
        // can spawn one synchronously from the (non-async) frame tick.  Covers
        // the whole library (presets plus any loaded from disk), so custom
        // definitions referencing other textures work too.
        let particle_texture_files: Vec<String> = self
            .particle_resources
            .iter()
            .map(|r| r.params.texture_file.clone())
            .collect();
        for texture_file in particle_texture_files {
            if self
                .particle_textures
                .iter()
                .any(|(path, _)| *path == texture_file)
            {
                continue; // Definitions may share a texture.
            }
            let handle = renderer.preload_texture(&texture_file).await;
            self.particle_textures.push((texture_file, handle));
        }

        // Texture library: every image under game_assets/textures/ (imports)
        // plus the bundled fx textures, loaded so materials can reference them.
        for path in resource_library::scan_textures().await {
            if self.texture_resources.iter().any(|t| t.path == path) {
                continue;
            }
            let handle = renderer.preload_texture(&path).await;
            self.texture_resources.push(TextureResource {
                name: resource_display_name(&path),
                path,
                handle,
            });
        }

        // Model catalog: every model under game_assets/models/, listed for the
        // browser but NOT loaded here -- geometry loads lazily on first use (a
        // browser select or a scene that references it), so a big model doesn't
        // cost anything at startup.  Scene-referenced models are loaded just
        // below, before the startup scene is applied.
        for path in resource_library::scan_models().await {
            if self.model_resources.iter().any(|m| m.path == path) {
                continue;
            }
            self.model_resources.push(ModelResource {
                name: resource_display_name(&path),
                path,
            });
        }

        // Build the splat pipeline now regardless of whether the startup scene
        // has any splats -- tick_frame_internal is sync and can only reach the
        // sync load_gaussian_splat_from_bytes (e.g. the [L] file picker), which
        // errors out if the pipeline doesn't exist yet.
        renderer.ensure_gaussian_splat_pipeline().await;

        // Open the startup scene: the user's saved one, or the built-in default
        // (the church) otherwise.  Since we're async here, splat clouds load
        // directly instead of through the pending queue.
        let scene = editor_config::load_startup_scene()
            .and_then(|json| match serde_json::from_str::<SceneFile>(&json) {
                Ok(scene) => Some(scene),
                Err(e) => {
                    log!("Bad startup scene ({e}); using the default");
                    None
                }
            })
            .unwrap_or_else(default_startup_scene);
        for dto in &scene.splats {
            if dto.path.is_empty() {
                continue; // No re-readable source (was a browser file pick).
            }
            let params = dto.params.to_params();
            if renderer.load_gaussian_splat(&dto.path, &params).await {
                self.scene_splats.push(SceneSplat {
                    name: dto.name.clone(),
                    path: dto.path.clone(),
                    params,
                    transform: Self::splat_dto_transform(dto),
                });
            }
        }
        // Eager-load the models this scene references (matched by display name
        // against the catalog) so apply_scene_objects can resolve their
        // handles.  Models the scene doesn't use stay lazy.
        for name in scene.actors.iter().filter_map(|a| a.model.as_ref()) {
            if let Some(path) = self.catalog_path_for(name) {
                renderer.load_model(&path, false).await;
            }
        }
        let model_resources: Vec<(String, ModelHandle)> = renderer
            .get_model_resources()
            .into_iter()
            .map(|(path, handle)| (resource_display_name(&path), handle))
            .collect();
        self.apply_scene_objects(&scene, &model_resources, renderer);
        if !self.scene_splats.is_empty() {
            self.activate_splat(0, renderer);
        }

        // Help is reachable from the menu bar's Help button, so drop the
        // engine's "Press [H]..." hint line.
        renderer.set_show_help_hint(false);

        renderer.set_camera(&self.game_camera);
    }

    fn get_game_objects(&self) -> &Vec<GameObject> {
        &self.game_objects
    }

    fn tick_frame_internal(
        &mut self,
        renderer: &mut Renderer,
        input_manager: &InputManager,
        game_config: &Config,
    ) {
        let delta_time = game_config.delta_time;
        // Forward/right basis for the touch pads below (keyboard movement uses
        // the fly camera directly).
        let (forward_dir, right_dir) = self.fly_camera.basis(&self.game_camera);

        // Movement + look (keyboard + right-drag mouse) come from the shared fly
        // camera.  The touch pads further add to `move_vec` / `camera_rot` before
        // they're committed below; pitch is clamped once, after every source.
        // In editor mode WASD only moves while the right mouse button is held
        // (Unreal-style flythrough): W/E/R are also the gizmo hotkeys, and the
        // held button is what disambiguates.  Game mode moves freely.
        let flythrough =
            !self.editor_mode || input_manager.get_key_state("mouse_right").is_down();
        let mut move_vec = if flythrough {
            self.fly_camera.wasd_direction(&self.game_camera, input_manager)
        } else {
            CG_VEC3_ZERO
        };
        let mut camera_rot = self.game_camera.get_rotation();
        self.fly_camera
            .apply_key_look(&mut camera_rot, input_manager, delta_time);
        self.fly_camera
            .apply_mouse_look(&mut camera_rot, input_manager, renderer);

        // --- GUI (egui, same on native + web) ---
        let ctx = renderer.egui_ctx().clone();

        // Web pixels-per-point: the canvas is sized in *physical* pixels
        // (`css_px * devicePixelRatio`, see index.html), so egui needs ppp =
        // DPR for one point to equal one CSS pixel -- which is what
        // egui_winit's window-scale-factor ppp gives the native build, so the
        // two match at any window size. Touch devices then get a flat zoom
        // bump plus enlarged interactive sizing (on the global style so
        // dropdown popups inherit it). Raise TOUCH_ZOOM for a bigger phone UI.
        #[cfg(target_arch = "wasm32")]
        {
            const TOUCH_ZOOM: f32 = 1.5;
            let touch = is_touch_device();
            let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio() as f32);
            let zoom = if touch { TOUCH_ZOOM } else { 1.0 };
            ctx.set_pixels_per_point((dpr * zoom).max(0.5));
            if touch {
                ctx.all_styles_mut(|s| {
                    s.spacing.button_padding = egui::vec2(16.0, 12.0);
                    s.spacing.interact_size.y = 38.0;
                    s.spacing.item_spacing = egui::vec2(14.0, 10.0);
                });
            }
        }

        let screen = ctx.content_rect();

        let mut do_load = false;
        // Editor actions collected from the menus / Scene tab this frame and
        // applied after the egui pass (avoids borrowing self inside closures).
        let mut do_save_scene = false;
        let mut do_load_scene = false;
        let mut do_new_scene = false;
        let mut do_set_startup = false;
        let mut do_clear_startup = false;
        let mut do_pick_startup = false;
        let mut do_add: Option<AddKind> = None;
        let mut delete_object: Option<Selection> = None;
        // A selection made this frame; the bool is "additive" (ctrl held), which
        // toggles membership in the multi-selection instead of replacing it.
        let mut select_object: Option<(Selection, bool)> = None;
        // Local rename accumulators (one per outliner section), applied after the
        // egui pass so the lists aren't mutably borrowed while iterating.
        let mut rename_actors: Vec<(usize, String)> = Vec::new();
        let mut rename_lights: Vec<(usize, String)> = Vec::new();
        let mut rename_particles: Vec<(usize, String)> = Vec::new();
        let mut rename_mujoco_actors: Vec<(usize, String)> = Vec::new();
        // True when the Details panel or gizmo changed the selected object this
        // frame; the renderer's copy (actor / particle) is refreshed afterwards.
        let mut selection_edited = false;

        // Loaded model resources for the Resources tab and the Details panel's
        // model dropdown, as (display name, handle).  Fetched up front so the
        // panel closure doesn't need to borrow the renderer.
        let model_resources: Vec<(String, ModelHandle)> = renderer
            .get_model_resources()
            .into_iter()
            .map(|(path, handle)| (resource_display_name(&path), handle))
            .collect();
        // Which catalog paths are currently loaded (path -> handle), so a tile
        // or dropdown row can resolve to a selectable handle or fall back to a
        // lazy load.
        let loaded_models: std::collections::HashMap<String, ModelHandle> =
            renderer.get_model_resources().into_iter().collect();
        // The full model catalog (display name, path, loaded handle) for the
        // content browser and the Details-panel model dropdown -- every
        // discovered model, loaded or not.  Unloaded ones load when picked.
        let model_catalog: Vec<(String, String, Option<ModelHandle>)> = self
            .model_resources
            .iter()
            .map(|m| (m.name.clone(), m.path.clone(), loaded_models.get(&m.path).copied()))
            .collect();
        // Loaded materials for the Details panel's Material dropdown, from the
        // owned library (the source of truth for names/handles) rather than the
        // renderer, so it stays in step with edits and saves.
        let material_resources: Vec<(String, MaterialHandle)> = self
            .material_library
            .iter()
            .map(|m| (m.name.clone(), m.handle))
            .collect();
        // Particle library names for the Add menus and the Resources panel.
        let particle_names: Vec<String> = self
            .particle_resources
            .iter()
            .map(|r| r.name.clone())
            .collect();
        // Texture library as (display name, relative path), for the material
        // inspector's texture pickers.  Built up front so the inspector can
        // mutably borrow material_library without also borrowing self here.
        let texture_options: Vec<(String, String)> = self
            .texture_resources
            .iter()
            .map(|t| (t.name.clone(), t.path.clone()))
            .collect();
        // Unified content-browser asset list: every asset -- model, material,
        // particle, texture, or scene splat -- as one entry tagged with the
        // virtual folder it lives in (its on-disk directory, or a synthetic
        // folder for path-less scene splats).  The browser shows them all
        // together and narrows by folder + type chips + name, like a file
        // browser.  Gathered up front so the browser closure can read it
        // without borrowing self while it mutates selection/create flags.
        let mut assets: Vec<AssetEntry> = Vec::new();
        for (name, path, loaded) in &model_catalog {
            assets.push(AssetEntry {
                kind: BrowserCategory::Models,
                name: name.clone(),
                folder: parent_dir(path),
                payload: AssetPayload::Model {
                    path: path.clone(),
                    loaded: *loaded,
                },
            });
        }
        for m in &self.material_library {
            let c = m.desc.color_constant;
            assets.push(AssetEntry {
                kind: BrowserCategory::Materials,
                name: m.name.clone(),
                folder: "resources/materials".to_string(),
                payload: AssetPayload::Material {
                    handle: m.handle,
                    dirty: m.dirty,
                    rgb: [c.x, c.y, c.z],
                },
            });
        }
        for (i, r) in self.particle_resources.iter().enumerate() {
            assets.push(AssetEntry {
                kind: BrowserCategory::Particles,
                name: r.name.clone(),
                folder: "resources/particles".to_string(),
                payload: AssetPayload::Particle {
                    index: i,
                    dirty: r.dirty,
                },
            });
        }
        for t in &self.texture_resources {
            assets.push(AssetEntry {
                kind: BrowserCategory::Textures,
                name: t.name.clone(),
                folder: parent_dir(&t.path),
                payload: AssetPayload::Texture {
                    handle: t.handle.clone(),
                },
            });
        }
        for s in &self.scene_splats {
            assets.push(AssetEntry {
                kind: BrowserCategory::Splats,
                name: s.name.clone(),
                folder: "Scene".to_string(),
                payload: AssetPayload::Splat,
            });
        }
        // The model offered to the Details panel's "use selected" button (only
        // model resources apply there).
        let selected_model = match self.selected_resource {
            Some(ResourceSelection::Model(handle)) => Some(handle),
            _ => None,
        };
        // Resources-panel actions, applied after the egui pass.
        let mut create_material = false;
        let mut create_particle = false;
        let mut import_texture = false;
        let mut import_model = false;
        // Set when the Details panel's Mujoco actor "Browse…" button is
        // clicked; applied after the egui pass (self is still borrowed by
        // the actor being inspected at that point).
        let mut mujoco_xml_browse_request: Option<String> = None;
        let mut mujoco_traj_browse_request: Option<String> = None;
        // Set to a catalog path when the user clicks a not-yet-loaded model tile
        // in the browser; applied after the egui pass (loads it, then selects).
        let mut model_load_request: Option<String> = None;
        // Set to a catalog path when the user picks a not-yet-loaded model from
        // the Details-panel dropdown; applied after the egui pass (loads it, then
        // assigns it to the selected actor).
        let mut model_pick_request: Option<String> = None;
        // A save requested from a browser tile's disk button (applied after the
        // pass, where self can be borrowed mutably to read the desc/params).
        let mut save_material_handle: Option<MaterialHandle> = None;
        let mut save_particle_index: Option<usize> = None;
        // A rename requested from a material tile's right-click menu (applied
        // after the pass, where material_library can be mutably borrowed).
        let mut rename_material: Option<(MaterialHandle, String)> = None;
        // Particle-resource edits from the Resource inspector this frame:
        // which entry changed (push params to its live emitters), and a
        // rename (propagate to scene emitters referencing the old name).
        let mut edited_particle: Option<usize> = None;
        let mut renamed_particle: Option<(String, String)> = None;

        // Editor vs game mode.  In editor mode the full bar is always shown --
        // the mode switch, File (open), Debug (scene cycle + future toggles),
        // Splat (parameter sliders), Help, and a right-aligned scene status.  In
        // game mode only the small mode switch remains, so the view is
        // unobstructed; that switch is the way back too (no keyboard needed, so
        // it works on touch).
        let editor = self.editor_mode;
        let menu_bar = egui::Area::new(egui::Id::new("menu_bar"))
            .fixed_pos(screen.left_top())
            .constrain(true)
            .show(&ctx, |ui| {
                egui::Frame::side_top_panel(ui.style()).show(ui, |ui| {
                    if !editor {
                        // Game mode: just the mode switch, kept small.  (MenuBar
                        // always claims full width, so it's only used in editor.)
                        ui.horizontal(|ui| draw_mode_switch(ui, &mut self.editor_mode));
                        return;
                    }
                    ui.set_width(screen.width());
                    egui::MenuBar::new().ui(ui, |ui| {
                        draw_mode_switch(ui, &mut self.editor_mode);
                        ui.separator();
                        ui.menu_button("File", |ui| {
                            // Buttons auto-close the menu on click.
                            if ui.button("New Scene").clicked() {
                                // Asks first (the modal below): it wipes
                                // everything, including unsaved work.
                                self.confirm_new_scene = true;
                            }
                            ui.separator();
                            do_save_scene |= ui.button("Save Scene…").clicked();
                            do_load_scene |= ui.button("Load Scene…").clicked();
                            ui.separator();
                            do_load |= ui.button("Load .ply…").clicked();
                        });
                        ui.menu_button("Edit", |ui| {
                            let can_undo = !self.undo_stack.undo.is_empty();
                            let can_redo = !self.undo_stack.redo.is_empty();
                            if ui
                                .add_enabled(
                                    can_undo,
                                    egui::Button::new("Undo").shortcut_text("Ctrl+Z"),
                                )
                                .clicked()
                            {
                                self.undo(renderer);
                            }
                            if ui
                                .add_enabled(
                                    can_redo,
                                    egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"),
                                )
                                .clicked()
                            {
                                self.redo(renderer);
                            }
                        });
                        ui.menu_button("Add", |ui| {
                            add_menu_ui(ui, &mut do_add, &particle_names);
                        });
                        // Splat params live in the Details panel and camera /
                        // keybindings in the right-hand Settings tab, so the old
                        // Debug / Splat / Camera / Settings menus are gone.
                        // Top-level toggle (not a dropdown) for the help text.
                        if ui.button("Help").clicked() {
                            renderer.enable_help_text();
                        }
                        // The loaded-splat name and splat count that used to sit
                        // here are shown in the Resources panel's Splats column /
                        // the Resource inspector instead.
                    });
                });
            });

        // Game mode hides the engine's debug/camera overlay too, for a clean
        // view.  In editor mode it's shown, pushed below the bar so the bar
        // doesn't cover it (bar rect is in points; the text is placed in physical
        // pixels, hence the pixels_per_point scale).
        renderer.set_allow_debug_text(editor);
        let bar_bottom_px = menu_bar.response.rect.bottom() * ctx.pixels_per_point();
        renderer.set_debug_text_top_offset(bar_bottom_px + 6.0);

        // Right-hand editor panel, tabbed: Scene (splats + actors/lights/
        // particles), Details (selected object's properties), Resources (loaded
        // assets).  Editor mode only; game mode keeps the view unobstructed.
        // Drawn as an Area (same as the menu bar) so it sits over the 3D view.
        // Collapsed to just the tab strip until a tab is clicked; clicking the
        // active tab collapses it again.  Open, it runs the full height of the
        // view like a docked sidebar.  Selection/deletion set the local action
        // flags above and are applied after the egui pass; Details edits mutate
        // the object directly and set `selection_edited` so the renderer's copy
        // is refreshed afterwards.
        // Gizmo mode toolbar, centered under the menu bar (both corners are
        // taken by the debug overlays).
        if editor {
            let top = menu_bar.response.rect.bottom();
            egui::Area::new(egui::Id::new("gizmo_toolbar"))
                .pivot(egui::Align2::CENTER_TOP)
                .fixed_pos(egui::pos2(screen.center().x, top))
                .constrain(true)
                .show(&ctx, |ui| {
                    egui::Frame::side_top_panel(ui.style()).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for (i, (mode, label)) in GIZMO_ACTIONS.iter().enumerate() {
                                let resp =
                                    ui.selectable_label(self.gizmo.mode == *mode, *label);
                                if resp.clicked() {
                                    self.gizmo.mode = *mode;
                                }
                                resp.on_hover_text(format!(
                                    "Hotkey: {}",
                                    self.editor_config.gizmo_keys[i].name()
                                ));
                            }
                            ui.separator();
                            // World/Local: fixed world axes, or the selected
                            // object's own rotated axes (averaged across a
                            // multi-selection).
                            for (space, label, hover) in [
                                (
                                    GizmoSpace::World,
                                    "World",
                                    "Translate/rotate/scale along fixed world axes",
                                ),
                                (
                                    GizmoSpace::Local,
                                    "Local",
                                    "Translate/rotate/scale along the object's own axes \
                                     (averaged across a multi-selection)",
                                ),
                            ] {
                                let resp = ui
                                    .selectable_label(self.gizmo.space == space, label)
                                    .on_hover_text(hover);
                                if resp.clicked() {
                                    self.gizmo.space = space;
                                }
                            }
                            // One snap box per mode, always visible (not just
                            // the active one) so all three can be dialed in
                            // without switching modes back and forth. Each is
                            // its own field on the gizmo: 0 = free
                            // (continuous) dragging, otherwise a drag in that
                            // mode commits in whole increments of the given
                            // size.
                            ui.separator();
                            for (label, snap_value, speed, range, suffix, hover) in [
                                (
                                    "T",
                                    &mut self.gizmo.translate_snap_units,
                                    0.05,
                                    0.0..=1000.0,
                                    "",
                                    "Translation snap increment, in world units (0 = free movement)",
                                ),
                                (
                                    "R",
                                    &mut self.gizmo.rotate_snap_degrees,
                                    1.0,
                                    0.0..=180.0,
                                    "°",
                                    "Rotation snap increment (0 = free rotation)",
                                ),
                                (
                                    "S",
                                    &mut self.gizmo.scale_snap_units,
                                    0.05,
                                    0.0..=100.0,
                                    "",
                                    "Scale snap increment (0 = free scaling)",
                                ),
                            ] {
                                ui.label(format!("{label} snap"));
                                ui.add(
                                    egui::DragValue::new(snap_value)
                                        .speed(speed)
                                        .range(range)
                                        .suffix(suffix),
                                )
                                .on_hover_text(hover);
                            }
                        });
                    });
                });
        }

        // Resources: a full-width bottom panel (content-browser style).
        // Closed, it collapses to a small "Resources" tab at the bottom-left
        // corner; open, the grab strip along its top edge drags to resize.
        // The right-hand editor panel stops above it (panel_bottom).
        const RESOURCES_COLLAPSE_HEIGHT: f32 = 60.0;
        let mut panel_bottom = screen.bottom();
        if editor {
            if self.resources_open {
                let max_height = (screen.height() * 0.7).max(120.0);
                self.resources_height = self.resources_height.clamp(120.0, max_height);
                panel_bottom = screen.bottom() - self.resources_height;
                // Pivoted on its bottom-left corner (not positioned by top_y)
                // so its bottom edge always sits exactly on the screen edge,
                // even if the measured content height ever drifts from
                // resources_height -- otherwise a gap opens below the panel
                // and the 3D view shows through it.
                egui::Area::new(egui::Id::new("resources_panel"))
                    .pivot(egui::Align2::LEFT_BOTTOM)
                    .fixed_pos(screen.left_bottom())
                    .constrain_to(screen)
                    .show(&ctx, |ui| {
                        let frame = egui::Frame::side_top_panel(ui.style());
                        let margin = frame.total_margin();
                        frame.show(ui, |ui| {
                            ui.set_width(screen.width() - margin.sum().x);
                            ui.set_height(self.resources_height - margin.sum().y);
                            // Grab strip: drag to resize (position catches up
                            // next frame), painted as a short handle line.
                            let (strip_rect, strip) = ui.allocate_exact_size(
                                egui::vec2(ui.available_width(), 8.0),
                                egui::Sense::drag(),
                            );
                            let strip = strip.on_hover_cursor(egui::CursorIcon::ResizeVertical);
                            if strip.dragged() {
                                let new_height =
                                    self.resources_height - strip.drag_delta().y;
                                // Dragging it down past the collapse threshold
                                // closes the panel instead of clamping to the
                                // minimum height, so dragging to the bottom of
                                // the screen hides it like the tab's own toggle.
                                if new_height < RESOURCES_COLLAPSE_HEIGHT {
                                    self.resources_open = false;
                                } else {
                                    self.resources_height = new_height.clamp(120.0, max_height);
                                }
                            }
                            ui.painter().line_segment(
                                [
                                    egui::pos2(strip_rect.center().x - 24.0, strip_rect.center().y),
                                    egui::pos2(strip_rect.center().x + 24.0, strip_rect.center().y),
                                ],
                                egui::Stroke::new(2.0, ui.visuals().weak_text_color()),
                            );
                            // Content browser header: collapse toggle, an "Add"
                            // menu (all kinds live together now, so create/import
                            // can't hang off a per-type tab), type-filter chips,
                            // and a name search box.
                            let mut pick: Option<Option<ResourceSelection>> = None;
                            let mut open_inspector = false;
                            ui.horizontal(|ui| {
                                if ui
                                    .selectable_label(
                                        true,
                                        egui::RichText::new("Resources").strong(),
                                    )
                                    .clicked()
                                {
                                    self.resources_open = false;
                                }
                                ui.separator();
                                // Create/import, gathered under one menu (buttons
                                // auto-close it on click).
                                ui.menu_button("➕ Add", |ui| {
                                    if ui.button("Import Model…").clicked() {
                                        import_model = true;
                                    }
                                    if ui.button("New Material").clicked() {
                                        create_material = true;
                                    }
                                    if ui.button("New Particle").clicked() {
                                        create_particle = true;
                                    }
                                    if ui.button("Import Texture…").clicked() {
                                        import_texture = true;
                                    }
                                });
                                ui.separator();
                                // Type-filter chips: each toggles its kind in the
                                // filter set.  An empty set shows every kind.
                                for (cat, label) in BROWSER_CATEGORIES {
                                    let on = self.browser_filters.contains(cat);
                                    if ui.selectable_label(on, *label).clicked() {
                                        if on {
                                            self.browser_filters.remove(cat);
                                        } else {
                                            self.browser_filters.insert(*cat);
                                        }
                                    }
                                }
                                ui.separator();
                                ui.label("🔍");
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.browser_filter)
                                        .hint_text("Search")
                                        .desired_width(120.0),
                                );
                                if !self.browser_filter.is_empty()
                                    && ui.small_button("×").clicked()
                                {
                                    self.browser_filter.clear();
                                }
                            });
                            ui.separator();

                            let filter = self.browser_filter.to_lowercase();
                            let current = self.selected_resource;
                            // click = select/deselect; double-click also opens
                            // the item in the right-hand Resource inspector.
                            let mut on_tile = |resp: &egui::Response,
                                               sel: ResourceSelection| {
                                if resp.double_clicked() {
                                    pick = Some(Some(sel));
                                    open_inspector = true;
                                } else if resp.clicked() {
                                    pick = Some(if current == Some(sel) {
                                        None
                                    } else {
                                        Some(sel)
                                    });
                                }
                            };
                            // Folder tree built from the folders the current
                            // assets occupy (ancestors included, since insert
                            // splits the path into every level).
                            let mut tree = FolderNode::default();
                            for a in &assets {
                                tree.insert(&a.folder);
                            }
                            // Scope + filters read into locals so the grid closure
                            // doesn't borrow self while the tree mutates
                            // self.browser_folder (the new selection lands next
                            // frame -- fine, like the resize strip).
                            let folder = self.browser_folder.clone();
                            let filters = self.browser_filters.clone();
                            let body_height = ui.available_height();
                            ui.horizontal_top(|ui| {
                                // Left: the folder tree.  "All" (root) clears the
                                // scope so every asset shows together.
                                ui.vertical(|ui| {
                                    ui.set_width(FOLDER_TREE_W);
                                    egui::ScrollArea::vertical()
                                        .id_salt("cb_tree")
                                        .max_height(body_height)
                                        .show(ui, |ui| {
                                            if ui
                                                .selectable_label(
                                                    self.browser_folder.is_empty(),
                                                    "📂 All",
                                                )
                                                .clicked()
                                            {
                                                self.browser_folder.clear();
                                            }
                                            draw_folder_tree(
                                                ui,
                                                &tree,
                                                "",
                                                &mut self.browser_folder,
                                            );
                                        });
                                });
                                ui.separator();
                                // Right: the asset grid -- every asset under the
                                // selected folder that passes the type + name
                                // filters, all kinds intermixed.
                                egui::ScrollArea::vertical()
                                    .id_salt("cb_grid")
                                    .max_height(body_height)
                                    .auto_shrink(false)
                                    .show(ui, |ui| {
                                        let mut shown = 0usize;
                                        ui.horizontal_wrapped(|ui| {
                                            for a in &assets {
                                                if !folder_contains(&folder, &a.folder) {
                                                    continue;
                                                }
                                                if !filters.is_empty()
                                                    && !filters.contains(&a.kind)
                                                {
                                                    continue;
                                                }
                                                if !name_matches(&a.name, &filter) {
                                                    continue;
                                                }
                                                shown += 1;
                                                match &a.payload {
                                                    AssetPayload::Model { path, loaded } => {
                                                        match *loaded {
                                                            // Loaded: a normal,
                                                            // selectable tile.
                                                            Some(handle) => {
                                                                let sel =
                                                                    ResourceSelection::Model(
                                                                        handle,
                                                                    );
                                                                let resp = browser_tile(
                                                                    ui,
                                                                    &a.name,
                                                                    &Thumb::Glyph("M"),
                                                                    current == Some(sel),
                                                                    false,
                                                                );
                                                                on_tile(&resp, sel);
                                                            }
                                                            // Not loaded: click
                                                            // starts a lazy load;
                                                            // it turns selectable
                                                            // once geometry is
                                                            // ready.
                                                            None => {
                                                                let resp = browser_tile(
                                                                    ui,
                                                                    &a.name,
                                                                    &Thumb::Glyph("M"),
                                                                    false,
                                                                    false,
                                                                );
                                                                if resp.clicked() {
                                                                    model_load_request =
                                                                        Some(path.clone());
                                                                }
                                                            }
                                                        }
                                                    }
                                                    AssetPayload::Material {
                                                        handle,
                                                        dirty,
                                                        rgb,
                                                    } => {
                                                        let sel =
                                                            ResourceSelection::Material(*handle);
                                                        let resp = browser_tile(
                                                            ui,
                                                            &a.name,
                                                            &Thumb::Color(*rgb),
                                                            current == Some(sel),
                                                            *dirty,
                                                        );
                                                        if *dirty
                                                            && tile_save_button(ui, &resp)
                                                                .clicked()
                                                        {
                                                            save_material_handle =
                                                                Some(*handle);
                                                        }
                                                        // Right-click to rename in
                                                        // place.
                                                        let menu = resp.context_menu(|ui| {
                                                            let active = matches!(
                                                                &self.material_rename,
                                                                Some((h, _)) if h == handle
                                                            );
                                                            if !active {
                                                                self.material_rename =
                                                                    Some((
                                                                        *handle,
                                                                        a.name.clone(),
                                                                    ));
                                                            }
                                                            ui.label("Rename material");
                                                            if let Some((_, buf)) =
                                                                &mut self.material_rename
                                                            {
                                                                let edit =
                                                                    ui.text_edit_singleline(
                                                                        buf,
                                                                    );
                                                                if !active {
                                                                    edit.request_focus();
                                                                }
                                                                if ui.input(|i| {
                                                                    i.key_pressed(
                                                                        egui::Key::Enter,
                                                                    )
                                                                }) {
                                                                    let new_name = buf
                                                                        .trim()
                                                                        .to_string();
                                                                    if !new_name.is_empty() {
                                                                        rename_material =
                                                                            Some((
                                                                                *handle,
                                                                                new_name,
                                                                            ));
                                                                    }
                                                                    self.material_rename =
                                                                        None;
                                                                    ui.close();
                                                                }
                                                            }
                                                        });
                                                        // Menu closed: drop any
                                                        // stale in-progress rename
                                                        // so a re-open re-seeds and
                                                        // refocuses.
                                                        if menu.is_none()
                                                            && matches!(
                                                                &self.material_rename,
                                                                Some((h, _)) if h == handle
                                                            )
                                                        {
                                                            self.material_rename = None;
                                                        }
                                                        on_tile(&resp, sel);
                                                    }
                                                    AssetPayload::Particle {
                                                        index,
                                                        dirty,
                                                    } => {
                                                        let sel =
                                                            ResourceSelection::Particle(*index);
                                                        let resp = browser_tile(
                                                            ui,
                                                            &a.name,
                                                            &Thumb::Glyph("P"),
                                                            current == Some(sel),
                                                            *dirty,
                                                        );
                                                        if *dirty
                                                            && tile_save_button(ui, &resp)
                                                                .clicked()
                                                        {
                                                            save_particle_index = Some(*index);
                                                        }
                                                        on_tile(&resp, sel);
                                                    }
                                                    AssetPayload::Texture { handle } => {
                                                        let thumb = renderer
                                                            .egui_texture_id(handle)
                                                            .map(Thumb::Image)
                                                            .unwrap_or(Thumb::Glyph("T"));
                                                        browser_tile(
                                                            ui, &a.name, &thumb, false, false,
                                                        );
                                                    }
                                                    AssetPayload::Splat => {
                                                        browser_tile(
                                                            ui,
                                                            &a.name,
                                                            &Thumb::Glyph("S"),
                                                            false,
                                                            false,
                                                        );
                                                    }
                                                }
                                            }
                                        });
                                        if shown == 0 {
                                            ui.label(
                                                egui::RichText::new("No assets here").weak(),
                                            );
                                        }
                                    });
                            });
                            if let Some(new_selection) = pick {
                                self.selected_resource = new_selection;
                                if open_inspector {
                                    self.active_tab = Some(EditorTab::Resource);
                                }
                            }
                        });
                    });
            } else {
                egui::Area::new(egui::Id::new("resources_tab"))
                    .pivot(egui::Align2::LEFT_BOTTOM)
                    .fixed_pos(screen.left_bottom())
                    .constrain(true)
                    .show(&ctx, |ui| {
                        egui::Frame::side_top_panel(ui.style()).show(ui, |ui| {
                            if ui.selectable_label(false, "Resources").clicked() {
                                self.resources_open = true;
                            }
                        });
                    });
            }
        }

        const PANEL_WIDTH: f32 = 260.0;
        if editor {
            let top = menu_bar.response.rect.bottom();
            egui::Area::new(egui::Id::new("editor_panel"))
                .fixed_pos(egui::pos2(screen.right() - PANEL_WIDTH, top))
                // Constrain to the region below the menu bar: if the panel
                // ever ends up too tall, egui slides a constrained Area back
                // inside its rect -- against the whole screen that shoved the
                // panel (tab strip included) up over the menu bar.
                .constrain_to(egui::Rect::from_min_max(
                    egui::pos2(screen.left(), top),
                    egui::pos2(screen.right(), panel_bottom),
                ))
                .show(&ctx, |ui| {
                    let frame = egui::Frame::side_top_panel(ui.style());
                    let frame_bottom = frame.total_margin().bottom;
                    frame.show(ui, |ui| {
                        ui.set_width(PANEL_WIDTH);
                        ui.horizontal(|ui| {
                            for (tab, label) in [
                                (EditorTab::Scene, "Scene"),
                                (EditorTab::Details, "Details"),
                                (EditorTab::Resource, "Resource"),
                                (EditorTab::Settings, "Settings"),
                            ] {
                                let is_active = self.active_tab == Some(tab);
                                if ui.selectable_label(is_active, label).clicked() {
                                    self.active_tab = if is_active { None } else { Some(tab) };
                                }
                            }
                        });
                        let Some(active_tab) = self.active_tab else {
                            return;
                        };
                        // Stretch down to the resources panel (or the screen
                        // bottom).  set_min_height reserves space from the
                        // cursor (i.e. below the tab strip), so measure the
                        // remaining space from there -- measuring from the
                        // panel top makes the Area overshoot and get shoved
                        // upward.
                        ui.set_min_height(
                            (panel_bottom - ui.cursor().top() - frame_bottom).max(80.0),
                        );
                        ui.separator();
                        let scroll_height =
                            (panel_bottom - ui.cursor().top() - frame_bottom).max(60.0);
                        egui::ScrollArea::vertical()
                            .max_height(scroll_height)
                            .show(ui, |ui| {
                                ui.set_width(PANEL_WIDTH);
                                match active_tab {
                                    EditorTab::Scene => {
                                        // Always-present scene-wide post-process
                                        // object (tonemap curve + exposure).
                                        {
                                            let is_selected =
                                                self.selected == Some(Selection::PostProcess);
                                            if ui
                                                .selectable_label(is_selected, "⚙ Post Process")
                                                .on_hover_text("Scene tonemap + exposure")
                                                .clicked()
                                            {
                                                select_object =
                                                    Some((Selection::PostProcess, false));
                                            }
                                        }
                                        ui.separator();
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("Splats").strong());
                                            if ui
                                                .small_button("+")
                                                .on_hover_text("Load a .ply splat")
                                                .clicked()
                                            {
                                                do_load = true;
                                            }
                                        });
                                        if self.scene_splats.is_empty() {
                                            ui.label("(none)");
                                        }
                                        // Clicking a splat selects it (its params
                                        // appear in Details) and makes it the
                                        // rendered cloud.  The "●" marks whichever
                                        // is currently rendered (only one shows).
                                        for (i, splat) in self.scene_splats.iter().enumerate() {
                                            let is_selected =
                                                self.selected == Some(Selection::Splat(i));
                                            let marker =
                                                if i == self.active_splat { "● " } else { "    " };
                                            let label = format!("{marker}{}", splat.name);
                                            if ui.selectable_label(is_selected, label).clicked() {
                                                select_object =
                                                    Some((Selection::Splat(i), false));
                                            }
                                        }

                                        // One outliner section per object kind,
                                        // each with its own "+" add control.
                                        // Names are collected first so the lists
                                        // aren't borrowed while the section
                                        // mutates self.name_edit etc.
                                        let actor_names: Vec<String> = self
                                            .scene_actors
                                            .iter()
                                            .map(|a| a.get_name().to_string())
                                            .collect();
                                        draw_outliner_section(
                                            ui,
                                            "Actors",
                                            Selection::Actor,
                                            &actor_names,
                                            self.selected,
                                            &self.multi_selected,
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_actors,
                                            &mut self.confirm_delete,
                                            |ui| {
                                                if ui
                                                    .small_button("+")
                                                    .on_hover_text("Add an actor")
                                                    .clicked()
                                                {
                                                    do_add = Some(AddKind::Actor);
                                                }
                                            },
                                        );

                                        let light_names: Vec<String> = self
                                            .scene_lights
                                            .iter()
                                            .map(|l| l.get_name().to_string())
                                            .collect();
                                        draw_outliner_section(
                                            ui,
                                            "Lights",
                                            Selection::Light,
                                            &light_names,
                                            self.selected,
                                            &self.multi_selected,
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_lights,
                                            &mut self.confirm_delete,
                                            |ui| {
                                                ui.menu_button("+", |ui| {
                                                    for (label, ty) in [
                                                        ("Directional", LightType::Directional),
                                                        ("Point", LightType::Point),
                                                        ("Spot", LightType::Spot),
                                                        ("Skylight", LightType::Skylight),
                                                    ] {
                                                        if ui.button(label).clicked() {
                                                            do_add = Some(AddKind::Light(ty));
                                                        }
                                                    }
                                                })
                                                .response
                                                .on_hover_text("Add a light");
                                            },
                                        );

                                        let particle_instance_names: Vec<String> = self
                                            .scene_particles
                                            .iter()
                                            .map(|p| p.name.clone())
                                            .collect();
                                        draw_outliner_section(
                                            ui,
                                            "Particles",
                                            Selection::Particle,
                                            &particle_instance_names,
                                            self.selected,
                                            &self.multi_selected,
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_particles,
                                            &mut self.confirm_delete,
                                            |ui| {
                                                ui.menu_button("+", |ui| {
                                                    for (i, preset) in
                                                        particle_names.iter().enumerate()
                                                    {
                                                        if ui.button(preset.as_str()).clicked() {
                                                            do_add = Some(AddKind::Particle(i));
                                                        }
                                                    }
                                                })
                                                .response
                                                .on_hover_text("Add a particle system");
                                            },
                                        );

                                        let mujoco_actor_names: Vec<String> = self
                                            .scene_mujoco_actors
                                            .iter()
                                            .map(|m| m.name.clone())
                                            .collect();
                                        draw_outliner_section(
                                            ui,
                                            "Mujoco Actors",
                                            Selection::MujocoActor,
                                            &mujoco_actor_names,
                                            self.selected,
                                            &self.multi_selected,
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_mujoco_actors,
                                            &mut self.confirm_delete,
                                            |ui| {
                                                if ui
                                                    .small_button("+")
                                                    .on_hover_text("Add a Mujoco actor")
                                                    .clicked()
                                                {
                                                    do_add = Some(AddKind::MujocoActor);
                                                }
                                            },
                                        );
                                    }
                                    EditorTab::Details => {
                                        // Speculative "before" snapshot: taken
                                        // only when no gesture is already in
                                        // progress, so it captures the PRE-edit
                                        // state.  Promoted to
                                        // pending_property_edit below only if
                                        // this turns out to be the first
                                        // changed frame of a new gesture.
                                        let speculative_snapshot =
                                            if self.pending_property_edit.is_none() {
                                                self.selected.and_then(|sel| {
                                                    self.snapshot_object(sel)
                                                        .map(|(obj, snap)| (sel, obj, snap))
                                                })
                                            } else {
                                                None
                                            };
                                        let mut details_edited = false;
                                        match self.selected {
                                        Some(Selection::Actor(i)) => {
                                            if let Some(actor) = self.scene_actors.get_mut(i) {
                                                let changed = editor::draw_properties(
                                                    ui,
                                                    actor,
                                                    &model_catalog,
                                                    &material_resources,
                                                    selected_model,
                                                    &mut model_pick_request,
                                                );
                                                selection_edited |= changed;
                                                details_edited |= changed;
                                            }
                                        }
                                        Some(Selection::Light(i)) => {
                                            if let Some(light) = self.scene_lights.get_mut(i) {
                                                let changed = editor::draw_properties(
                                                    ui,
                                                    light,
                                                    &model_catalog,
                                                    &material_resources,
                                                    selected_model,
                                                    &mut model_pick_request,
                                                );
                                                selection_edited |= changed;
                                                details_edited |= changed;
                                            }
                                        }
                                        Some(Selection::Particle(i)) => {
                                            if let Some(particle) =
                                                self.scene_particles.get_mut(i)
                                            {
                                                let changed = editor::draw_properties(
                                                    ui,
                                                    particle,
                                                    &model_catalog,
                                                    &material_resources,
                                                    selected_model,
                                                    &mut model_pick_request,
                                                );
                                                selection_edited |= changed;
                                                details_edited |= changed;
                                            }
                                        }
                                        Some(Selection::Splat(i)) => {
                                            if let Some(splat) = self.scene_splats.get_mut(i) {
                                                ui.label(
                                                    egui::RichText::new(splat.name.as_str())
                                                        .strong(),
                                                );
                                                ui.separator();
                                                let changed = editor::draw_properties(
                                                    ui,
                                                    splat,
                                                    &model_catalog,
                                                    &material_resources,
                                                    selected_model,
                                                    &mut model_pick_request,
                                                );
                                                selection_edited |= changed;
                                                details_edited |= changed;
                                            }
                                        }
                                        Some(Selection::MujocoActor(i)) => {
                                            if let Some(m) = self.scene_mujoco_actors.get_mut(i) {
                                                let changed = editor::draw_properties(
                                                    ui,
                                                    m,
                                                    &model_catalog,
                                                    &material_resources,
                                                    selected_model,
                                                    &mut model_pick_request,
                                                );
                                                selection_edited |= changed;
                                                details_edited |= changed;
                                                ui.label("XML Path");
                                                ui.horizontal(|ui| {
                                                    // Display-only: xml_path is set exclusively via
                                                    // Browse…, never typed, since on web it's an
                                                    // IndexedDB cache key rather than a real path.
                                                    let mut display = m.xml_path.clone();
                                                    ui.add_enabled(
                                                        false,
                                                        egui::TextEdit::singleline(&mut display),
                                                    );
                                                    if ui.button("Browse…").clicked() {
                                                        mujoco_xml_browse_request = Some(m.name.clone());
                                                    }
                                                });
                                                ui.label("Trajectory");
                                                ui.horizontal(|ui| {
                                                    // Display-only, same as XML Path above.
                                                    let mut display = m.trajectory_path.clone();
                                                    ui.add_enabled(
                                                        false,
                                                        egui::TextEdit::singleline(&mut display),
                                                    );
                                                    if ui.button("Browse…").clicked() {
                                                        mujoco_traj_browse_request = Some(m.name.clone());
                                                    }
                                                    if ui
                                                        .add_enabled(
                                                            !m.trajectory_path.is_empty(),
                                                            egui::Button::new("Clear"),
                                                        )
                                                        .clicked()
                                                    {
                                                        m.trajectory_path = String::new();
                                                        selection_edited = true;
                                                        details_edited = true;
                                                    }
                                                });
                                                if let Some(clip) = &m.clip {
                                                    let total =
                                                        clip.frame_count() as f32 * clip.dt;
                                                    ui.label(format!(
                                                        "{} joints, {} frames ({:.1}s) — replaying \
                                                         recorded qpos, physics off",
                                                        clip.joints.len(),
                                                        clip.frame_count(),
                                                        total,
                                                    ));
                                                    let mut t = m.clip_time;
                                                    if ui
                                                        .add(
                                                            egui::Slider::new(&mut t, 0.0..=total)
                                                                .text("Time"),
                                                        )
                                                        .changed()
                                                    {
                                                        // Scrubbing is transient sim state, not a
                                                        // property edit -- deliberately not
                                                        // flagged for undo.
                                                        m.clip_time = t;
                                                    }
                                                } else if !m.trajectory_path.is_empty() {
                                                    ui.label("Loading trajectory…");
                                                }
                                                ui.separator();
                                                ui.horizontal(|ui| {
                                                    if ui.button("Step").clicked() {
                                                        if let Some(scene) = &mut m.scene {
                                                            scene.step_once();
                                                        }
                                                    }
                                                    if ui.button("Reset").clicked() {
                                                        m.clip_time = 0.0;
                                                        if let Some(scene) = &mut m.scene {
                                                            scene.reset();
                                                        }
                                                    }
                                                });
                                                if m.scene.is_none() && !m.xml_path.is_empty() {
                                                    ui.label("Loading…");
                                                } else if m.xml_path.is_empty() {
                                                    ui.label("Set an XML Path to load a scene.");
                                                }
                                            }
                                        }
                                        Some(Selection::PostProcess) => {
                                            ui.label(
                                                egui::RichText::new("Post Process").strong(),
                                            );
                                            ui.separator();
                                            let changed = editor::draw_properties(
                                                ui,
                                                &mut self.scene_post_process,
                                                &model_catalog,
                                                &material_resources,
                                                selected_model,
                                                &mut model_pick_request,
                                            );
                                            if changed {
                                                // Keep the curve invertible so the
                                                // splat inverse stays well-defined
                                                // (bad values when midtone is high).
                                                self.scene_post_process.enforce_invertible();
                                            }
                                            selection_edited |= changed;
                                            details_edited |= changed;

                                            ui.separator();
                                            ui.label(
                                                egui::RichText::new("Screen-Space Ambient")
                                                    .strong(),
                                            );
                                            let ambient_changed = editor::draw_properties(
                                                ui,
                                                &mut self.scene_ambient,
                                                &model_catalog,
                                                &material_resources,
                                                selected_model,
                                                &mut model_pick_request,
                                            );
                                            if ambient_changed {
                                                renderer.set_ambient_settings(&self.scene_ambient);
                                            }
                                        }
                                        None => {
                                            ui.label("Nothing selected.");
                                            ui.label("Pick something in the Scene tab.");
                                        }
                                        }
                                        // Changed-edge commit: the first changed
                                        // frame promotes the speculative snapshot
                                        // to a pending gesture; a not-changed
                                        // frame while one is pending means the
                                        // gesture just ended, so commit it.
                                        if details_edited {
                                            if self.pending_property_edit.is_none() {
                                                self.pending_property_edit = speculative_snapshot;
                                            }
                                        } else if let Some((sel, obj, before)) =
                                            self.pending_property_edit.take()
                                        {
                                            if let Some((_, after)) = self.snapshot_object(sel) {
                                                self.undo_stack.push(UndoAction::Edit {
                                                    obj,
                                                    before,
                                                    after,
                                                });
                                            }
                                        }
                                    }
                                    // Resource inspector: edits the resource
                                    // highlighted in the bottom Resources
                                    // panel.  Material constants apply live
                                    // (the G-buffer reads them every frame);
                                    // particle edits are pushed to live
                                    // emitters after the pass.
                                    EditorTab::Resource => match self.selected_resource {
                                        Some(ResourceSelection::Model(handle)) => {
                                            let name = model_resources
                                                .iter()
                                                .find(|(_, h)| *h == handle)
                                                .map_or("(unknown model)", |(n, _)| n.as_str());
                                            ui.label(egui::RichText::new(name).strong());
                                            ui.separator();
                                            ui.label("Models aren't editable yet.");
                                        }
                                        Some(ResourceSelection::Material(handle)) => {
                                            if let Some(mat) = self
                                                .material_library
                                                .iter_mut()
                                                .find(|m| m.handle == handle)
                                            {
                                                // Title row: name, dirty marker,
                                                // and a save-to-disk button.
                                                if resource_header(ui, &mat.name, mat.dirty) {
                                                    match save_material_file(mat) {
                                                        Ok(()) => mat.dirty = false,
                                                        Err(e) => log!("Save failed: {e}"),
                                                    }
                                                }
                                                ui.separator();
                                                // Materials are referenced by
                                                // handle, so a rename only touches
                                                // the display name / save file --
                                                // no scene references to fix up.
                                                ui.label("Name");
                                                if ui
                                                    .text_edit_singleline(&mut mat.name)
                                                    .changed()
                                                {
                                                    mat.dirty = true;
                                                }
                                                ui.separator();
                                                let mut changed = false;
                                                ui.label("Color");
                                                let c = &mut mat.desc.color_constant;
                                                let mut rgb = [c.x, c.y, c.z];
                                                if ui
                                                    .color_edit_button_rgb(&mut rgb)
                                                    .changed()
                                                {
                                                    *c = CgVec4::new(
                                                        rgb[0], rgb[1], rgb[2], c.w,
                                                    );
                                                    changed = true;
                                                }
                                                changed |= ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut mat.desc.mr_constant.x,
                                                            0.0..=1.0,
                                                        )
                                                        .text("metallic"),
                                                    )
                                                    .changed();
                                                changed |= ui
                                                    .add(
                                                        egui::Slider::new(
                                                            &mut mat.desc.mr_constant.y,
                                                            0.0..=1.0,
                                                        )
                                                        .text("roughness"),
                                                    )
                                                    .changed();
                                                if changed {
                                                    renderer.update_material(
                                                        &handle,
                                                        &mat.desc.color_constant,
                                                        &mat.desc.mr_constant,
                                                    );
                                                    mat.dirty = true;
                                                }
                                                // Texture assignment.  Changing a
                                                // texture rebuilds the material's
                                                // bind group (not just its
                                                // constants), so it goes through
                                                // reload_material rather than
                                                // update_material.
                                                ui.separator();
                                                let mut tex_changed = texture_combo(
                                                    ui,
                                                    "mat_color_tex",
                                                    "Color texture",
                                                    &mut mat.desc.color_texture,
                                                    &texture_options,
                                                );
                                                tex_changed |= texture_combo(
                                                    ui,
                                                    "mat_metal_tex",
                                                    "Metallic texture",
                                                    &mut mat.desc.metal_texture,
                                                    &texture_options,
                                                );
                                                tex_changed |= texture_combo(
                                                    ui,
                                                    "mat_rough_tex",
                                                    "Roughness texture",
                                                    &mut mat.desc.rough_texture,
                                                    &texture_options,
                                                );
                                                if tex_changed {
                                                    // Native import path already
                                                    // preloaded these; blocking
                                                    // the reload here is fine.
                                                    pollster::block_on(
                                                        renderer.reload_material(
                                                            &handle, &mat.name, &mat.desc,
                                                        ),
                                                    );
                                                    mat.dirty = true;
                                                }
                                                if texture_options.is_empty() {
                                                    ui.label(
                                                        egui::RichText::new(
                                                            "Import textures in the \
                                                             Resources panel to assign them.",
                                                        )
                                                        .weak(),
                                                    );
                                                }
                                            } else {
                                                ui.label("(unknown material)");
                                            }
                                        }
                                        Some(ResourceSelection::Particle(i)) => {
                                            // Cloned so the params borrow below
                                            // doesn't collide.
                                            let textures = self.particle_textures.clone();
                                            if let Some(resource) =
                                                self.particle_resources.get_mut(i)
                                            {
                                                // Save row (dirty marker + disk).
                                                if resource_header(
                                                    ui,
                                                    &resource.name,
                                                    resource.dirty,
                                                ) {
                                                    match save_particle_file(resource) {
                                                        Ok(()) => resource.dirty = false,
                                                        Err(e) => log!("Save failed: {e}"),
                                                    }
                                                }
                                                ui.separator();
                                                let old_name = resource.name.clone();
                                                ui.label("Name");
                                                if ui
                                                    .text_edit_singleline(&mut resource.name)
                                                    .changed()
                                                {
                                                    renamed_particle = Some((
                                                        old_name,
                                                        resource.name.clone(),
                                                    ));
                                                    resource.dirty = true;
                                                }
                                                ui.separator();
                                                if draw_particle_params_ui(
                                                    ui,
                                                    &mut resource.params,
                                                    &textures,
                                                ) {
                                                    edited_particle = Some(i);
                                                    resource.dirty = true;
                                                }
                                            } else {
                                                ui.label("(missing particle resource)");
                                            }
                                        }
                                        None => {
                                            ui.label("No resource selected.");
                                            ui.label(
                                                "Pick one in the Resources panel below.",
                                            );
                                        }
                                    },
                                    EditorTab::Settings => {
                                        ui.label(egui::RichText::new("Camera").strong());
                                        ui.spacing_mut().slider_width = 150.0;
                                        ui.add(
                                            egui::Slider::new(
                                                &mut self.fly_camera.move_rate,
                                                0.2..=50.0,
                                            )
                                            .logarithmic(true)
                                            .text("speed"),
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut self.fly_camera.shift_move_multiplier,
                                                1.0..=10.0,
                                            )
                                            .text("sprint ×"),
                                        );

                                        // Global shadow map size; casting and
                                        // the cascade controls stay on the
                                        // light's own properties in Details.
                                        ui.add_space(12.0);
                                        ui.label(egui::RichText::new("Shadows").strong());
                                        let mut shadows_changed = false;
                                        egui::ComboBox::from_label("Resolution")
                                            .selected_text(format!(
                                                "{}",
                                                self.editor_config.shadow_resolution
                                            ))
                                            .show_ui(ui, |ui| {
                                                for res in [512u32, 1024, 2048] {
                                                    shadows_changed |= ui
                                                        .selectable_value(
                                                            &mut self
                                                                .editor_config
                                                                .shadow_resolution,
                                                            res,
                                                            res.to_string(),
                                                        )
                                                        .changed();
                                                }
                                            });
                                        shadows_changed |= ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut self.editor_config.shadow_depth_bias,
                                                    0.0..=0.02,
                                                )
                                                .logarithmic(true)
                                                .text("depth bias"),
                                            )
                                            .on_hover_text(
                                                "Too low: shadow acne on flat surfaces. \
                                                 Too high: shadows detach from thin casters \
                                                 (looks like the shadow is floating).",
                                            )
                                            .changed();
                                        if shadows_changed {
                                            renderer.set_shadow_settings(
                                                &shadow_settings_from_config(
                                                    &self.editor_config,
                                                ),
                                            );
                                            self.editor_config.save();
                                        }

                                        ui.add_space(12.0);
                                        ui.label(egui::RichText::new("Editor").strong());
                                        if ui.button("Keybindings…").clicked() {
                                            self.show_settings = true;
                                        }

                                        // What loads when the editor opens: the
                                        // built-in default (church) until the
                                        // user saves their own.
                                        ui.add_space(12.0);
                                        ui.label(
                                            egui::RichText::new("Start-up Scene").strong(),
                                        );
                                        ui.label(if self.has_custom_startup {
                                            "Current: your saved scene"
                                        } else {
                                            "Current: default (church)"
                                        });
                                        if ui
                                            .button("Choose start-up scene…")
                                            .on_hover_text(
                                                "Pick a scene .json to open on \
                                                 every launch",
                                            )
                                            .clicked()
                                        {
                                            do_pick_startup = true;
                                        }
                                        if ui
                                            .button("Use current scene as start-up")
                                            .on_hover_text(
                                                "Opens this scene on every launch",
                                            )
                                            .clicked()
                                        {
                                            do_set_startup = true;
                                        }
                                        if ui
                                            .add_enabled(
                                                self.has_custom_startup,
                                                egui::Button::new("Reset to default"),
                                            )
                                            .clicked()
                                        {
                                            do_clear_startup = true;
                                        }
                                    }
                                }
                            });
                    });
                });
        }

        // Delete confirmation for the Scene tab's ✕ button.  A modal blocks
        // the rest of the UI until answered; clicking the backdrop cancels.
        if let Some(sel) = self.confirm_delete {
            let name = self.selection_name(sel);
            match name {
                None => self.confirm_delete = None, // Stale selection.
                Some(name) => {
                    let modal =
                        egui::Modal::new(egui::Id::new("confirm_delete")).show(&ctx, |ui| {
                            ui.label(format!("Delete \"{name}\"?"));
                            ui.add_space(8.0);
                            ui.horizontal(|ui| {
                                if ui
                                    .button(
                                        egui::RichText::new("Delete")
                                            .color(egui::Color32::from_rgb(235, 80, 80)),
                                    )
                                    .clicked()
                                {
                                    delete_object = Some(sel);
                                    self.confirm_delete = None;
                                }
                                if ui.button("Cancel").clicked() {
                                    self.confirm_delete = None;
                                }
                            });
                        });
                    if modal.should_close() {
                        self.confirm_delete = None;
                    }
                }
            }
        }

        // New Scene confirmation: it empties everything, unsaved work included.
        if self.confirm_new_scene {
            let modal = egui::Modal::new(egui::Id::new("confirm_new_scene")).show(&ctx, |ui| {
                ui.label("Start a new scene?  Unsaved changes will be lost.");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("New Scene").clicked() {
                        do_new_scene = true;
                        self.confirm_new_scene = false;
                    }
                    if ui.button("Cancel").clicked() {
                        self.confirm_new_scene = false;
                    }
                });
            });
            if modal.should_close() {
                self.confirm_new_scene = false;
            }
        }

        // Keybindings window (Settings > Keybindings…): one rebindable hotkey
        // per gizmo mode, plus a reset-to-defaults that asks first.  A binding
        // is picked by clicking it and pressing a key, captured from egui's
        // event stream below.  Editor only; changes are saved to disk.
        let mut rebound_this_frame = false;
        if editor && self.show_settings {
            let mut open = self.show_settings;
            egui::Window::new("Keybindings")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(&ctx, |ui| {
                    ui.label("Gizmo mode hotkeys (editor only):");
                    ui.add_space(6.0);
                    egui::Grid::new("keybindings_grid")
                        .num_columns(2)
                        .spacing(egui::vec2(12.0, 6.0))
                        .show(ui, |ui| {
                            for (i, (_mode, label)) in GIZMO_ACTIONS.iter().enumerate() {
                                ui.label(*label);
                                let listening = self.rebinding == Some(i);
                                let text = if listening {
                                    "press a key…".to_string()
                                } else {
                                    self.editor_config.gizmo_keys[i].name().to_string()
                                };
                                // Click to (re)bind; click again to cancel.
                                if ui.selectable_label(listening, text).clicked() {
                                    self.rebinding = if listening { None } else { Some(i) };
                                }
                                ui.end_row();
                            }
                        });
                    ui.add_space(8.0);
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Reset to Defaults").clicked() {
                            self.confirm_reset = true;
                        }
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| ui.label(egui::RichText::new("saved automatically").weak()),
                        );
                    });
                });
            self.show_settings = open;
            // Closing the window abandons any half-finished rebind.
            if !self.show_settings {
                self.rebinding = None;
            }

            // Capture the next key press for a pending rebind (Esc cancels).
            if let Some(slot) = self.rebinding {
                let key = ctx.input(|input| {
                    input.events.iter().find_map(|event| match event {
                        egui::Event::Key { key, pressed: true, .. } => Some(*key),
                        _ => None,
                    })
                });
                if let Some(key) = key {
                    if key != egui::Key::Escape {
                        self.editor_config.rebind(slot, key);
                        self.editor_config.save();
                        rebound_this_frame = true;
                    }
                    self.rebinding = None;
                }
            }
        }

        // Reset-to-defaults confirmation for the keybindings window.
        if editor && self.confirm_reset {
            let modal =
                egui::Modal::new(egui::Id::new("confirm_reset_keybinds")).show(&ctx, |ui| {
                    ui.label("Reset all keybindings to their defaults?");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Reset").clicked() {
                            self.editor_config = EditorConfig::default();
                            self.editor_config.save();
                            self.rebinding = None;
                            self.confirm_reset = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.confirm_reset = false;
                        }
                    });
                });
            if modal.should_close() {
                self.confirm_reset = false;
            }
        }

        // Editor-only gizmo hotkeys (default W/E/R).  Read from egui so any key
        // can be bound and typing in a field is naturally ignored; also
        // suppressed while rebinding and during a right-drag flythrough, where
        // W/A/S/D is driving the camera.
        if editor
            && self.rebinding.is_none()
            && !rebound_this_frame
            && !self.confirm_reset
            && !ctx.egui_wants_keyboard_input()
            && !input_manager.get_key_state("mouse_right").is_down()
        {
            for (i, (mode, _label)) in GIZMO_ACTIONS.iter().enumerate() {
                if ctx.input(|input| input.key_pressed(self.editor_config.gizmo_keys[i])) {
                    self.gizmo.mode = *mode;
                }
            }
            if ctx.input(|input| {
                input.key_pressed(egui::Key::S) && (input.modifiers.ctrl || input.modifiers.command)
            }) {
                do_save_scene = true;
            }
        }

        // In-world editor icons for lights (clicking one selects it).  Kept for
        // the pick block below, which only runs when a light icon didn't take
        // the click.
        let clicked_light = if editor {
            self.draw_light_icons(&ctx, game_config)
        } else {
            None
        };
        if let Some(i) = clicked_light {
            // Ctrl+click joins the multi-selection, like the outliner rows.
            let additive =
                ctx.input(|input| input.modifiers.ctrl || input.modifiers.command);
            select_object = Some((Selection::Light(i), additive));
        }

        // Translate/rotate/scale gizmo on the selected object, drawn over the 3D
        // view.  Its edits ride the same `selection_edited` path as Details.
        // Lights have no scale (only translate/rotate apply); particles use
        // translate/scale (rotation is ignored); splats have no gizmo.
        //
        // With a multi-selection the gizmo sits at the selection's centroid
        // and edits every selected object: the gizmo is fed an identity pivot
        // each frame, so what comes back is this frame's delta -- positions
        // orbit/scale about the centroid, and the rotation/scale compose onto
        // each object's own transform (see apply_pivot_delta).
        //
        // Undo capture is edge-triggered off the gizmo's active state (see the
        // commit block after this if/else): `gizmo_was_active` is its state at
        // the START of this frame, and `drag_candidate` holds the pre-drag
        // snapshot(s) taken speculatively while NOT dragging -- promoted to the
        // pending gesture only if a drag actually starts this frame.
        let gizmo_was_active = self.gizmo.is_active();
        let mut drag_candidate: Option<Vec<(Selection, ObjectRef, ObjectSnapshot)>> = None;
        if editor && self.multi_selected.len() >= 2 {
            let objects: Vec<(Selection, CgVec3)> = self
                .multi_selected
                .iter()
                .filter_map(|sel| self.selection_position(*sel).map(|p| (*sel, p)))
                .collect();
            if !objects.is_empty() {
                let centroid = objects
                    .iter()
                    .fold(CG_VEC3_ZERO, |acc, (_, p)| acc + *p)
                    / objects.len() as f32;
                // Local space has no single frame for a group: each object's
                // own local axis is averaged and renormalized (see
                // average_local_axes), giving one representative frame.
                let rotations: Vec<CgQuat> = self
                    .multi_selected
                    .iter()
                    .filter_map(|sel| self.selection_rotation(*sel))
                    .collect();
                let local_axes = TransformGizmo::average_local_axes(&rotations);
                let mut position = centroid;
                let mut rotation = CG_QUAT_IDENT;
                let mut scale = CG_VEC3_ONE;
                // Speculative pre-drag snapshot of every selected object (only
                // while not already dragging); promoted to the pending gesture
                // below if a drag starts this frame.
                if !gizmo_was_active {
                    let snaps: Vec<(Selection, ObjectRef, ObjectSnapshot)> = self
                        .multi_selected
                        .iter()
                        .filter_map(|sel| {
                            self.snapshot_object(*sel).map(|(obj, snap)| (*sel, obj, snap))
                        })
                        .collect();
                    if !snaps.is_empty() {
                        drag_candidate = Some(snaps);
                    }
                }
                if self.gizmo.ui(
                    &ctx,
                    &self.game_camera,
                    game_config,
                    &mut position,
                    &mut rotation,
                    &mut scale,
                    local_axes,
                ) {
                    // The axes ui() actually dragged along this call, so the
                    // returned per-component `scale` can be decomposed back
                    // onto the same directions (exact for an orthonormal
                    // frame -- world axes, or Local with one shared
                    // orientation; an approximation for a Local multi-select
                    // whose averaged axes aren't quite orthogonal).
                    let axes = self.gizmo.effective_axes(local_axes);
                    for (sel, old_pos) in objects {
                        let offset = old_pos - centroid;
                        let scaled_offset = axes[0] * (offset.dot(axes[0]) * scale.x)
                            + axes[1] * (offset.dot(axes[1]) * scale.y)
                            + axes[2] * (offset.dot(axes[2]) * scale.z);
                        let new_pos = position + rotation * scaled_offset;
                        self.apply_pivot_delta(sel, new_pos, rotation, scale, renderer);
                    }
                }
            }
        } else if editor {
            if let Some(sel) = self.selected {
                let current = match sel {
                    Selection::Actor(i) => self
                        .scene_actors
                        .get(i)
                        .map(|a| (a.get_position(), a.get_rotation(), a.get_scale())),
                    Selection::Light(i) => self
                        .scene_lights
                        .get(i)
                        .map(|l| (l.get_position(), l.get_rotation(), CG_VEC3_ONE)),
                    Selection::Particle(i) => self
                        .scene_particles
                        .get(i)
                        .map(|p| (p.position, CG_QUAT_IDENT, p.scale)),
                    // Splats use uniform scale only (non-uniform cloud scale is
                    // only approximate), so feed a uniform scale to the gizmo.
                    Selection::Splat(i) => self.scene_splats.get(i).map(|s| {
                        let u = s.transform.scale.x;
                        (s.transform.position, s.transform.rotation, CgVec3::new(u, u, u))
                    }),
                    // No scale of its own -- feed the gizmo a fixed 1.0 so a
                    // scale drag is a no-op rather than corrupting the sim.
                    Selection::MujocoActor(i) => self
                        .scene_mujoco_actors
                        .get(i)
                        .map(|m| (m.position, m.rotation, CG_VEC3_ONE)),
                    Selection::PostProcess => None,
                };
                if let Some((mut position, mut rotation, mut scale)) = current {
                    let local_axes = TransformGizmo::local_axes(rotation);
                    // Speculative pre-drag snapshot (only while not already
                    // dragging); promoted to the pending gesture below if a
                    // drag starts this frame.
                    if !gizmo_was_active {
                        if let Some((obj, snap)) = self.snapshot_object(sel) {
                            drag_candidate = Some(vec![(sel, obj, snap)]);
                        }
                    }
                    if self.gizmo.ui(
                        &ctx,
                        &self.game_camera,
                        game_config,
                        &mut position,
                        &mut rotation,
                        &mut scale,
                        local_axes,
                    ) {
                        match sel {
                            Selection::Actor(i) => {
                                if let Some(a) = self.scene_actors.get_mut(i) {
                                    a.set_position(&position);
                                    a.set_rotation(&rotation);
                                    a.set_scale(&scale);
                                }
                            }
                            Selection::Light(i) => {
                                if let Some(l) = self.scene_lights.get_mut(i) {
                                    l.set_position(&position);
                                    l.set_rotation(&rotation);
                                }
                            }
                            Selection::Particle(i) => {
                                if let Some(p) = self.scene_particles.get_mut(i) {
                                    p.position = position;
                                    p.scale = scale;
                                }
                            }
                            Selection::Splat(i) => {
                                if let Some(s) = self.scene_splats.get_mut(i) {
                                    s.transform.position = position;
                                    s.transform.rotation = rotation;
                                    // Uniform scale only: collapse the gizmo's
                                    // per-axis result to the axis that moved most
                                    // (the center handle moves them equally), and
                                    // apply it to all three.
                                    let prev = s.transform.scale.x;
                                    let mut u = prev;
                                    for a in [scale.x, scale.y, scale.z] {
                                        if (a - prev).abs() > (u - prev).abs() {
                                            u = a;
                                        }
                                    }
                                    let u = u.max(0.001);
                                    s.transform.scale = CgVec3::new(u, u, u);
                                }
                            }
                            Selection::MujocoActor(i) => {
                                if let Some(m) = self.scene_mujoco_actors.get_mut(i) {
                                    m.position = position;
                                    m.rotation = rotation;
                                }
                            }
                            Selection::PostProcess => {}
                        }
                        selection_edited = true;
                    }
                }
            }
        }

        // Commit undo history on gizmo-drag *edges* only -- never on idle
        // frames, which would otherwise flood the stack with no-op edits and
        // evict the real ones:
        //   * drag start (was inactive, now active): stash the pre-drag
        //     snapshot(s) captured above as the pending gesture.
        //   * drag end (was active, now inactive): diff each object against its
        //     current state and push one Edit (or a Batch for multi-select).
        let gizmo_now_active = self.gizmo.is_active();
        if !gizmo_was_active && gizmo_now_active {
            self.pending_gizmo_edit = drag_candidate;
        } else if gizmo_was_active && !gizmo_now_active {
            if let Some(pending) = self.pending_gizmo_edit.take() {
                let edits: Vec<UndoAction> = pending
                    .into_iter()
                    .filter_map(|(sel, obj, before)| {
                        self.snapshot_object(sel)
                            .map(|(_, after)| UndoAction::Edit { obj, before, after })
                    })
                    .collect();
                match edits.len() {
                    0 => {}
                    1 => self.undo_stack.push(edits.into_iter().next().unwrap()),
                    _ => self.undo_stack.push(UndoAction::Batch(edits)),
                }
            }
        }

        // Viewport click-to-select: after the gizmo (so grabbing a handle wins)
        // and only when a light icon didn't take the click.  Picks the front-most
        // actor/particle under the pointer; failing that, the active splat -- so
        // splats are only pickable when no 3D object was.  Ctrl+click adds to /
        // removes from the multi-selection (and never falls through to the
        // splat, so a missed ctrl+click doesn't wipe the set).
        if editor && clicked_light.is_none() && !self.gizmo.is_active() {
            let (pointer, pressed, additive) = ctx.input(|i| {
                (
                    i.pointer.interact_pos(),
                    i.pointer.primary_pressed(),
                    i.modifiers.ctrl || i.modifiers.command,
                )
            });
            if pressed && !ctx.egui_wants_pointer_input() {
                if let Some(p) = pointer {
                    if let Some(sel) = self.pick_object(&ctx, game_config, p) {
                        select_object = Some((sel, additive));
                    } else if !additive && !self.scene_splats.is_empty() {
                        select_object = Some((Selection::Splat(self.active_splat), false));
                    }
                }
            }
        }

        // Right-click (a click, not a look-drag) opens the context menu when
        // two or more objects (of any kind) are multi-selected.  "Click" is
        // defined by the same signal that drives the camera: if the fly
        // camera's look never engaged during the hold (see
        // FlyCamera::is_looking), the release is a click -- so the menu and
        // the pointer grab can never disagree, whatever the platform's
        // raw-delta behavior (the browser's differs from native).
        {
            let rmb = input_manager.get_key_state("mouse_right");
            if rmb.just_pressed() {
                self.rmb_drag_accum = 0.0;
                self.rmb_look_engaged = false;
            }
            if rmb.is_down() {
                let (dx, dy) = input_manager.get_mouse_raw_delta();
                self.rmb_drag_accum += dx.abs() as f32 + dy.abs() as f32;
                self.rmb_look_engaged |= self.fly_camera.is_looking();
            }
            let released = ctx
                .input(|i| i.pointer.button_released(egui::PointerButton::Secondary));
            if editor && released && self.multi_selected.len() >= 2 {
                let pointer = ctx.input(|i| i.pointer.interact_pos().or(i.pointer.latest_pos()));
                // "Over UI" = an egui Area above the Background painter layer
                // (panels, menus, modals) under the pointer.  Deliberately not
                // egui_wants_pointer_input(): on the release frame its
                // pointer-over check flips on (`!any_down`) and can claim
                // empty-viewport clicks.
                let over_ui = pointer
                    .and_then(|p| ctx.layer_id_at(p))
                    .is_some_and(|l| l.order != egui::Order::Background);
                if !self.rmb_look_engaged && !over_ui {
                    if let Some(p) = pointer {
                        self.context_menu = Some(p);
                    }
                }
            }
        }

        // The context menu itself: currently one action, snapping every other
        // selected object (any kind -- actors, lights, particles, splats)
        // onto the first-selected one.
        let mut do_snap = false;
        if editor {
            if let Some(menu_pos) = self.context_menu {
                // First selection = the snap anchor; the rest move to it.
                // Names resolved up front (stale selections dismiss the menu).
                let label = match self.multi_selected.split_first() {
                    Some((anchor, rest @ [_, ..])) => {
                        self.selection_name(*anchor).map(|anchor_name| {
                            if let [only] = rest {
                                let moved = self
                                    .selection_name(*only)
                                    .unwrap_or_else(|| "object".to_string());
                                format!("Snap \"{moved}\" to \"{anchor_name}\"")
                            } else {
                                format!("Snap {} objects to \"{anchor_name}\"", rest.len())
                            }
                        })
                    }
                    _ => None,
                };
                match label {
                    None => self.context_menu = None, // Selection changed under it.
                    Some(label) => {
                        let menu = egui::Area::new(egui::Id::new("viewport_context_menu"))
                            .fixed_pos(menu_pos)
                            .constrain(true)
                            .show(&ctx, |ui| {
                                egui::Frame::menu(ui.style()).show(ui, |ui| {
                                    if ui.button(label).clicked() {
                                        do_snap = true;
                                        self.context_menu = None;
                                    }
                                });
                            });
                        // Click anywhere else or Escape dismisses.
                        let (clicked, pointer, escape) = ctx.input(|i| {
                            (
                                i.pointer.any_pressed(),
                                i.pointer.interact_pos(),
                                i.key_pressed(egui::Key::Escape),
                            )
                        });
                        let outside = clicked
                            && pointer
                                .is_none_or(|p| !menu.response.rect.expand(4.0).contains(p));
                        if outside || escape {
                            self.context_menu = None;
                        }
                    }
                }
            }
        }

        // Apply the editor actions gathered from the menus / Scene tab.
        if let Some((sel, additive)) = select_object {
            if additive {
                // Ctrl+click toggles membership; the newest pick becomes the
                // primary selection (gizmo/Details target).
                if let Some(at) = self.multi_selected.iter().position(|s| *s == sel) {
                    self.multi_selected.remove(at);
                    self.selected = self.multi_selected.last().copied();
                } else {
                    self.multi_selected.push(sel);
                    self.selected = Some(sel);
                }
            } else {
                self.multi_selected.clear();
                self.multi_selected.push(sel);
                self.selected = Some(sel);
                self.context_menu = None;
            }
            // Selecting a splat also makes it the rendered cloud.
            if let Selection::Splat(i) = sel {
                self.activate_splat(i, renderer);
            }
        }
        // Snap from the context menu: move every other selected object onto
        // the first-selected one.
        if do_snap {
            let anchor_pos = self
                .multi_selected
                .first()
                .and_then(|sel| self.selection_position(*sel));
            if let Some(pos) = anchor_pos {
                let targets: Vec<Selection> = self.multi_selected[1..].to_vec();
                for sel in targets {
                    self.set_selection_position(sel, &pos, renderer);
                }
            }
        }
        // Apply outliner renames (accumulated per section during the pass).
        for (i, new_name) in rename_actors {
            if let Some(actor) = self.scene_actors.get_mut(i) {
                actor.set_name(&new_name);
            }
        }
        for (i, new_name) in rename_lights {
            if let Some(light) = self.scene_lights.get_mut(i) {
                light.set_name(&new_name);
            }
        }
        for (i, new_name) in rename_particles {
            if let Some(particle) = self.scene_particles.get_mut(i) {
                particle.name = new_name;
            }
        }
        for (i, new_name) in rename_mujoco_actors {
            if let Some(actor) = self.scene_mujoco_actors.get_mut(i) {
                actor.name = new_name;
            }
        }
        if let Some(kind) = do_add {
            match kind {
                AddKind::Actor => self.add_actor(renderer),
                AddKind::Light(light_type) => self.add_light(light_type, renderer),
                AddKind::Particle(preset) => self.add_particle(preset, renderer),
                // Splats come from a file, so "Add > Gaussian Splat" opens the
                // same .ply picker as the load button.
                AddKind::Splat => do_load = true,
                AddKind::MujocoActor => self.add_mujoco_actor(),
            }
        }
        if let Some(sel) = delete_object {
            self.delete_selected(sel, renderer);
        }
        // Resources-panel actions: create a library resource and open it in
        // the Resource inspector.
        if create_material {
            self.next_object_num += 1;
            let name = format!("Material {}", self.next_object_num);
            let desc = MaterialDesc::default();
            let handle = renderer.create_material(&name, &desc);
            // New, unsaved: dirty so the browser shows its save icon until the
            // user writes it to disk (saved_name None until then).
            self.material_library.push(MaterialResource {
                name,
                desc,
                handle,
                dirty: true,
                saved_name: None,
            });
            self.selected_resource = Some(ResourceSelection::Material(handle));
            self.active_tab = Some(EditorTab::Resource);
        }
        if create_particle {
            self.next_object_num += 1;
            self.particle_resources.push(ParticleResource {
                name: format!("Particle {}", self.next_object_num),
                params: preset_particle_params("Fire"),
                dirty: true,
                saved_name: None,
            });
            self.selected_resource =
                Some(ResourceSelection::Particle(self.particle_resources.len() - 1));
            self.active_tab = Some(EditorTab::Resource);
        }
        if import_texture {
            self.open_texture_picker();
        }
        if import_model {
            self.open_model_picker();
        }
        if let Some(actor_name) = mujoco_xml_browse_request {
            self.open_mujoco_xml_picker(actor_name);
        }
        if let Some(actor_name) = mujoco_traj_browse_request {
            self.open_mujoco_trajectory_picker(actor_name);
        }
        // A model tile clicked while still unloaded: load it now (lazy), then
        // select it.  Native loads synchronously; web fetches in the background
        // and the tick that finishes the upload does the select.
        if let Some(path) = model_load_request {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let handle = pollster::block_on(renderer.load_model(&path, false));
                self.selected_resource = Some(ResourceSelection::Model(handle));
            }
            #[cfg(target_arch = "wasm32")]
            {
                let pending = self.pending_model_uploads.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(bytes) = load_binary(&path).await {
                        pending.lock().unwrap().push((path, bytes, true));
                    }
                });
            }
        }
        // A model chosen from the Details-panel dropdown while still unloaded:
        // load it and assign it to the selected actor.
        if let (Some(path), Some(Selection::Actor(i))) = (model_pick_request, self.selected) {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let handle = pollster::block_on(renderer.load_model(&path, false));
                if let Some(actor) = self.scene_actors.get_mut(i) {
                    actor.set_model(&handle);
                    renderer.add_or_update_actor(actor);
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                // Fetch in the background; the model-upload drain assigns it to
                // this actor once ready (matched by name via pending_actor_models).
                self.pending_actor_models
                    .push((i, resource_display_name(&path)));
                let pending = self.pending_model_uploads.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    if let Ok(bytes) = load_binary(&path).await {
                        pending.lock().unwrap().push((path, bytes, false));
                    }
                });
            }
        }
        // A MuJoCo scene XML picked via the Details panel's Browse… button:
        // assign it to the actor it was opened for. tick_mujoco_actors picks
        // up the new xml_path next frame and (re)loads the scene.
        let picked_mujoco_xml = self.picked_mujoco_xml.lock().unwrap().take();
        if let Some((actor_name, path)) = picked_mujoco_xml {
            if let Some(actor) = self
                .scene_mujoco_actors
                .iter_mut()
                .find(|a| a.name == actor_name)
            {
                actor.xml_path = path;
            }
        }
        // Likewise for a trajectory clip picked via its own Browse… button;
        // tick_mujoco_actors retargets it against the loaded sim next frame.
        let picked_mujoco_trajectory = self.picked_mujoco_trajectory.lock().unwrap().take();
        if let Some((actor_name, path)) = picked_mujoco_trajectory {
            if let Some(actor) = self
                .scene_mujoco_actors
                .iter_mut()
                .find(|a| a.name == actor_name)
            {
                actor.trajectory_path = path;
            }
        }
        // Save requests from the browser tiles' disk buttons: write the file
        // and clear the dirty flag.
        if let Some(handle) = save_material_handle {
            if let Some(mat) = self.material_library.iter_mut().find(|m| m.handle == handle) {
                match save_material_file(mat) {
                    Ok(()) => mat.dirty = false,
                    Err(e) => log!("Save failed: {e}"),
                }
            }
        }
        if let Some(i) = save_particle_index {
            if let Some(resource) = self.particle_resources.get_mut(i) {
                match save_particle_file(resource) {
                    Ok(()) => resource.dirty = false,
                    Err(e) => log!("Save failed: {e}"),
                }
            }
        }
        // Rename requested from a material tile's right-click menu.  Materials
        // are referenced by handle, so only the display name / save file change.
        if let Some((handle, new_name)) = rename_material {
            if let Some(mat) = self.material_library.iter_mut().find(|m| m.handle == handle) {
                mat.name = new_name;
                mat.dirty = true;
            }
        }
        // Renaming a particle resource keeps the scene emitters that reference
        // it pointing at the new name.
        if let Some((old_name, new_name)) = renamed_particle {
            for particle in &mut self.scene_particles {
                if particle.preset == old_name {
                    particle.preset = new_name.clone();
                }
            }
        }
        // Inspector edits to a particle resource push to every live emitter
        // spawned from it (texture changes only affect future spawns).
        if let Some(i) = edited_particle {
            if let Some(resource) = self.particle_resources.get(i) {
                let params = resource.params.clone();
                let handles: Vec<ParticleHandle> = self
                    .scene_particles
                    .iter()
                    .filter(|p| p.preset == resource.name)
                    .map(|p| p.handle.clone())
                    .collect();
                for handle in handles {
                    renderer.update_particle_params(&handle, &params);
                }
            }
        }
        // Push Details-panel / gizmo edits to the renderer's copy of the object.
        if selection_edited {
            match self.selected {
                Some(Selection::Actor(i)) => {
                    if let Some(actor) = self.scene_actors.get(i) {
                        renderer.add_or_update_actor(actor);
                    }
                }
                Some(Selection::Particle(i)) => {
                    if let Some(p) = self.scene_particles.get(i) {
                        renderer.update_particle_transform(&p.handle, &p.position, &Some(p.scale));
                    }
                }
                Some(Selection::Light(i)) => {
                    if let Some(light) = self.scene_lights.get(i) {
                        renderer.add_or_update_light(light);
                    }
                }
                Some(Selection::Splat(i)) => {
                    if let Some(splat) = self.scene_splats.get(i) {
                        renderer.set_gaussian_splat_params(&splat.params);
                        renderer.set_gaussian_splat_transform(&splat.transform);
                    }
                }
                Some(Selection::PostProcess) => {
                    renderer.set_post_process_settings(&self.scene_post_process);
                }
                _ => {}
            }
        }
        if do_new_scene {
            self.clear_scene(renderer);
            // Every new scene starts with a skylight (deletable like any light).
            self.add_default_skylight(renderer);
            self.status = Some(("New scene".to_string(), STATUS_WHITE, 3.0));
        }
        if do_set_startup {
            // Snapshot the current scene as the startup scene.  Clouds picked
            // through a browser dialog have no re-readable path and won't
            // reload; everything else restores on the next launch.
            if let Ok(json) = serde_json::to_string_pretty(&self.build_scene_file(&model_resources))
            {
                editor_config::save_startup_scene(&json);
                self.has_custom_startup = editor_config::load_startup_scene().is_some();
                self.status = Some(if self.has_custom_startup {
                    ("Start-up scene saved".to_string(), STATUS_WHITE, 4.0)
                } else {
                    (
                        "Couldn't save the start-up scene".to_string(),
                        STATUS_RED,
                        6.0,
                    )
                });
            }
        }
        if do_clear_startup {
            editor_config::clear_startup_scene();
            self.has_custom_startup = false;
            self.status = Some((
                "Start-up scene reset to default".to_string(),
                STATUS_WHITE,
                4.0,
            ));
        }
        if do_save_scene {
            self.save_scene(&model_resources);
        }
        if do_load_scene {
            Self::open_scene_picker_into(self.picked_scene.clone());
        }
        if do_pick_startup {
            Self::open_scene_picker_into(self.picked_startup.clone());
        }
        // Apply an imported texture once its file has been read: copy it into
        // game_assets/textures/, load it, and add it to the texture library.
        let picked_texture = self.picked_texture.lock().unwrap().take();
        if let Some((file_name, bytes)) = picked_texture {
            self.status = Some(match resource_library::import_texture(&file_name, &bytes) {
                Ok(path) => {
                    if self.texture_resources.iter().any(|t| t.path == path) {
                        (format!("Texture already imported: {file_name}"), STATUS_WHITE, 4.0)
                    } else {
                        // Native-only import path, so a blocking load here is
                        // fine (rfd already ran the pick off-thread).
                        let handle = pollster::block_on(renderer.preload_texture(&path));
                        self.texture_resources.push(TextureResource {
                            name: resource_display_name(&path),
                            path,
                            handle,
                        });
                        (format!("Imported texture: {file_name}"), STATUS_WHITE, 4.0)
                    }
                }
                Err(reason) => (format!("Couldn't import texture: {reason}"), STATUS_RED, 8.0),
            });
        }
        // Apply an imported model once its file has been read: persist it and
        // register it so it shows in the model picker this session.  Persist =
        // game_assets/models on native, IndexedDB on web (survives reloads);
        // registration is synchronous on both (block_on native, from-bytes web).
        let picked_model = self.picked_model.lock().unwrap().take();
        if let Some((file_name, bytes)) = picked_model {
            let path = format!("game_assets/models/{file_name}");
            if self.model_resources.iter().any(|m| m.path == path) {
                self.status = Some((
                    format!("Model already imported: {file_name}"),
                    STATUS_WHITE,
                    4.0,
                ));
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                let handle = match resource_library::save_model(&file_name, &bytes) {
                    Ok(_) => Some(pollster::block_on(renderer.load_model(&path, false))),
                    Err(e) => {
                        log!("Model import failed: {e}");
                        None
                    }
                };
                // Web import: only self-contained binary glb is supported (a
                // .gltf's external buffers/textures can't be resolved without a
                // filesystem).  Reject anything else rather than panic in the
                // glb parser.
                #[cfg(target_arch = "wasm32")]
                let handle = if bytes.starts_with(b"glTF") {
                    // Persist to IndexedDB in the background (bytes already in
                    // hand), and upload from those bytes for this session.
                    let (p, b) = (path.clone(), bytes.clone());
                    wasm_bindgen_futures::spawn_local(async move {
                        black_splat::idb::put(&p, &b).await;
                    });
                    Some(renderer.load_model_from_bytes(&path, &bytes, false))
                } else {
                    None
                };

                self.status = Some(match handle {
                    // The model is registered in the renderer's AssetManager by
                    // the load above; add its catalog entry so it lists too.
                    Some(_) => {
                        self.model_resources.push(ModelResource {
                            name: resource_display_name(&path),
                            path,
                        });
                        (format!("Imported model: {file_name}"), STATUS_WHITE, 4.0)
                    }
                    None => {
                        #[cfg(target_arch = "wasm32")]
                        let msg = format!("Couldn't import {file_name}: web import needs a binary .glb");
                        #[cfg(not(target_arch = "wasm32"))]
                        let msg = format!("Couldn't import {file_name} (see log)");
                        (msg, STATUS_RED, 8.0)
                    }
                });
            }
        }
        // Apply a loaded scene once its file has been read (async, like the .ply).
        let scene_bytes = self.picked_scene.lock().unwrap().take();
        if let Some(bytes) = scene_bytes {
            self.status = Some(match self.load_scene_from_bytes(&bytes, &model_resources, renderer)
            {
                Ok(summary) => (format!("Loaded scene: {summary}"), STATUS_WHITE, 5.0),
                Err(reason) => (format!("Couldn't load scene: {reason}"), STATUS_RED, 10.0),
            });
        }
        // Persist a picked startup scene once its file has been read.  Validated
        // first so a bad pick can't brick the next launch.
        let startup_bytes = self.picked_startup.lock().unwrap().take();
        if let Some(bytes) = startup_bytes {
            let valid = std::str::from_utf8(&bytes)
                .ok()
                .filter(|text| serde_json::from_str::<SceneFile>(text).is_ok())
                .map(|text| text.to_string());
            self.status = Some(match valid {
                Some(json) => {
                    editor_config::save_startup_scene(&json);
                    self.has_custom_startup = editor_config::load_startup_scene().is_some();
                    (
                        "Start-up scene set (loads on next launch)".to_string(),
                        STATUS_WHITE,
                        5.0,
                    )
                }
                None => (
                    "That file isn't a valid scene .json".to_string(),
                    STATUS_RED,
                    8.0,
                ),
            });
        }
        // Upload splat clouds whose .ply bytes arrived from a scene load.
        self.drain_pending_splats(renderer);

        // Reconcile + step + draw every MuJoCo actor.
        self.tick_mujoco_actors(renderer, game_config);

        // A skylight's "Bake Environment Cubemap" checkbox is a one-shot
        // trigger (see Light::take_cubemap_bake_request): fire the bake and
        // let the checkbox pop back up.
        for light in &mut self.scene_lights {
            if light.take_cubemap_bake_request() {
                renderer.bake_skylight_cubemap(light.id, game_config);
                renderer.add_or_update_light(light);
            }
            // Same one-shot pattern for "Clear Environment Cubemap": drop the
            // baked GPU texture and fall back to the analytic gradient.
            if light.take_cubemap_clear_request() {
                renderer.clear_skylight_cubemap(light.id);
                renderer.add_or_update_light(light);
            }
        }

        // Upload lazily-fetched model bytes that arrived this frame (web): browser
        // selects and scene actors whose model was still loading pop in here.
        #[cfg(target_arch = "wasm32")]
        {
            let uploads: Vec<(String, Vec<u8>, bool)> =
                std::mem::take(&mut *self.pending_model_uploads.lock().unwrap());
            for (path, bytes, select) in uploads {
                let handle = renderer.load_model_from_bytes(&path, &bytes, false);
                if select {
                    self.selected_resource = Some(ResourceSelection::Model(handle));
                }
                // Assign the freshly-loaded mesh to any scene actor waiting on it.
                let name = resource_display_name(&path);
                let mut pending = std::mem::take(&mut self.pending_actor_models);
                pending.retain(|(idx, want)| {
                    if *want == name {
                        if let Some(actor) = self.scene_actors.get_mut(*idx) {
                            actor.set_model(&handle);
                            renderer.add_or_update_actor(actor);
                        }
                        false
                    } else {
                        true
                    }
                });
                self.pending_actor_models = pending;
            }
        }

        // On-screen move/look touch pads (shared TouchPads controller).  Left
        // pad adds to movement, right pad turns the camera.
        let pads = self.touch_pads.update(&ctx, input_manager, delta_time);
        move_vec += right_dir * pads.move_deflection.x - forward_dir * pads.move_deflection.y;
        camera_rot.x += pads.yaw_delta_deg;
        camera_rot.y += pads.pitch_delta_deg;

        if move_vec.magnitude2() > 0.001 {
            let speed = self.fly_camera.move_speed(input_manager);
            let new_pos =
                self.game_camera.get_position() + move_vec.normalize() * delta_time * speed;
            self.game_camera.set_position(&new_pos);
        }

        self.fly_camera.clamp_pitch(&mut camera_rot);
        self.game_camera.set_rotation(&camera_rot);

        // Frame the selected object ([F]).  Runs after the rotation is committed
        // so it dollies along the final view direction; editor only, and ignored
        // while typing in a field or rebinding a key.
        if editor
            && self.rebinding.is_none()
            && !ctx.egui_wants_keyboard_input()
            && ctx.input(|i| i.key_pressed(egui::Key::F))
        {
            self.frame_selected();
        }

        // The [Space]/[L] shortcuts below read the keyboard directly (via
        // input_manager, bypassing egui), so -- like the gizmo and 1-9 param
        // keys -- they must ignore key presses while a text field has focus or a
        // rebind is in progress.  Otherwise typing an "l" into a rename box
        // (e.g. "Wall", "Pillar", "Floor") pops the .ply file dialog, and a
        // space cycles the active splat.
        let raw_hotkeys_ok = self.rebinding.is_none() && !ctx.egui_wants_keyboard_input();

        // Undo / redo ([Ctrl+Z] / [Ctrl+Shift+Z] or [Ctrl+Y]).  Gated the same
        // as the other raw hotkeys so typing "z"/"y" into a rename box or a
        // Details text field doesn't trigger it.
        if raw_hotkeys_ok {
            let (ctrl, shift) = ctx.input(|i| {
                (i.modifiers.ctrl || i.modifiers.command, i.modifiers.shift)
            });
            if ctrl && input_manager.get_key_state("z").just_pressed() {
                if shift {
                    self.redo(renderer);
                } else {
                    self.undo(renderer);
                }
            } else if ctrl && input_manager.get_key_state("y").just_pressed() {
                self.redo(renderer);
            }
        }

        // Cycle to the next loaded splat cloud ([Space]), applying its params.
        // If a splat is the current selection, keep the selection on the newly
        // shown cloud so Details follows it.
        if raw_hotkeys_ok
            && input_manager.get_key_state("space").just_pressed()
            && !self.scene_splats.is_empty()
        {
            let next = (self.active_splat + 1) % self.scene_splats.len();
            self.activate_splat(next, renderer);
            if matches!(self.selected, Some(Selection::Splat(_))) {
                self.selected = Some(Selection::Splat(next));
            }
        }

        // Load a user .ply ([L] or the GUI button): opens the async file picker,
        // whose result arrives via `picked_ply` on a later tick.
        if do_load || (raw_hotkeys_ok && input_manager.get_key_state("l").just_pressed()) {
            self.open_ply_picker();
        }

        // [B] cycles the splat perf bisect (see SplatBisect): each step drops one
        // stage of the splat pass so the fps delta prices that stage.  The HUD
        // banners whichever stage is currently being skipped.
        if raw_hotkeys_ok && input_manager.get_key_state("b").just_pressed() {
            if let Some(label) = renderer.cycle_splat_bisect() {
                log!("Splat bisect: {label}");
            }
        }
        let picked = self.picked_ply.lock().unwrap().take();
        if let Some((file_name, path, bytes)) = picked {
            let name = file_name.trim_end_matches(".ply").to_string();
            match renderer.load_gaussian_splat_from_bytes(&bytes, &name, &default_splat_params()) {
                Ok(info) => {
                    self.scene_splats.push(SceneSplat {
                        name: name.clone(),
                        // Native: the file's real path. Web: an IndexedDB
                        // cache key (see `open_ply_picker`). Either way,
                        // re-readable by `load_binary` when the scene reloads.
                        path: path.unwrap_or_default(),
                        params: default_splat_params(),
                        transform: ActorTransform::from_position(CG_VEC3_ZERO),
                    });
                    let idx = self.scene_splats.len() - 1;
                    self.activate_splat(idx, renderer);
                    self.selected = Some(Selection::Splat(idx));
                    self.status = info.clamped_from.map(|original| {
                        (
                            format!(
                                "{name}.ply is too large: showing {} of {} splats",
                                group_digits(info.num_splats as usize),
                                group_digits(original),
                            ),
                            STATUS_RED,
                            10.0,
                        )
                    });
                }
                Err(reason) => {
                    self.status = Some((
                        format!("Couldn't load {file_name}: {reason}"),
                        STATUS_RED,
                        10.0,
                    ));
                }
            }
        }

        // Status line: an in-flight read reports progress; otherwise show (and
        // age out) the latest transient message.
        match &*self.picker_state.lock().unwrap() {
            PickerState::Reading(name) => {
                renderer.set_status_msg(&format!("Loading {name}..."), &STATUS_WHITE);
            }
            _ => match &mut self.status {
                Some((msg, color, time_left)) => {
                    *time_left -= delta_time;
                    let expired = *time_left <= 0.0;
                    renderer.set_status_msg(if expired { "" } else { msg }, color);
                    if expired {
                        self.status = None;
                    }
                }
                None => renderer.set_status_msg("", &STATUS_WHITE),
            },
        }

        // Splat param adjustments: keyboard nudges for the active cloud's params
        // (a quick alternative to the Details panel drags).  Suppressed while
        // egui has keyboard focus, so typing a value into the Settings/Details
        // panels (e.g. a shadow distance of "300") doesn't also drive the
        // active cloud's params through the 1-9 keys -- which is how "3" was
        // silently shrinking Splat Scale while adjusting shadow settings.
        let adj = delta_time * PARAM_RATE;
        let hotkeys_ok = !ctx.egui_wants_keyboard_input();
        if let Some(splat) = self
            .scene_splats
            .get_mut(self.active_splat)
            .filter(|_| hotkeys_ok)
        {
            let p = &mut splat.params;
            let mut changed = false;

            if input_manager.get_key_state("1").is_down() {
                p.falloff = (p.falloff - adj).max(0.01);
                changed = true;
            }
            if input_manager.get_key_state("2").is_down() {
                p.falloff = (p.falloff + adj).min(20.0);
                changed = true;
            }
            if input_manager.get_key_state("3").is_down() {
                p.scale = (p.scale - adj).max(0.1);
                changed = true;
            }
            if input_manager.get_key_state("4").is_down() {
                p.scale = (p.scale + adj).min(20.0);
                changed = true;
            }
            if input_manager.get_key_state("5").is_down() {
                p.contrast = (p.contrast - adj).max(0.1);
                changed = true;
            }
            if input_manager.get_key_state("6").is_down() {
                p.contrast = (p.contrast + adj).min(5.0);
                changed = true;
            }
            if input_manager.get_key_state("7").is_down() {
                p.overall_scale = (p.overall_scale - adj).max(0.1);
                changed = true;
            }
            if input_manager.get_key_state("8").is_down() {
                p.overall_scale = (p.overall_scale + adj).min(10.0);
                changed = true;
            }
            if input_manager.get_key_state("9").just_pressed() {
                // The splat record carries 8 "rest" coefficients (degrees 1+2), so
                // 2 is the highest degree that changes anything.
                p.max_sh_degree = match p.max_sh_degree as u32 {
                    0 => 1.0,
                    1 => 2.0,
                    _ => 0.0,
                };
                changed = true;
            }

            if changed {
                renderer.set_gaussian_splat_params(p);
            }
        }

        let pos = self.game_camera.get_position();
        let rot = self.game_camera.get_rotation();
        let active_name = self
            .scene_splats
            .get(self.active_splat)
            .map_or("none", |s| s.name.as_str());
        let splat_count = renderer.active_gaussian_splat_count();
        renderer.set_debug_game_msg(&format!(
            "Move: [W][A][S][D] (editor: hold right mouse)   [Shift] sprint   Look: [Arrow Keys]\n\
             Touch: left pad = move,  right pad = look\n\n\
             [Space]     Next scene  {} ({}/{})   {} splats\n\
             [L]         Load your own .ply\n\
             [F]         Frame the selected object",
            active_name,
            self.active_splat + 1,
            self.scene_splats.len().max(1),
            splat_count,
        ));
        renderer.set_debug_topright_msg(&format!(
            "Camera\npos ({:.2}, {:.2}, {:.2})\nrot ({:.1}, {:.1})",
            pos.x, pos.y, pos.z, rot.x, rot.y,
        ));

        renderer.set_camera(&self.game_camera);
    }
}
