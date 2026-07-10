use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use serde::{Deserialize, Serialize};

use black_splat::{
    egui, assets::*, config::*, editor::{self, EditorChoice, TransformGizmo}, engine::*,
    fly_camera::*, game_object::*, input::*, renderer::*, resource::SceneLayer, touch_pads::*,
    utils::*, log,
    passes::gaussian_splat::SplatParams,
};

use crate::editor_config::{self, EditorConfig, GIZMO_ACTIONS};

// No clouds are hardcoded: startup opens the user's saved startup scene, or
// `default_startup_scene()` (the church) if none is saved yet.

// Model resources preloaded for the editor (the Resources tab and the Details
// panel's model dropdown).  Loaded in initialize_world -- model loading is
// async and the frame tick isn't -- so they're always available to assign.
const PRELOADED_MODELS: &[&str] = &[
    "game_assets/models/barrel.glb",
    "game_assets/models/shotgun.glb",
];

// How far in front of the camera a newly added game object is dropped.
const ADD_OBJECT_DISTANCE: f32 = 5.0;

// Keyboard/mouse fly-camera movement and look come from the shared FlyCamera and
// the on-screen touch pads from the shared TouchPads (black_splat::fly_camera /
// ::touch_pads), whose defaults already match this viewer's feel.
const PARAM_RATE: f32 = 1.5;

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

/// Whether this browser reports a touch screen.  Finger-friendly GUI sizing
/// should only kick in on actual touch devices; desktop browsers keep egui's
/// defaults so the GUI matches the native desktop build.
#[cfg(target_arch = "wasm32")]
fn is_touch_device() -> bool {
    web_sys::window().is_some_and(|w| w.navigator().max_touch_points() > 0)
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
    Settings,
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
}

/// What the "Add" menu asked to create this frame (applied after the egui pass).
#[derive(Clone, Copy)]
enum AddKind {
    Actor,
    Light(LightType),
    Particle(usize), // index into PARTICLE_PRESETS
    Splat,           // opens the .ply picker
}

/// A loaded splat cloud as a scene object: a display name plus its own render
/// params (shown in the Details panel when selected).  Index-aligned with the
/// renderer's splat clouds.
struct SceneSplat {
    name: String,
    // Where the .ply came from (stored in saved scenes so they can reload it).
    // Empty when there is no re-readable source (browser file picks).
    path: String,
    params: SplatParams,
    // World transform (editor gizmo): lets a cloud be dragged/rotated/scaled.
    transform: ActorTransform,
}

/// Render params a freshly loaded splat cloud starts with (tuned for this
/// viewer's clouds; each splat can then be tweaked in Details).
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
// A flexible, human-readable snapshot of the editable scene.  Kept deliberately
// decoupled from the engine types (plain arrays + names/indices) so the format
// can evolve independently; bump SCENE_FORMAT_VERSION on breaking changes.

const SCENE_FORMAT_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SceneFile {
    version: u32,
    // Absent in e.g. the built-in startup scene, which keeps the config's
    // start camera.
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
    layer: u32, // SceneLayer choice index
}

#[derive(Serialize, Deserialize)]
struct LightDto {
    name: String,
    position: [f32; 3],
    rotation: [f32; 4],
    #[serde(rename = "type", default)]
    light_type: u32, // LightType choice index
    color: [f32; 3],
    intensity: f32,
    casts_shadow: bool,
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
    // Where the .ply came from, so loading the scene can reload the cloud.
    // Empty for clouds picked via the browser file dialog (no re-readable
    // path exists); those can't be restored by a scene load.
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

/// The scene the editor opens when the user hasn't saved a startup scene of
/// their own: just the church cloud, camera left at the config's start pose.
fn default_startup_scene() -> SceneFile {
    SceneFile {
        version: SCENE_FORMAT_VERSION,
        camera: None,
        actors: Vec::new(),
        lights: Vec::new(),
        particles: Vec::new(),
        splats: vec![SplatDto {
            name: "church".to_string(),
            path: "game_assets/splats/church.ply".to_string(),
            params: SplatParamsDto::from_params(&default_splat_params()),
            position: [0.0; 3],
            rotation: ident_quat(),
            scale: ones3(),
        }],
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
fn add_menu_ui(ui: &mut egui::Ui, add: &mut Option<AddKind>) {
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
    });
    ui.menu_button("Particle System", |ui| {
        for (i, preset) in PARTICLE_PRESETS.iter().enumerate() {
            if ui.button(*preset).clicked() {
                *add = Some(AddKind::Particle(i));
            }
        }
    });
    ui.separator();
    if ui.button("Gaussian Splat…").clicked() {
        *add = Some(AddKind::Splat);
    }
}

/// A particle system placed in the scene.  The live emitter lives in the
/// renderer (keyed by `handle`); its name and transform are mirrored here so the
/// outliner, Details panel and gizmo can edit it -- transform edits are pushed
/// back with Renderer::update_particle_transform.
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
        let mut uniform = self.transform.scale.x;
        if visitor.edit_float("Scale", &mut uniform) {
            self.transform.scale = CgVec3::new(uniform, uniform, uniform);
            changed = true;
        }
        // Render params.
        changed |= visitor.edit_float("Falloff", &mut self.params.falloff);
        changed |= visitor.edit_float("Splat Scale", &mut self.params.scale);
        changed |= visitor.edit_float("Contrast", &mut self.params.contrast);
        changed |= visitor.edit_float("Overall Scale", &mut self.params.overall_scale);
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
    // Actors the editor has placed, listed in the Scene tab.  An actor
    // carries its own display name (see Actor's editor markup).
    scene_actors: Vec<Actor>,
    // Lights the editor has placed (editor data only for now -- nothing samples
    // them yet; they show an in-world icon and are editable in Details).
    scene_lights: Vec<Light>,
    // Particle systems the editor has placed.  The live emitters live in the
    // renderer; SceneParticle mirrors name + transform for the editor.
    scene_particles: Vec<SceneParticle>,
    // Preloaded particle-preset textures, as (texture path, handle), so presets
    // can be spawned synchronously from the tick (see PARTICLE_PRESETS).
    particle_textures: Vec<(String, TextureHandle)>,
    // Scene-tab selection (actor / light / particle), if any.
    selected: Option<Selection>,
    // Ctrl+click multi-selection, in click order (`selected` is its last
    // entry).  With exactly two actors selected, right-click offers snapping
    // the second to the first.
    multi_selected: Vec<Selection>,
    // Viewport context menu (right-click without dragging), at screen pos.
    context_menu: Option<egui::Pos2>,
    // Mouse travel while the right button is held, to tell a right-click
    // (context menu) from a right-drag (camera look).
    rmb_drag_accum: f32,
    // Object awaiting delete confirmation (the Scene tab's ✕ button).
    confirm_delete: Option<Selection>,
    // File > New Scene awaiting its confirmation modal (it wipes unsaved work).
    confirm_new_scene: bool,
    // Whether a user-saved startup scene exists (drives the Settings tab UI;
    // cached so native doesn't stat the config file every frame).
    has_custom_startup: bool,
    // Model highlighted in the resources browser, offered to the Details
    // panel's Model row as a one-click assignment.
    selected_resource: Option<ModelHandle>,
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
    // Monotonic counter so added objects get unique default names.
    next_object_num: u32,
    // Splat clouds that loaded, each carrying its own render params, aligned with
    // the renderer's splat indices.  Selecting one (in the outliner or by picking
    // empty space) shows its params in Details; `active_splat` is the one being
    // shown (only one renders at a time).
    scene_splats: Vec<SceneSplat>,
    active_splat: usize,
    // "Load .ply" plumbing.  The file dialog must run asynchronously (a browser
    // file input can't block the frame loop), so it drops its result here and
    // tick_frame picks it up: (file name, source path if one exists -- native
    // only, browsers don't expose one -- and the bytes).  `picker_state` keeps a
    // held key / double tap from stacking dialogs and reports read progress.
    picked_ply: Arc<Mutex<Option<(String, Option<String>, Vec<u8>)>>>,
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
                // Remember where the file lives when the platform can tell us
                // (native), so saved scenes can reload the cloud by path.
                #[cfg(not(target_arch = "wasm32"))]
                let path = Some(file.path().to_string_lossy().to_string());
                #[cfg(target_arch = "wasm32")]
                let path = None;
                *picker_state.lock().unwrap() = PickerState::Reading(name.clone());
                let bytes = file.read().await;
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

    /// Drops a fresh empty Actor into the scene ahead of the camera and selects
    /// it, so its model and transform can be set right away in Details.
    fn add_actor(&mut self, renderer: &mut Renderer) {
        let mut actor = Actor::new();
        actor.set_name(&format!("Actor {}", self.next_object_num));
        actor.set_position(&self.spawn_point());
        renderer.add_or_update_actor(&actor);

        self.next_object_num += 1;
        self.scene_actors.push(actor);
        self.select_after_add(Selection::Actor(self.scene_actors.len() - 1));
    }

    /// Drops a new light of the given type into the scene ahead of the camera.
    fn add_light(&mut self, light_type: LightType) {
        let mut light = Light::new();
        light.set_light_type(light_type);
        light.set_position(&self.spawn_point());
        self.scene_lights.push(light);
        self.select_after_add(Selection::Light(self.scene_lights.len() - 1));
    }

    /// Spawns a particle system from a preset (see PARTICLE_PRESETS) at the given
    /// transform, recording it as a scene object.  Uses the preloaded texture, so
    /// no async is needed.  Returns false if that texture wasn't preloaded.
    fn spawn_particle(
        &mut self,
        preset_name: &str,
        name: String,
        position: CgVec3,
        scale: CgVec3,
        renderer: &mut Renderer,
    ) -> bool {
        let params = preset_particle_params(preset_name);
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

    /// Spawns a preset particle system ahead of the camera and selects it (Add
    /// menu).
    fn add_particle(&mut self, preset: usize, renderer: &mut Renderer) {
        let Some(preset_name) = PARTICLE_PRESETS.get(preset).copied() else {
            return;
        };
        self.next_object_num += 1;
        let name = format!("{preset_name} {}", self.next_object_num);
        let spawn_pos = self.spawn_point();
        if self.spawn_particle(preset_name, name, spawn_pos, CG_VEC3_ONE, renderer) {
            self.select_after_add(Selection::Particle(self.scene_particles.len() - 1));
        }
    }

    /// Removes the selected object from its list and the renderer, keeping the
    /// Scene-tab selection valid.
    fn delete_selected(&mut self, sel: Selection, renderer: &mut Renderer) {
        match sel {
            Selection::Actor(i) => {
                if i >= self.scene_actors.len() {
                    return;
                }
                let actor = self.scene_actors.remove(i);
                renderer.remove_actor(&actor);
            }
            Selection::Light(i) => {
                if i >= self.scene_lights.len() {
                    return;
                }
                self.scene_lights.remove(i);
            }
            Selection::Particle(i) => {
                if i >= self.scene_particles.len() {
                    return;
                }
                let particle = self.scene_particles.remove(i);
                renderer.remove_particle_actor(&particle.handle);
            }
            // Splats have no delete button in the outliner, so this is never
            // reached; unloading a cloud from the renderer isn't supported yet.
            Selection::Splat(_) => return,
        }
        self.selected = selection_after_delete(self.selected, sel);
        // Deleting shifts indices; drop the multi-selection rather than remap.
        self.multi_selected.clear();
        self.context_menu = None;
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
            // stays a constant on-screen size.
            if !matches!(light.get_light_type(), LightType::Point) {
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
                    layer: a.get_layer().0.choice_index() as u32,
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
                    intensity: l.get_intensity(),
                    casts_shadow: l.casts_shadow(),
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
        self.scene_splats.clear();
        renderer.clear_gaussian_splats();
        self.active_splat = 0;
        self.pending_splats.lock().unwrap().clear();
        self.selected = None;
        self.multi_selected.clear();
        self.context_menu = None;
        self.confirm_delete = None;
        self.name_edit = None;
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
        let model_handle = |name: &str| -> ModelHandle {
            model_resources
                .iter()
                .find(|(n, _)| n == name)
                .map_or_else(ModelHandle::make_invalid, |(_, h)| *h)
        };
        for dto in &scene.actors {
            let mut actor = Actor::new();
            actor.set_name(&dto.name);
            actor.set_position(&arr_vec3(dto.position));
            actor.set_rotation(&arr_quat(dto.rotation));
            actor.set_scale(&arr_vec3(dto.scale));
            if let Some(model) = &dto.model {
                actor.set_model(&model_handle(model));
            }
            actor.set_layer(&SceneLayer::from_choice_index(dto.layer as usize), &None);
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
            light.set_intensity(dto.intensity);
            light.set_casts_shadow(dto.casts_shadow);
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
                    if self.scene_splats.len() == 1 {
                        self.activate_splat(0, renderer);
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
        model_resources: &[(String, ModelHandle)],
        renderer: &mut Renderer,
    ) -> Result<String, String> {
        let scene: SceneFile = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;

        self.clear_scene(renderer);
        self.apply_scene_objects(&scene, model_resources, renderer);

        let mut skipped_splats = 0;
        for dto in &scene.splats {
            if dto.path.is_empty() {
                // A cloud picked through a browser dialog: no re-readable path.
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
            scene_actors: Vec::new(),
            scene_lights: Vec::new(),
            scene_particles: Vec::new(),
            particle_textures: Vec::new(),
            selected: None,
            multi_selected: Vec::new(),
            context_menu: None,
            rmb_drag_accum: 0.0,
            confirm_delete: None,
            confirm_new_scene: false,
            has_custom_startup: editor_config::load_startup_scene().is_some(),
            selected_resource: None,
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
            resources_height: 200.0,
            next_object_num: 1,
            picked_ply: Arc::new(Mutex::new(None)),
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

        // Preload the editor's model resources (Resources tab / the Details
        // panel's model dropdown).  Done here because loading is async.
        for path in PRELOADED_MODELS {
            renderer.load_model(path, false).await;
        }

        // Preload each particle preset's texture so "Add > Particle System" can
        // spawn one synchronously from the (non-async) frame tick.
        for preset in PARTICLE_PRESETS {
            let texture_file = preset_particle_params(preset).texture_file;
            if self
                .particle_textures
                .iter()
                .any(|(path, _)| *path == texture_file)
            {
                continue; // Presets may share a texture.
            }
            let handle = renderer.preload_texture(&texture_file).await;
            self.particle_textures.push((texture_file, handle));
        }

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

        // Touch web only: on a high-DPR phone/tablet, egui's default
        // pixels-per-point (the devicePixelRatio) leaves the fixed 1280x720
        // canvas with very few layout "points" and the panels balloon -- pin
        // ppp to a 480-point-tall design space instead.  Then enlarge egui's
        // interactive sizing for finger-friendly tap targets (set on the
        // global style so dropdown popups get it too).  Desktop web is left
        // alone: the devicePixelRatio default mirrors the OS scaling that the
        // native desktop build honors, so both look the same.
        #[cfg(target_arch = "wasm32")]
        if is_touch_device() {
            ctx.set_pixels_per_point((game_config.window_height as f32 / 480.0).max(0.5));
            ctx.all_styles_mut(|s| {
                s.spacing.button_padding = egui::vec2(16.0, 12.0);
                s.spacing.interact_size.y = 38.0;
                s.spacing.item_spacing = egui::vec2(14.0, 10.0);
            });
        }

        let screen = ctx.content_rect();

        // Scene status shown on the right of the menu bar.
        let active_scene = self
            .scene_splats
            .get(self.active_splat)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "none".to_string());
        let scene_total = self.scene_splats.len().max(1);
        let splat_count = renderer.active_gaussian_splat_count();

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
                        ui.menu_button("Add", |ui| {
                            add_menu_ui(ui, &mut do_add);
                        });
                        // Splat params live in the Details panel and camera /
                        // keybindings in the right-hand Settings tab, so the old
                        // Debug / Splat / Camera / Settings menus are gone.
                        // Top-level toggle (not a dropdown) for the help text.
                        if ui.button("Help").clicked() {
                            renderer.enable_help_text();
                        }
                        // Right-aligned scene status.
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(format!(
                                "{active_scene}  ({}/{scene_total})   {splat_count} splats",
                                self.active_splat + 1
                            ));
                        });
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
                        });
                    });
                });
        }

        // Resources: a full-width bottom panel (content-browser style).
        // Closed, it collapses to a small "Resources" tab at the bottom-left
        // corner; open, the grab strip along its top edge drags to resize.
        // The right-hand editor panel stops above it (panel_bottom).
        let mut panel_bottom = screen.bottom();
        if editor {
            if self.resources_open {
                let max_height = (screen.height() * 0.7).max(120.0);
                self.resources_height = self.resources_height.clamp(120.0, max_height);
                let top_y = screen.bottom() - self.resources_height;
                panel_bottom = top_y;
                egui::Area::new(egui::Id::new("resources_panel"))
                    .fixed_pos(egui::pos2(screen.left(), top_y))
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
                                self.resources_height = (self.resources_height
                                    - strip.drag_delta().y)
                                    .clamp(120.0, max_height);
                            }
                            ui.painter().line_segment(
                                [
                                    egui::pos2(strip_rect.center().x - 24.0, strip_rect.center().y),
                                    egui::pos2(strip_rect.center().x + 24.0, strip_rect.center().y),
                                ],
                                egui::Stroke::new(2.0, ui.visuals().weak_text_color()),
                            );
                            if ui
                                .selectable_label(true, egui::RichText::new("Resources").strong())
                                .clicked()
                            {
                                self.resources_open = false;
                            }
                            ui.separator();
                            egui::ScrollArea::vertical()
                                .max_height(ui.available_height())
                                .show(ui, |ui| {
                                    ui.columns(2, |columns| {
                                        let ui = &mut columns[0];
                                        ui.label(egui::RichText::new("Models").strong());
                                        if model_resources.is_empty() {
                                            ui.label("(none)");
                                        }
                                        // Click to highlight; the Details
                                        // panel's Model row can then apply the
                                        // highlighted model with one click.
                                        for (name, handle) in &model_resources {
                                            let is_selected =
                                                self.selected_resource == Some(*handle);
                                            if ui
                                                .selectable_label(is_selected, name.as_str())
                                                .clicked()
                                            {
                                                self.selected_resource =
                                                    if is_selected { None } else { Some(*handle) };
                                            }
                                        }
                                        let ui = &mut columns[1];
                                        ui.label(egui::RichText::new("Splats").strong());
                                        if self.scene_splats.is_empty() {
                                            ui.label("(none)");
                                        }
                                        for splat in &self.scene_splats {
                                            ui.label(splat.name.as_str());
                                        }
                                    });
                                });
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

                                        let particle_names: Vec<String> = self
                                            .scene_particles
                                            .iter()
                                            .map(|p| p.name.clone())
                                            .collect();
                                        draw_outliner_section(
                                            ui,
                                            "Particles",
                                            Selection::Particle,
                                            &particle_names,
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
                                                        PARTICLE_PRESETS.iter().enumerate()
                                                    {
                                                        if ui.button(*preset).clicked() {
                                                            do_add = Some(AddKind::Particle(i));
                                                        }
                                                    }
                                                })
                                                .response
                                                .on_hover_text("Add a particle system");
                                            },
                                        );
                                    }
                                    EditorTab::Details => match self.selected {
                                        Some(Selection::Actor(i)) => {
                                            if let Some(actor) = self.scene_actors.get_mut(i) {
                                                selection_edited |= editor::draw_properties(
                                                    ui,
                                                    actor,
                                                    &model_resources,
                                                    self.selected_resource,
                                                );
                                            }
                                        }
                                        Some(Selection::Light(i)) => {
                                            if let Some(light) = self.scene_lights.get_mut(i) {
                                                selection_edited |= editor::draw_properties(
                                                    ui,
                                                    light,
                                                    &model_resources,
                                                    self.selected_resource,
                                                );
                                            }
                                        }
                                        Some(Selection::Particle(i)) => {
                                            if let Some(particle) =
                                                self.scene_particles.get_mut(i)
                                            {
                                                selection_edited |= editor::draw_properties(
                                                    ui,
                                                    particle,
                                                    &model_resources,
                                                    self.selected_resource,
                                                );
                                            }
                                        }
                                        Some(Selection::Splat(i)) => {
                                            if let Some(splat) = self.scene_splats.get_mut(i) {
                                                ui.label(
                                                    egui::RichText::new(splat.name.as_str())
                                                        .strong(),
                                                );
                                                ui.separator();
                                                selection_edited |= editor::draw_properties(
                                                    ui,
                                                    splat,
                                                    &model_resources,
                                                    self.selected_resource,
                                                );
                                            }
                                        }
                                        None => {
                                            ui.label("Nothing selected.");
                                            ui.label("Pick something in the Scene tab.");
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
            let name = match sel {
                Selection::Actor(i) => self.scene_actors.get(i).map(|a| a.get_name().to_string()),
                Selection::Light(i) => self.scene_lights.get(i).map(|l| l.get_name().to_string()),
                Selection::Particle(i) => self.scene_particles.get(i).map(|p| p.name.clone()),
                Selection::Splat(i) => self.scene_splats.get(i).map(|s| s.name.clone()),
            };
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
            select_object = Some((Selection::Light(i), false));
        }

        // Translate/rotate/scale gizmo on the selected object, drawn over the 3D
        // view.  Its edits ride the same `selection_edited` path as Details.
        // Lights have no scale (only translate/rotate apply); particles use
        // translate/scale (rotation is ignored); splats have no gizmo.
        if editor {
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
                };
                if let Some((mut position, mut rotation, mut scale)) = current {
                    if self.gizmo.ui(
                        &ctx,
                        &self.game_camera,
                        game_config,
                        &mut position,
                        &mut rotation,
                        &mut scale,
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
                        }
                        selection_edited = true;
                    }
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
        // exactly two actors are multi-selected.  Drag distance is accumulated
        // from raw mouse motion because the cursor is grabbed while looking.
        {
            let rmb = input_manager.get_key_state("mouse_right");
            if rmb.just_pressed() {
                self.rmb_drag_accum = 0.0;
            }
            if rmb.is_down() {
                let (dx, dy) = input_manager.get_mouse_raw_delta();
                self.rmb_drag_accum += dx.abs() as f32 + dy.abs() as f32;
            }
            let released = ctx
                .input(|i| i.pointer.button_released(egui::PointerButton::Secondary));
            if editor
                && released
                && self.rmb_drag_accum < 6.0
                && !ctx.egui_wants_pointer_input()
                && matches!(
                    self.multi_selected.as_slice(),
                    [Selection::Actor(_), Selection::Actor(_)]
                )
            {
                if let Some(p) = ctx.input(|i| i.pointer.interact_pos()) {
                    self.context_menu = Some(p);
                }
            }
        }

        // The context menu itself: currently one action, snapping the second
        // selected actor onto the first.
        let mut do_snap: Option<(usize, usize)> = None; // (target, source)
        if editor {
            if let Some(menu_pos) = self.context_menu {
                let pair = match self.multi_selected.as_slice() {
                    [Selection::Actor(first), Selection::Actor(second)] => self
                        .scene_actors
                        .get(*first)
                        .zip(self.scene_actors.get(*second))
                        .map(|(a, b)| {
                            (*first, *second, a.get_name().to_string(), b.get_name().to_string())
                        }),
                    _ => None,
                };
                match pair {
                    None => self.context_menu = None, // Selection changed under it.
                    Some((first, second, first_name, second_name)) => {
                        let menu = egui::Area::new(egui::Id::new("viewport_context_menu"))
                            .fixed_pos(menu_pos)
                            .constrain(true)
                            .show(&ctx, |ui| {
                                egui::Frame::menu(ui.style()).show(ui, |ui| {
                                    if ui
                                        .button(format!(
                                            "Snap \"{second_name}\" to \"{first_name}\""
                                        ))
                                        .clicked()
                                    {
                                        do_snap = Some((second, first));
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
        // Snap from the context menu: move the target actor onto the source.
        if let Some((target, source)) = do_snap {
            if target != source {
                if let Some(pos) = self.scene_actors.get(source).map(|a| a.get_position()) {
                    if let Some(actor) = self.scene_actors.get_mut(target) {
                        actor.set_position(&pos);
                        renderer.add_or_update_actor(actor);
                    }
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
        if let Some(kind) = do_add {
            match kind {
                AddKind::Actor => self.add_actor(renderer),
                AddKind::Light(light_type) => self.add_light(light_type),
                AddKind::Particle(preset) => self.add_particle(preset, renderer),
                // Splats come from a file, so "Add > Gaussian Splat" opens the
                // same .ply picker as the load button.
                AddKind::Splat => do_load = true,
            }
        }
        if let Some(sel) = delete_object {
            self.delete_selected(sel, renderer);
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
                Some(Selection::Splat(i)) => {
                    if let Some(splat) = self.scene_splats.get(i) {
                        renderer.set_gaussian_splat_params(&splat.params);
                        renderer.set_gaussian_splat_transform(&splat.transform);
                    }
                }
                // Lights are editor-only for now -- nothing to push.
                _ => {}
            }
        }
        if do_new_scene {
            self.clear_scene(renderer);
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

        // Cycle to the next loaded splat cloud ([Space]), applying its params.
        // If a splat is the current selection, keep the selection on the newly
        // shown cloud so Details follows it.
        if input_manager.get_key_state("space").just_pressed() && !self.scene_splats.is_empty() {
            let next = (self.active_splat + 1) % self.scene_splats.len();
            self.activate_splat(next, renderer);
            if matches!(self.selected, Some(Selection::Splat(_))) {
                self.selected = Some(Selection::Splat(next));
            }
        }

        // Load a user .ply ([L] or the GUI button): opens the async file picker,
        // whose result arrives via `picked_ply` on a later tick.
        if do_load || input_manager.get_key_state("l").just_pressed() {
            self.open_ply_picker();
        }
        let picked = self.picked_ply.lock().unwrap().take();
        if let Some((file_name, path, bytes)) = picked {
            let name = file_name.trim_end_matches(".ply").to_string();
            match renderer.load_gaussian_splat_from_bytes(&bytes, &name, &default_splat_params()) {
                Ok(info) => {
                    self.scene_splats.push(SceneSplat {
                        name: name.clone(),
                        // Native picks give a re-readable path (stored in saved
                        // scenes); browser picks don't.
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
        // (a quick alternative to the Details panel drags).
        let adj = delta_time * PARAM_RATE;
        if let Some(splat) = self.scene_splats.get_mut(self.active_splat) {
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
