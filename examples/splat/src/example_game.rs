use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use black_splat::{
    egui, assets::*, config::*, editor::{self, TransformGizmo}, engine::*,
    fly_camera::*, game_object::*, input::*, renderer::*, touch_pads::*, utils::*, log,
    passes::gaussian_splat::SplatParams,
};

use crate::editor_config::{EditorConfig, GIZMO_ACTIONS};

// Splat clouds the demo preloads and cycles between with [Space].  Missing files
// are skipped at load, so the demo still runs with whatever is present.
const SPLAT_PLY_PATHS: &[&str] = &[
    "game_assets/splats/church.ply",
    "game_assets/splats/cabin.ply",
    "game_assets/splats/opera.ply",
];

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
}

/// The currently selected scene object.  The three lists (actors, lights,
/// particle systems) are kept separate, so a selection names both the list and
/// the index into it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Selection {
    Actor(usize),
    Light(usize),
    Particle(usize),
}

/// What the "Add" menu asked to create this frame (applied after the egui pass).
#[derive(Clone, Copy)]
enum AddKind {
    Actor,
    Light(LightType),
    Particle(usize), // index into PARTICLE_PRESETS
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
}

/// A particle system placed in the scene.  The live emitter lives in the
/// renderer (keyed by `handle`); its name and transform are mirrored here so the
/// outliner, Details panel and gizmo can edit it -- transform edits are pushed
/// back with Renderer::update_particle_transform.
struct SceneParticle {
    name: String,
    handle: ParticleHandle,
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
/// Rows support click-to-select, double-click-to-rename and a delete button;
/// picks are reported through the `*_out` accumulators (applied after the pass).
#[allow(clippy::too_many_arguments)]
fn draw_outliner_section(
    ui: &mut egui::Ui,
    header: &str,
    make_sel: fn(usize) -> Selection,
    names: &[String],
    selected: Option<Selection>,
    name_edit: &mut Option<Selection>,
    name_edit_buffer: &mut String,
    name_edit_focus: &mut bool,
    select_out: &mut Option<Selection>,
    rename_out: &mut Vec<(usize, String)>,
    delete_out: &mut Option<Selection>,
) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(header).strong());
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
                let label_resp = ui.selectable_label(selected == Some(this), name.as_str());
                if label_resp.clicked() {
                    *select_out = Some(this);
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
    splat_params: SplatParams,
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
    // Object awaiting delete confirmation (the Scene tab's ✕ button).
    confirm_delete: Option<Selection>,
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
    // Keybindings window (opened from the menu bar's Settings menu).
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
    // Display names of the clouds that actually loaded, aligned with the
    // renderer's splat indices; `active_splat` is the one being shown.
    splat_names: Vec<String>,
    active_splat: usize,
    // "Load .ply" plumbing.  The file dialog must run asynchronously (a browser
    // file input can't block the frame loop), so it drops its result here and
    // tick_frame picks it up.  `picker_state` keeps a held key / double tap from
    // stacking dialogs and reports read progress.
    picked_ply: Arc<Mutex<Option<(String, Vec<u8>)>>>,
    picker_state: Arc<Mutex<PickerState>>,
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
                *picked.lock().unwrap() = Some((name, bytes));
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

    /// Selects `sel` and flips to the Details tab so it can be edited right away.
    fn select_and_show_details(&mut self, sel: Selection) {
        self.selected = Some(sel);
        self.active_tab = Some(EditorTab::Details);
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
        self.select_and_show_details(Selection::Actor(self.scene_actors.len() - 1));
    }

    /// Drops a new light of the given type into the scene ahead of the camera.
    fn add_light(&mut self, light_type: LightType) {
        let mut light = Light::new();
        light.set_light_type(light_type);
        light.set_position(&self.spawn_point());
        self.scene_lights.push(light);
        self.select_and_show_details(Selection::Light(self.scene_lights.len() - 1));
    }

    /// Spawns a particle system from the given preset (see PARTICLE_PRESETS)
    /// ahead of the camera, using the preloaded texture so no async is needed.
    fn add_particle(&mut self, preset: usize, renderer: &mut Renderer) {
        let Some(preset_name) = PARTICLE_PRESETS.get(preset) else {
            return;
        };
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
            return;
        };
        let texture = *texture;
        let spawn_pos = self.spawn_point();
        let transform = ActorTransform::from_position(spawn_pos);
        let handle = renderer.spawn_particle_actor(&transform, &params, &texture, true);

        self.next_object_num += 1;
        self.scene_particles.push(SceneParticle {
            name: format!("{preset_name} {}", self.next_object_num),
            handle,
            position: spawn_pos,
            scale: CG_VEC3_ONE,
        });
        self.select_and_show_details(Selection::Particle(self.scene_particles.len() - 1));
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
        }
        self.selected = selection_after_delete(self.selected, sel);
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
            splat_params: SplatParams {
                falloff: 4.65,
                scale: 3.0,
                contrast: 1.0,
                max_sh_degree: 2.0,
                overall_scale: 1.0,
            },
            splat_names: Vec::new(),
            active_splat: 0,
            scene_actors: Vec::new(),
            scene_lights: Vec::new(),
            scene_particles: Vec::new(),
            particle_textures: Vec::new(),
            selected: None,
            confirm_delete: None,
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

        // Preload every available cloud; record the names that loaded so the
        // cycle hotkey and HUD only reference real ones.
        for path in SPLAT_PLY_PATHS {
            if renderer.load_gaussian_splat(path, &self.splat_params).await {
                let name = path
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or(path)
                    .trim_end_matches(".ply")
                    .to_string();
                self.splat_names.push(name);
            }
        }
        renderer.set_active_gaussian_splat(0);
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
        let mut move_vec = self.fly_camera.wasd_direction(&self.game_camera, input_manager);
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
            .splat_names
            .get(self.active_splat)
            .cloned()
            .unwrap_or_else(|| "none".to_string());
        let scene_total = self.splat_names.len().max(1);
        let splat_count = renderer.active_gaussian_splat_count();

        let mut do_load = false;
        let mut do_cycle = false;
        let mut params_changed = false;
        // Editor actions collected from the menus / Scene tab this frame and
        // applied after the egui pass (avoids borrowing self inside closures).
        let mut do_save_scene = false;
        let mut do_load_scene = false;
        let mut do_add: Option<AddKind> = None;
        let mut delete_object: Option<Selection> = None;
        let mut select_object: Option<Selection> = None;
        let mut select_splat: Option<usize> = None;
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
                            do_save_scene |= ui.button("Save Scene…").clicked();
                            do_load_scene |= ui.button("Load Scene…").clicked();
                            ui.separator();
                            do_load |= ui.button("Load .ply…").clicked();
                        });
                        ui.menu_button("Add", |ui| {
                            add_menu_ui(ui, &mut do_add);
                        });
                        ui.menu_button("Debug", |ui| {
                            do_cycle |= ui.button("Next scene").clicked();
                        });
                        ui.menu_button("Splat", |ui| {
                            // Sliders keep the menu open while dragged.
                            ui.set_min_width(240.0);
                            ui.spacing_mut().slider_width = 150.0;
                            let p = &mut self.splat_params;
                            params_changed |= ui
                                .add(egui::Slider::new(&mut p.falloff, 0.01..=20.0).text("falloff"))
                                .changed();
                            params_changed |= ui
                                .add(
                                    egui::Slider::new(&mut p.scale, 0.1..=20.0).text("splat scale"),
                                )
                                .changed();
                            params_changed |= ui
                                .add(egui::Slider::new(&mut p.contrast, 0.1..=5.0).text("contrast"))
                                .changed();
                            params_changed |= ui
                                .add(
                                    egui::Slider::new(&mut p.overall_scale, 0.1..=10.0)
                                        .text("overall scale"),
                                )
                                .changed();
                            // The splat record carries 8 "rest" coefficients (degrees
                            // 1+2), so 2 is the highest degree that changes anything.
                            let mut sh = p.max_sh_degree as u32;
                            if ui
                                .add(egui::Slider::new(&mut sh, 0..=2).text("SH degree"))
                                .changed()
                            {
                                p.max_sh_degree = sh as f32;
                                params_changed = true;
                            }
                        });
                        ui.menu_button("Settings", |ui| {
                            if ui.button("Keybindings…").clicked() {
                                self.show_settings = true;
                            }
                        });
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
                                        if self.splat_names.is_empty() {
                                            ui.label("(none)");
                                        }
                                        for name in &self.splat_names {
                                            ui.label(name.as_str());
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
                                        if self.splat_names.is_empty() {
                                            ui.label("(none)");
                                        }
                                        for (i, name) in self.splat_names.iter().enumerate() {
                                            let is_active = i == self.active_splat
                                                && self.selected.is_none();
                                            if ui.selectable_label(is_active, name).clicked() {
                                                select_splat = Some(i);
                                            }
                                        }

                                        // Unified "Add" menu (actor / light / particle).
                                        ui.add_space(8.0);
                                        ui.menu_button("➕ Add", |ui| add_menu_ui(ui, &mut do_add));

                                        // One outliner section per object kind.  Names are
                                        // collected first so the lists aren't borrowed while
                                        // the section mutates self.name_edit etc.
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
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_actors,
                                            &mut self.confirm_delete,
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
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_lights,
                                            &mut self.confirm_delete,
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
                                            &mut self.name_edit,
                                            &mut self.name_edit_buffer,
                                            &mut self.name_edit_focus,
                                            &mut select_object,
                                            &mut rename_particles,
                                            &mut self.confirm_delete,
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
                                        None => {
                                            ui.label("Nothing selected.");
                                            ui.label("Pick something in the Scene tab.");
                                        }
                                    },
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

        // In-world editor icons for lights (clicking one selects it).
        if editor {
            if let Some(i) = self.draw_light_icons(&ctx, game_config) {
                select_object = Some(Selection::Light(i));
            }
        }

        // Translate/rotate/scale gizmo on the selected object, drawn over the 3D
        // view.  Its edits ride the same `selection_edited` path as Details.
        // Lights have no scale (only translate/rotate apply); particles use
        // translate/scale (rotation is ignored).
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
                        }
                        selection_edited = true;
                    }
                }
            }
        }

        // Apply the editor actions gathered from the menus / Scene tab.
        if let Some(sel) = select_splat {
            if sel < self.splat_names.len() {
                self.active_splat = sel;
                renderer.set_active_gaussian_splat(sel);
            }
            self.selected = None;
        }
        if let Some(sel) = select_object {
            self.selected = Some(sel);
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
                // Lights are editor-only for now -- nothing to push.
                _ => {}
            }
        }
        if do_save_scene {
            self.status = Some(("Save Scene: not implemented yet".to_string(), STATUS_WHITE, 4.0));
        }
        if do_load_scene {
            self.status = Some(("Load Scene: not implemented yet".to_string(), STATUS_WHITE, 4.0));
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

        // Cycle to the next loaded splat cloud ([Space] or the GUI button).
        if (do_cycle || input_manager.get_key_state("space").just_pressed())
            && !self.splat_names.is_empty()
        {
            self.active_splat = (self.active_splat + 1) % self.splat_names.len();
            renderer.set_active_gaussian_splat(self.active_splat);
        }

        // Load a user .ply ([L] or the GUI button): opens the async file picker,
        // whose result arrives via `picked_ply` on a later tick.
        if do_load || input_manager.get_key_state("l").just_pressed() {
            self.open_ply_picker();
        }
        let picked = self.picked_ply.lock().unwrap().take();
        if let Some((file_name, bytes)) = picked {
            let name = file_name.trim_end_matches(".ply").to_string();
            match renderer.load_gaussian_splat_from_bytes(&bytes, &name, &self.splat_params) {
                Ok(info) => {
                    self.splat_names.push(name.clone());
                    self.active_splat = self.splat_names.len() - 1;
                    renderer.set_active_gaussian_splat(self.active_splat);
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

        // Splat param adjustments (keyboard mirrors of the sliders)
        let adj = delta_time * PARAM_RATE;
        let p = &mut self.splat_params;
        let mut changed = params_changed;

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
            // The splat record carries 8 "rest" coefficients (degrees 1+2), so 2
            // is the highest degree that changes anything.
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

        let pos = self.game_camera.get_position();
        let rot = self.game_camera.get_rotation();
        let active_name = self
            .splat_names
            .get(self.active_splat)
            .map_or("none", |s| s.as_str());
        let splat_count = renderer.active_gaussian_splat_count();
        renderer.set_debug_game_msg(&format!(
            "Move: [W][A][S][D]   [Shift] sprint   Look: [Arrow Keys]\n\
             Touch: left pad = move,  right pad = look\n\n\
             [Space]     Next scene  {} ({}/{})   {} splats\n\
             [L]         Load your own .ply",
            active_name,
            self.active_splat + 1,
            self.splat_names.len().max(1),
            splat_count,
        ));
        renderer.set_debug_topright_msg(&format!(
            "Camera\npos ({:.2}, {:.2}, {:.2})\nrot ({:.1}, {:.1})",
            pos.x, pos.y, pos.z, rot.x, rot.y,
        ));

        renderer.set_camera(&self.game_camera);
    }
}
