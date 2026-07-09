use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use black_splat::{
    egui, assets::*, config::*, engine::*, fly_camera::*, game_object::*, input::*, renderer::*,
    touch_pads::*, utils::*, log, passes::gaussian_splat::SplatParams,
};

// Splat clouds the demo preloads and cycles between with [Space].  Missing files
// are skipped at load, so the demo still runs with whatever is present.
const SPLAT_PLY_PATHS: &[&str] = &[
    "game_assets/splats/church.ply",
    "game_assets/splats/cabin.ply",
    "game_assets/splats/opera.ply",
];

// Models the editor's "Add" menu can drop into the scene, as (display name,
// path) pairs.  These are preloaded in initialize_world -- model loading is
// async and the frame tick isn't -- so the menu can instance them synchronously.
const MODEL_PALETTE: &[(&str, &str)] = &[
    ("Barrel", "game_assets/models/barrel.glb"),
    ("Shotgun", "game_assets/models/shotgun.glb"),
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

/// A model-backed game object the editor placed in the scene.  Owns its
/// renderable `Actor` (transform + model) and the display name shown in the
/// outliner.
struct SceneObject {
    name: String,
    actor: Actor,
}

pub struct SplatGame {
    game_objects: Vec<GameObject>,
    game_camera: Camera,
    splat_params: SplatParams,
    // Models the "Add" menu can instance, loaded once in initialize_world.
    // Aligned with MODEL_PALETTE.
    model_palette: Vec<(String, ModelHandle)>,
    // Game objects the editor has placed, listed in the outliner.
    scene_objects: Vec<SceneObject>,
    // Outliner selection: index into `scene_objects`, if any.
    selected_object: Option<usize>,
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

    /// Drops a fresh instance of palette model `palette_index` into the scene a
    /// few units ahead of the camera, registers it with the renderer, and
    /// selects it in the outliner.
    fn add_model_object(&mut self, palette_index: usize, renderer: &mut Renderer) {
        let (model_name, model_handle) = match self.model_palette.get(palette_index) {
            Some((name, handle)) => (name.clone(), *handle),
            None => return,
        };

        let (_view, view_dir, _right) = self.game_camera.calculate_view_matrix();
        let spawn_pos = self.game_camera.get_position() + view_dir * ADD_OBJECT_DISTANCE;

        let mut actor = Actor::new();
        actor.set_model(&model_handle);
        actor.set_position(&spawn_pos);
        renderer.add_or_update_actor(&actor);

        let name = format!("{model_name} {}", self.next_object_num);
        self.next_object_num += 1;
        self.scene_objects.push(SceneObject { name, actor });
        self.selected_object = Some(self.scene_objects.len() - 1);
    }

    /// Removes the scene object at `index` from both the scene list and the
    /// renderer, keeping the outliner selection valid.
    fn delete_scene_object(&mut self, index: usize, renderer: &mut Renderer) {
        if index >= self.scene_objects.len() {
            return;
        }
        let object = self.scene_objects.remove(index);
        renderer.remove_actor(&object.actor);
        self.selected_object = match self.selected_object {
            Some(sel) if sel == index => None,
            Some(sel) if sel > index => Some(sel - 1),
            other => other,
        };
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
            model_palette: Vec::new(),
            scene_objects: Vec::new(),
            selected_object: None,
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

        // Preload the models the editor's "Add" menu can instance.  Done here
        // (async) so placing them from the frame tick is a synchronous clone.
        for (name, path) in MODEL_PALETTE {
            let handle = renderer.load_model(path, false).await;
            self.model_palette.push((name.to_string(), handle));
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

        // On web, egui's pixels-per-point follows the browser devicePixelRatio,
        // which on a high-DPR display left the fixed 1280x720 canvas with very
        // few layout "points" -- the panels ballooned.  Pin ppp to the surface
        // instead.  A 480-point-tall design space gives ppp ~1.5, matching the
        // native desktop look (which was fine) so the GUI reads at a comfortable
        // size instead of tiny.  Native is left alone (honors OS scaling).
        #[cfg(target_arch = "wasm32")]
        {
            ctx.set_pixels_per_point((game_config.window_height as f32 / 480.0).max(0.5));
            // Finger-friendly tap targets (iOS especially): enlarge egui's
            // interactive sizing so menu items, dropdown entries and sliders are
            // easy to tap.  Set on the global style so dropdown popups -- which
            // don't inherit a per-`ui` spacing tweak -- get it too.  Desktop web
            // gets slightly larger controls as well, which is fine.
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
        // Editor actions collected from the menus / outliner this frame and
        // applied after the egui pass (avoids borrowing self inside closures).
        let mut do_save_scene = false;
        let mut do_load_scene = false;
        let mut add_model_index: Option<usize> = None;
        let mut delete_object_index: Option<usize> = None;
        let mut select_object: Option<Option<usize>> = None;
        let mut select_splat: Option<usize> = None;

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
                            if self.model_palette.is_empty() {
                                ui.label("(no models loaded)");
                            }
                            for (i, (name, _)) in self.model_palette.iter().enumerate() {
                                if ui.button(name).clicked() {
                                    add_model_index = Some(i);
                                }
                            }
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

        // Outliner: a right-anchored panel listing every game object in the
        // scene -- the loaded splat clouds and the models the editor has placed.
        // Editor mode only; game mode keeps the view unobstructed.  Drawn as an
        // Area (same as the menu bar) so it sits over the 3D view.  Its widgets
        // only set the local action flags above; they're applied after the egui
        // pass so the closure never has to borrow `self`/`renderer` mutably.
        const OUTLINER_WIDTH: f32 = 220.0;
        if editor {
            let top = menu_bar.response.rect.bottom();
            egui::Area::new(egui::Id::new("outliner"))
                .fixed_pos(egui::pos2(screen.right() - OUTLINER_WIDTH, top))
                .constrain(true)
                .show(&ctx, |ui| {
                    egui::Frame::side_top_panel(ui.style()).show(ui, |ui| {
                        ui.set_width(OUTLINER_WIDTH);
                        ui.heading("Outliner");
                        ui.separator();
                        egui::ScrollArea::vertical()
                            .max_height((screen.bottom() - top - 40.0).max(80.0))
                            .show(ui, |ui| {
                                ui.set_width(OUTLINER_WIDTH);
                                ui.label(egui::RichText::new("Splats").strong());
                                if self.splat_names.is_empty() {
                                    ui.label("(none)");
                                }
                                for (i, name) in self.splat_names.iter().enumerate() {
                                    let is_active =
                                        i == self.active_splat && self.selected_object.is_none();
                                    if ui.selectable_label(is_active, name).clicked() {
                                        select_splat = Some(i);
                                    }
                                }

                                ui.add_space(8.0);
                                ui.label(egui::RichText::new("Models").strong());
                                if self.scene_objects.is_empty() {
                                    ui.label("(none)");
                                }
                                for (i, object) in self.scene_objects.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        let selected = self.selected_object == Some(i);
                                        if ui.selectable_label(selected, &object.name).clicked() {
                                            select_object = Some(Some(i));
                                        }
                                        if ui.small_button("✕").clicked() {
                                            delete_object_index = Some(i);
                                        }
                                    });
                                }
                            });
                    });
                });
        }

        // Apply the editor actions gathered from the menus / outliner.
        if let Some(sel) = select_splat {
            if sel < self.splat_names.len() {
                self.active_splat = sel;
                renderer.set_active_gaussian_splat(sel);
            }
            self.selected_object = None;
        }
        if let Some(sel) = select_object {
            self.selected_object = sel;
        }
        if let Some(i) = add_model_index {
            self.add_model_object(i, renderer);
        }
        if let Some(i) = delete_object_index {
            self.delete_scene_object(i, renderer);
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
