use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use black_splat::{
    egui, assets::*, config::*, editor::{self, GizmoMode, TransformGizmo}, engine::*,
    fly_camera::*, game_object::*, input::*, renderer::*, touch_pads::*, utils::*, log,
    passes::gaussian_splat::SplatParams,
};

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
    // Scene-tab selection: index into `scene_actors`, if any.
    selected_object: Option<usize>,
    // Actor index awaiting delete confirmation (the Scene tab's ✕ button).
    confirm_delete: Option<usize>,
    // Model highlighted in the resources browser, offered to the Details
    // panel's Model row as a one-click assignment.
    selected_resource: Option<ModelHandle>,
    // Actor index currently being renamed (double-click in the Scene tab).
    name_edit: Option<usize>,
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

    /// Drops a fresh empty Actor into the scene a few units ahead of the
    /// camera and selects it, flipping to the Details tab so its model and
    /// transform can be set right away.
    fn add_actor(&mut self, renderer: &mut Renderer) {
        let (_view, view_dir, _right) = self.game_camera.calculate_view_matrix();
        let spawn_pos = self.game_camera.get_position() + view_dir * ADD_OBJECT_DISTANCE;

        let mut actor = Actor::new();
        actor.set_name(&format!("Actor {}", self.next_object_num));
        actor.set_position(&spawn_pos);
        renderer.add_or_update_actor(&actor);

        self.next_object_num += 1;
        self.scene_actors.push(actor);
        self.selected_object = Some(self.scene_actors.len() - 1);
    }

    /// Removes the scene actor at `index` from both the scene list and the
    /// renderer, keeping the Scene-tab selection valid.
    fn delete_scene_object(&mut self, index: usize, renderer: &mut Renderer) {
        if index >= self.scene_actors.len() {
            return;
        }
        let actor = self.scene_actors.remove(index);
        renderer.remove_actor(&actor);
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
            scene_actors: Vec::new(),
            selected_object: None,
            confirm_delete: None,
            selected_resource: None,
            name_edit: None,
            name_edit_buffer: String::new(),
            name_edit_focus: false,
            gizmo: TransformGizmo::default(),
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
        let mut do_add_actor = false;
        let mut delete_object_index: Option<usize> = None;
        let mut select_object: Option<Option<usize>> = None;
        let mut select_splat: Option<usize> = None;
        // True when the Details panel changed the selected actor this frame;
        // the renderer's copy is refreshed after the egui pass.
        let mut actor_edited = false;

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
                            // Just empty actors for now; a model is assigned
                            // afterwards in the Details panel.
                            do_add_actor |= ui.button("Actor").clicked();
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

        // Right-hand editor panel, tabbed: Scene (splats + actors), Details
        // (selected actor's properties), Resources (loaded assets).  Editor mode
        // only; game mode keeps the view unobstructed.  Drawn as an Area (same
        // as the menu bar) so it sits over the 3D view.  Collapsed to just the
        // tab strip until a tab is clicked; clicking the active tab collapses
        // it again.  Open, it runs the full height of the view like a docked
        // sidebar.  Selection/deletion set the local action flags above and are
        // applied after the egui pass; Details edits mutate the actor directly
        // and set `actor_edited` so the renderer's copy is refreshed afterwards.
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
                            for (mode, label) in [
                                (GizmoMode::Translate, "Move"),
                                (GizmoMode::Rotate, "Rotate"),
                                (GizmoMode::Scale, "Scale"),
                            ] {
                                if ui.selectable_label(self.gizmo.mode == mode, label).clicked()
                                {
                                    self.gizmo.mode = mode;
                                }
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
                                                && self.selected_object.is_none();
                                            if ui.selectable_label(is_active, name).clicked() {
                                                select_splat = Some(i);
                                            }
                                        }

                                        ui.add_space(8.0);
                                        ui.horizontal(|ui| {
                                            ui.label(egui::RichText::new("Actors").strong());
                                            if ui
                                                .small_button("+")
                                                .on_hover_text("Add an actor")
                                                .clicked()
                                            {
                                                do_add_actor = true;
                                            }
                                        });
                                        if self.scene_actors.is_empty() {
                                            ui.label("(none)");
                                        }
                                        // Accumulate names to update after the UI pass
                                        // (no mutable borrow of scene_actors while iterating).
                                        let mut updated_names = Vec::new();
                                        for (i, actor) in self.scene_actors.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                let selected = self.selected_object == Some(i);
                                                let is_editing = self.name_edit == Some(i);
                                                if is_editing {
                                                    // Inline text edit: save on Enter/blur,
                                                    // cancel on Escape.  The buffer lives in
                                                    // self so it survives across frames.
                                                    let edit_resp = ui.text_edit_singleline(
                                                        &mut self.name_edit_buffer,
                                                    );
                                                    // Focus the field the frame it appears.
                                                    if self.name_edit_focus {
                                                        edit_resp.request_focus();
                                                        self.name_edit_focus = false;
                                                    }
                                                    let finish_edit = edit_resp.lost_focus()
                                                        || ui.input(|i| {
                                                            i.key_pressed(egui::Key::Enter)
                                                        });
                                                    let cancel_edit =
                                                        ui.input(|i| i.key_pressed(egui::Key::Escape));
                                                    if finish_edit || cancel_edit {
                                                        let new_name =
                                                            self.name_edit_buffer.trim();
                                                        if finish_edit && !new_name.is_empty() {
                                                            updated_names
                                                                .push((i, new_name.to_string()));
                                                        }
                                                        self.name_edit = None;
                                                    }
                                                } else {
                                                    let label_resp = ui.selectable_label(
                                                        selected,
                                                        actor.get_name(),
                                                    );
                                                    if label_resp.clicked() {
                                                        select_object = Some(Some(i));
                                                    }
                                                    if label_resp.double_clicked() {
                                                        self.name_edit = Some(i);
                                                        self.name_edit_focus = true;
                                                        self.name_edit_buffer =
                                                            actor.get_name().to_string();
                                                    }
                                                }
                                                // Deletion asks first (the
                                                // modal below).
                                                if ui
                                                    .small_button(
                                                        egui::RichText::new("✕").color(
                                                            egui::Color32::from_rgb(235, 80, 80),
                                                        ),
                                                    )
                                                    .clicked()
                                                {
                                                    self.confirm_delete = Some(i);
                                                    self.name_edit = None; // Cancel any edit.
                                                }
                                            });
                                        }
                                        // Apply name updates.
                                        for (i, new_name) in updated_names {
                                            if let Some(actor) = self.scene_actors.get_mut(i) {
                                                actor.set_name(&new_name);
                                            }
                                        }
                                    }
                                    EditorTab::Details => match self.selected_object {
                                        Some(index) => {
                                            actor_edited |= editor::draw_properties(
                                                ui,
                                                &mut self.scene_actors[index],
                                                &model_resources,
                                                self.selected_resource,
                                            );
                                        }
                                        None => {
                                            ui.label("Nothing selected.");
                                            ui.label("Pick an actor in the Scene tab.");
                                        }
                                    },
                                }
                            });
                    });
                });
        }

        // Delete confirmation for the Scene tab's ✕ button.  A modal blocks
        // the rest of the UI until answered; clicking the backdrop cancels.
        if let Some(index) = self.confirm_delete {
            match self.scene_actors.get(index) {
                None => self.confirm_delete = None, // Stale index.
                Some(actor) => {
                    let name = actor.get_name().to_string();
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
                                    delete_object_index = Some(index);
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

        // Translate/rotate/scale gizmo on the selected actor, drawn over the
        // 3D view.  Its edits ride the same actor_edited path as the Details
        // panel's.
        if editor {
            if let Some(actor) = self
                .selected_object
                .and_then(|i| self.scene_actors.get_mut(i))
            {
                let mut position = actor.get_position();
                let mut rotation = actor.get_rotation();
                let mut scale = actor.get_scale();
                if self.gizmo.ui(
                    &ctx,
                    &self.game_camera,
                    game_config,
                    &mut position,
                    &mut rotation,
                    &mut scale,
                ) {
                    actor.set_position(&position);
                    actor.set_rotation(&rotation);
                    actor.set_scale(&scale);
                    actor_edited = true;
                }
            }
        }

        // Apply the editor actions gathered from the menus / Scene tab.
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
        if do_add_actor {
            self.add_actor(renderer);
        }
        if let Some(i) = delete_object_index {
            self.delete_scene_object(i, renderer);
        }
        // Push Details-panel / gizmo edits to the renderer's copy of the actor.
        if actor_edited {
            if let Some(actor) = self.selected_object.and_then(|i| self.scene_actors.get(i)) {
                renderer.add_or_update_actor(actor);
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
