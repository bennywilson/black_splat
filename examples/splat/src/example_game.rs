use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use black_splat::{
    egui, config::*, engine::*, game_object::*, input::*, renderer::*, utils::*,
    log, passes::gaussian_splat::SplatParams,
};

// Splat clouds the demo preloads and cycles between with [Space].  Missing files
// are skipped at load, so the demo still runs with whatever is present.
const SPLAT_PLY_PATHS: &[&str] = &[
    "game_assets/splats/church.ply",
    "game_assets/splats/cabin.ply",
    "game_assets/splats/opera.ply",
];

const CAMERA_MOVE_RATE: f32 = 1.0;
const CAMERA_ROTATION_RATE: f32 = 30.0;
const PARAM_RATE: f32 = 1.5;

// On-screen touch pads (drawn with egui, fed by raw touches so both work at
// once): left pad = move, right pad = look.  Sizes are fractions of the
// shorter screen axis, in egui points, so they scale with any resolution.
const PAD_RADIUS_FRAC: f32 = 0.16;
const PAD_MARGIN_FRAC: f32 = 0.05;
// Fraction of the pad radius that counts as full deflection.
const PAD_SPAN_FRAC: f32 = 0.75;
const PAD_DEAD_ZONE: f32 = 0.12;
const PAD_LOOK_RATE: f32 = 70.0; // degrees/sec at full deflection

// Right-click + drag look: camera degrees turned per pixel of mouse movement.
const MOUSE_LOOK_SENS: f32 = 0.18;


/// Deflection of one touch pad: finger offset from the pad center, saturating
/// at `span` and zeroed inside the dead zone.
fn pad_deflection(finger: egui::Pos2, center: egui::Pos2, span: f32) -> egui::Vec2 {
    let mut defl = (finger - center) / span;
    let len = defl.length();
    if len < PAD_DEAD_ZONE {
        return egui::Vec2::ZERO;
    }
    if len > 1.0 {
        defl /= len;
    }
    defl
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
    // The move/look pads draw once the first touch proves this is a touch
    // device (there's no reliable "has touchscreen" query through winit).
    touch_pads_visible: bool,
    // True while the right button is held for mouse look.  Drives grabbing +
    // hiding the cursor on the first frame and restoring it once on release.
    looking: bool,
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
            picked_ply: Arc::new(Mutex::new(None)),
            picker_state: Arc::new(Mutex::new(PickerState::Idle)),
            status: None,
            touch_pads_visible: false,
            looking: false,
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
        let (_view, view_dir, right_dir) = self.game_camera.calculate_view_matrix();
        let forward_dir = CgVec3::new(view_dir.x, view_dir.y, view_dir.z).normalize();

        // Movement (keyboard)
        let mut move_vec = CG_VEC3_ZERO;
        if input_manager.get_key_state("w").is_down() {
            move_vec += forward_dir;
        }
        if input_manager.get_key_state("s").is_down() {
            move_vec += -forward_dir;
        }
        if input_manager.get_key_state("d").is_down() {
            move_vec += right_dir;
        }
        if input_manager.get_key_state("a").is_down() {
            move_vec += -right_dir;
        }
        let sprint = input_manager.get_key_state("left_shift").is_down();

        // Look (keyboard)
        let mut camera_rot = self.game_camera.get_rotation();
        let rot_amount = delta_time * CAMERA_ROTATION_RATE;
        if input_manager.get_key_state("left_arrow").is_down() {
            camera_rot.x += rot_amount;
        }
        if input_manager.get_key_state("right_arrow").is_down() {
            camera_rot.x -= rot_amount;
        }
        if input_manager.get_key_state("up_arrow").is_down() {
            camera_rot.y -= rot_amount;
        }
        if input_manager.get_key_state("down_arrow").is_down() {
            camera_rot.y += rot_amount;
        }

        // Look (mouse): hold the right button to look.  While held the cursor
        // is hidden and grabbed to the window so it can't slide off-screen, and
        // the camera turns from RAW mouse motion -- position-independent, so you
        // can sweep as far as you like without the pointer hitting an edge.
        // Matches the touch look pad: move right = look right, move up = look up.
        let rmb = input_manager.get_key_state("mouse_right");
        if rmb.is_down() || rmb.just_pressed() {
            if !self.looking {
                self.looking = true;
                renderer.set_cursor_visible(false);
                renderer.set_cursor_grabbed(true);
            }
            let (dx, dy) = input_manager.get_mouse_raw_delta();
            camera_rot.x -= dx as f32 * MOUSE_LOOK_SENS;
            camera_rot.y += dy as f32 * MOUSE_LOOK_SENS;
        } else if self.looking {
            self.looking = false;
            renderer.set_cursor_grabbed(false);
            renderer.set_cursor_visible(true);
        }

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
                            do_load |= ui.button("Load .ply…").clicked();
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

        // Touch pads.  egui's pointer only tracks one touch, so the pads read
        // the engine's raw touch map -- move and look then work simultaneously
        // -- and use egui purely as the painter.
        let touch_map = input_manager.get_touch_map();
        if !touch_map.is_empty() {
            self.touch_pads_visible = true;
        }
        if self.touch_pads_visible {
            let ppp = ctx.pixels_per_point();
            let min_axis = screen.width().min(screen.height());
            let radius = min_axis * PAD_RADIUS_FRAC;
            let margin = min_axis * PAD_MARGIN_FRAC;
            let span = radius * PAD_SPAN_FRAC;
            let move_center = egui::pos2(
                screen.left() + margin + radius,
                screen.bottom() - margin - radius,
            );
            let look_center = egui::pos2(
                screen.right() - margin - radius,
                screen.bottom() - margin - radius,
            );

            // A touch belongs to the pad it STARTED on, so a held drag can
            // wander outside the circle without hopping pads.
            let mut move_defl = egui::Vec2::ZERO;
            let mut look_defl = egui::Vec2::ZERO;
            for (_id, touch) in touch_map.iter() {
                if !(touch.touch_state.is_down() || touch.touch_state.just_pressed()) {
                    continue;
                }
                let start = egui::pos2(
                    touch.start_pos.0 as f32 / ppp,
                    touch.start_pos.1 as f32 / ppp,
                );
                let finger = egui::pos2(
                    touch.current_pos.0 as f32 / ppp,
                    touch.current_pos.1 as f32 / ppp,
                );
                if start.distance(move_center) <= radius {
                    move_defl = pad_deflection(finger, move_center, span);
                } else if start.distance(look_center) <= radius {
                    look_defl = pad_deflection(finger, look_center, span);
                }
            }

            // Left pad: drag right = strafe right, drag up = move forward.
            // Right pad: drag right = look right, drag up = look up.
            move_vec += right_dir * move_defl.x - forward_dir * move_defl.y;
            camera_rot.x -= look_defl.x * PAD_LOOK_RATE * delta_time;
            camera_rot.y += look_defl.y * PAD_LOOK_RATE * delta_time;

            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("touch_pads"),
            ));
            for (center, defl) in [(move_center, move_defl), (look_center, look_defl)] {
                painter.circle(
                    center,
                    radius,
                    egui::Color32::from_rgba_unmultiplied(10, 23, 15, 110),
                    egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(64, 160, 90, 150)),
                );
                painter.circle(
                    center + defl * span,
                    radius * 0.35,
                    egui::Color32::from_rgba_unmultiplied(38, 115, 57, 200),
                    egui::Stroke::new(
                        2.0,
                        egui::Color32::from_rgba_unmultiplied(120, 255, 145, 220),
                    ),
                );
            }
        }

        if move_vec.magnitude2() > 0.001 {
            let speed = if sprint {
                CAMERA_MOVE_RATE * 3.0
            } else {
                CAMERA_MOVE_RATE
            };
            let new_pos =
                self.game_camera.get_position() + move_vec.normalize() * delta_time * speed;
            self.game_camera.set_position(&new_pos);
        }

        camera_rot.y = camera_rot.y.clamp(-89.0, 89.0);
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
