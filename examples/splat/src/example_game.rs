use std::sync::{Arc, Mutex};

use cgmath::InnerSpace;

use kb_engine3::{
    kb_config::*, kb_engine::*, kb_game_object::*, kb_input::*, kb_renderer::*, kb_utils::*, log,
    render_groups::{kb_gaussian_splat_group::KbSplatParams, kb_ui_group::KbUiButton},
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

// Touch (mobile / iOS): the screen splits into a left "move" stick and a right
// "look" stick.  A drag of this fraction of the screen's smaller axis counts as
// full deflection; smaller drags scale proportionally.  Using a fraction (not
// fixed pixels) keeps the feel consistent across canvas resolutions.
const TOUCH_STICK_SPAN: f64 = 0.18;
const TOUCH_DEAD_ZONE: f32 = 0.12;
const TOUCH_LOOK_RATE: f32 = 70.0; // degrees/sec at full deflection
// Top strip of the screen toggles the help text when tapped, and is kept out of
// the sticks.  (Load/cycle live on real GUI buttons in the bottom-right.)
const TOUCH_UI_BAND: f64 = 0.14;

fn point_in_rect(rect: (f32, f32, f32, f32), x: f32, y: f32) -> bool {
    x >= rect.0 && x <= rect.0 + rect.2 && y >= rect.1 && y <= rect.1 + rect.3
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
    game_camera: KbCamera,
    splat_params: KbSplatParams,
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

impl KbGameEngine for SplatGame {
    fn new(game_config: &KbConfig) -> Self {
        log!("SplatGame::new()");
        let mut game_camera = KbCamera::new();
        game_camera.set_position(&game_config.start_position);
        game_camera.set_rotation(&game_config.start_rotation);
        Self {
            game_objects: Vec::new(),
            game_camera,
            splat_params: KbSplatParams {
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
        }
    }

    async fn initialize_world(
        &mut self,
        renderer: &mut KbRenderer<'_>,
        game_config: &mut KbConfig,
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

        renderer.set_camera(&self.game_camera);
    }

    fn get_game_objects(&self) -> &Vec<GameObject> {
        &self.game_objects
    }

    fn tick_frame_internal(
        &mut self,
        renderer: &mut KbRenderer,
        input_manager: &KbInputManager,
        game_config: &KbConfig,
    ) {
        let delta_time = game_config.delta_time;
        let (_view, view_dir, right_dir) = self.game_camera.calculate_view_matrix();
        let forward_dir = CgVec3::new(view_dir.x, view_dir.y, view_dir.z).normalize();

        // Movement (keyboard)
        let mut move_vec = CG_VEC3_ZERO;
        if input_manager.get_key_state("w").is_down() { move_vec += forward_dir; }
        if input_manager.get_key_state("s").is_down() { move_vec += -forward_dir; }
        if input_manager.get_key_state("d").is_down() { move_vec += right_dir; }
        if input_manager.get_key_state("a").is_down() { move_vec += -right_dir; }
        let sprint = input_manager.get_key_state("left_shift").is_down();

        // Look (keyboard)
        let mut camera_rot = self.game_camera.get_rotation();
        let rot_amount = delta_time * CAMERA_ROTATION_RATE;
        if input_manager.get_key_state("left_arrow").is_down() { camera_rot.x += rot_amount; }
        if input_manager.get_key_state("right_arrow").is_down() { camera_rot.x -= rot_amount; }
        if input_manager.get_key_state("up_arrow").is_down() { camera_rot.y -= rot_amount; }
        if input_manager.get_key_state("down_arrow").is_down() { camera_rot.y += rot_amount; }

        // Touch (mobile / iOS): left half of the screen is a move stick, right
        // half is a look stick.  Each reads the finger's offset from where it
        // first landed as an analog deflection, so a finger held still doesn't
        // drift and a drag-and-hold keeps moving/turning.  Spans are screen
        // fractions, so this behaves the same at any canvas resolution.
        let screen_w = game_config.window_width.max(1) as f64;
        let screen_h = game_config.window_height.max(1) as f64;
        let span = screen_w.min(screen_h) * TOUCH_STICK_SPAN;
        let ui_band = screen_h * TOUCH_UI_BAND;

        // GUI buttons, bottom-right: [Load .ply] above [Next splat].  Sized off
        // the smaller screen axis (clamped to finger-friendly bounds) so they
        // stay tappable on phones without dominating a desktop window.
        let btn_h = ((screen_w.min(screen_h) as f32) * 0.08).clamp(40.0, 96.0);
        let btn_w = btn_h * 3.6;
        let margin = btn_h * 0.3;
        let btn_x = screen_w as f32 - btn_w - margin;
        let cycle_rect = (btn_x, screen_h as f32 - btn_h - margin, btn_w, btn_h);
        let load_rect = (btn_x, cycle_rect.1 - btn_h - margin * 0.5, btn_w, btn_h);

        // Mouse: hover highlights, left-click triggers.
        let (mx, my) = {
            let pos = input_manager.get_mouse_position();
            (pos.0 as f32, pos.1 as f32)
        };
        let mut load_hot = point_in_rect(load_rect, mx, my);
        let mut cycle_hot = point_in_rect(cycle_rect, mx, my);
        let clicked = input_manager.get_key_state("mouse_left").just_pressed();
        let mut do_load = clicked && load_hot;
        let mut do_cycle = clicked && cycle_hot;

        for (_id, touch) in input_manager.get_touch_map().iter() {
            if !(touch.touch_state.is_down() || touch.touch_state.just_pressed()) {
                continue;
            }

            // Touches that started on a GUI button belong to it (highlight while
            // held, trigger on press) and never feed the movement/look sticks --
            // start_pos anchors them even if the finger drifts.
            let (tx, ty) = (touch.start_pos.0 as f32, touch.start_pos.1 as f32);
            if point_in_rect(load_rect, tx, ty) {
                load_hot = true;
                do_load |= touch.touch_state.just_pressed();
                continue;
            }
            if point_in_rect(cycle_rect, tx, ty) {
                cycle_hot = true;
                do_cycle |= touch.touch_state.just_pressed();
                continue;
            }

            // Top strip toggles help (tap only), kept out of the sticks.
            if touch.start_pos.1 < ui_band {
                if touch.touch_state.just_pressed() {
                    renderer.enable_help_text();
                }
                continue;
            }

            let dx = ((touch.current_pos.0 - touch.start_pos.0) / span).clamp(-1.0, 1.0) as f32;
            let dy = ((touch.current_pos.1 - touch.start_pos.1) / span).clamp(-1.0, 1.0) as f32;
            if dx * dx + dy * dy < TOUCH_DEAD_ZONE * TOUCH_DEAD_ZONE {
                continue;
            }
            if touch.start_pos.0 < screen_w * 0.5 {
                // Left stick: drag right = strafe right, drag up = move forward.
                move_vec += right_dir * dx;
                move_vec -= forward_dir * dy;
            } else {
                // Right stick: drag right = look right, drag up = look up.
                camera_rot.x -= dx * TOUCH_LOOK_RATE * delta_time;
                camera_rot.y += dy * TOUCH_LOOK_RATE * delta_time;
            }
        }

        if move_vec.magnitude2() > 0.001 {
            let speed = if sprint { CAMERA_MOVE_RATE * 3.0 } else { CAMERA_MOVE_RATE };
            let new_pos = self.game_camera.get_position()
                + move_vec.normalize() * delta_time * speed;
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
                    self.status =
                        Some((format!("Couldn't load {file_name}: {reason}"), STATUS_RED, 10.0));
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

        // Splat param adjustments
        let adj = delta_time * PARAM_RATE;
        let p = &mut self.splat_params;
        let mut changed = false;

        if input_manager.get_key_state("1").is_down() { p.falloff = (p.falloff - adj).max(0.01); changed = true; }
        if input_manager.get_key_state("2").is_down() { p.falloff = (p.falloff + adj).min(20.0); changed = true; }
        if input_manager.get_key_state("3").is_down() { p.scale = (p.scale - adj).max(0.1); changed = true; }
        if input_manager.get_key_state("4").is_down() { p.scale = (p.scale + adj).min(20.0); changed = true; }
        if input_manager.get_key_state("5").is_down() { p.contrast = (p.contrast - adj).max(0.1); changed = true; }
        if input_manager.get_key_state("6").is_down() { p.contrast = (p.contrast + adj).min(5.0); changed = true; }
        if input_manager.get_key_state("7").is_down() { p.overall_scale = (p.overall_scale - adj).max(0.1); changed = true; }
        if input_manager.get_key_state("8").is_down() { p.overall_scale = (p.overall_scale + adj).min(10.0); changed = true; }
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
        // Button styling: dark translucent fill, green accents to match the
        // debug text; brightened while hovered or held.
        let button = |label: &str, rect: (f32, f32, f32, f32), hot: bool| KbUiButton {
            label: label.to_string(),
            rect,
            background: if hot {
                [0.10, 0.35, 0.16, 0.92]
            } else {
                [0.04, 0.09, 0.06, 0.78]
            },
            border: if hot {
                [0.45, 1.0, 0.55, 0.95]
            } else {
                [0.20, 0.55, 0.30, 0.65]
            },
            text_color: [0.0, 1.0, 0.0, 1.0],
        };
        renderer.set_ui_buttons(vec![
            button("Load .ply", load_rect, load_hot),
            button("Next scene", cycle_rect, cycle_hot),
        ]);

        renderer.set_debug_game_msg(&format!(
            "Move: [W][A][S][D]   [Shift] sprint   Look: [Arrow Keys]\n\
             Touch: left half = move,  right half = look,  top strip = help\n\n\
             [Space]     Next scene  {} ({}/{})   {} splats\n\
             [L]         Load your own .ply",
            active_name, self.active_splat + 1, self.splat_names.len().max(1), splat_count,
        ));
        renderer.set_debug_topright_msg(&format!(
            "Camera\npos ({:.2}, {:.2}, {:.2})\nrot ({:.1}, {:.1})",
            pos.x, pos.y, pos.z, rot.x, rot.y,
        ));

        renderer.set_camera(&self.game_camera);
    }
}
