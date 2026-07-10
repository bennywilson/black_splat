//! A reusable free-fly / walk camera controller.
//!
//! Reads the engine's [`InputManager`] and moves a [`Camera`]. It is split into
//! small, composable steps rather than a single `update()` so a game can slot
//! its own logic -- collision, arena clamping, custom touch controls -- between
//! reading input and committing the result. For example the splat viewer applies
//! [`wasd_direction`](FlyCamera::wasd_direction) straight to the camera, while
//! the 3D game feeds it through a collision raycast and arena clamp first.
//!
//! A [`Camera`]'s rotation is `(yaw = rotation.x, pitch = rotation.y)` in
//! degrees, matching [`Camera::calculate_view_matrix`]; the look helpers here
//! operate on that convention.
//!
//! The config fields are public -- construct with [`FlyCamera::default`] (whose
//! defaults match a fly-through viewer) and tweak the fields you need.

use cgmath::InnerSpace;

use crate::{game_object::Camera, input::InputManager, renderer::Renderer, utils::*};

/// Controls a [`Camera`] from keyboard, mouse and (via the caller) touch input.
///
/// All fields are configuration except the private mouse-look state; build one
/// with [`FlyCamera::default`] and adjust fields as needed.
#[derive(Clone)]
pub struct FlyCamera {
    /// Base movement speed, in world units per second.
    pub move_rate: f32,
    /// Multiplier applied to [`move_rate`](Self::move_rate) while the shift key
    /// is held.
    pub shift_move_multiplier: f32,

    /// [todo]: remove
    /// Arrow-key look speed, in degrees per second.
    pub key_look_rate: f32,
    /// Right-drag mouse-look sensitivity, in degrees per pixel of raw motion.
    pub mouse_look_sensitivity: f32,
    /// When true, movement is flattened onto the ground plane (you walk); when
    /// false the camera flies along its full view direction.
    pub walk_on_plane: bool,
    /// Invert the pitch (up/down) look direction for both keys and mouse.
    pub invert_pitch: bool,
    /// Inclusive pitch clamp range, in degrees, applied by [`clamp_pitch`](Self::clamp_pitch).
    /// [todo] - use a range here?
    pub pitch_min: f32,
    pub pitch_max: f32,

    /// True while a right-drag look is engaged: drives the one-shot cursor
    /// grab/hide when the drag starts and the restore on release.
    looking: bool,
    /// Raw mouse travel accumulated while the right button is held, to tell a
    /// right-click from a look-drag (see [`LOOK_DRAG_THRESHOLD`]).
    look_drag_accum: f64,
}

/// Raw mouse travel (physical px) a right-button hold must cover before look
/// engages and the cursor is grabbed/hidden.  Below this the press stays an
/// ordinary right-click (e.g. the editor's context menu) -- important in the
/// browser, where grabbing on the bare press fires a pointer-lock "press Esc
/// to show your cursor" toast on every right-click and swallows the release.
const LOOK_DRAG_THRESHOLD: f64 = 4.0;

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            move_rate: 1.0,
            shift_move_multiplier: 3.0,
            key_look_rate: 30.0,
            mouse_look_sensitivity: 0.18,
            walk_on_plane: false,
            invert_pitch: false,
            pitch_min: -89.0,
            pitch_max: 89.0,
            looking: false,
            look_drag_accum: 0.0,
        }
    }
}

impl FlyCamera {
    /// The camera's forward and right basis vectors. `forward` is flattened onto
    /// the ground plane when [`walk_on_plane`](Self::walk_on_plane) is set. Handy
    /// for building custom movement (e.g. touch pads) in the same frame.
    pub fn basis(&self, camera: &Camera) -> (CgVec3, CgVec3) {
        let (_view, view_dir, right_dir) = camera.calculate_view_matrix();
        let forward = if self.walk_on_plane {
            let flat = CgVec3::new(view_dir.x, 0.0, view_dir.z);
            if flat.magnitude2() > 1e-6 {
                flat.normalize()
            } else {
                view_dir.normalize()
            }
        } else {
            view_dir.normalize()
        };
        (forward, right_dir)
    }

    /// The un-normalized WASD movement direction for this frame.
    pub fn wasd_direction(&self, camera: &Camera, input: &InputManager) -> CgVec3 {
        let (forward, right) = self.basis(camera);
        let mut dir = CG_VEC3_ZERO;
        if input.get_key_state("w").is_down() {
            dir += forward;
        }
        if input.get_key_state("s").is_down() {
            dir -= forward;
        }
        if input.get_key_state("d").is_down() {
            dir += right;
        }
        if input.get_key_state("a").is_down() {
            dir -= right;
        }
        dir
    }

    /// Movement speed for this frame: [`move_rate`](Self::move_rate), multiplied
    /// by [`shift_move_multiplier`](Self::shift_move_multiplier) while shift is held.
    pub fn move_speed(&self, input: &InputManager) -> f32 {
        if input.get_key_state("left_shift").is_down() {
            self.move_rate * self.shift_move_multiplier
        } else {
            self.move_rate
        }
    }

    /// [todo: remove]
    /// Applies arrow-key look to a rotation `(yaw = .x, pitch = .y)`, honoring
    /// [`invert_pitch`](Self::invert_pitch). Does not clamp -- call
    /// [`clamp_pitch`](Self::clamp_pitch) once you've applied every look source
    /// (keys, mouse, touch) for the frame.
    pub fn apply_key_look(&self, rotation: &mut CgVec3, input: &InputManager, delta_time: f32) {
        let step = delta_time * self.key_look_rate;
        let pitch_step = if self.invert_pitch { -step } else { step };
        if input.get_key_state("left_arrow").is_down() {
            rotation.x += step;
        }
        if input.get_key_state("right_arrow").is_down() {
            rotation.x -= step;
        }
        if input.get_key_state("up_arrow").is_down() {
            rotation.y -= pitch_step;
        }
        if input.get_key_state("down_arrow").is_down() {
            rotation.y += pitch_step;
        }
    }

    /// Applies right-drag mouse look to a rotation `(yaw = .x, pitch = .y)`.
    pub fn apply_mouse_look(
        &mut self,
        rotation: &mut CgVec3,
        input: &InputManager,
        renderer: &Renderer,
    ) {
        let rmb = input.get_key_state("mouse_right");
        if rmb.is_down() || rmb.just_pressed() {
            if rmb.just_pressed() {
                self.look_drag_accum = 0.0;
            }
            let (dx, dy) = input.get_mouse_raw_delta();
            self.look_drag_accum += dx.abs() + dy.abs();
            // Only engage look (and the cursor grab) once the hold actually
            // drags; a sub-threshold press-release stays a plain right-click.
            if !self.looking && self.look_drag_accum >= LOOK_DRAG_THRESHOLD {
                self.looking = true;
                renderer.set_cursor_visible(false);
                renderer.set_cursor_grabbed(true);
            }
            if self.looking {
                rotation.x -= dx as f32 * self.mouse_look_sensitivity;
                let dy = dy as f32 * self.mouse_look_sensitivity;
                rotation.y += if self.invert_pitch { -dy } else { dy };
            }
        } else if self.looking {
            self.looking = false;
            renderer.set_cursor_grabbed(false);
            renderer.set_cursor_visible(true);
        }
    }

    /// True while a right-drag look is engaged (the cursor is grabbed/hidden).
    /// Lets the editor tell a look-drag from a plain right-click: if look never
    /// engaged during a right-button hold, the release is a click (e.g. opens
    /// the context menu).
    pub fn is_looking(&self) -> bool {
        self.looking
    }

    /// Clamps a rotation's pitch (`.y`) to `[pitch_min, pitch_max]`.
    pub fn clamp_pitch(&self, rotation: &mut CgVec3) {
        rotation.y = rotation.y.clamp(self.pitch_min, self.pitch_max);
    }
}
