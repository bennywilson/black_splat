//! On-screen dual thumb-pads for touch devices, drawn with egui.
//!
//! egui's own pointer only tracks a single touch, so these pads read the
//! engine's raw touch map instead -- that way move and look work at the same
//! time -- and use egui purely as the painter. The left pad drives movement, the
//! right pad drives look; [`TouchPads::update`] returns this frame's deflections
//! for the game to feed into its camera. Shared by the splat viewer and the 3D
//! demo so both feel identical.

use crate::input::InputManager;

/// Per-frame result from [`TouchPads::update`].
#[derive(Clone, Copy, Default)]
pub struct TouchPadsOutput {
    /// Left-pad deflection, each axis in `[-1, 1]`: `x` = strafe right, `y` =
    /// move forward. Combine with the camera basis, e.g. `right * x - forward * y`.
    pub move_deflection: egui::Vec2,
    /// Yaw change from the right pad this frame, in degrees; add to the camera's
    /// yaw. Already scaled by [`look_rate`](TouchPads::look_rate) and delta time.
    pub yaw_delta_deg: f32,
    /// Pitch change from the right pad this frame, in degrees; add to the
    /// camera's pitch. Already scaled by look rate and delta time.
    pub pitch_delta_deg: f32,
}

/// Two on-screen thumb-pads (left = move, right = look) for touch control.
///
/// Sizes are fractions of the shorter screen axis (in egui points) so the pads
/// scale to any resolution. Construct with [`TouchPads::default`] and tweak the
/// fields you need.
#[derive(Clone)]
pub struct TouchPads {
    /// Pad radius, as a fraction of the shorter screen axis.
    pub radius_frac: f32,
    /// Gap from the screen edge to the pad, as a fraction of the shorter axis.
    pub margin_frac: f32,
    /// Fraction of the pad radius that counts as full stick deflection.
    pub span_frac: f32,
    /// Deflections shorter than this fraction of full are treated as zero.
    pub dead_zone: f32,
    /// Look speed at full right-pad deflection, in degrees per second.
    pub look_rate: f32,
    /// When true the pads stay hidden (and inert) until the first touch -- handy
    /// on desktop, where a mouse user never needs them. When false (the default)
    /// they are shown from the start.
    pub reveal_on_touch: bool,

    /// Set once the first touch is seen; only consulted when `reveal_on_touch`.
    revealed: bool,
}

impl Default for TouchPads {
    fn default() -> Self {
        Self {
            radius_frac: 0.16,
            margin_frac: 0.05,
            span_frac: 0.75,
            dead_zone: 0.12,
            look_rate: 70.0,
            reveal_on_touch: false,
            revealed: false,
        }
    }
}

impl TouchPads {
    /// Reads the touch map, draws the pads (when visible), and returns this
    /// frame's move/look deflections. Call once per frame from `tick`, passing
    /// the shared egui context (`renderer.egui_ctx()`).
    pub fn update(
        &mut self,
        ctx: &egui::Context,
        input: &InputManager,
        delta_time: f32,
    ) -> TouchPadsOutput {
        let touch_map = input.get_touch_map();
        if !touch_map.is_empty() {
            self.revealed = true;
        }
        if self.reveal_on_touch && !self.revealed {
            return TouchPadsOutput::default();
        }

        let screen = ctx.content_rect();
        let ppp = ctx.pixels_per_point();
        let min_axis = screen.width().min(screen.height());
        let radius = min_axis * self.radius_frac;
        let margin = min_axis * self.margin_frac;
        let span = radius * self.span_frac;
        let move_center = egui::pos2(
            screen.left() + margin + radius,
            screen.bottom() - margin - radius,
        );
        let look_center = egui::pos2(
            screen.right() - margin - radius,
            screen.bottom() - margin - radius,
        );

        // A touch belongs to the pad it STARTED on, so a held drag can wander
        // outside the circle without hopping pads.
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
                move_defl = self.deflection(finger, move_center, span);
            } else if start.distance(look_center) <= radius {
                look_defl = self.deflection(finger, look_center, span);
            }
        }

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
                egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(120, 255, 145, 220)),
            );
        }

        // Left pad: drag right = strafe right, drag up = move forward.
        // Right pad: drag right = look right, drag up = look up.
        TouchPadsOutput {
            move_deflection: move_defl,
            yaw_delta_deg: -look_defl.x * self.look_rate * delta_time,
            pitch_delta_deg: look_defl.y * self.look_rate * delta_time,
        }
    }

    /// Stick deflection: the finger's offset from the pad center, saturating at
    /// `span` and zeroed within the dead zone. Each axis ends up in `[-1, 1]`.
    fn deflection(&self, finger: egui::Pos2, center: egui::Pos2, span: f32) -> egui::Vec2 {
        let mut defl = (finger - center) / span;
        let len = defl.length();
        if len < self.dead_zone {
            return egui::Vec2::ZERO;
        }
        if len > 1.0 {
            defl /= len;
        }
        defl
    }
}
