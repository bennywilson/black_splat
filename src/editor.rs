//! Editor property inspection.  Game types mark up their tweakable fields with
//! the `editor_properties!` macro, which generates an [`EditorInspect`] impl;
//! an editor then walks those properties through a [`PropertyVisitor`] without
//! knowing the type's internals.  [`draw_properties`] is the egui visitor used
//! by the in-engine editor's Details panel.

use cgmath::{InnerSpace, Rotation3};

use crate::assets::{MaterialHandle, ModelHandle};
use crate::config::Config;
use crate::game_object::Camera;
use crate::utils::{CgQuat, CgVec3, CgVec4};

/// Receives one callback per marked-up property, with a mutable view of the
/// value.  Each method returns true if it changed the value.
pub trait PropertyVisitor {
    fn edit_text(&mut self, name: &str, value: &mut String) -> bool;
    /// A single scalar, edited as a drag value (never taken below zero -- the
    /// engine's floats are physical quantities like intensity or scale).
    fn edit_float(&mut self, name: &str, value: &mut f32) -> bool;
    /// A checkbox.
    fn edit_bool(&mut self, name: &str, value: &mut bool) -> bool;
    fn edit_vec3(&mut self, name: &str, value: &mut CgVec3) -> bool;
    /// An RGB color (each channel 0..1), edited with a swatch/color picker.
    fn edit_color(&mut self, name: &str, value: &mut CgVec3) -> bool;
    /// Rotation is stored as a quaternion but presented as XYZ euler degrees.
    fn edit_rotation(&mut self, name: &str, value: &mut CgQuat) -> bool;
    /// A fixed set of named choices (an enum): `index` selects into `options`.
    fn edit_choice(
        &mut self,
        name: &str,
        index: &mut usize,
        options: &'static [&'static str],
    ) -> bool;
    /// A reference to a loaded model resource.
    fn edit_model(&mut self, name: &str, value: &mut ModelHandle) -> bool;
    /// A reference to a loaded material resource.
    fn edit_material(&mut self, name: &str, value: &mut MaterialHandle) -> bool;
}

/// Implemented (via `editor_properties!`) by types the editor can inspect.
/// Returns true if the visitor changed any property.
pub trait EditorInspect {
    fn inspect_properties(&mut self, visitor: &mut dyn PropertyVisitor) -> bool;
}

/// Enums editable as a `choice(...)` property: a fixed name list plus
/// index <-> variant mapping.
pub trait EditorChoice: Sized {
    const NAMES: &'static [&'static str];
    fn choice_index(&self) -> usize;
    fn from_choice_index(index: usize) -> Self;
}

// Free-function shims so `editor_properties!` can reach EditorChoice through
// type inference on the field (a macro can't name the field's type directly).
pub fn choice_names<T: EditorChoice>(_value: &T) -> &'static [&'static str] {
    T::NAMES
}
pub fn choice_index<T: EditorChoice>(value: &T) -> usize {
    value.choice_index()
}
pub fn set_choice_index<T: EditorChoice>(value: &mut T, index: usize) {
    *value = T::from_choice_index(index);
}

/// Marks up which fields of a type the editor shows and how each is edited:
///
/// ```ignore
/// crate::editor_properties!(Actor {
///     position: vec3("Position"),
///     rotation: rotation("Rotation"),
///     layer: choice("Scene Layer"),      // field type must impl EditorChoice
///     model_handle: model("Model"),
/// });
/// ```
///
/// Invoke it where the fields are visible (typically the defining module);
/// it generates the type's [`EditorInspect`] impl.
#[macro_export]
macro_rules! editor_properties {
    ($type:ty { $( $field:ident : $kind:ident ( $label:literal ) ),+ $(,)? }) => {
        impl $crate::editor::EditorInspect for $type {
            fn inspect_properties(
                &mut self,
                visitor: &mut dyn $crate::editor::PropertyVisitor,
            ) -> bool {
                let mut changed = false;
                $( changed |= $crate::editor_property!(self, visitor, $field, $kind, $label); )+
                changed
            }
        }
    };
}

/// One field of an `editor_properties!` block; dispatches on the kind keyword.
#[doc(hidden)]
#[macro_export]
macro_rules! editor_property {
    ($self:ident, $visitor:ident, $field:ident, text, $label:literal) => {
        $visitor.edit_text($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, float, $label:literal) => {
        $visitor.edit_float($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, bool, $label:literal) => {
        $visitor.edit_bool($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, vec3, $label:literal) => {
        $visitor.edit_vec3($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, color, $label:literal) => {
        $visitor.edit_color($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, rotation, $label:literal) => {
        $visitor.edit_rotation($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, model, $label:literal) => {
        $visitor.edit_model($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, material, $label:literal) => {
        $visitor.edit_material($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, choice, $label:literal) => {{
        let mut index = $crate::editor::choice_index(&$self.$field);
        let names = $crate::editor::choice_names(&$self.$field);
        if $visitor.edit_choice($label, &mut index, names) {
            $crate::editor::set_choice_index(&mut $self.$field, index);
            true
        } else {
            false
        }
    }};
}

/// Draws `object`'s marked-up properties into `ui` and returns true if any
/// changed this frame.  `model_resources` / `material_resources` are the
/// (display name, handle) pairs offered by `model(...)` / `material(...)`
/// property dropdowns; `selected_resource` is the model currently picked in
/// the editor's resource browser, if any, which the model dropdown offers to
/// apply with one click.
pub fn draw_properties(
    ui: &mut egui::Ui,
    object: &mut dyn EditorInspect,
    model_catalog: &[(String, String, Option<ModelHandle>)],
    material_resources: &[(String, MaterialHandle)],
    selected_resource: Option<ModelHandle>,
    model_pick_request: &mut Option<String>,
) -> bool {
    object.inspect_properties(&mut EguiPropertyEditor {
        ui,
        model_catalog,
        material_resources,
        selected_resource,
        model_pick_request,
    })
}

/// The egui-widget PropertyVisitor behind [`draw_properties`].  `model_catalog`
/// is every discovered model as (display name, path, loaded handle) -- the
/// dropdown lists them all, and picking one that isn't loaded yet writes its
/// path to `model_pick_request` so the caller can lazily load it (see the
/// editor's lazy model loading).
struct EguiPropertyEditor<'a> {
    ui: &'a mut egui::Ui,
    model_catalog: &'a [(String, String, Option<ModelHandle>)],
    material_resources: &'a [(String, MaterialHandle)],
    selected_resource: Option<ModelHandle>,
    model_pick_request: &'a mut Option<String>,
}

impl EguiPropertyEditor<'_> {
    /// A row of xyz drag values under a label; returns true if any changed.
    fn drag_row(&mut self, name: &str, values: [&mut f32; 3], speed: f32, suffix: &str) -> bool {
        let mut changed = false;
        self.ui.label(name);
        self.ui.horizontal(|ui| {
            for (axis, value) in ["x", "y", "z"].into_iter().zip(values) {
                changed |= ui
                    .add(
                        egui::DragValue::new(value)
                            .speed(speed)
                            .prefix(format!("{axis} "))
                            .suffix(suffix)
                            .max_decimals(3),
                    )
                    .changed();
            }
        });
        changed
    }
}

impl PropertyVisitor for EguiPropertyEditor<'_> {
    fn edit_text(&mut self, name: &str, value: &mut String) -> bool {
        self.ui.label(name);
        self.ui.text_edit_singleline(value).changed()
    }

    fn edit_float(&mut self, name: &str, value: &mut f32) -> bool {
        self.ui.label(name);
        self.ui
            .add(
                egui::DragValue::new(value)
                    .speed(0.05)
                    .range(0.0..=f32::MAX)
                    .max_decimals(3),
            )
            .changed()
    }

    fn edit_bool(&mut self, name: &str, value: &mut bool) -> bool {
        self.ui.checkbox(value, name).changed()
    }

    fn edit_vec3(&mut self, name: &str, value: &mut CgVec3) -> bool {
        self.drag_row(name, [&mut value.x, &mut value.y, &mut value.z], 0.05, "")
    }

    fn edit_color(&mut self, name: &str, value: &mut CgVec3) -> bool {
        self.ui.label(name);
        // egui edits gamma-space 0..1 rgb; the light color is stored the same way.
        let mut rgb = [value.x, value.y, value.z];
        if self.ui.color_edit_button_rgb(&mut rgb).changed() {
            *value = CgVec3::new(rgb[0], rgb[1], rgb[2]);
            true
        } else {
            false
        }
    }

    fn edit_rotation(&mut self, name: &str, value: &mut CgQuat) -> bool {
        // Presented as euler degrees.  The quat is only rebuilt on frames where
        // a drag actually changed a value, so idle frames don't accumulate
        // quat -> euler -> quat round-trip error.
        let euler = cgmath::Euler::from(*value);
        let mut degrees = [
            cgmath::Deg::from(euler.x).0,
            cgmath::Deg::from(euler.y).0,
            cgmath::Deg::from(euler.z).0,
        ];
        let [x, y, z] = &mut degrees;
        if self.drag_row(name, [x, y, z], 0.5, "°") {
            *value = CgQuat::from(cgmath::Euler::new(
                cgmath::Deg(degrees[0]),
                cgmath::Deg(degrees[1]),
                cgmath::Deg(degrees[2]),
            ));
            true
        } else {
            false
        }
    }

    fn edit_choice(
        &mut self,
        name: &str,
        index: &mut usize,
        options: &'static [&'static str],
    ) -> bool {
        let mut changed = false;
        self.ui.label(name);
        egui::ComboBox::from_id_salt(name)
            .selected_text(options.get(*index).copied().unwrap_or("?"))
            .show_ui(self.ui, |ui| {
                for (i, option) in options.iter().enumerate() {
                    changed |= ui.selectable_value(index, i, *option).changed();
                }
            });
        changed
    }

    fn edit_model(&mut self, name: &str, value: &mut ModelHandle) -> bool {
        let selected_resource = self.selected_resource;
        let Self {
            ui,
            model_catalog,
            model_pick_request,
            ..
        } = self;
        let mut changed = false;
        ui.label(name);
        let selected = model_catalog
            .iter()
            .find(|(_, _, handle)| *handle == Some(*value))
            .map_or("(none)", |(res_name, _, _)| res_name.as_str());
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt(name)
                .selected_text(selected)
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(value, ModelHandle::make_invalid(), "(none)")
                        .changed();
                    // The dropdown always lists the full catalog.  Loaded models
                    // assign their handle directly; ones not loaded yet request a
                    // lazy load (the caller loads + assigns to this actor).
                    for (res_name, path, handle) in model_catalog.iter() {
                        match handle {
                            Some(h) => {
                                changed |= ui
                                    .selectable_value(value, *h, res_name.as_str())
                                    .changed();
                            }
                            None => {
                                if ui.selectable_label(false, res_name.as_str()).clicked() {
                                    **model_pick_request = Some(path.clone());
                                }
                            }
                        }
                    }
                });
            // One-click apply of the resource browser's selection
            // (Unreal-style "use selected asset").
            let use_selected = ui
                .add_enabled(selected_resource.is_some(), egui::Button::new("◀").small())
                .on_hover_text("Use the model selected in the Resources panel");
            if use_selected.clicked() {
                if let Some(handle) = selected_resource {
                    if *value != handle {
                        *value = handle;
                        changed = true;
                    }
                }
            }
        });
        changed
    }

    fn edit_material(&mut self, name: &str, value: &mut MaterialHandle) -> bool {
        let Self {
            ui,
            material_resources,
            ..
        } = self;
        let mut changed = false;
        ui.label(name);
        let selected = material_resources
            .iter()
            .find(|(_, handle)| handle == value)
            .map_or("(none)", |(res_name, _)| res_name.as_str());
        egui::ComboBox::from_id_salt(name)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(value, MaterialHandle::make_invalid(), "(none)")
                    .changed();
                for (res_name, res_handle) in material_resources.iter() {
                    changed |= ui
                        .selectable_value(value, *res_handle, res_name.as_str())
                        .changed();
                }
            });
        changed
    }
}

/// Which transform the viewport gizmo edits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

const GIZMO_AXES: [CgVec3; 3] = [
    CgVec3::new(1.0, 0.0, 0.0),
    CgVec3::new(0.0, 1.0, 0.0),
    CgVec3::new(0.0, 0.0, 1.0),
];
const GIZMO_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(235, 75, 75),
    egui::Color32::from_rgb(115, 210, 75),
    egui::Color32::from_rgb(85, 135, 245),
];
// Hovered or dragged handles light up yellow.
const GIZMO_HIGHLIGHT: egui::Color32 = egui::Color32::from_rgb(255, 220, 60);
// How close (in points) the pointer must be to a handle to grab it.
const GIZMO_HIT_RADIUS: f32 = 12.0;
// A larger grab radius around each axis' tip (arrowhead / scale box), so the
// ends of the handles are big, forgiving tap targets on touch.
const GIZMO_TIP_HIT_RADIUS: f32 = 24.0;
// Radius (in points) of the filled knob drawn at each translate arrow tip; a
// visible mark for the enlarged tip tap target above.
const GIZMO_TIP_KNOB_RADIUS: f32 = 6.0;
// On-screen gizmo size: world size = distance to camera * this.
const GIZMO_SCALE: f32 = 0.2;
// Side (in points) of the square tips on the scale handles.
const GIZMO_SCALE_HANDLE_SIZE: f32 = 10.0;
// Scale drags never shrink an axis past this, so the actor's world matrix
// stays invertible and the actor stays clickable/recoverable in the viewport.
const GIZMO_MIN_SCALE: f32 = 0.001;
// drag_axis value for the scale gizmo's uniform-scale center handle
// (0..2 are the entries of GIZMO_AXES).
const GIZMO_CENTER_HANDLE: usize = 3;
// Rotate-drag gain: how much the object turns per unit of pointer angle swept
// around the gizmo center.  1.0 tracks the pointer exactly (a full lap around
// the center = one turn), but that makes a straight drag across the view worth
// only ~half a turn; >1 trades exact tracking for a livelier feel so a
// screen-wide drag spins the object a bit more than the pointer sweep.
const GIZMO_ROTATE_SENSITIVITY: f32 = 1.5;

/// A screen-space translate/rotate/scale gizmo drawn over the 3D view for the
/// selected actor.  Translate shows three world-axis arrows; dragging one
/// slides the position along that axis.  Rotate shows three axis rings;
/// dragging one spins the rotation about that axis by the pointer's angle
/// change around the gizmo center.  Scale shows three square-tipped axis
/// handles plus a center square; dragging a tip scales that axis, dragging
/// the center scales all three uniformly.
pub struct TransformGizmo {
    pub mode: GizmoMode,
    /// Rotate-mode snap increment in degrees; 0 disables snapping so rotation
    /// is continuous.  When set, a rotate drag commits only in whole
    /// increments of this many degrees.
    pub rotate_snap_degrees: f32,
    /// Translate-mode snap increment in world units; 0 disables snapping so
    /// translation is continuous.  When set, a translate drag commits only
    /// in whole increments of this many units along the dragged axis.
    pub translate_snap_units: f32,
    /// Scale-mode snap increment; 0 disables snapping so scaling is
    /// continuous.  When set, a scale drag commits only in whole increments
    /// of this much scale, applied to the axis' (or, for the uniform center
    /// handle, each axis') absolute scale value.
    pub scale_snap_units: f32,
    // Handle currently being dragged: an index into GIZMO_AXES, or
    // GIZMO_CENTER_HANDLE for the uniform-scale center square.
    drag_axis: Option<usize>,
    // Rotate-drag accumulation: the raw angle dragged since the drag began
    // and how much of it has been committed (after snapping).  Snapping
    // quantizes the *cumulative* turn, so a small wiggle never jumps a full
    // increment -- the object only turns once the drag passes a half-step.
    rotate_accum: f32,
    rotate_applied: f32,
    // Translate-drag accumulation, same shape as the rotate fields above but
    // in world units along the dragged axis.
    translate_accum: f32,
    translate_applied: f32,
    // Scale-drag accumulation: unlike rotate/translate this tracks the raw
    // (unsnapped) *absolute* scale value per axis rather than a delta, since
    // a scale drag is multiplicative (feel stays the same at any current
    // scale) while the snap increment is a fixed absolute step.  Seeded from
    // the actor's current scale on press; the center handle drives all three
    // entries at once, an axis handle drives just its own.
    scale_accum: [f32; 3],
    scale_applied: [f32; 3],
}

impl Default for TransformGizmo {
    fn default() -> Self {
        TransformGizmo {
            mode: GizmoMode::Translate,
            rotate_snap_degrees: 0.0,
            translate_snap_units: 0.0,
            scale_snap_units: 0.0,
            drag_axis: None,
            rotate_accum: 0.0,
            rotate_applied: 0.0,
            translate_accum: 0.0,
            translate_applied: 0.0,
            scale_accum: [0.0; 3],
            scale_applied: [0.0; 3],
        }
    }
}

impl TransformGizmo {
    /// True while a handle is being dragged.  Lets the caller suppress
    /// viewport click-to-select when a drag started on the gizmo this frame.
    pub fn is_active(&self) -> bool {
        self.drag_axis.is_some()
    }

    /// Draws the gizmo at `position` and applies any drag to
    /// `position`/`rotation`/`scale` (depending on mode).  Returns true if it
    /// changed them this frame.
    pub fn ui(
        &mut self,
        ctx: &egui::Context,
        camera: &Camera,
        config: &Config,
        position: &mut CgVec3,
        rotation: &mut CgQuat,
        scale: &mut CgVec3,
    ) -> bool {
        // The same view/projection the model pass renders with, so the gizmo
        // lines up with the actor on screen.
        let (view, _, _) = camera.calculate_view_matrix();
        let proj = cgmath::perspective(
            cgmath::Deg(config.fov),
            config.window_width as f32 / config.window_height as f32,
            0.1,
            10000.0,
        );
        let view_proj = proj * view;
        let screen = ctx.content_rect();
        let project = move |world: CgVec3| -> Option<egui::Pos2> {
            let clip = view_proj * CgVec4::new(world.x, world.y, world.z, 1.0);
            if clip.w < 0.01 {
                return None; // Behind the camera.
            }
            Some(egui::pos2(
                screen.left() + (clip.x / clip.w + 1.0) * 0.5 * screen.width(),
                screen.top() + (1.0 - clip.y / clip.w) * 0.5 * screen.height(),
            ))
        };

        let Some(center) = project(*position) else {
            self.drag_axis = None;
            return false;
        };

        // Sized by camera distance so the gizmo stays constant on screen.
        let world_size = (*position - camera.get_position()).magnitude().max(0.01) * GIZMO_SCALE;

        // Background layer: over the 3D scene (all egui painting is) but
        // under the editor panels and menus.
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("transform_gizmo"),
        ));

        let (pointer, pressed, down, delta, any_down) = ctx.input(|i| {
            (
                i.pointer.interact_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.delta(),
                // Any button, so hover highlights stay off during e.g. a
                // right-drag camera look sweeping across the gizmo.
                i.pointer.any_down(),
            )
        });
        if !down {
            self.drag_axis = None;
        }
        // A fresh press starts a new drag: reset every mode's snap
        // accumulation.  Rotate/translate track a delta-since-press (start at
        // 0); scale tracks an absolute value (seed from the current scale).
        if pressed {
            self.rotate_accum = 0.0;
            self.rotate_applied = 0.0;
            self.translate_accum = 0.0;
            self.translate_applied = 0.0;
            self.scale_accum = [scale.x, scale.y, scale.z];
            self.scale_applied = [scale.x, scale.y, scale.z];
        }
        // Neither grabs nor hover highlights reach through the editor UI.  (A
        // drag that started on the gizmo and passes over a panel keeps going:
        // egui_wants_pointer_input stays false for drags started outside it.)
        let over_ui = ctx.egui_wants_pointer_input();
        let can_grab = pressed && !over_ui;

        let mut changed = false;

        // Scale mode's uniform-scale center square.  Grab-checked before the
        // axis handles so it wins presses near the center, where all three
        // axis lines converge; painted after them so it sits on top.
        let mut center_active = false;
        let mut center_hovered = false;
        if self.mode == GizmoMode::Scale {
            let center_dist = pointer.map_or(f32::MAX, |p| p.distance(center));
            if can_grab && center_dist < GIZMO_HIT_RADIUS {
                self.drag_axis = Some(GIZMO_CENTER_HANDLE);
            }
            center_active = self.drag_axis == Some(GIZMO_CENTER_HANDLE);
            center_hovered = !any_down && !over_ui && center_dist < GIZMO_HIT_RADIUS;
            if center_active && down {
                // Drag right/up to grow, left/down to shrink (screen y points
                // down).  Multiplicative, so the feel is the same at any
                // current scale and an axis can never cross zero.
                let factor = 1.0 + (delta.x - delta.y) * 0.01;
                for i in 0..3 {
                    self.scale_accum[i] = (self.scale_accum[i] * factor).max(GIZMO_MIN_SCALE);
                    let target = snap_value(self.scale_accum[i], self.scale_snap_units)
                        .max(GIZMO_MIN_SCALE);
                    if target != self.scale_applied[i] {
                        scale[i] = target;
                        self.scale_applied[i] = target;
                        changed = true;
                    }
                }
            }
        }

        for (axis, dir) in GIZMO_AXES.iter().enumerate() {
            match self.mode {
                GizmoMode::Translate => {
                    let Some(tip) = project(*position + *dir * world_size) else {
                        continue;
                    };
                    // Grab along the whole shaft, but with a much larger radius
                    // right at the tip so the arrowhead is an easy touch target.
                    let hit = pointer.map_or(false, |p| {
                        distance_to_segment(p, center, tip) < GIZMO_HIT_RADIUS
                            || p.distance(tip) < GIZMO_TIP_HIT_RADIUS
                    });
                    if can_grab && hit {
                        self.drag_axis = Some(axis);
                    }
                    let active = self.drag_axis == Some(axis);
                    let hovered = !any_down && !over_ui && hit;
                    if active && down {
                        // Pointer movement along the axis' screen direction,
                        // converted back to world units.  Accumulated and
                        // snapped the same way as rotate above: the raw drag
                        // is tracked in full, but only the snapped portion of
                        // it is ever committed to `position`.
                        let screen_axis = tip - center;
                        let len2 = screen_axis.length_sq();
                        if len2 > 1.0 {
                            let t = delta.dot(screen_axis) / len2;
                            self.translate_accum += t * world_size;
                            let target =
                                snap_value(self.translate_accum, self.translate_snap_units);
                            let to_apply = target - self.translate_applied;
                            if to_apply != 0.0 {
                                *position += *dir * to_apply;
                                self.translate_applied = target;
                                changed = true;
                            }
                        }
                    }
                    let color = if active || hovered {
                        GIZMO_HIGHLIGHT
                    } else {
                        GIZMO_COLORS[axis]
                    };
                    painter.arrow(
                        center,
                        tip - center,
                        egui::Stroke::new(if active { 4.5 } else { 3.0 }, color),
                    );
                    // Knob at the tip: a visible mark for the enlarged tap target.
                    painter.circle_filled(
                        tip,
                        if active || hovered {
                            GIZMO_TIP_KNOB_RADIUS + 1.5
                        } else {
                            GIZMO_TIP_KNOB_RADIUS
                        },
                        color,
                    );
                }
                GizmoMode::Rotate => {
                    // Ring in the plane perpendicular to the axis, spanned by
                    // the other two axes.
                    let u = GIZMO_AXES[(axis + 1) % 3];
                    let v = GIZMO_AXES[(axis + 2) % 3];
                    let mut points = Vec::with_capacity(49);
                    let mut min_dist = f32::MAX;
                    for i in 0..=48 {
                        let t = i as f32 / 48.0 * std::f32::consts::TAU;
                        let world = *position + (u * t.cos() + v * t.sin()) * world_size;
                        if let Some(p) = project(world) {
                            if let Some(ptr) = pointer {
                                min_dist = min_dist.min(p.distance(ptr));
                            }
                            points.push(p);
                        }
                    }
                    if points.len() < 2 {
                        continue;
                    }
                    if can_grab && min_dist < GIZMO_HIT_RADIUS {
                        self.drag_axis = Some(axis);
                    }
                    let active = self.drag_axis == Some(axis);
                    let hovered = !any_down && !over_ui && min_dist < GIZMO_HIT_RADIUS;
                    if active && down {
                        if let Some(ptr) = pointer {
                            // Rotate by the pointer's angle change around the
                            // gizmo center.
                            let prev = ptr - delta;
                            let mut d_angle = (ptr - center).angle() - (prev - center).angle();
                            if d_angle > std::f32::consts::PI {
                                d_angle -= std::f32::consts::TAU;
                            } else if d_angle < -std::f32::consts::PI {
                                d_angle += std::f32::consts::TAU;
                            }
                            // Screen y points down and the ring can face
                            // either way; flip so the drag follows the ring.
                            if dir.dot(camera.get_position() - *position) >= 0.0 {
                                d_angle = -d_angle;
                            }
                            // Amplify so a screen-wide drag is worth several
                            // turns rather than the ~half-turn exact pointer
                            // tracking would give.
                            d_angle *= GIZMO_ROTATE_SENSITIVITY;
                            // Accumulate the raw drag, then commit only the
                            // snapped delta.  With no snap the target tracks
                            // the accumulation exactly (continuous rotation);
                            // with snap the object turns a whole increment only
                            // once the cumulative drag crosses a half-step, so
                            // a minor drag never over-rotates.
                            self.rotate_accum += d_angle;
                            let target = snap_value(
                                self.rotate_accum,
                                self.rotate_snap_degrees.to_radians(),
                            );
                            let to_apply = target - self.rotate_applied;
                            if to_apply != 0.0 {
                                *rotation = (CgQuat::from_axis_angle(*dir, cgmath::Rad(to_apply))
                                    * *rotation)
                                    .normalize();
                                self.rotate_applied = target;
                                changed = true;
                            }
                        }
                    }
                    painter.add(egui::Shape::line(
                        points,
                        egui::Stroke::new(
                            if active { 5.0 } else { 3.5 },
                            if active || hovered {
                                GIZMO_HIGHLIGHT
                            } else {
                                GIZMO_COLORS[axis]
                            },
                        ),
                    ));
                }
                GizmoMode::Scale => {
                    let Some(tip) = project(*position + *dir * world_size) else {
                        continue;
                    };
                    // Grab along the shaft, with a bigger radius at the box tip
                    // so it's an easy touch target.
                    let hit = pointer.map_or(false, |p| {
                        distance_to_segment(p, center, tip) < GIZMO_HIT_RADIUS
                            || p.distance(tip) < GIZMO_TIP_HIT_RADIUS
                    });
                    // is_none() keeps a center grab (checked above) from being
                    // stolen by the axis lines that pass through it.
                    if can_grab && self.drag_axis.is_none() && hit {
                        self.drag_axis = Some(axis);
                    }
                    let active = self.drag_axis == Some(axis);
                    let hovered = !any_down && !over_ui && hit;
                    if active && down {
                        // Pointer movement along the axis' screen direction as
                        // a fraction of the gizmo length, applied
                        // multiplicatively: dragging outward grows, inward
                        // shrinks, same feel at any current scale.  The raw
                        // (unsnapped) value keeps accumulating so the drag
                        // feel never changes; only the snapped value is ever
                        // written to `scale`.
                        let screen_axis = tip - center;
                        let len2 = screen_axis.length_sq();
                        if len2 > 1.0 {
                            let t = delta.dot(screen_axis) / len2;
                            self.scale_accum[axis] =
                                (self.scale_accum[axis] * (1.0 + t)).max(GIZMO_MIN_SCALE);
                            let target = snap_value(self.scale_accum[axis], self.scale_snap_units)
                                .max(GIZMO_MIN_SCALE);
                            if target != self.scale_applied[axis] {
                                scale[axis] = target;
                                self.scale_applied[axis] = target;
                                changed = true;
                            }
                        }
                    }
                    let color = if active || hovered {
                        GIZMO_HIGHLIGHT
                    } else {
                        GIZMO_COLORS[axis]
                    };
                    painter.line_segment(
                        [center, tip],
                        egui::Stroke::new(if active { 4.5 } else { 3.0 }, color),
                    );
                    painter.rect_filled(
                        egui::Rect::from_center_size(
                            tip,
                            egui::Vec2::splat(GIZMO_SCALE_HANDLE_SIZE),
                        ),
                        1.0,
                        color,
                    );
                }
            }
        }

        if self.mode == GizmoMode::Scale {
            painter.rect_filled(
                egui::Rect::from_center_size(
                    center,
                    egui::Vec2::splat(GIZMO_SCALE_HANDLE_SIZE),
                ),
                1.0,
                if center_active || center_hovered {
                    GIZMO_HIGHLIGHT
                } else {
                    egui::Color32::from_gray(220)
                },
            );
        }
        changed
    }
}

// Rounds `value` to the nearest multiple of `step`, or returns it unchanged
// if `step` is 0 (snapping disabled).
fn snap_value(value: f32, step: f32) -> f32 {
    if step > 0.0 {
        (value / step).round() * step
    } else {
        value
    }
}

fn distance_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 <= f32::EPSILON {
        return a.distance(p);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (a + ab * t).distance(p)
}
