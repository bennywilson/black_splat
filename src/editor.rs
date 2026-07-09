//! Editor property inspection.  Game types mark up their tweakable fields with
//! the `editor_properties!` macro, which generates an [`EditorInspect`] impl;
//! an editor then walks those properties through a [`PropertyVisitor`] without
//! knowing the type's internals.  [`draw_properties`] is the egui visitor used
//! by the in-engine editor's Details panel.

use cgmath::{InnerSpace, Rotation3};

use crate::assets::ModelHandle;
use crate::config::Config;
use crate::game_object::Camera;
use crate::utils::{CgQuat, CgVec3, CgVec4};

/// Receives one callback per marked-up property, with a mutable view of the
/// value.  Each method returns true if it changed the value.
pub trait PropertyVisitor {
    fn edit_text(&mut self, name: &str, value: &mut String) -> bool;
    fn edit_vec3(&mut self, name: &str, value: &mut CgVec3) -> bool;
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
    ($self:ident, $visitor:ident, $field:ident, vec3, $label:literal) => {
        $visitor.edit_vec3($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, rotation, $label:literal) => {
        $visitor.edit_rotation($label, &mut $self.$field)
    };
    ($self:ident, $visitor:ident, $field:ident, model, $label:literal) => {
        $visitor.edit_model($label, &mut $self.$field)
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
/// changed this frame.  `model_resources` are the (display name, handle) pairs
/// offered by `model(...)` property dropdowns.
pub fn draw_properties(
    ui: &mut egui::Ui,
    object: &mut dyn EditorInspect,
    model_resources: &[(String, ModelHandle)],
) -> bool {
    object.inspect_properties(&mut EguiPropertyEditor {
        ui,
        model_resources,
    })
}

/// The egui-widget PropertyVisitor behind [`draw_properties`].
struct EguiPropertyEditor<'a> {
    ui: &'a mut egui::Ui,
    model_resources: &'a [(String, ModelHandle)],
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

    fn edit_vec3(&mut self, name: &str, value: &mut CgVec3) -> bool {
        self.drag_row(name, [&mut value.x, &mut value.y, &mut value.z], 0.05, "")
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
        let Self {
            ui,
            model_resources,
        } = self;
        let mut changed = false;
        ui.label(name);
        let selected = model_resources
            .iter()
            .find(|(_, handle)| handle == value)
            .map_or("(none)", |(res_name, _)| res_name.as_str());
        egui::ComboBox::from_id_salt(name)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(value, ModelHandle::make_invalid(), "(none)")
                    .changed();
                for (res_name, res_handle) in model_resources.iter() {
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
// How close (in points) the pointer must be to a handle to grab it.
const GIZMO_HIT_RADIUS: f32 = 12.0;
// On-screen gizmo size: world size = distance to camera * this.
const GIZMO_SCALE: f32 = 0.2;

/// A screen-space translate/rotate gizmo drawn over the 3D view for the
/// selected actor.  Translate shows three world-axis arrows; dragging one
/// slides the position along that axis.  Rotate shows three axis rings;
/// dragging one spins the rotation about that axis by the pointer's angle
/// change around the gizmo center.
pub struct TransformGizmo {
    pub mode: GizmoMode,
    // Axis (index into GIZMO_AXES) currently being dragged.
    drag_axis: Option<usize>,
}

impl Default for TransformGizmo {
    fn default() -> Self {
        TransformGizmo {
            mode: GizmoMode::Translate,
            drag_axis: None,
        }
    }
}

impl TransformGizmo {
    /// Draws the gizmo at `position` and applies any drag to
    /// `position`/`rotation` (depending on mode).  Returns true if it changed
    /// them this frame.
    pub fn ui(
        &mut self,
        ctx: &egui::Context,
        camera: &Camera,
        config: &Config,
        position: &mut CgVec3,
        rotation: &mut CgQuat,
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

        let (pointer, pressed, down, delta) = ctx.input(|i| {
            (
                i.pointer.interact_pos(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.pointer.delta(),
            )
        });
        if !down {
            self.drag_axis = None;
        }
        // A grab must start on a handle and not through the editor UI.  (A
        // drag that started on the gizmo and passes over a panel keeps going:
        // egui_wants_pointer_input stays false for drags started outside it.)
        let can_grab = pressed && !ctx.egui_wants_pointer_input();

        let mut changed = false;
        for (axis, dir) in GIZMO_AXES.iter().enumerate() {
            match self.mode {
                GizmoMode::Translate => {
                    let Some(tip) = project(*position + *dir * world_size) else {
                        continue;
                    };
                    if can_grab {
                        if let Some(p) = pointer {
                            if distance_to_segment(p, center, tip) < GIZMO_HIT_RADIUS {
                                self.drag_axis = Some(axis);
                            }
                        }
                    }
                    let active = self.drag_axis == Some(axis);
                    if active && down {
                        // Pointer movement along the axis' screen direction,
                        // converted back to world units.
                        let screen_axis = tip - center;
                        let len2 = screen_axis.length_sq();
                        if len2 > 1.0 {
                            let t = delta.dot(screen_axis) / len2;
                            *position += *dir * (t * world_size);
                            changed = true;
                        }
                    }
                    painter.arrow(
                        center,
                        tip - center,
                        egui::Stroke::new(if active { 4.0 } else { 2.5 }, GIZMO_COLORS[axis]),
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
                            if d_angle != 0.0 {
                                *rotation = (CgQuat::from_axis_angle(*dir, cgmath::Rad(d_angle))
                                    * *rotation)
                                    .normalize();
                                changed = true;
                            }
                        }
                    }
                    painter.add(egui::Shape::line(
                        points,
                        egui::Stroke::new(if active { 4.0 } else { 2.0 }, GIZMO_COLORS[axis]),
                    ));
                }
            }
        }
        changed
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
