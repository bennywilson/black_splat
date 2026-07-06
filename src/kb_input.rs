use std::collections::HashMap;

use winit::{
    event::{ElementState, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum KbButtonState {
    #[default]
    None,
    JustPressed,
    Down {
        mouse_start: (i32, i32),
    },
}

impl KbButtonState {
    pub fn is_down(&self) -> bool {
        matches!(*self, KbButtonState::Down { .. })
    }

    pub fn just_pressed(&self) -> bool {
        *self == KbButtonState::JustPressed
    }
}

#[derive(Debug, Default)]
pub struct KbTouchInfo {
    pub start_pos: (f64, f64),
    pub current_pos: (f64, f64),
    pub frame_delta: (f64, f64),
    pub touch_state: KbButtonState,
}

#[derive(Debug, Default)]
pub struct KbInputManager {
    touch_id_to_info: HashMap<u64, KbTouchInfo>,
    mouse_scroll_delta: f32,
    cursor_position: (i32, i32),
    // Raw mouse motion (winit DeviceEvent::MouseMotion) accumulated this frame.
    // Independent of the cursor position, so it keeps reporting movement even
    // when the cursor is grabbed/hidden at a window edge -- the basis for
    // FPS-style mouse look.  Reset each frame in update_key_states.
    mouse_raw_delta: (f64, f64),
    key_map: HashMap<&'static str, KbButtonState>,
}

#[allow(dead_code)]
impl KbInputManager {
    pub fn new() -> Self {
        let key_map = HashMap::<&str, KbButtonState>::new();
        KbInputManager {
            key_map,
            ..Default::default()
        }
    }

    pub fn update_mouse_scroll(&mut self, y_delta: f32) {
        self.mouse_scroll_delta += y_delta;
    }

    pub fn add_mouse_raw_delta(&mut self, dx: f64, dy: f64) {
        self.mouse_raw_delta.0 += dx;
        self.mouse_raw_delta.1 += dy;
    }

    /// Raw mouse movement accumulated since the last frame, in device units
    /// (~pixels).  Use this for mouse look while the cursor is grabbed.
    pub fn get_mouse_raw_delta(&self) -> (f64, f64) {
        self.mouse_raw_delta
    }

    pub fn update_touch(
        &mut self,
        phase: winit::event::TouchPhase,
        id: u64,
        location: winit::dpi::PhysicalPosition<f64>,
    ) {
        if phase == winit::event::TouchPhase::Started {
            let touch_info = KbTouchInfo {
                start_pos: location.into(),
                current_pos: location.into(),
                frame_delta: (0.0, 0.0),
                touch_state: KbButtonState::JustPressed,
            };
            self.touch_id_to_info.insert(id, touch_info);
        } else if phase == winit::event::TouchPhase::Cancelled
            || phase == winit::event::TouchPhase::Ended
        {
            self.touch_id_to_info.remove(&id);
        } else if phase == winit::event::TouchPhase::Moved {
            // A Moved for a touch we never saw start (its Started was dropped or
            // already Ended) must not panic -- just ignore it.
            if let Some(touch_info) = self.touch_id_to_info.get_mut(&id) {
                touch_info.frame_delta.0 = touch_info.current_pos.0 - location.x;
                touch_info.frame_delta.1 = touch_info.current_pos.1 - location.y;
                touch_info.current_pos.0 = location.x;
                touch_info.current_pos.1 = location.y;
            }
        }
    }

    pub fn set_mouse_button_state(&mut self, button: &MouseButton, state: &ElementState) -> bool {
        let button_name = match button {
            MouseButton::Left => "mouse_left",
            MouseButton::Right => "mouse_right",
            MouseButton::Middle => "mouse_middle",
            _ => "none",
        };
        if button_name == "none" {
            return false;
        }

        let key_pair = self.key_map.get_mut(&button_name);
        if key_pair.is_some() {
            let key_pair = key_pair.unwrap();
            if *state == ElementState::Pressed {
                if *key_pair == KbButtonState::None {
                    *key_pair = KbButtonState::JustPressed;
                }
            } else {
                *key_pair = KbButtonState::None;
            };
        } else if *state == ElementState::Pressed {
            // Only a press creates a new entry.  A release for a button we
            // never recorded (e.g. its press was swallowed elsewhere) must not
            // fabricate a JustPressed.
            self.key_map.insert(button_name, KbButtonState::JustPressed);
        }

        true
    }

    pub fn set_key_state(&mut self, key: PhysicalKey, state: ElementState) -> bool {
        let key_name = match key {
            PhysicalKey::Code(KeyCode::ArrowUp) => "up_arrow",
            PhysicalKey::Code(KeyCode::ArrowDown) => "down_arrow",
            PhysicalKey::Code(KeyCode::ArrowLeft) => "left_arrow",
            PhysicalKey::Code(KeyCode::ArrowRight) => "right_arrow",
            PhysicalKey::Code(KeyCode::Digit1) => "1",
            PhysicalKey::Code(KeyCode::Digit2) => "2",
            PhysicalKey::Code(KeyCode::Digit3) => "3",
            PhysicalKey::Code(KeyCode::Digit4) => "4",
            PhysicalKey::Code(KeyCode::Digit5) => "5",
            PhysicalKey::Code(KeyCode::Digit6) => "6",
            PhysicalKey::Code(KeyCode::Digit7) => "7",
            PhysicalKey::Code(KeyCode::Digit8) => "8",
            PhysicalKey::Code(KeyCode::Digit9) => "9",
            PhysicalKey::Code(KeyCode::Digit0) => "0",
            PhysicalKey::Code(KeyCode::Equal) => "+",
            PhysicalKey::Code(KeyCode::Minus) => "-",
            PhysicalKey::Code(KeyCode::KeyW) => "w",
            PhysicalKey::Code(KeyCode::KeyA) => "a",
            PhysicalKey::Code(KeyCode::KeyS) => "s",
            PhysicalKey::Code(KeyCode::KeyD) => "d",
            PhysicalKey::Code(KeyCode::Space) => "space",
            PhysicalKey::Code(KeyCode::KeyH) => "h",
            PhysicalKey::Code(KeyCode::KeyL) => "l",
            PhysicalKey::Code(KeyCode::KeyM) => "m",
            PhysicalKey::Code(KeyCode::KeyY) => "y",
            PhysicalKey::Code(KeyCode::KeyV) => "v",
            PhysicalKey::Code(KeyCode::ShiftLeft) => "left_shift",
            _ => "none",
        };
        if key_name == "none" {
            return false;
        }

        let key_pair = self.key_map.get_mut(&key_name);
        if key_pair.is_some() {
            let key_pair = key_pair.unwrap();
            if state == ElementState::Pressed {
                if *key_pair == KbButtonState::None {
                    *key_pair = KbButtonState::JustPressed;
                }
            } else {
                *key_pair = KbButtonState::None;
            };
        } else {
            self.key_map.insert(key_name, KbButtonState::JustPressed);
        }
        true
    }

    pub fn update_key_states(&mut self) {
        for button_pair in &mut self.key_map {
            if *button_pair.1 == KbButtonState::JustPressed {
                *button_pair.1 = KbButtonState::Down {
                    mouse_start: (self.cursor_position.0, self.cursor_position.1),
                }
            }
        }

        for touch in &mut self.touch_id_to_info {
            touch.1.frame_delta.0 = 0.0;
            touch.1.frame_delta.1 = 0.0;
            touch.1.touch_state = KbButtonState::Down {
                mouse_start: (0, 0),
            }
        }

        self.mouse_scroll_delta = 0.0;
        self.mouse_raw_delta = (0.0, 0.0);
    }

    pub fn get_key_state(&self, key: &str) -> KbButtonState {
        let key_state_pair = self.key_map.get(key);
        if let Some(key_state) = key_state_pair {
            return key_state.clone();
        }

        KbButtonState::None
    }

    pub fn get_touch_map(&self) -> &HashMap<u64, KbTouchInfo> {
        &self.touch_id_to_info
    }

    pub fn get_mouse_scroll_delta(&self) -> f32 {
        self.mouse_scroll_delta
    }

    pub fn set_mouse_position(&mut self, position: &winit::dpi::PhysicalPosition<f64>) {
        self.cursor_position.0 = position.x as i32;
        self.cursor_position.1 = position.y as i32;
    }

    pub fn get_mouse_position(&self) -> (i32, i32) {
        self.cursor_position
    }
}
