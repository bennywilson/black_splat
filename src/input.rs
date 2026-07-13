use std::collections::HashMap;

use winit::{
    event::{ElementState, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ButtonState {
    #[default]
    None,
    JustPressed,
    Down {
        mouse_start: (i32, i32),
    },
}

impl ButtonState {
    pub fn is_down(&self) -> bool {
        matches!(*self, ButtonState::Down { .. })
    }

    pub fn just_pressed(&self) -> bool {
        *self == ButtonState::JustPressed
    }
}

#[derive(Debug, Default)]
pub struct TouchInfo {
    pub start_pos: (f64, f64),
    pub current_pos: (f64, f64),
    pub frame_delta: (f64, f64),
    pub touch_state: ButtonState,
}

#[derive(Debug, Default)]
pub struct InputManager {
    touch_id_to_info: HashMap<u64, TouchInfo>,
    mouse_scroll_delta: f32,
    cursor_position: (i32, i32),
    // Raw mouse motion (winit DeviceEvent::MouseMotion) accumulated this frame in device units.
    mouse_raw_delta: (f64, f64),
    key_map: HashMap<&'static str, ButtonState>,
}

#[allow(dead_code)]
impl InputManager {
    pub fn new() -> Self {
        let key_map = HashMap::<&str, ButtonState>::new();
        InputManager {
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
            let touch_info = TouchInfo {
                start_pos: location.into(),
                current_pos: location.into(),
                frame_delta: (0.0, 0.0),
                touch_state: ButtonState::JustPressed,
            };
            self.touch_id_to_info.insert(id, touch_info);
        } else if phase == winit::event::TouchPhase::Cancelled
            || phase == winit::event::TouchPhase::Ended
        {
            self.touch_id_to_info.remove(&id);
        } else if phase == winit::event::TouchPhase::Moved {
            // Ignore a touch move that we never saw start (its Started was dropped or
            // already Ended) 
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

        if let Some(key_pair) = self.key_map.get_mut(&button_name) {
            if *state == ElementState::Pressed {
                if *key_pair == ButtonState::None {
                    *key_pair = ButtonState::JustPressed;
                }
            } else {
                *key_pair = ButtonState::None;
            };
        } else if *state == ElementState::Pressed {
            self.key_map.insert(button_name, ButtonState::JustPressed);
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

        if let Some(key_pair) = self.key_map.get_mut(&key_name) {
            if state == ElementState::Pressed {
                if *key_pair == ButtonState::None {
                    *key_pair = ButtonState::JustPressed;
                }
            } else {
                *key_pair = ButtonState::None;
            };
        } else if state == ElementState::Pressed {
            self.key_map.insert(key_name, ButtonState::JustPressed);
        }
        true
    }

    pub fn update_key_states(&mut self) {
        for button_pair in &mut self.key_map {
            if *button_pair.1 == ButtonState::JustPressed {
                *button_pair.1 = ButtonState::Down {
                    mouse_start: (self.cursor_position.0, self.cursor_position.1),
                }
            }
        }

        for touch in &mut self.touch_id_to_info {
            touch.1.frame_delta.0 = 0.0;
            touch.1.frame_delta.1 = 0.0;
            touch.1.touch_state = ButtonState::Down {
                mouse_start: (0, 0),
            }
        }

        self.mouse_scroll_delta = 0.0;
        self.mouse_raw_delta = (0.0, 0.0);
    }

    pub fn get_key_state(&self, key: &str) -> ButtonState {
        let key_state_pair = self.key_map.get(key);
        if let Some(key_state) = key_state_pair {
            return key_state.clone();
        }

        ButtonState::None
    }

    pub fn get_touch_map(&self) -> &HashMap<u64, TouchInfo> {
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
