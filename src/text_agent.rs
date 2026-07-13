//! Mobile-web virtual keyboard support.  Browsers only show the on-screen
//! keyboard while a real DOM editable has focus -- keystrokes aimed at
//! winit's canvas never summon it.  So a hidden `<input>` (the "text agent")
//! is kept focused whenever egui has a text field focused, and whatever it
//! receives is forwarded to egui as events.  Desktop browsers route through
//! it too when a field is focused, which is harmless: typed text, Backspace
//! and Enter all come through (fancier shortcuts don't -- acceptable for the
//! editor's simple fields).

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

const AGENT_ID: &str = "egui-text-agent";

pub struct TextAgent {
    input: web_sys::HtmlInputElement,
    // Events captured by the DOM listeners, drained each frame.
    events: Rc<RefCell<Vec<egui::Event>>>,
}

// egui expects a press+release pair per keystroke.
fn push_key(queue: &mut Vec<egui::Event>, key: egui::Key) {
    for pressed in [true, false] {
        queue.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
    }
}

impl TextAgent {
    /// Creates the hidden input and hooks its listeners.  Returns None if the
    /// DOM isn't available, creation fails, or this isn't a touch device.
    ///
    /// Desktop web is deliberately opted out: there, keyboard input already
    /// flows through the canvas (winit's default path), and focusing a hidden
    /// input to capture it instead would only *steal* focus from the canvas
    /// and break typing.  The agent exists solely to summon the on-screen
    /// keyboard on touch, so it's installed only where that keyboard exists.
    pub fn install() -> Option<Self> {
        let window = web_sys::window()?;
        if window.navigator().max_touch_points() <= 0 {
            return None;
        }
        let document = window.document()?;
        let input: web_sys::HtmlInputElement =
            document.create_element("input").ok()?.dyn_into().ok()?;
        input.set_id(AGENT_ID);
        input.set_type("text");
        input.set_autocomplete("off");
        let _ = input.set_attribute("autocapitalize", "off");
        // Invisible but still focusable (display:none can't take focus).
        let style = input.style();
        for (key, value) in [
            ("position", "absolute"),
            ("top", "0"),
            ("left", "0"),
            ("opacity", "0"),
            ("width", "1px"),
            ("height", "1px"),
            ("z-index", "-1"),
        ] {
            let _ = style.set_property(key, value);
        }
        document.body()?.append_child(&input).ok()?;

        let events = Rc::new(RefCell::new(Vec::new()));

        // Typed characters arrive as "input" events.  Android virtual
        // keyboards often report Backspace only here (keydown says
        // "Unidentified"), so deletes are handled in both listeners.  They
        // can't double up: the input is cleared after every event, and an
        // empty input never produces a deleteContentBackward.
        let queue = events.clone();
        let field = input.clone();
        let on_input =
            Closure::<dyn FnMut(web_sys::InputEvent)>::new(move |event: web_sys::InputEvent| {
                let mut queue = queue.borrow_mut();
                if event.input_type() == "deleteContentBackward" {
                    push_key(&mut queue, egui::Key::Backspace);
                } else if let Some(text) = event.data() {
                    if !text.is_empty() {
                        queue.push(egui::Event::Text(text));
                    }
                }
                field.set_value("");
            });
        input
            .add_event_listener_with_callback("input", on_input.as_ref().unchecked_ref())
            .ok()?;
        on_input.forget();

        let queue = events.clone();
        let on_keydown = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(
            move |event: web_sys::KeyboardEvent| {
                let mut queue = queue.borrow_mut();
                match event.key().as_str() {
                    "Backspace" => push_key(&mut queue, egui::Key::Backspace),
                    "Enter" => push_key(&mut queue, egui::Key::Enter),
                    _ => {}
                }
            },
        );
        input
            .add_event_listener_with_callback("keydown", on_keydown.as_ref().unchecked_ref())
            .ok()?;
        on_keydown.forget();

        Some(TextAgent { input, events })
    }

    /// Appends whatever the agent captured since last frame to egui's raw
    /// input.  Call before handing the frame's input to egui.
    pub fn drain_into(&self, raw_input: &mut egui::RawInput) {
        raw_input.events.append(&mut self.events.borrow_mut());
    }

    /// Keeps the agent's DOM focus in step with egui: focused while egui has
    /// a text field focused (this is what summons the on-screen keyboard),
    /// blurred otherwise (dropping it and returning keys to the canvas).
    pub fn set_focus(&self, focus: bool) {
        let agent_focused = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.active_element())
            .is_some_and(|e| e.id() == AGENT_ID);
        if focus && !agent_focused {
            let _ = self.input.focus();
        } else if !focus && agent_focused {
            let _ = self.input.blur();
        }
    }
}
