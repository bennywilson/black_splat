//! Persisted, user-editable editor preferences for the splat demo: viewport
//! gizmo hotkeys, shadow settings, and the startup scene JSON.
//!
//! On native it's saved to a per-user config file (see [`config_file_path`]);
//! on web there's no filesystem, so [`EditorConfig::load`]/[`save`](EditorConfig::save)
//! degrade to "defaults in memory / no-op".
//!
//! Keys are stored as [`egui::Key`] and serialized by their
//! [`name`](egui::Key::name) (`"W"`, `"E"`, `"Left"`, ...), which
//! [`egui::Key::from_name`] parses back, so the file stays human-readable and
//! editable by hand.

use black_splat::{editor::GizmoMode, egui};

/// Gizmo modes a hotkey can switch to, paired with its toolbar label.
/// Slot order is shared: the gizmo toolbar, the keybindings window,
/// [`EditorConfig::gizmo_keys`], and [`CONFIG_KEYS`] all index by it, so slot
/// `i` means the same action everywhere.
pub const GIZMO_ACTIONS: [(GizmoMode, &str); 3] = [
    (GizmoMode::Translate, "Move"),
    (GizmoMode::Rotate, "Rotate"),
    (GizmoMode::Scale, "Scale"),
];

/// On-disk key for each entry of [`GIZMO_ACTIONS`] (same order).  Kept separate
/// from the toolbar labels so renaming a button doesn't invalidate saved files.
/// Only the native build reads/writes the file, so it's unused on web.
#[cfg(not(target_arch = "wasm32"))]
const CONFIG_KEYS: [&str; 3] = ["gizmo_translate", "gizmo_rotate", "gizmo_scale"];

/// Editor preferences that persist across runs.
#[derive(Clone, Copy, PartialEq)]
pub struct EditorConfig {
    /// Hotkey per [`GIZMO_ACTIONS`] slot.  Kept distinct by [`rebind`](Self::rebind).
    pub gizmo_keys: [egui::Key; 3],
    /// Shadow tile resolution (Settings tab > Shadows), pushed to the
    /// renderer's `ShadowSettings` at startup and whenever edited.  The rest of
    /// the shadow controls are per-light and live in the scene, not here.
    pub shadow_resolution: u32,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            // The Unity-style W / E / R.
            gizmo_keys: [egui::Key::W, egui::Key::E, egui::Key::R],
            // Mirrors black_splat::passes::deferred::ShadowSettings::default().
            shadow_resolution: 1024,
        }
    }
}

impl EditorConfig {
    /// Binds `key` to `slot`. If another action already uses `key`, the two
    /// slots swap keys, so every action keeps a hotkey and none collide.
    pub fn rebind(&mut self, slot: usize, key: egui::Key) {
        if slot >= self.gizmo_keys.len() {
            return;
        }
        let previous = self.gizmo_keys[slot];
        if let Some(other) = self.gizmo_keys.iter().position(|k| *k == key) {
            self.gizmo_keys[other] = previous;
        }
        self.gizmo_keys[slot] = key;
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl EditorConfig {
    /// Serializes to the simple `name = value` line format (also the on-disk
    /// format), e.g. `gizmo_translate = W`.
    fn serialize(&self) -> String {
        let mut text = String::from("# black_splat editor preferences\n");
        for (slot, id) in CONFIG_KEYS.iter().enumerate() {
            text.push_str(&format!("{id} = {}\n", self.gizmo_keys[slot].name()));
        }
        text.push_str(&format!("shadow_resolution = {}\n", self.shadow_resolution));
        text
    }

    /// Parses the [`serialize`](Self::serialize) format on top of the defaults,
    /// so unknown/missing/renamed lines simply keep their default value.
    fn parse(text: &str) -> Self {
        let mut config = Self::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((name, value)) = line.split_once('=') else {
                continue;
            };
            let (name, value) = (name.trim(), value.trim());
            if let Some(slot) = CONFIG_KEYS.iter().position(|id| *id == name) {
                if let Some(key) = egui::Key::from_name(value) {
                    config.gizmo_keys[slot] = key;
                }
            } else if name == "shadow_resolution" {
                if let Ok(v) = value.parse() {
                    config.shadow_resolution = v;
                }
            }
        }
        config
    }

    /// Loads the saved config, falling back to defaults if the file is missing
    /// or unreadable (first run, no home dir, etc.).
    pub fn load() -> Self {
        config_file_path()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|text| Self::parse(&text))
            .unwrap_or_default()
    }

    /// Writes the config to disk, creating the parent directory if needed.
    pub fn save(&self) {
        let Some(path) = config_file_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, self.serialize());
    }
}

#[cfg(target_arch = "wasm32")]
impl EditorConfig {
    /// No filesystem in the browser: always start from defaults.
    pub fn load() -> Self {
        Self::default()
    }

    /// No filesystem in the browser: nothing to persist to.
    pub fn save(&self) {}
}

// --- Startup scene -----------------------------------------------------------
// The scene JSON loaded when the editor starts, stored as a sibling file of the
// keybindings (`startup_scene.json`).  Same persistence rules as the rest of
// the config: native reads/writes the per-user file, web keeps the built-in
// default (no filesystem).

/// The user's saved startup scene JSON, if any.  `None` means "use the built-in
/// default scene".
#[cfg(not(target_arch = "wasm32"))]
pub fn load_startup_scene() -> Option<String> {
    let path = config_file_path()?.with_file_name("startup_scene.json");
    std::fs::read_to_string(path).ok()
}

/// Saves `json` as the startup scene.  Best-effort, like [`EditorConfig::save`].
#[cfg(not(target_arch = "wasm32"))]
pub fn save_startup_scene(json: &str) {
    let Some(path) = config_file_path() else {
        return;
    };
    let path = path.with_file_name("startup_scene.json");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, json);
}

/// Removes the saved startup scene, reverting to the built-in default.
#[cfg(not(target_arch = "wasm32"))]
pub fn clear_startup_scene() {
    if let Some(path) = config_file_path() {
        let _ = std::fs::remove_file(path.with_file_name("startup_scene.json"));
    }
}

// On the web the scene JSON persists in localStorage: the browser's small
// per-site key-value store (a few MB, kept on the user's disk, scoped to this
// origin).  Content, not a path -- browsers don't expose file paths at all --
// so the picker-chosen file's text is what's stored.
#[cfg(target_arch = "wasm32")]
const STARTUP_SCENE_KEY: &str = "black_splat_startup_scene";

#[cfg(target_arch = "wasm32")]
fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

#[cfg(target_arch = "wasm32")]
pub fn load_startup_scene() -> Option<String> {
    local_storage()?.get_item(STARTUP_SCENE_KEY).ok().flatten()
}

#[cfg(target_arch = "wasm32")]
pub fn save_startup_scene(json: &str) {
    if let Some(storage) = local_storage() {
        let _ = storage.set_item(STARTUP_SCENE_KEY, json);
    }
}

#[cfg(target_arch = "wasm32")]
pub fn clear_startup_scene() {
    if let Some(storage) = local_storage() {
        let _ = storage.remove_item(STARTUP_SCENE_KEY);
    }
}

/// The per-user config file: `<config dir>/black_splat/editor_config.txt`,
/// where the config dir is the platform's standard location (`%APPDATA%` on
/// Windows, `~/Library/Application Support` on macOS, `$XDG_CONFIG_HOME` or
/// `~/.config` elsewhere).  `None` only if the home/config env var is unset.
#[cfg(not(target_arch = "wasm32"))]
fn config_file_path() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let base: Option<PathBuf> = {
        #[cfg(target_os = "windows")]
        {
            std::env::var_os("APPDATA").map(PathBuf::from)
        }
        #[cfg(target_os = "macos")]
        {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        }
    };
    base.map(|dir| dir.join("black_splat").join("editor_config.txt"))
}
