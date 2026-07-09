//! Persisted, user-editable editor preferences for the splat demo.  Right now
//! that's just the viewport gizmo hotkeys, but the file is meant to grow.
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

/// The gizmo modes a hotkey can switch to, paired with their toolbar label, in
/// a fixed order.  This order is the single source of truth: the gizmo toolbar,
/// the keybindings window, and [`EditorConfig::gizmo_keys`] are all indexed by
/// it, and [`CONFIG_KEYS`] gives each slot its on-disk name.
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct EditorConfig {
    /// Hotkey per [`GIZMO_ACTIONS`] slot.  Kept distinct by [`rebind`](Self::rebind).
    pub gizmo_keys: [egui::Key; 3],
}

impl Default for EditorConfig {
    fn default() -> Self {
        // The Unity-style W / E / R.
        Self {
            gizmo_keys: [egui::Key::W, egui::Key::E, egui::Key::R],
        }
    }
}

impl EditorConfig {
    /// Binds `key` to `slot`.  Bindings are kept unique: if another action
    /// already uses `key`, it inherits the key `slot` is giving up (a swap), so
    /// no action is ever left without a hotkey and no two share one.
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
    /// Serializes to the simple `name = Key` line format (also the on-disk
    /// format), e.g. `gizmo_translate = W`.
    fn serialize(&self) -> String {
        let mut text = String::from("# black_splat editor keybindings\n");
        for (slot, id) in CONFIG_KEYS.iter().enumerate() {
            text.push_str(&format!("{id} = {}\n", self.gizmo_keys[slot].name()));
        }
        text
    }

    /// Parses the [`serialize`](Self::serialize) format on top of the defaults,
    /// so unknown/missing/renamed lines simply keep their default binding.
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
            if let Some(slot) = CONFIG_KEYS.iter().position(|id| *id == name.trim()) {
                if let Some(key) = egui::Key::from_name(value.trim()) {
                    config.gizmo_keys[slot] = key;
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
    /// Best-effort: a failure (read-only home, etc.) is silently ignored so it
    /// never takes down the editor.
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
