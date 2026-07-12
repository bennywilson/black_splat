//! Persistence for the editor's material and particle libraries.  Native: each
//! resource is its own JSON file under `resources/` (UE-style one-file-per-
//! asset), so they can be version-controlled and shared individually.  Web: the
//! same JSON is kept in localStorage (one key per resource), since the browser
//! has no writable filesystem.  Texture import is native-only either way (it
//! copies into game_assets/).

use black_splat::assets::MaterialDesc;
use black_splat::game_object::ParticleParams;
use serde::{Deserialize, Serialize};

// Native-only: the on-disk layout.  The web build keys off localStorage and
// has no writable game_assets, so these are unused there.
#[cfg(not(target_arch = "wasm32"))]
const MATERIALS_DIR: &str = "resources/materials";
#[cfg(not(target_arch = "wasm32"))]
const PARTICLES_DIR: &str = "resources/particles";
/// Where imported textures are copied so scenes stay portable (relative paths).
#[cfg(not(target_arch = "wasm32"))]
const TEXTURES_DIR: &str = "game_assets/textures";
/// Also surfaced as texture resources: the bundled particle/fx textures.
#[cfg(not(target_arch = "wasm32"))]
const FX_DIR: &str = "game_assets/fx";
#[cfg(not(target_arch = "wasm32"))]
const MODELS_DIR: &str = "game_assets/models";
// Not gated: the wasm scan also classifies IndexedDB keys by these (see the
// wasm `scan`), so a model's own texture maps aren't listed as models.
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg"];
const MODEL_EXTS: &[&str] = &["glb", "gltf"];

/// One material on disk: a display name plus its full description.
#[derive(Serialize, Deserialize)]
pub struct MaterialFile {
    pub name: String,
    pub desc: MaterialDesc,
}

/// One particle definition on disk: a display name plus its full params.
#[derive(Serialize, Deserialize)]
pub struct ParticleFile {
    pub name: String,
    pub params: ParticleParams,
}

/// Turns a resource's display name into a filesystem-safe file stem, so names
/// with spaces or punctuation still land on disk predictably.  (Two names that
/// differ only in punctuation could collide; acceptable for an editor library.)
#[cfg(not(target_arch = "wasm32"))]
fn file_stem(name: &str) -> String {
    let stem: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if stem.is_empty() {
        "unnamed".to_string()
    } else {
        stem
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::*;
    use std::path::{Path, PathBuf};

    pub fn material_path(name: &str) -> PathBuf {
        Path::new(MATERIALS_DIR).join(format!("{}.json", file_stem(name)))
    }
    pub fn particle_path(name: &str) -> PathBuf {
        Path::new(PARTICLES_DIR).join(format!("{}.json", file_stem(name)))
    }

    pub fn save_material(name: &str, desc: &MaterialDesc) -> Result<(), String> {
        std::fs::create_dir_all(MATERIALS_DIR).map_err(|e| e.to_string())?;
        let file = MaterialFile {
            name: name.to_string(),
            desc: desc.clone(),
        };
        let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        std::fs::write(material_path(name), json).map_err(|e| e.to_string())
    }

    pub fn save_particle(name: &str, params: &ParticleParams) -> Result<(), String> {
        std::fs::create_dir_all(PARTICLES_DIR).map_err(|e| e.to_string())?;
        let file = ParticleFile {
            name: name.to_string(),
            params: params.clone(),
        };
        let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        std::fs::write(particle_path(name), json).map_err(|e| e.to_string())
    }

    fn load_dir<T: for<'de> Deserialize<'de>>(dir: &str) -> Vec<T> {
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else {
            return out; // Missing folder just means an empty library.
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |x| x == "json"))
            .collect();
        paths.sort(); // Stable, name-sorted order in the browser.
        for path in paths {
            match std::fs::read_to_string(&path) {
                Ok(text) => match serde_json::from_str::<T>(&text) {
                    Ok(v) => out.push(v),
                    Err(e) => eprintln!("Skipping bad resource {}: {e}", path.display()),
                },
                Err(e) => eprintln!("Can't read resource {}: {e}", path.display()),
            }
        }
        out
    }

    pub fn load_materials() -> Vec<MaterialFile> {
        load_dir(MATERIALS_DIR)
    }
    pub fn load_particles() -> Vec<ParticleFile> {
        load_dir(PARTICLES_DIR)
    }

    // Used by rename-save to remove a resource's previous file.
    pub fn delete_material(name: &str) {
        let _ = std::fs::remove_file(material_path(name));
    }
    pub fn delete_particle(name: &str) {
        let _ = std::fs::remove_file(particle_path(name));
    }

    /// Relative paths of every image the editor offers as a texture resource:
    /// imports under game_assets/textures/, the bundled fx textures, and any
    /// model's own source maps (e.g. game_assets/models/Barrel/*.jpg), so a
    /// per-asset folder's textures show up in the browser next to its model.
    /// Recurses like `scan_models`, since those per-asset folders nest.
    /// (async only to share a signature with the wasm build, which reads a
    /// manifest + IndexedDB; native just walks the folders.)
    pub async fn scan_textures() -> Vec<String> {
        let mut out = Vec::new();
        for dir in [TEXTURES_DIR, FX_DIR, MODELS_DIR] {
            collect_images(Path::new(dir), &mut out);
        }
        out.sort();
        out.dedup();
        out
    }

    fn collect_images(dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return; // Missing folder just means an empty library.
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_images(&path, out);
                continue;
            }
            let is_image = path
                .extension()
                .and_then(|x| x.to_str())
                .map_or(false, |ext| {
                    IMAGE_EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext))
                });
            if is_image {
                // Forward slashes so paths match the wasm manifest / load keys.
                out.push(path.to_string_lossy().replace('\\', "/"));
            }
        }
    }

    /// Copies an imported image into game_assets/textures/ and returns the
    /// relative path materials should reference.
    pub fn import_texture(file_name: &str, bytes: &[u8]) -> Result<String, String> {
        std::fs::create_dir_all(TEXTURES_DIR).map_err(|e| e.to_string())?;
        let rel = format!("{TEXTURES_DIR}/{file_name}");
        std::fs::write(&rel, bytes).map_err(|e| e.to_string())?;
        Ok(rel)
    }

    /// Writes an imported model into game_assets/models/ so it persists and is
    /// picked up by `scan_models` on the next launch; returns its relative path.
    pub fn save_model(file_name: &str, bytes: &[u8]) -> Result<String, String> {
        std::fs::create_dir_all(MODELS_DIR).map_err(|e| e.to_string())?;
        let rel = format!("{MODELS_DIR}/{file_name}");
        std::fs::write(&rel, bytes).map_err(|e| e.to_string())?;
        Ok(rel)
    }

    /// Relative paths of every model under game_assets/models/, recursing into
    /// subfolders (per-asset folders like Barrel/ that bundle a glb with its
    /// source textures), so nested models show up in the editor too.
    pub async fn scan_models() -> Vec<String> {
        let mut out = Vec::new();
        collect_models(Path::new(MODELS_DIR), &mut out);
        out.sort();
        out
    }

    fn collect_models(dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return; // Missing folder just means an empty library.
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_models(&path, out);
                continue;
            }
            let is_model = path
                .extension()
                .and_then(|x| x.to_str())
                .map_or(false, |ext| {
                    MODEL_EXTS.iter().any(|e| e.eq_ignore_ascii_case(ext))
                });
            if is_model {
                // Forward slashes so paths match the wasm manifest / load keys.
                out.push(path.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::*;

    // The web build has no writable filesystem, so materials/particles persist
    // in localStorage instead -- one key per resource, prefixed by type.  It's
    // the same per-origin store the startup scene uses (a few MB, on the user's
    // disk, scoped to this site).
    const MATERIAL_PREFIX: &str = "black_splat_material:";
    const PARTICLE_PREFIX: &str = "black_splat_particle:";

    fn storage() -> Option<web_sys::Storage> {
        web_sys::window()?.local_storage().ok().flatten()
    }

    fn save(prefix: &str, name: &str, json: &str) -> Result<(), String> {
        storage()
            .ok_or_else(|| "localStorage unavailable".to_string())?
            .set_item(&format!("{prefix}{name}"), json)
            .map_err(|_| "localStorage write failed (quota?)".to_string())
    }

    pub fn save_material(name: &str, desc: &MaterialDesc) -> Result<(), String> {
        let file = MaterialFile {
            name: name.to_string(),
            desc: desc.clone(),
        };
        let json = serde_json::to_string(&file).map_err(|e| e.to_string())?;
        save(MATERIAL_PREFIX, name, &json)
    }

    pub fn save_particle(name: &str, params: &ParticleParams) -> Result<(), String> {
        let file = ParticleFile {
            name: name.to_string(),
            params: params.clone(),
        };
        let json = serde_json::to_string(&file).map_err(|e| e.to_string())?;
        save(PARTICLE_PREFIX, name, &json)
    }

    // Reads every localStorage entry whose key starts with `prefix` and parses
    // it as a `T` (skipping any that fail).
    fn load_prefixed<T: for<'de> Deserialize<'de>>(prefix: &str) -> Vec<T> {
        let mut out = Vec::new();
        let Some(storage) = storage() else {
            return out;
        };
        let len = storage.length().unwrap_or(0);
        for i in 0..len {
            let Ok(Some(key)) = storage.key(i) else {
                continue;
            };
            if !key.starts_with(prefix) {
                continue;
            }
            if let Ok(Some(json)) = storage.get_item(&key) {
                if let Ok(value) = serde_json::from_str::<T>(&json) {
                    out.push(value);
                }
            }
        }
        out
    }

    pub fn load_materials() -> Vec<MaterialFile> {
        let mut v = load_prefixed::<MaterialFile>(MATERIAL_PREFIX);
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }
    pub fn load_particles() -> Vec<ParticleFile> {
        let mut v = load_prefixed::<ParticleFile>(PARTICLE_PREFIX);
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    pub fn delete_material(name: &str) {
        if let Some(s) = storage() {
            let _ = s.remove_item(&format!("{MATERIAL_PREFIX}{name}"));
        }
    }
    pub fn delete_particle(name: &str) {
        if let Some(s) = storage() {
            let _ = s.remove_item(&format!("{PARTICLE_PREFIX}{name}"));
        }
    }

    // The browser can't list a served directory, so the set of bundled assets
    // comes from a manifest the build step emits (see build.py); user imports
    // live in IndexedDB.  Both are keyed by the same game_assets-relative paths
    // the engine loads assets by, so scans are just "manifest + stored keys".
    #[derive(Deserialize, Default)]
    struct Manifest {
        #[serde(default)]
        models: Vec<String>,
        #[serde(default)]
        textures: Vec<String>,
    }

    async fn load_manifest() -> Manifest {
        // A missing/served-404 manifest yields non-JSON, which parses to the
        // empty default -- the browser then simply shows no bundled assets.
        match black_splat::assets::load_string("manifest.json").await {
            Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
            Err(_) => Manifest::default(),
        }
    }

    /// Whether `path`'s extension is one of `exts` (case-insensitive).
    fn has_ext(path: &str, exts: &[&str]) -> bool {
        std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map_or(false, |ext| exts.iter().any(|e| e.eq_ignore_ascii_case(ext)))
    }

    // Stored keys under any of `prefixes` whose extension is in `exts`, unioned
    // with `bundled`, deduped.  The extension gate is essential: load_binary
    // caches every fetched asset into IndexedDB keyed by its full path, and a
    // model's own texture maps live under the same game_assets/models/ prefix
    // as the .glb -- so matching on prefix alone would list a model's cached
    // textures as models (and cached models as textures) on the next launch.
    // Mirrors build.py's by-extension manifest classification.
    async fn scan(bundled: Vec<String>, prefixes: &[&str], exts: &[&str]) -> Vec<String> {
        let mut out = bundled;
        for key in black_splat::idb::keys().await {
            if prefixes.iter().any(|p| key.starts_with(p)) && has_ext(&key, exts) {
                out.push(key);
            }
        }
        out.sort();
        out.dedup();
        out
    }

    pub async fn scan_textures() -> Vec<String> {
        scan(
            load_manifest().await.textures,
            &["game_assets/textures/", "game_assets/fx/", "game_assets/models/"],
            IMAGE_EXTS,
        )
        .await
    }

    pub async fn scan_models() -> Vec<String> {
        scan(
            load_manifest().await.models,
            &["game_assets/models/"],
            MODEL_EXTS,
        )
        .await
    }

    // Texture import copies a file into game_assets/ -- impossible in the
    // browser sandbox -- so it stays native-only for now.  (Models import via
    // IndexedDB; see example_game's picked_model handling.)
    pub fn import_texture(_file_name: &str, _bytes: &[u8]) -> Result<String, String> {
        Err("texture import isn't supported on the web build yet".to_string())
    }
}

pub use imp::*;
