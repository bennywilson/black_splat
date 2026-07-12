use std::{collections::HashMap, path::Path, result::Result::Ok};
use image::GenericImageView;
use wgpu::ShaderModule;

use crate::{resource::*, log, make_handle, passes::model::*, utils::*};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let origin = location.origin().unwrap();
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            // Serve from the IndexedDB cache when present -- also the only source
            // for user-imported assets, which were never on the server -- else
            // fetch from /rust_assets/ and cache for next time.  Keyed by the
            // caller's relative path so imports and bundled assets stay distinct.
            let data = if let Some(cached) = crate::idb::get(file_name).await {
                cached
            } else {
                let base = Path::new(file_name).file_name().unwrap().to_str().unwrap();
                let url = format_url(&format!("/rust_assets/{}", base));
                let resp = reqwest::get(url).await?;
                // A 404 still yields a body (the error page); treat non-success
                // as an error so callers can fall back instead of decoding junk,
                // and so we don't cache the error page under this key.
                if !resp.status().is_success() {
                    anyhow::bail!("fetch {base} failed: HTTP {}", resp.status());
                }
                let bytes = resp.bytes().await?.to_vec();
                crate::idb::put(file_name, &bytes).await;
                bytes
            };
        } else {
            let data = std::fs::read(file_name)?;
        }
    }
    Ok(data)
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let path = Path::new(file_name);
            let file_name = format!("/rust_assets/{}", path.file_name().unwrap().to_str().unwrap());

            let url = format_url(&file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;
        } else {
            let txt = std::fs::read_to_string(file_name)?;
        }
    }

    Ok(txt)
}

make_handle!(Texture, TextureHandle, TextureAssetMappings);
make_handle!(ShaderModule, ShaderHandle, ShaderAssetMappings);
type ByteVec = Vec<u8>;
make_handle!(ByteVec, ByteFileHandle, ByteMappings);
make_handle!(Model, ModelHandle, ModelMappings);
make_handle!(Material, MaterialHandle, MaterialMappings);

/// How to build a [`Material`]: optional color / metallic / roughness texture
/// paths (a missing one falls back to the built-in 1x1 white) plus the
/// constants each is multiplied by.  `metal_texture` and `rough_texture` are
/// independent grayscale maps (read from their red channel) -- the material
/// editor's separate Metallic/Roughness inputs -- packed on load into a single
/// glTF-layout GPU texture (G = roughness, B = metallic) so the G-buffer
/// shader only ever samples one combined texture; `mr_constant` is (x
/// metallic multiplier, y roughness multiplier) -- with no texture the
/// constants pass through as the final PBR values.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MaterialDesc {
    pub color_texture: Option<String>,
    #[serde(default)]
    pub metal_texture: Option<String>,
    #[serde(default)]
    pub rough_texture: Option<String>,
    pub color_constant: CgVec4,
    pub mr_constant: CgVec4,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        MaterialDesc {
            color_texture: None,
            metal_texture: None,
            rough_texture: None,
            color_constant: CG_VEC4_ONE,
            mr_constant: CgVec4::new(0.0, 0.85, 0.0, 0.0),
        }
    }
}

/// A reusable surface description an actor can override its model's textures
/// with: a color texture and a packed metallic-roughness texture (see
/// [`MaterialDesc`]), each multiplied by a constant in the G-buffer shader
/// (output = texture * constant).  The bind group is laid out identically to
/// a model's texture bind group (0: color texture, 1: sampler, 2: second
/// texture), so passes can bind either interchangeably.
pub struct Material {
    pub color_texture: TextureHandle,
    pub mr_texture: TextureHandle,
    pub color_constant: CgVec4,
    pub mr_constant: CgVec4,
    pub bind_group: wgpu::BindGroup,
}

#[allow(dead_code)]
pub struct AssetManager {
    texture_mappings: TextureAssetMappings,
    shader_mappings: ShaderAssetMappings,
    model_mappings: ModelMappings,
    material_mappings: MaterialMappings,
    // Built-in 1x1 white texture backing material slots with no texture
    // assigned (constant-only materials).  Created on first material load.
    white_texture: Option<TextureHandle>,
    // Built-in checkerboard, used as the visible placeholder for a model with
    // no texture or material of its own (see Model::from_bytes).  Created on
    // first use.
    checker_texture: Option<TextureHandle>,

    file_to_string_buffer: HashMap<String, String>,
    file_to_byte_buffer: HashMap<String, ByteVec>,
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AssetManager {
    pub fn new() -> Self {
        let mut file_to_byte_buffer = HashMap::<String, ByteVec>::new();
        let mut file_to_string_buffer = HashMap::<String, String>::new();

        file_to_byte_buffer.insert(
            "postprocess_filter.png".to_string(),
            include_bytes!("../engine_assets/textures/postprocess_filter.png").to_vec(),
        );
        file_to_byte_buffer.insert(
            "sprite_sheet.png".to_string(),
            include_bytes!("../engine_assets/textures/sprite_sheet.png").to_vec(),
        );
        file_to_string_buffer.insert(
            "decal.wgsl".to_string(),
            include_str!("../engine_assets/shaders/decal.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "line.wgsl".to_string(),
            include_str!("../engine_assets/shaders/line.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "model.wgsl".to_string(),
            include_str!("../engine_assets/shaders/model.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "gaussian_splat.wgsl".to_string(),
            include_str!("../engine_assets/shaders/gaussian_splat.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "gaussian_splat_radix.wgsl".to_string(),
            include_str!("../engine_assets/shaders/gaussian_splat_radix.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "splat_composite.wgsl".to_string(),
            include_str!("../engine_assets/shaders/splat_composite.wgsl").to_string(),
        );
        file_to_byte_buffer.insert(
            "scorch_t.png".to_string(),
            include_bytes!("../engine_assets/textures/scorch_t.png").to_vec(),
        );
        file_to_string_buffer.insert(
            "model_with_holes.wgsl".to_string(),
            include_str!("../engine_assets/shaders/model_with_holes.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "bullet_hole.wgsl".to_string(),
            include_str!("../engine_assets/shaders/bullet_hole.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "particle.wgsl".to_string(),
            include_str!("../engine_assets/shaders/particle.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "postprocess_uber.wgsl".to_string(),
            include_str!("../engine_assets/shaders/postprocess_uber.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "basic_sprite.wgsl".to_string(),
            include_str!("../engine_assets/shaders/basic_sprite.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "cloud_sprite.wgsl".to_string(),
            include_str!("../engine_assets/shaders/cloud_sprite.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "sky_dome_draw.wgsl".to_string(),
            include_str!("../engine_assets/shaders/sky_dome_draw.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "sky_dome_occlude.wgsl".to_string(),
            include_str!("../engine_assets/shaders/sky_dome_occlude.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "postprocess_uber.wgsl".to_string(),
            include_str!("../engine_assets/shaders/postprocess_uber.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "sunbeam_draw.wgsl".to_string(),
            include_str!("../engine_assets/shaders/sunbeam_draw.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "sunbeam_mask.wgsl".to_string(),
            include_str!("../engine_assets/shaders/sunbeam_mask.wgsl").to_string(),
        );
        file_to_byte_buffer.insert(
            "lens_flare.png".to_string(),
            include_bytes!("../engine_assets/textures/lens_flare.png").to_vec(),
        );
        file_to_string_buffer.insert(
            "gbuffer.wgsl".to_string(),
            include_str!("../engine_assets/shaders/gbuffer.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "light_skylight.wgsl".to_string(),
            include_str!("../engine_assets/shaders/light_skylight.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "light_directional.wgsl".to_string(),
            include_str!("../engine_assets/shaders/light_directional.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "light_point.wgsl".to_string(),
            include_str!("../engine_assets/shaders/light_point.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "light_spot.wgsl".to_string(),
            include_str!("../engine_assets/shaders/light_spot.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "shadow_depth.wgsl".to_string(),
            include_str!("../engine_assets/shaders/shadow_depth.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "shadow_mask_cascades.wgsl".to_string(),
            include_str!("../engine_assets/shaders/shadow_mask_cascades.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "shadow_mask_spot.wgsl".to_string(),
            include_str!("../engine_assets/shaders/shadow_mask_spot.wgsl").to_string(),
        );
        file_to_string_buffer.insert(
            "shadow_catcher_overlay.wgsl".to_string(),
            include_str!("../engine_assets/shaders/shadow_catcher_overlay.wgsl").to_string(),
        );

        #[cfg(feature = "wasm_include_3d")]
        {
            file_to_byte_buffer.insert(
                "ember_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/ember_t.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "fire_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/fire_t.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "smoke_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/smoke_t.png").to_vec(),
            );

            file_to_byte_buffer.insert(
                "muzzle_flash_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/muzzle_flash_t.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "monster_gibs_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/monster_gibs_t.png").to_vec(),
            );

            file_to_byte_buffer.insert(
                "barrel.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/barrel.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "decal.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/decal.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "fp_hands.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/fp_hands.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "level.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/level.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "monster.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/monster.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "pinky.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/pinky.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "sign.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/sign.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "sky_dome.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/sky_dome.glb").to_vec(),
            );
            file_to_byte_buffer.insert(
                "shotgun.glb".to_string(),
                include_bytes!("./../examples/3d/game_assets/models/shotgun.glb").to_vec(),
            );
            file_to_string_buffer.insert(
                "first_person.wgsl".to_string(),
                include_str!("./../examples/3d/game_assets/shaders/first_person.wgsl").to_string(),
            );
            file_to_string_buffer.insert(
                "first_person_outline.wgsl".to_string(),
                include_str!("./../examples/3d/game_assets/shaders/first_person_outline.wgsl")
                    .to_string(),
            );
            file_to_string_buffer.insert(
                "monster.wgsl".to_string(),
                include_str!("./../examples/3d/game_assets/shaders/monster.wgsl").to_string(),
            );
        }

        #[cfg(feature = "wasm_include_key")]
        {
            file_to_byte_buffer.insert(
                "ember_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/ember_t.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "smoke_t.png".to_string(),
                include_bytes!("./../examples/3d/game_assets/fx/smoke_t.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "map_00.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/map_00.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "map_01.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/map_01.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "map_10.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/map_10.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "map_11.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/map_11.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "map_20.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/map_20.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "map_21.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/map_21.png").to_vec(),
            );
            file_to_byte_buffer.insert(
                "timeline_atlas_1.png".to_string(),
                include_bytes!("./../../key/game_assets/textures/timeline_atlas_1.png").to_vec(),
            );
        }

        AssetManager {
            texture_mappings: TextureAssetMappings::new(),
            shader_mappings: ShaderAssetMappings::new(),
            model_mappings: ModelMappings::new(),
            material_mappings: MaterialMappings::new(),
            white_texture: None,
            checker_texture: None,

            file_to_string_buffer,
            file_to_byte_buffer,
        }
    }

    // Resolves and reads the raw bytes for `file_path` the same way a texture
    // load does: native prefixes the cwd and special-cases `engine_assets`/
    // `game_assets`; wasm prefers a build-time-baked buffer (keyed by
    // basename) and falls back to fetching. Shared with texture packing
    // (see load_metal_rough_texture), which needs to decode bytes on the CPU
    // rather than hand them straight to a GPU upload.
    async fn read_asset_bytes(&self, file_path: &str) -> Option<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut cwd: String = "".to_string();
            match std::env::current_dir() {
                Ok(dir) => {
                    cwd = format!("{}", dir.display());
                }
                _ => { /* todo use default texture*/ }
            };

            let final_file_path = {
                if file_path.chars().nth(1).unwrap() == ':' {
                    file_path.to_string()
                } else if file_path.contains("engine_assets") {
                    if Path::new("/./engine_assets").exists() {
                        format!("{cwd}/./{file_path}")
                    } else {
                        #[cfg(feature = "wasm_include_key")]
                        let path = format!("{cwd}/../kbengine3/{file_path}");

                        #[cfg(not(feature = "wasm_include_key"))]
                        let path = format!("{cwd}/../../{file_path}");

                        path
                    }
                } else if file_path.contains("game_assets") {
                    format!("{cwd}/./{file_path}")
                } else {
                    file_path.to_string()
                }
            };
            load_binary(&final_file_path).await.ok()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let path = Path::new(&file_path);
            let file_name = path.file_name().unwrap().to_str().unwrap();
            log!("Path returned {} ", file_name);

            // Prefer bytes baked in at build time (include_bytes! above,
            // gated by the wasm_include_* features). Textures a build didn't
            // compile in -- e.g. ones the editor loads at runtime -- are
            // fetched from /rust_assets/ instead, the same place the splat
            // .ply/.glb files are served from (mirrors load_model).
            match self.file_to_byte_buffer.get(file_name) {
                Some(buffer) => Some(buffer.clone()),
                None => load_binary(file_path).await.ok(),
            }
        }
    }

    pub async fn load_texture(
        &mut self,
        file_path: &str,
        device_resource: &DeviceResources<'_>,
    ) -> TextureHandle {
        if let Some(handle) = self.texture_mappings.names_to_handles.get(file_path) {
            return *handle;
        }

        log!("AssetManager loading texture {file_path}");

        // Attempt the load without panicking: a missing or undecodable texture
        // yields None, and we substitute the checkerboard below.
        let label = Path::new(file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file_path);
        let loaded: Option<Texture> = self.read_asset_bytes(file_path).await.and_then(|bytes| {
            Texture::from_bytes(
                &device_resource.device,
                &device_resource.queue,
                &bytes,
                label,
            )
            .ok()
        });

        let Some(new_texture) = loaded else {
            // Missing or undecodable: reuse the built-in checkerboard so it reads
            // as an obvious "missing texture" placeholder rather than sampling
            // garbage.  Cache it under this path so we don't retry every frame.
            log!("Texture {file_path} missing/failed to load; using checkerboard");
            let checker = self.checker_texture(device_resource);
            self.texture_mappings
                .names_to_handles
                .insert(file_path.to_string(), checker);
            return checker;
        };

        let mappings = &mut self.texture_mappings;
        if !mappings.next_handle.is_valid() {
            mappings.next_handle.index = 0;
        }
        let new_handle = mappings.next_handle;
        mappings.next_handle.index += 1;
        mappings.handles_to_assets.insert(new_handle, new_texture);
        mappings
            .names_to_handles
            .insert(file_path.to_string(), new_handle);

        new_handle
    }

    pub fn get_texture(&self, texture_handle: &TextureHandle) -> &Texture {
        &self.texture_mappings.handles_to_assets[texture_handle]
    }

    pub async fn load_shader(
        &mut self,
        file_path: &str,
        device_resources: &DeviceResources<'_>,
    ) -> ShaderHandle {
        let mappings = &mut self.shader_mappings;
        if let Some(handle) = mappings.names_to_handles.get(file_path) {
            return *handle;
        }

        log!("AssetManager loading shader {file_path}");
        let new_handle = {
            if !mappings.next_handle.is_valid() {
                mappings.next_handle.index = 0;
            }
            let new_handle = mappings.next_handle;
            mappings.next_handle.index += 1;
            new_handle
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        let shader_str = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut cwd: String = "".to_string();
                match std::env::current_dir() {
                    Ok(dir) => {
                        cwd = format!("{}", dir.display());
                    }
                    _ => { /* todo use default texture*/ }
                };
                let final_file_path = {
                    if file_path.chars().nth(1).unwrap() == ':' {
                        file_path.to_string()
                    } else if file_path.contains("engine_assets") {
                        if Path::new("/./engine_assets").exists() {
                            format!("{cwd}/./{file_path}")
                        } else {
                            #[cfg(feature = "wasm_include_key")]
                            let path = format!("{cwd}/../kbengine3/{file_path}");

                            #[cfg(not(feature = "wasm_include_key"))]
                            let path = format!("{cwd}/../../{file_path}");

                            path
                        }
                    } else {
                        file_path.to_string()
                    }
                };
                load_string(&final_file_path).await.unwrap()
            }
            #[cfg(target_arch = "wasm32")]
            {
                let path = Path::new(&file_path);
                let file_name = path.file_name().unwrap().to_str().unwrap();
                log!("Path returned {} ", file_name);
                self.file_to_string_buffer.get(file_name).unwrap()
            }
        };

        let new_shader =
            device_resources
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(file_path),
                    source: wgpu::ShaderSource::Wgsl(shader_str.into()),
                });

        mappings.handles_to_assets.insert(new_handle, new_shader);
        mappings
            .names_to_handles
            .insert(file_path.to_string(), new_handle);
        new_handle
    }

    pub fn get_shader(&self, shader_handle: &ShaderHandle) -> &ShaderModule {
        &self.shader_mappings.handles_to_assets[shader_handle]
    }

    pub async fn load_model(
        &mut self,
        file_path: &str,
        device_resource: &mut DeviceResources<'_>,
        use_holes: bool,
    ) -> ModelHandle {
        let new_model = {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut cwd: String = "".to_string();
                match std::env::current_dir() {
                    Ok(dir) => {
                        cwd = format!("{}", dir.display());
                    }
                    _ => { /* todo use default texture*/ }
                };

                let final_file_path = {
                    if file_path.chars().nth(1).unwrap() == ':' {
                        file_path.to_string()
                    } else if file_path.contains("engine_assets") {
                        if Path::new("/./engine_assets").exists() {
                            format!("{cwd}/./{file_path}")
                        } else {
                            format!("{cwd}/../../{file_path}")
                        }
                    } else if file_path.contains("game_assets") {
                        format!("{cwd}/./{file_path}")
                    } else {
                        file_path.to_string()
                    }
                };
                let bytes = load_binary(&final_file_path).await.unwrap();
                Model::from_bytes(&bytes, device_resource, self, use_holes).await
            }
            #[cfg(target_arch = "wasm32")]
            {
                let path = Path::new(&file_path);
                let file_name = path.file_name().unwrap().to_str().unwrap();
                // Prefer bytes baked in at build time (include_bytes! above,
                // gated by the wasm_include_* features). Models a build didn't
                // compile in -- e.g. ones the editor loads at runtime -- are
                // fetched from /rust_assets/ instead, the same place the splat
                // .ply files are served from.
                let byte_buffer = match self.file_to_byte_buffer.get(file_name) {
                    Some(buffer) => buffer.clone(),
                    None => load_binary(file_path).await.unwrap(),
                };
                Model::from_bytes(&byte_buffer, device_resource, self, use_holes).await
            }
        };
        log!("Model loaded");

        let mappings = &mut self.model_mappings;
        if let Some(handle) = mappings.names_to_handles.get(file_path) {
            return *handle;
        }

        log!("AssetManager loading model {file_path}");
        let new_handle = {
            if !mappings.next_handle.is_valid() {
                mappings.next_handle.index = 0;
            }
            let new_handle = mappings.next_handle;
            mappings.next_handle.index += 1;
            new_handle
        };
        mappings.handles_to_assets.insert(new_handle, new_model);
        mappings
            .names_to_handles
            .insert(file_path.to_string(), new_handle);

        new_handle
    }

    /// Registers a model directly from in-memory glb/gltf bytes, keyed by
    /// `file_path` (a later `load_model` with the same path returns this
    /// handle).  Synchronous counterpart to [`load_model`](Self::load_model)
    /// for the wasm frame tick, which can't `.await`: used by the editor's
    /// web model import.  `Model::from_bytes` only awaits when resolving
    /// URI-referenced external textures, which needs a filesystem and so never
    /// happens on wasm, so the future is driven to completion in place.
    #[cfg(target_arch = "wasm32")]
    pub fn add_model_from_bytes(
        &mut self,
        file_path: &str,
        bytes: &[u8],
        device_resource: &mut DeviceResources<'_>,
        use_holes: bool,
    ) -> ModelHandle {
        let bytes_vec = bytes.to_vec();
        let new_model = crate::utils::now_or_never(Model::from_bytes(
            &bytes_vec,
            device_resource,
            self,
            use_holes,
        ))
        .expect("glb import future must not suspend on wasm");

        let mappings = &mut self.model_mappings;
        // Re-importing a name overwrites the model but keeps its handle valid.
        if let Some(handle) = mappings.names_to_handles.get(file_path).copied() {
            mappings.handles_to_assets.insert(handle, new_model);
            return handle;
        }
        if !mappings.next_handle.is_valid() {
            mappings.next_handle.index = 0;
        }
        let new_handle = mappings.next_handle;
        mappings.next_handle.index += 1;
        mappings.handles_to_assets.insert(new_handle, new_model);
        mappings
            .names_to_handles
            .insert(file_path.to_string(), new_handle);
        new_handle
    }

    pub fn get_model(&mut self, model_handle: &ModelHandle) -> Option<&mut Model> {
        self.model_mappings.handles_to_assets.get_mut(model_handle)
    }

    pub fn get_model_mappings(&mut self) -> &mut HashMap<ModelHandle, Model> {
        &mut self.model_mappings.handles_to_assets
    }

    /// Every loaded model as (file path, handle), sorted by path -- feeds the
    /// editor's resource list and model dropdowns.
    pub fn get_model_resources(&self) -> Vec<(String, ModelHandle)> {
        let mut resources: Vec<(String, ModelHandle)> = self
            .model_mappings
            .get_names_to_handles()
            .iter()
            .map(|(name, handle)| (name.clone(), *handle))
            .collect();
        resources.sort_by(|a, b| a.0.cmp(&b.0));
        resources
    }

    /// The built-in 1x1 white texture (created on first use); backs material
    /// slots that have no texture assigned, so constants pass through as-is.
    fn white_texture(&mut self, device_resources: &DeviceResources<'_>) -> TextureHandle {
        if let Some(handle) = self.white_texture {
            return handle;
        }
        let texture = Texture::from_rgba(
            &[255, 255, 255, 255],
            true,
            1,
            1,
            device_resources,
            Some("white 1x1"),
        )
        .unwrap();
        let mappings = &mut self.texture_mappings;
        if !mappings.next_handle.is_valid() {
            mappings.next_handle.index = 0;
        }
        let handle = mappings.next_handle;
        mappings.next_handle.index += 1;
        mappings.handles_to_assets.insert(handle, texture);
        mappings
            .names_to_handles
            .insert("<white>".to_string(), handle);
        self.white_texture = Some(handle);
        handle
    }

    /// The built-in checkerboard texture (created on first use), an engine asset
    /// baked in so every project has it.  Used as the visible placeholder for a
    /// model with no texture/material, instead of leaving it garbled or blank.
    pub fn checker_texture(&mut self, device_resources: &DeviceResources<'_>) -> TextureHandle {
        if let Some(handle) = self.checker_texture {
            return handle;
        }
        let bytes = include_bytes!("../engine_assets/textures/checker_board.png");
        let texture = Texture::from_bytes(
            &device_resources.device,
            &device_resources.queue,
            bytes,
            "checker_board",
        )
        .unwrap();
        let mappings = &mut self.texture_mappings;
        if !mappings.next_handle.is_valid() {
            mappings.next_handle.index = 0;
        }
        let handle = mappings.next_handle;
        mappings.next_handle.index += 1;
        mappings.handles_to_assets.insert(handle, texture);
        mappings
            .names_to_handles
            .insert("<checker>".to_string(), handle);
        self.checker_texture = Some(handle);
        handle
    }

    /// Registers a named material, loading its textures (see [`MaterialDesc`]).
    /// Materials are keyed by name; loading a name again returns the existing
    /// handle.  The bind group mirrors a model's texture bind group layout
    /// (0: color texture, 1: sampler, 2: packed metal/rough texture), so
    /// passes can bind a material in place of the model's own textures.
    pub async fn load_material(
        &mut self,
        name: &str,
        desc: &MaterialDesc,
        device_resources: &DeviceResources<'_>,
    ) -> MaterialHandle {
        if let Some(handle) = self.material_mappings.names_to_handles.get(name) {
            return *handle;
        }

        let white = self.white_texture(device_resources);
        let color_texture = match &desc.color_texture {
            Some(path) => self.load_texture(path, device_resources).await,
            None => white,
        };
        let mr_texture = self
            .load_metal_rough_texture(
                desc.metal_texture.as_deref(),
                desc.rough_texture.as_deref(),
                device_resources,
            )
            .await;
        self.build_material(name, desc, color_texture, mr_texture, device_resources)
    }

    /// Synchronous material creation for texture-less (constant-only)
    /// materials -- e.g. ones made in the editor's Resources panel from the
    /// non-async frame tick.  Any texture paths in `desc` are ignored (the
    /// built-in white is used); use [`load_material`](Self::load_material) to
    /// load textures.
    pub fn create_material(
        &mut self,
        name: &str,
        desc: &MaterialDesc,
        device_resources: &DeviceResources<'_>,
    ) -> MaterialHandle {
        if let Some(handle) = self.material_mappings.names_to_handles.get(name) {
            return *handle;
        }
        let white = self.white_texture(device_resources);
        self.build_material(name, desc, white, white, device_resources)
    }

    /// Overwrites a material's color and metallic/roughness constants.  Takes
    /// effect immediately: the G-buffer pass reads them every frame.
    pub fn update_material_constants(
        &mut self,
        handle: &MaterialHandle,
        color_constant: &CgVec4,
        mr_constant: &CgVec4,
    ) {
        if let Some(material) = self.material_mappings.handles_to_assets.get_mut(handle) {
            material.color_constant = *color_constant;
            material.mr_constant = *mr_constant;
        }
    }

    /// Rebuilds an existing material in place from a new description, keeping
    /// its handle valid (so actors referencing it pick up the change) --
    /// unlike [`load_material`](Self::load_material), which early-returns the
    /// old material untouched when the name already exists.  Used by the editor
    /// when a material's textures or constants change.
    pub async fn reload_material(
        &mut self,
        handle: &MaterialHandle,
        name: &str,
        desc: &MaterialDesc,
        device_resources: &DeviceResources<'_>,
    ) {
        let white = self.white_texture(device_resources);
        let color_texture = match &desc.color_texture {
            Some(path) => self.load_texture(path, device_resources).await,
            None => white,
        };
        let mr_texture = self
            .load_metal_rough_texture(
                desc.metal_texture.as_deref(),
                desc.rough_texture.as_deref(),
                device_resources,
            )
            .await;
        let material =
            self.make_material(name, desc, color_texture, mr_texture, device_resources);
        self.material_mappings
            .handles_to_assets
            .insert(*handle, material);
    }

    /// Loads (or reuses) the packed glTF-layout metallic-roughness texture
    /// (G = roughness, B = metallic) for a material's independent Metallic/
    /// Roughness inputs.  Each side is a plain grayscale image (read from its
    /// red channel); a missing side fills its channel with 255 so its
    /// constant multiplier passes through unchanged, matching the built-in
    /// white texture's no-texture behavior.  Cached by the (metal, rough)
    /// path pair, so re-picking an already-seen combination skips the
    /// decode/pack.  Mismatched source resolutions are resized to match (the
    /// larger of the two) rather than left to silently misalign.
    async fn load_metal_rough_texture(
        &mut self,
        metal_path: Option<&str>,
        rough_path: Option<&str>,
        device_resources: &DeviceResources<'_>,
    ) -> TextureHandle {
        if metal_path.is_none() && rough_path.is_none() {
            return self.white_texture(device_resources);
        }

        let cache_key = format!(
            "<mr>{}|{}",
            metal_path.unwrap_or(""),
            rough_path.unwrap_or("")
        );
        if let Some(handle) = self.texture_mappings.names_to_handles.get(&cache_key) {
            return *handle;
        }

        let metal_img = match metal_path {
            Some(path) => self
                .read_asset_bytes(path)
                .await
                .and_then(|b| image::load_from_memory(&b).ok()),
            None => None,
        };
        let rough_img = match rough_path {
            Some(path) => self
                .read_asset_bytes(path)
                .await
                .and_then(|b| image::load_from_memory(&b).ok()),
            None => None,
        };

        let Some((width, height)) = metal_img
            .as_ref()
            .map(|i| i.dimensions())
            .or_else(|| rough_img.as_ref().map(|i| i.dimensions()))
        else {
            log!("Metal/roughness textures missing/failed to load; using checkerboard");
            let checker = self.checker_texture(device_resources);
            self.texture_mappings
                .names_to_handles
                .insert(cache_key, checker);
            return checker;
        };
        let to_gray = |img: image::DynamicImage| -> image::GrayImage {
            if img.dimensions() == (width, height) {
                img.to_luma8()
            } else {
                img.resize_exact(width, height, image::imageops::FilterType::Triangle)
                    .to_luma8()
            }
        };
        let metal_gray = metal_img.map(to_gray);
        let rough_gray = rough_img.map(to_gray);

        let mut packed = vec![0u8; (width * height * 4) as usize];
        for i in 0..(width * height) as usize {
            packed[i * 4 + 1] = rough_gray.as_ref().map_or(255, |g| g.as_raw()[i]);
            packed[i * 4 + 2] = metal_gray.as_ref().map_or(255, |g| g.as_raw()[i]);
            packed[i * 4 + 3] = 255;
        }

        let Ok(texture) = Texture::from_rgba(
            &packed,
            true,
            width,
            height,
            device_resources,
            Some(&cache_key),
        ) else {
            let checker = self.checker_texture(device_resources);
            self.texture_mappings
                .names_to_handles
                .insert(cache_key, checker);
            return checker;
        };

        let mappings = &mut self.texture_mappings;
        if !mappings.next_handle.is_valid() {
            mappings.next_handle.index = 0;
        }
        let handle = mappings.next_handle;
        mappings.next_handle.index += 1;
        mappings.handles_to_assets.insert(handle, texture);
        mappings.names_to_handles.insert(cache_key, handle);
        handle
    }

    // Shared tail of load_material/create_material: builds the bind group
    // from already-resolved textures and registers the material under `name`.
    fn build_material(
        &mut self,
        name: &str,
        desc: &MaterialDesc,
        color_texture: TextureHandle,
        mr_texture: TextureHandle,
        device_resources: &DeviceResources<'_>,
    ) -> MaterialHandle {
        let material =
            self.make_material(name, desc, color_texture, mr_texture, device_resources);
        let mappings = &mut self.material_mappings;
        if !mappings.next_handle.is_valid() {
            mappings.next_handle.index = 0;
        }
        let new_handle = mappings.next_handle;
        mappings.next_handle.index += 1;
        mappings.handles_to_assets.insert(new_handle, material);
        mappings
            .names_to_handles
            .insert(name.to_string(), new_handle);
        new_handle
    }

    // Builds a Material (bind group + constants) from already-resolved
    // textures, without registering it -- callers place it under a new or
    // existing handle (see build_material / reload_material).
    fn make_material(
        &self,
        name: &str,
        desc: &MaterialDesc,
        color_texture: TextureHandle,
        mr_texture: TextureHandle,
        device_resources: &DeviceResources<'_>,
    ) -> Material {
        log!("AssetManager loading material {name}");
        let device = &device_resources.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
            label: Some("Material::bind_group_layout"),
        });
        let color = self.get_texture(&color_texture);
        let mr = self.get_texture(&mr_texture);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&color.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&mr.view),
                },
            ],
            label: Some(&format!("Material::{name}")),
        });

        Material {
            color_texture,
            mr_texture,
            color_constant: desc.color_constant,
            mr_constant: desc.mr_constant,
            bind_group,
        }
    }

    pub fn get_material(&self, handle: &MaterialHandle) -> Option<&Material> {
        self.material_mappings.handles_to_assets.get(handle)
    }

    /// Every loaded material as (name, handle), sorted by name -- feeds the
    /// editor's material dropdown.
    pub fn get_material_resources(&self) -> Vec<(String, MaterialHandle)> {
        let mut resources: Vec<(String, MaterialHandle)> = self
            .material_mappings
            .get_names_to_handles()
            .iter()
            .map(|(name, handle)| (name.clone(), *handle))
            .collect();
        resources.sort_by(|a, b| a.0.cmp(&b.0));
        resources
    }

    /// Simultaneous access to the model map (mutable, for per-frame uniform
    /// allocation) and the material map (read-only) -- the G-buffer pass needs
    /// both while recording draws.
    pub fn get_models_and_materials(
        &mut self,
    ) -> (
        &mut HashMap<ModelHandle, Model>,
        &HashMap<MaterialHandle, Material>,
    ) {
        (
            &mut self.model_mappings.handles_to_assets,
            &self.material_mappings.handles_to_assets,
        )
    }
}
