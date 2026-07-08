use std::{collections::HashMap, path::Path, result::Result::Ok};
use wgpu::ShaderModule;

use crate::{resource::*, log, make_handle, passes::model::*};

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
            let path = Path::new(file_name);
            let file_name = format!("/rust_assets/{}", path.file_name().unwrap().to_str().unwrap());

            let url = format_url(&file_name);
            let data = reqwest::get(url)
                .await?
                .bytes()
                .await?
                .to_vec();
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

#[allow(dead_code)]
pub struct AssetManager {
    texture_mappings: TextureAssetMappings,
    shader_mappings: ShaderAssetMappings,
    model_mappings: ModelMappings,

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

            file_to_string_buffer,
            file_to_byte_buffer,
        }
    }

    pub async fn load_texture(
        &mut self,
        file_path: &str,
        device_resource: &DeviceResources<'_>,
    ) -> TextureHandle {
        let mappings = &mut self.texture_mappings;
        if let Some(handle) = mappings.names_to_handles.get(file_path) {
            return *handle;
        }

        log!("AssetManager loading texture {file_path}");
        let new_handle = {
            if !mappings.next_handle.is_valid() {
                mappings.next_handle.index = 0;
            }
            let new_handle = mappings.next_handle;
            mappings.next_handle.index += 1;
            new_handle
        };

        let new_texture = {
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
                Texture::from_file(&final_file_path, device_resource)
                    .await
                    .unwrap()

                /*let current_exe = std::env::current_exe();
                let exe_path = current_exe.as_ref().unwrap().parent().unwrap();
                let final_file_path = format!("{}", exe_path.to_string_lossy());
                let final_file_path = format!("{final_file_path}/{file_path}");
                Texture::from_file(&final_file_path, device_resource).await.unwrap()*/
            }
            #[cfg(target_arch = "wasm32")]
            {
                let path = Path::new(&file_path);
                let file_name = path.file_name().unwrap().to_str().unwrap();
                log!("Path returned {} ", file_name);

                let byte_buffer = self.file_to_byte_buffer.get(file_name).unwrap();
                Texture::from_bytes(
                    &device_resource.device,
                    &device_resource.queue,
                    byte_buffer,
                    file_name,
                )
                .unwrap()
            }
        };

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
                log!(
                    "Path returned {} {}",
                    file_name,
                    self.file_to_byte_buffer.len()
                );
                let byte_buffer = self.file_to_byte_buffer.get(file_name).unwrap().clone(); // cloning here.
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

    pub fn get_model(&mut self, model_handle: &ModelHandle) -> Option<&mut Model> {
        self.model_mappings.handles_to_assets.get_mut(model_handle)
    }

    pub fn get_model_mappings(&mut self) -> &mut HashMap<ModelHandle, Model> {
        &mut self.model_mappings.handles_to_assets
    }
}
