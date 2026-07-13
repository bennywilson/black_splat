use cgmath::SquareMatrix;
use std::{collections::HashMap, mem::size_of, result::Result::Ok};
use wgpu::{
    util::DeviceExt, BindGroupLayoutEntry, BindingType, SamplerBindingType, ShaderStages,
    TextureSampleType, TextureViewDimension,
};

use crate::{assets::*, game_object::*, resource::*, log};

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniform {
    pub world: [[f32; 4]; 4],
    pub inv_world: [[f32; 4]; 4],
    pub mvp_matrix: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub camera_dir: [f32; 4],
    pub screen_dimensions: [f32; 4],
    pub time: [f32; 4],
    pub model_color: [f32; 4],
    pub custom_data_1: [f32; 4],
    pub sun_color: [f32; 4],
    // Material specular constant: rgb tint, a gloss (0..1).  Appended last so
    // shaders declaring the older, shorter struct still bind this buffer.
    pub spec_color: [f32; 4],
}
pub const MAX_UNIFORMS: usize = 100;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelDrawInstance {
    pub position: [f32; 4],
    pub scale: [f32; 4],
    pub color: [f32; 4],
}
pub const MAX_MODEL_INSTANCES: usize = 1000;
pub const MAX_PARTICLE_INSTANCES: usize = 1000;

impl ModelDrawInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<ModelDrawInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 2 * size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct Model {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub num_indices: u32,

    pub textures: Vec<TextureHandle>,
    pub tex_bind_group: wgpu::BindGroup,

    pub empty_texture: Option<Texture>,
    pub hole_texture: Option<Texture>,

    uniform_buffers: Vec<wgpu::Buffer>,
    uniform_bind_groups: Vec<wgpu::BindGroup>,
    next_uniform_buffer: usize,
}

impl Model {
    pub async fn new_particle(
        texture_file_path: &str,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        // Loading the texture is the only async step; the GPU resources are all
        // built synchronously in `new_particle_with_texture`.
        let texture_handle = asset_manager
            .load_texture(texture_file_path, device_resources, TextureFilter::Linear)
            .await;
        Self::new_particle_with_texture(&texture_handle, device_resources, asset_manager)
    }

    /// Builds a particle model from an already-loaded texture (see
    /// `AssetManager::load_texture`).  Synchronous, so callers outside an async
    /// context (e.g. the frame tick) can spawn particles once the texture has
    /// been preloaded.
    pub fn new_particle_with_texture(
        texture_handle: &TextureHandle,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        let device = &device_resources.device;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model::vertex_buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model::index_buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Model::instance_buffer"),
            mapped_at_creation: false,
            size: (size_of::<ModelDrawInstance>() * MAX_PARTICLE_INSTANCES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("Model::texture_bind_group_layout"),
            });

        let textures = vec![*texture_handle];
        // Resolve the built-in white (a &mut borrow) before the shared texture
        // borrows below. Bound to the metallic/roughness slots (2 and 3) so
        // this model reads as constant-only PBR through the G-buffer shader.
        let white_handle = asset_manager.white_texture(device_resources);
        let texture = asset_manager.get_texture(&textures[0]);
        let white = asset_manager.get_texture(&white_handle);

        let tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&white.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&white.view),
                },
            ],
            label: Some("Model::tex_bind_group"),
        });

        // Uniform buffer
        let mut uniform_buffers = Vec::<wgpu::Buffer>::new();
        let mut uniform_bind_groups = Vec::<wgpu::BindGroup>::new();
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Model::uniform_bind_group_layout"),
            });

        let empty_uniform = ModelUniform {
            ..Default::default()
        };
        //let mut uniforms: Vec<ModelUniform> = Vec::with_capacity(MAX_UNIFORMS);

        for i in 0..MAX_UNIFORMS {
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Model::uniform_buffer_{i}")),
                contents: bytemuck::cast_slice(&[empty_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
                label: Some(&format!("Model::uniform_bind_group_{i}")),
            });

            uniform_buffers.push(uniform_buffer);
            uniform_bind_groups.push(uniform_bind_group);
        }

        Model {
            vertex_buffer,
            index_buffer,
            instance_buffer,
            num_indices: 6,
            textures,
            hole_texture: None,
            empty_texture: None,
            tex_bind_group,
            uniform_buffers,
            uniform_bind_groups,
            next_uniform_buffer: 0,
        }
    }

    pub async fn from_bytes(
        bytes: &Vec<u8>,
        device_resources: &mut DeviceResources<'_>,
        asset_manager: &mut AssetManager,
        use_holes: bool,
    ) -> Self {
        log!("Loading Model from bytes");

        let device = &device_resources.device;
        let mut indices = Vec::<u16>::new();
        let mut vertices = Vec::<Vertex>::new();
        let mut textures = Vec::<TextureHandle>::new();
        // https://stackoverflow.com/questions/75846989/how-to-load-gltf-files-with-gltf-rs-crate

        let (gltf_doc, buffers, gltf_images) = gltf::import_slice(bytes).unwrap();

        for gltf_texture in gltf_doc.textures() {
            match gltf_texture.source().source() {
                gltf::image::Source::View {
                    view: _,
                    mime_type: _,
                } => {}
                gltf::image::Source::Uri { uri, mime_type: _ } => {
                    if let Ok(dir) = std::env::current_dir() {
                        let file_path = format!("{}\\game_assets\\{}", dir.display(), uri);
                        let texture_handle = asset_manager
                            .load_texture(&file_path, device_resources, TextureFilter::Linear)
                            .await;
                        textures.push(texture_handle);
                    }
                }
            }
        }

        for m in gltf_doc.meshes() {
            for p in m.primitives() {
                let r = p.reader(|buffer| Some(&buffers[buffer.index()]));
                if let Some(gltf::mesh::util::ReadIndices::U16(gltf::accessor::Iter::Standard(
                    iter,
                ))) = r.read_indices()
                {
                    for v in iter {
                        indices.push(v);
                    }
                }

                let mut positions = Vec::new();
                if let Some(iter) = r.read_positions() {
                    for v in iter {
                        positions.push(v);
                    }
                }

                let mut uvs = Vec::new();
                if let Some(gltf::mesh::util::ReadTexCoords::F32(gltf::accessor::Iter::Standard(
                    iter,
                ))) = r.read_tex_coords(0)
                {
                    for v in iter {
                        uvs.push(v);
                    }
                }

                let mut normals = Vec::new();
                if let Some(iter) = r.read_normals() {
                    for v in iter {
                        normals.push(v);
                    }
                }

                let mut i = 0;
                while i < positions.len() {
                    let vertex = Vertex {
                        position: positions[i],
                        tex_coords: uvs[i],
                        normal: normals[i],
                        color: [1.0, 1.0, 1.0, 1.0],
                    };
                    vertices.push(vertex);
                    i += 1;
                }
            }
        }

        let num_indices = indices.len() as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Model_vertex_buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("Model_texture_bind_group_layout"),
            });

        // Resolve the built-in checkerboard and white up front (a &mut borrow of
        // the asset manager) so those borrows are released before we take the
        // shared &Texture below. Cheap after the first call (cached). White backs
        // the metallic/roughness slots so an untextured model reads as
        // constant-only PBR through the G-buffer shader.
        let checker_handle = asset_manager.checker_texture(device_resources);
        let white_handle = asset_manager.white_texture(device_resources);

        // Follow the material's baseColorTexture -> texture -> image chain to
        // pick the *correct* embedded image. Blindly using gltf_images[0] binds
        // whichever image happens to be first (often a normal/roughness map),
        // which renders as a garbled surface.
        let base_color_image_index = gltf_doc
            .materials()
            .find_map(|m| m.pbr_metallic_roughness().base_color_texture())
            .map(|info| info.texture().source().index());

        // Which embedded image to decode, if any: the material's base color if
        // it declares one; otherwise (only when there's no explicit material to
        // tell us) the first image as a best-effort guess.
        let image_index = base_color_image_index.or_else(|| {
            if gltf_doc.materials().next().is_none() && !gltf_images.is_empty() {
                Some(0)
            } else {
                None
            }
        });

        let mut empty_texture = None;
        if textures.is_empty() {
            if let Some(image) = image_index.and_then(|i| gltf_images.get(i)) {
                match image.format {
                    gltf::image::Format::R8G8B8 | gltf::image::Format::R8G8B8A8 => {
                        empty_texture = Texture::from_rgba(
                            &image.pixels,
                            image.format == gltf::image::Format::R8G8B8A8,
                            image.width,
                            image.height,
                            device_resources,
                            Some("gltf base color"),
                            TextureFilter::Linear,
                        )
                        .ok();
                    }
                    other => {
                        log!("gltf image format {other:?} unsupported; using checkerboard");
                    }
                }
            }
        }

        let texture = {
            if !textures.is_empty() {
                // A URI-referenced texture was loaded (load_texture already
                // substitutes the checkerboard for a missing/undecodable file).
                asset_manager.get_texture(&textures[0])
            } else if let Some(tex) = empty_texture.as_ref() {
                tex
            } else {
                // Material references a texture we couldn't produce, or the
                // model is untextured: show the checkerboard so it reads as an
                // obvious placeholder rather than a garbled/blank surface.
                asset_manager.get_texture(&checker_handle)
            }
        };

        let white = asset_manager.get_texture(&white_handle);

        let mut surface_config = device_resources.surface_config.clone();
        surface_config.width = 1024;
        surface_config.height = 1024;
        let mut hole_texture = None; //
        // Binding 2 is the metallic slot for the G-buffer shader (white =
        // constant passes through), but the bullet-hole forward shader
        // (model_with_holes.wgsl) samples this same slot as its hole mask -- so
        // with holes it takes the hole render texture instead.
        let mut tex_2_bind = wgpu::BindingResource::TextureView(&white.view);
        if use_holes {
            hole_texture = Some(
                Texture::new_render_texture(
                    device,
                    &surface_config,
                    surface_config.width,
                    surface_config.height,
                )
                .unwrap(),
            );
            tex_2_bind = wgpu::BindingResource::TextureView(&hole_texture.as_ref().unwrap().view);
        }
        let tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tex_2_bind,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&white.view),
                },
            ],
            label: Some("Model::tex_bind_group"),
        });

        // Uniform buffer
        let mut uniform_buffers = Vec::<wgpu::Buffer>::new();
        let mut uniform_bind_groups = Vec::<wgpu::BindGroup>::new();
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("ModelPass_uniform_bind_group_layout"),
            });

        let uniform = ModelUniform {
            ..Default::default()
        };

        for _ in 0..MAX_UNIFORMS {
            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("kbModelPipeline_uniform_buffer"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
                label: Some("ModelPass_uniform_bind_group"),
            });

            uniform_buffers.push(uniform_buffer);
            uniform_bind_groups.push(uniform_bind_group);
        }

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            mapped_at_creation: false,
            size: (size_of::<ModelDrawInstance>() * MAX_UNIFORMS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Model {
            vertex_buffer,
            index_buffer,
            instance_buffer,
            num_indices,
            uniform_bind_groups,
            uniform_buffers,
            textures,
            hole_texture,
            empty_texture,
            tex_bind_group,
            next_uniform_buffer: 0,
        }
    }

    pub fn alloc_uniform_buffer(&mut self) -> &mut wgpu::Buffer {
        if self.next_uniform_buffer > 80 {
            self.next_uniform_buffer -= 1;
            for _ in 0..32 {
                log!("Wear the AP don't slam my door!");
            }
        }

        let ret_val = &mut self.uniform_buffers[self.next_uniform_buffer];
        self.next_uniform_buffer += 1;
        ret_val
    }

    pub fn get_uniform_bind_group(&self, index: usize) -> &wgpu::BindGroup {
        &self.uniform_bind_groups[index]
    }

    pub fn get_uniform_info_count(&self) -> usize {
        self.next_uniform_buffer
    }

    // Call after submitting the Model's draw calls for the frame
    pub fn free_uniform_buffers(&mut self) {
        self.next_uniform_buffer = 0;
    }
}

pub struct ModelPass {
    pub model_pipeline: wgpu::RenderPipeline,
    pub alpha_blend_pipeline: wgpu::RenderPipeline,
    pub additive_pipeline: wgpu::RenderPipeline,
    pub uniform: ModelUniform,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub blend_mode: BlendMode,
}

impl ModelPass {
    pub async fn new(
        shader_path: &str,
        blend_mode: &BlendMode,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        log!("Creating ModelPass with shader {shader_path}");
        let device = &device_resources.device;

        // Uniform buffer
        let uniform = ModelUniform {
            ..Default::default()
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kbModelPipeline_uniform_buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("ModelPass_uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("ModelPass_uniform_bind_group"),
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            view_dimension: TextureViewDimension::D2,
                            sample_type: TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("ModelPass_texture_bind_group_layout"),
            });

        log!("  Creating pipeline");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ModelPass_render_pipeline_layout"),
                bind_group_layouts: &[Some(&texture_bind_group_layout), Some(&uniform_bind_group_layout)],
                immediate_size: 0,
            });

        let shader_handle = asset_manager
            .load_shader(shader_path, device_resources)
            .await;
        let model_shader = asset_manager.get_shader(&shader_handle);
        let blend = Some(match blend_mode {
            BlendMode::None => wgpu::BlendState::REPLACE,
            BlendMode::Alpha => wgpu::BlendState::ALPHA_BLENDING,
            BlendMode::Additive => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            },
        });

        let mut cull_mode = Some(wgpu::Face::Back);
        if shader_path.contains("decal") {
            cull_mode = None;
        }

        let mut depth_write_enabled = true;
        if shader_path.contains("first_person_outline")
            || shader_path.contains("sky_dome_draw")
            || shader_path.contains("decal")
        {
            depth_write_enabled = false;
        }

        let mut write_mask = wgpu::ColorWrites::ALL;
        if shader_path.contains("sky_dome_occlude") {
            write_mask = wgpu::ColorWrites::ALPHA;
        }
        let model_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ModelPass_opaque_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: model_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: model_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::resource::SCENE_COLOR_FORMAT,
                    blend,
                    write_mask,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(depth_write_enabled),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let particle_shader_handle = asset_manager
            .load_shader("/engine_assets/shaders/particle.wgsl", device_resources)
            .await;
        let particle_shader = asset_manager.get_shader(&particle_shader_handle);
        let alpha_blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ModelPass::alpha_blend_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: particle_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), ModelDrawInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: particle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::resource::SCENE_COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let additive_blend_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };

        let additive_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ModelPass_additive_pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: particle_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc(), ModelDrawInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: particle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::resource::SCENE_COLOR_FORMAT,
                    blend: Some(additive_blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        ModelPass {
            model_pipeline,
            alpha_blend_pipeline,
            additive_pipeline,
            uniform,
            uniform_buffer,
            uniform_bind_group,
            blend_mode: blend_mode.clone(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        ctx: &mut RenderContext,
        layer: &SceneLayer,
        custom_pass_handle: Option<usize>,
        actors: &HashMap<u32, Actor>,
    ) {
        let device_resources = &mut *ctx.device;
        let asset_manager = &mut *ctx.assets;
        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ModelPass::render()"),
                });

        let layer = (*layer).clone();
        let (color_attachment, depth_attachment) = {
            let (color_ops, depth_ops) = {
                let clear_color = game_config.clear_color;
                if layer == SceneLayer::World {
                    (
                        wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: clear_color.x as f64,
                                g: clear_color.y as f64,
                                b: clear_color.z as f64,
                                a: clear_color.w as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        },
                    )
                } else if layer == SceneLayer::Foreground {
                    (
                        wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        },
                    )
                } else {
                    (
                        wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    )
                }
            };
            (
                wgpu::RenderPassColorAttachment {
                    view: &device_resources.render_textures[0].view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: color_ops,
                },
                wgpu::RenderPassDepthStencilAttachment {
                    view: &device_resources.render_textures[1].view,
                    depth_ops: Some(depth_ops),
                    stencil_ops: None,
                },
            )
        };

        let render_pass_label = format!("{:?} {:?}", layer, self.blend_mode);
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&render_pass_label),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(depth_attachment),
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.model_pipeline);

        let (view_matrix, view_dir, _) = game_camera.calculate_view_matrix();
        let view_pos = game_camera.get_position();
        let view_pos = [view_pos.x, view_pos.y, view_pos.z, 1.0];
        let fov = if layer == SceneLayer::Foreground
            || layer == SceneLayer::ForegroundCustom
        {
            game_config.foreground_fov
        } else {
            game_config.fov
        };
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            10000.0,
        );

        // Iterate over actors and add their uniform info to their corresponding Models
        let mut models_to_render = Vec::<ModelHandle>::new();
        let actor_iter = actors.iter();
        for actor_key_value in actor_iter {
            let (actor_layer, pass_handle) = actor_key_value.1.get_layer();
            if actor_layer != layer {
                continue;
            }
            if actor_layer == SceneLayer::ForegroundCustom
                || actor_layer == SceneLayer::WorldCustom
            {
                // Custom-layer actors only draw in their matching custom pass;
                // ones without a pass handle assigned yet don't draw at all.
                if custom_pass_handle.is_none() || custom_pass_handle != pass_handle {
                    continue;
                }
            }
            let actor = actor_key_value.1;
            let model_handle = actor.get_model();
            // Editor-placed actors can exist before a model is assigned.
            let Some(model) = asset_manager.get_model(&model_handle) else {
                continue;
            };

            if !models_to_render.contains(&model_handle) {
                models_to_render.push(model_handle);
            }

            let uniform_buffer = model.alloc_uniform_buffer();
            let mut uniform_data = ModelUniform {
                ..Default::default()
            };
            let world_matrix = cgmath::Matrix4::from_translation(actor.get_position())
                * cgmath::Matrix4::from(actor.get_rotation())
                * cgmath::Matrix4::from_nonuniform_scale(
                    actor.get_scale().x,
                    actor.get_scale().y,
                    actor.get_scale().z,
                );
            uniform_data.world = world_matrix.into();
            // A zero scale on any axis (easy to hit while dragging the editor's
            // Scale field) makes the matrix singular: render the degenerate
            // frame with an identity inverse instead of panicking.
            uniform_data.inv_world = world_matrix
                .invert()
                .unwrap_or_else(cgmath::Matrix4::identity)
                .into();
            uniform_data.mvp_matrix = (proj_matrix * view_matrix * world_matrix).into();
            uniform_data.view_proj = (proj_matrix * view_matrix).into();
            uniform_data.camera_dir = [view_dir.x, view_dir.y, view_dir.z, 0.0];
            uniform_data.camera_pos = view_pos;
            uniform_data.screen_dimensions = [
                game_config.window_width as f32,
                game_config.window_height as f32,
                (game_config.window_height as f32) / (game_config.window_width as f32),
                0.0,
            ];
            uniform_data.time[0] = game_config.start_time.elapsed().as_secs_f32();
            uniform_data.time[1] = 1.0;
            uniform_data.model_color = [
                actor.get_color().x,
                actor.get_color().y,
                actor.get_color().z,
                actor.get_color().w,
            ];
            uniform_data.custom_data_1 = [
                actor.get_custom_data_1().x,
                actor.get_custom_data_1().y,
                actor.get_custom_data_1().z,
                actor.get_custom_data_1().w,
            ];
            uniform_data.sun_color = [
                game_config.sun_color.x,
                game_config.sun_color.y,
                game_config.sun_color.z,
                0.0,
            ];
            device_resources.queue.write_buffer(
                uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniform_data]),
            );
        }

        // Render Models now that uniforms are set
        let model_mappings = asset_manager.get_model_mappings();
        for model_handle in &mut models_to_render {
            let model = &model_mappings[model_handle];
            render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
            render_pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            for i in 0..model.get_uniform_info_count() {
                let uniform_bind_group = model.get_uniform_bind_group(i);
                render_pass.set_bind_group(1, uniform_bind_group, &[]);
                render_pass.set_bind_group(0, &model.tex_bind_group, &[]);
                render_pass.draw_indexed(0..model.num_indices, 0, 0..1);
            }
        }

        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));

        for model_handle in &mut models_to_render {
            let model = &mut model_mappings.get_mut(model_handle).unwrap();
            model.free_uniform_buffers();
        }
    }

    pub fn render_particles(
        &mut self,
        ctx: &mut RenderContext,
        blend_mode: ParticleBlendMode,
        particles: &mut HashMap<ParticleHandle, ParticleActor>,
    ) {
        let device_resources = &mut *ctx.device;
        let game_camera = ctx.camera;
        let game_config = ctx.config;
        let mut command_encoder =
            device_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ModelPass::render_particles()"),
                });

        // Create instances
        let label = format!("Particle {:?}", blend_mode);
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &device_resources.render_textures[0].view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &device_resources.render_textures[1].view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            multiview_mask: None,
            timestamp_writes: None,
        });

        let (view_matrix, view_dir, _) = game_camera.calculate_view_matrix();
        let view_pos = game_camera.get_position();
        let view_pos = [view_pos.x, view_pos.y, view_pos.z, 1.0];
        let proj_matrix = cgmath::perspective(
            cgmath::Deg(game_config.fov),
            game_config.window_width as f32 / game_config.window_height as f32,
            0.1,
            1000000.0,
        );
        let view_proj_matrix = proj_matrix * view_matrix;

        match blend_mode {
            ParticleBlendMode::AlphaBlend => {
                render_pass.set_pipeline(&self.alpha_blend_pipeline);
            }
            ParticleBlendMode::Additive => render_pass.set_pipeline(&self.additive_pipeline),
        };

        let particle_iter = particles.iter_mut();
        for mut particle_val in particle_iter {
            let particle_actor = &mut particle_val.1;
            if particle_actor.params.blend_mode != blend_mode {
                continue;
            }

            if !particle_actor.is_active() {
                continue;
            }

            let position = particle_actor.get_position();
            let scale = particle_actor.get_scale();
            let model = &mut particle_val.1.model;

            // Uniform data
            model.free_uniform_buffers();
            let uniform_buffer = model.alloc_uniform_buffer();

            let world_matrix =
                cgmath::Matrix4::from_translation(position) * cgmath::Matrix4::from_scale(scale.x);
            let mut uniform = ModelUniform {
                ..Default::default()
            };
            // Same zero-scale guard as the actor path above.
            uniform.inv_world = world_matrix
                .invert()
                .unwrap_or_else(cgmath::Matrix4::identity)
                .into();
            uniform.mvp_matrix = (view_proj_matrix * world_matrix).into();
            uniform.view_proj = (proj_matrix * view_matrix).into();
            uniform.camera_pos = view_pos;
            uniform.camera_dir = [view_dir.x, view_dir.y, view_dir.z, 0.0];
            uniform.screen_dimensions = [
                game_config.window_width as f32,
                game_config.window_height as f32,
                (game_config.window_height as f32) / (game_config.window_width as f32),
                0.0,
            ];
            uniform.time[0] = game_config.start_time.elapsed().as_secs_f32();
            uniform.time[1] = 1.0;
            uniform.custom_data_1 = [0.0, 0.0, 0.0, 0.0];
            uniform.model_color = [1.0, 1.0, 1.0, 1.0];
            device_resources.queue.write_buffer(
                uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniform]),
            );

            // Instances
            let particles = &particle_val.1.particles;
            if particles.is_empty() {
                continue;
            }
            let mut particle_instances = Vec::<ModelDrawInstance>::new();
            for particle in particles {
                let new_instance = ModelDrawInstance {
                    position: [
                        particle.position.x,
                        particle.position.y,
                        particle.position.z,
                        particle.rotation,
                    ],
                    scale: [particle.scale.x, particle.scale.y, 0.0, 0.0],
                    color: particle.color.into(),
                };
                particle_instances.push(new_instance);
            }
            device_resources.queue.write_buffer(
                &model.instance_buffer,
                0,
                bytemuck::cast_slice(particle_instances.as_slice()),
            );

            render_pass.set_vertex_buffer(0, model.vertex_buffer.slice(..));
            render_pass.set_index_buffer(model.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.set_vertex_buffer(1, model.instance_buffer.slice(..));
            let uniform_bind_group = model.get_uniform_bind_group(0);
            render_pass.set_bind_group(1, uniform_bind_group, &[]);
            render_pass.set_bind_group(0, &model.tex_bind_group, &[]);
            render_pass.draw_indexed(0..model.num_indices, 0, 0..particle_instances.len() as u32);
        }
        drop(render_pass);
        device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }
}
