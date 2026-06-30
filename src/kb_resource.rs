use std::{mem::size_of, result::Result::Ok, sync::Arc};

use ab_glyph::FontRef;
use anyhow::*;
use image::GenericImageView;
use wgpu::{Device, DeviceDescriptor, Queue, SurfaceConfiguration};
use wgpu_text::{BrushBuilder, TextBrush};

use crate::{kb_assets::*, kb_config::*, log};

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum KbBlendMode {
    None,
    Alpha,
    Additive,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum KbRenderGroupType {
    World,
    WorldHole,
    WorldCustom,
    Foreground,
    ForegroundCustom,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KbVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

impl KbVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<KbVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub const VERTICES: &[KbVertex] = &[
    KbVertex {
        position: [1.0, 1.0, 0.0],
        tex_coords: [1.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    },
    KbVertex {
        position: [-1.0, 1.0, 0.0],
        tex_coords: [0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    },
    KbVertex {
        position: [-1.0, -1.0, 0.0],
        tex_coords: [0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    },
    KbVertex {
        position: [1.0, -1.0, 0.0],
        tex_coords: [1.0, 1.0],
        normal: [0.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0, 1.0],
    },
];

pub const INDICES: &[u16] = &[0, 1, 3, 3, 1, 2];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KbSpriteDrawInstance {
    pub pos_scale: [f32; 4],
    pub uv_scale_bias: [f32; 4],
    pub per_instance_data: [f32; 4],
}

impl KbSpriteDrawInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<KbSpriteDrawInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 10, // Corresponds to @location in the shader
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 2 * size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteUniform {
    pub screen_dimensions: [f32; 4],
    pub time: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessUniform {
    // x: time, y: postprocess mode, z: 1.0 when the surface is non-sRGB (encode
    // in-shader), w: 1.0 to apply the ACES tonemap.
    pub time_mode_srgb_tonemap: [f32; 4],
}

#[allow(dead_code)]
pub struct KbTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl KbTexture {
    pub fn new_depth_texture(
        device: &Device,
        _surface_config: &SurfaceConfiguration,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            // 4.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });
        Ok(KbTexture {
            texture,
            view,
            sampler,
        })
    }

    pub fn new_render_texture(
        device: &Device,
        surface_config: &wgpu::SurfaceConfiguration,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Always render the offscreen scene into an sRGB target so alpha
            // blending happens in linear space (the hardware decodes/encodes).
            // On native the surface is already sRGB so this is a no-op; on web
            // Chrome's canvas is non-sRGB `Bgra8Unorm`, and without this the
            // scene would blend in gamma space -- making stacked transparent
            // splats composite far too dark.  The postprocess pass converts back
            // to the (possibly non-sRGB) surface on present.
            format: surface_config.format.add_srgb_suffix(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(KbTexture {
            texture,
            view,
            sampler,
        })
    }

    pub async fn from_file(
        file_path: &str,
        device_resources: &KbDeviceResources<'_>,
    ) -> Result<Self> {
        log!("Loading texture {}", file_path);
        let texture_bytes = load_binary(file_path).await.unwrap(); //load_bytes!(file_path);
        KbTexture::from_bytes(
            &device_resources.device,
            &device_resources.queue,
            &texture_bytes,
            file_path,
        )
    }

    pub fn from_bytes(device: &Device, queue: &Queue, bytes: &[u8], label: &str) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Self::from_image(device, queue, &img, Some(label))
    }

    pub fn from_rgba(
        rgba: &[u8],
        is_rgba: bool,
        width: u32,
        height: u32,
        device_resources: &KbDeviceResources<'_>,
        label: Option<&str>,
    ) -> Result<Self> {
        let queue = &device_resources.queue;
        let device = &device_resources.device;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let mut new_rgba = Vec::<u8>::new();
        let mut i = 0;
        while i < rgba.len() {
            new_rgba.push(rgba[i]);
            new_rgba.push(rgba[i + 1]);
            new_rgba.push(rgba[i + 2]);
            if !is_rgba {
                new_rgba.push(255);
                i += 3;
            } else {
                new_rgba.push(rgba[i + 3]);
                i += 4;
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &new_rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }

    pub fn from_image(
        device: &Device,
        queue: &Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Result<Self> {
        let dimensions = img.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let rgba = img.to_rgba8();

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum KbRenderPassType {
    Opaque,
    Transparent,
    PostProcess,
}

#[derive(Clone)]
pub enum KbPostProcessMode {
    Passthrough,
    Desaturation,
    ScanLines,
    Warp,
}

#[allow(dead_code)]
pub struct KbDeviceResources<'a> {
    pub surface: wgpu::Surface<'a>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub adapter: wgpu::Adapter,
    pub device: Device,
    pub queue: Queue,

    pub instance_buffer: wgpu::Buffer,
    pub brush: TextBrush<FontRef<'a>>,
    pub render_textures: Vec<KbTexture>, // [0] is color, [1] is depth
}

impl<'a> KbDeviceResources<'a> {
    pub fn resize(&mut self, game_config: &KbConfig) {
        assert!(game_config.window_width > 0 && game_config.window_height > 0);

        self.surface_config.width = game_config.window_width;
        self.surface_config.height = game_config.window_height;
        self.surface.configure(&self.device, &self.surface_config);

        // Offscreen scene targets render at render_scale; the postprocess pass
        // upscales them onto the full-size surface.
        let (rw, rh) = game_config.render_resolution();
        self.render_textures[0] =
            KbTexture::new_render_texture(&self.device, &self.surface_config, rw, rh).unwrap();
        self.render_textures[1] =
            KbTexture::new_depth_texture(&self.device, &self.surface_config, rw, rh).unwrap();
        self.render_textures[2] =
            KbTexture::new_render_texture(&self.device, &self.surface_config, rw, rh).unwrap();
    }

    pub async fn new(window: Arc<winit::window::Window>, game_config: &KbConfig) -> Self {
        log!("KbDeviceResources::new() called...");

        log!("  Creating instance");
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: game_config.graphics_backend,
            ..Default::default()
        });

        log!("  Creating surface + adapter");
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: game_config.graphics_power_pref,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        log!(
            "  Adapter: {:?} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        log!("  Requesting Device");
        // WebGL2 can't do shader storage buffers, so it gets the most restrictive
        // limit set.  For every other backend (native + browser WebGPU) we request
        // exactly what the adapter reports -- that can never fail a limit check, and
        // it still gives the gaussian splat path the storage buffers it needs.
        let required_limits = if game_config.graphics_backend == wgpu::Backends::GL {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            adapter.limits()
        };
        let (device, queue) = match adapter
            .request_device(
                &DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    label: Some("Device Descriptor"),
                },
                None, // Trace path
            )
            .await
        {
            Ok(dq) => dq,
            Err(e) => {
                log!("  request_device() FAILED: {e:?}");
                panic!("request_device() failed: {e:?}");
            }
        };

        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer an sRGB format, but some WebGPU surfaces (e.g. Chrome's canvas)
        // only expose a non-sRGB format -- fall back to whatever is offered.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| {
                log!(
                    "  No sRGB surface format; using {:?}",
                    surface_caps.formats.first()
                );
                surface_caps.formats[0]
            });

        let mut surface_config = surface
            .get_default_config(
                &adapter,
                game_config.window_width,
                game_config.window_height,
            )
            .unwrap_or_else(|| {
                log!("  get_default_config() returned None (surface/adapter mismatch)");
                panic!("surface.get_default_config() returned None");
            });
        surface_config.format = surface_format;
        // Keep at most one frame queued ahead of the GPU.  With heavy per-frame GPU
        // work (e.g. the gaussian splat sort + overdraw) the default of 2+ lets the
        // CPU race ahead, so input shows up seconds late even at a steady frame rate.
        surface_config.desired_maximum_frame_latency = 1;
        surface.configure(&device, &surface_config);

        let max_instances = game_config.max_render_instances;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_buffer"),
            mapped_at_creation: false,
            size: (size_of::<KbSpriteDrawInstance>() * max_instances as usize) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Offscreen scene targets render at render_scale; the postprocess pass
        // upscales them onto the full-size surface.
        let (rw, rh) = game_config.render_resolution();
        let mut render_textures = Vec::<KbTexture>::new();
        let render_texture =
            KbTexture::new_render_texture(&device, &surface_config, rw, rh).unwrap();
        render_textures.push(render_texture);

        let depth_texture = KbTexture::new_depth_texture(&device, &surface_config, rw, rh).unwrap();
        render_textures.push(depth_texture);

        let render_texture =
            KbTexture::new_render_texture(&device, &surface_config, rw, rh).unwrap();
        render_textures.push(render_texture);

        log!("  Creating Font");
        let brush =
            BrushBuilder::using_font_bytes(include_bytes!("../engine_assets/fonts/bold.ttf"))
                .unwrap()
                .build(
                    &device,
                    surface_config.width,
                    surface_config.height,
                    surface_config.format,
                );

        log!("KbDeviceResources allocated");
        KbDeviceResources {
            surface_config,
            surface,
            adapter,
            device,
            queue,
            instance_buffer,
            brush,
            render_textures,
        }
    }
}
