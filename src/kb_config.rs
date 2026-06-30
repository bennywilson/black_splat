use crate::kb_utils::*;
use crate::KbPostProcessMode;

#[derive(Clone)]
pub struct KbConfig {
    // From file
    pub enemy_spawn_delay: f32,
    pub enemy_move_speed: f32,
    pub max_render_instances: u32,
    pub window_width: u32,
    pub window_height: u32,
    // Offscreen scene render resolution as a fraction of the window size (1.0 =
    // full res).  The window/surface stays at window_width x window_height; the
    // scene renders into smaller targets and the postprocess pass upscales it.
    pub render_scale: f32,
    pub fov: f32,
    pub foreground_fov: f32,
    pub graphics_backend: wgpu::Backends,
    pub graphics_power_pref: wgpu::PowerPreference,
    pub vsync: bool,

    // Dynamic
    pub start_time: instant::Instant,
    pub delta_time: f32,
    pub last_frame_time: f32,
    pub postprocess_mode: KbPostProcessMode,
    pub sunbeams_enabled: bool,

    pub clear_color: CgVec4,
    pub sun_color: CgVec4,
    pub sun_beam_pos_scale: CgVec4,
    pub bullet_holes: bool,

    pub start_position: CgVec3,
    pub start_rotation: CgVec3,
}

impl KbConfig {
    pub fn new(config_file_text: &str) -> Self {
        let mut json_file = json::parse(config_file_text).unwrap();

        let json_val = json_file["enemy_spawn_delay"].as_f32();
        let enemy_spawn_delay = json_val.unwrap_or(1.0);

        let json_val = json_file["enemy_move_speed"].as_f32();
        let enemy_move_speed = json_val.unwrap_or(0.01);

        let json_val = json_file["max_instances"].as_u32();
        let max_render_instances = json_val.unwrap_or(10000);

        let json_val = json_file["window_width"].as_u32();
        let window_width: u32 = json_val.unwrap_or(1280);

        let json_val = json_file["window_height"].as_u32();
        let window_height: u32 = json_val.unwrap_or(720);

        // Optional; clamp to a sane range so a typo can't make 0-sized targets.
        let render_scale = json_file["render_scale"].as_f32().unwrap_or(1.0).clamp(0.1, 1.0);

        let graphics_backend = {
            #[cfg(target_arch = "wasm32")]
            {
                // WebGL2 can't bind storage buffers in shaders, which the gaussian
                // splat path requires.  Honor an explicit "webgpu" request so that
                // example can run in the browser; otherwise default to GL like before.
                let json_val = json_file["graphics_back_end"].as_str();
                match json_val {
                    Some("webgpu") => wgpu::Backends::BROWSER_WEBGPU,
                    _ => wgpu::Backends::GL,
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let json_val = json_file["graphics_back_end"].as_str();
                match json_val {
                    Some(val) => match val {
                        "dx12" => wgpu::Backends::DX12,
                        // No native "WebGPU" backend exists; use whatever primary
                        // backend the platform offers (dx12/vulkan/metal).
                        "webgpu" => wgpu::Backends::PRIMARY,
                        "vulkan" => wgpu::Backends::VULKAN,
                        "gl" => wgpu::Backends::GL,
                        _ => wgpu::Backends::all(),
                    },
                    None => wgpu::Backends::PRIMARY,
                }
            }
        };

        let json_val = json_file["graphics_power_pref"].as_str();
        let graphics_power_pref = match json_val {
            Some(val) => match val {
                "high" => wgpu::PowerPreference::HighPerformance,
                "low" => wgpu::PowerPreference::LowPower,
                _ => wgpu::PowerPreference::None,
            },
            None => wgpu::PowerPreference::None,
        };

        let json_val = json_file["vsync"].as_bool();
        let vsync = json_val.unwrap_or(true);

        let sunbeams_enabled = json_file["sunbeams"].as_bool().unwrap_or(false);
        let sun_beam_pos_scale = {
            if json_file["sun_beam_pos_scale"].is_array() {
                let x = json_file["sun_beam_pos_scale"].pop().as_f32().unwrap();
                let y = json_file["sun_beam_pos_scale"].pop().as_f32().unwrap();
                let z = json_file["sun_beam_pos_scale"].pop().as_f32().unwrap();
                let w = json_file["sun_beam_pos_scale"].pop().as_f32().unwrap();
                CgVec4::new(x, y, z, w)
            } else {
                CgVec4::new(500.0, 550.0, 500.0, 1550.0)
            }
        };

        let json_val = json_file["bullet_holes"].as_bool();
        let bullet_holes = json_val.unwrap_or_default();

        let start_position = {
            let arr = &json_file["start_position"];
            if arr.is_array() {
                CgVec3::new(
                    arr[0].as_f32().unwrap_or(0.0),
                    arr[1].as_f32().unwrap_or(0.0),
                    arr[2].as_f32().unwrap_or(0.0),
                )
            } else {
                CgVec3::new(0.0, 0.0, 0.0)
            }
        };

        let start_rotation = {
            let arr = &json_file["start_rotation"];
            if arr.is_array() {
                CgVec3::new(
                    arr[0].as_f32().unwrap_or(0.0),
                    arr[1].as_f32().unwrap_or(0.0),
                    arr[2].as_f32().unwrap_or(0.0),
                )
            } else {
                CgVec3::new(0.0, 0.0, 0.0)
            }
        };

        KbConfig {
            enemy_spawn_delay,
            enemy_move_speed,
            max_render_instances,
            window_width,
            window_height,
            render_scale,
            fov: 75.0,
            foreground_fov: 50.0,
            graphics_backend,
            graphics_power_pref,
            vsync,

            start_time: instant::Instant::now(),
            delta_time: 0.0,
            last_frame_time: 0.0,
            postprocess_mode: KbPostProcessMode::Passthrough,
            sunbeams_enabled,
            clear_color: CG_VEC4_ZERO,
            sun_color: CgVec4::new(1.0, 1.0, 1.0, 1.0),
            sun_beam_pos_scale,
            bullet_holes,
            start_position,
            start_rotation,
        }
    }

    /// Offscreen scene render resolution (window size * render_scale), clamped to
    /// at least 1x1.  The window/surface stays at window_width x window_height.
    pub fn render_resolution(&self) -> (u32, u32) {
        let w = ((self.window_width as f32 * self.render_scale).round() as u32).max(1);
        let h = ((self.window_height as f32 * self.render_scale).round() as u32).max(1);
        (w, h)
    }

    pub fn update_frame_times(&mut self) {
        let elapsed_time = self.start_time.elapsed().as_secs_f32();
        self.delta_time = elapsed_time - self.last_frame_time;
        self.last_frame_time = elapsed_time;
    }
}
