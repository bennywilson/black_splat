use instant::Instant;
use std::{collections::HashMap, sync::Arc};
use wgpu_text::glyph_brush::{HorizontalAlign, Layout, Section as TextSection, Text, VerticalAlign};

use crate::{
    assets::*,
    config::*,
    game_object::*,
    resource::*,
    utils::*,
    log,
    passes::{
        bullet_hole::*, deferred::*, gaussian_splat::*, line::*, model::*,
        postprocess::*, splat_composite::*, sprite::*, sunbeam::*,
    },
    PERF_SCOPE,
};

#[allow(dead_code)]
pub struct Renderer<'a> {
    device_resources: DeviceResources<'a>,

    default_sprite_pass: SpritePass,
    custom_sprite_passes: Vec<SpritePass>,
    postprocess_pass: PostprocessPass,
    model_pass: ModelPass,
    model_with_holes_pass: ModelPass,
    // Deferred world rendering: the G-buffer pass draws the World-layer actors
    // (color/normal/metallic-roughness), then the lighting pass accumulates
    // light_map's lights onto the scene color target, with the shadow pass
    // rendering shadow maps + screen-space masks for the lights that cast.
    // None on the GL backend: the light/shadow shaders textureLoad from depth
    // textures, which naga can't translate to GLSL, and GL is treated as a
    // basic backend anyway, so the whole deferred trio is skipped rather than
    // chasing GL fallbacks for every feature (same reasoning as
    // gaussian_splat_pass below).
    gbuffer_pass: Option<GBufferPass>,
    shadow_pass: Option<ShadowPass>,
    lighting_pass: Option<LightingPass>,
    // Games opt in via set_deferred_world_enabled (the splat editor does; it
    // registers scene lights).  Off, World-layer actors render through the
    // classic forward model pass -- a game without lights would otherwise get
    // a flat clear_color world, since the lighting pass is what composites
    // G-buffer contents onto the screen.
    deferred_world_enabled: bool,
    line_pass: LinePass,
    sunbeam_pass: SunbeamPass,
    // Built lazily on first load_gaussian_splat() call: its pipeline binds storage
    // buffers in the vertex stage, which WebGL2 can't do, so we must not create it
    // for the GL-backed 2D/3D demos.
    gaussian_splat_pass: Option<GaussianSplatPass>,
    // Converts the splat pass's display-space composite into the linear HDR scene
    // (see splat_composite.wgsl).  Built alongside gaussian_splat_pass -- no
    // splats, no need for it.
    splat_composite_pass: Option<SplatCompositePass>,
    // In-engine GUI (egui).  The context is shared with the host loop's
    // egui-winit State (which feeds it window input); games build widgets
    // against it during tick_frame, and render_frame paints the tessellated
    // output onto the swapchain as the final pass.
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    // egui TextureIds for loaded textures the GUI wants to draw (e.g. the
    // editor's content-browser thumbnails), registered on first request and
    // cached so we don't re-register every frame.
    egui_texture_ids: HashMap<TextureHandle, egui::TextureId>,
    // True between begin_egui_pass() and the end_pass in render_frame; used
    // to keep the begin/end pairing balanced when frames are skipped.
    egui_pass_active: bool,
    // Clipboard/cursor/link requests from the last rendered pass, picked up
    // by the host loop and handed to egui-winit.
    egui_platform_output: Option<egui::PlatformOutput>,
    // Status line drawn bottom-center in its own color (e.g. red load errors),
    // independent of the help text.  Empty = hidden.
    status_msg: String,
    status_msg_color: CgVec4,

    bullet_hole_pass: BulletHolePass,
    bullet_hole_actor_index: Option<u32>,
    bullet_hole_trace: (CgVec3, CgVec3),
    custom_world_passes: Vec<ModelPass>,
    custom_foreground_passes: Vec<ModelPass>,

    asset_manager: AssetManager,
    actor_map: HashMap<u32, Actor>,
    // Scene lights mirrored from the game/editor (see add_or_update_light),
    // consumed by the deferred lighting pass each frame.
    light_map: HashMap<u32, Light>,

    particle_map: HashMap<ParticleHandle, ParticleActor>,
    next_particle_id: ParticleHandle,
    active_particles: usize,

    debug_lines: Vec<Line>,

    game_camera: Camera,
    postprocess_mode: PostProcessMode,
    frame_times: Vec<f32>,
    frame_timer: Instant,
    frame_count: u32,
    window_id: winit::window::WindowId,
    // Kept so games can control the cursor (grab/hide) for mouse look.
    window: Arc<winit::window::Window>,

    allow_debug_text: bool,
    display_debug_msg: bool,
    // Y (physical px) where the top-left/top-right debug text begins.  A game
    // with a top menu bar pushes this down so the bar doesn't cover the text.
    debug_text_top_offset: f32,
    // Whether the auto-generated "Press [H] for help" hint is drawn.  Games
    // that expose help another way (e.g. a menu button) turn it off.
    show_help_hint: bool,
    game_debug_msg: String,
    game_hud_msg: String,
    // Shown right-aligned in the top-right corner while help is up (e.g. camera pos).
    game_topright_msg: String,
    debug_msg_color: CgVec4,
}

impl<'a> Renderer<'a> {
    pub async fn new(window: Arc<winit::window::Window>, game_config: &Config) -> Self {
        log!("Renderer::new() called...");

        let mut asset_manager = AssetManager::new();
        let device_resources = DeviceResources::new(window.clone(), game_config).await;
        let default_sprite_pass = SpritePass::new(
            "/engine_assets/textures/sprite_sheet.png".to_string(),
            0,
            &device_resources,
            &mut asset_manager,
            game_config,
        )
        .await;
        let postprocess_pass =
            PostprocessPass::new(&device_resources, &mut asset_manager).await;
        let model_pass = ModelPass::new(
            "/engine_assets/shaders/model.wgsl",
            &BlendMode::None,
            &device_resources,
            &mut asset_manager,
        )
        .await;

        let line_pass = LinePass::new(
            "/engine_assets/shaders/line.wgsl",
            &device_resources,
            &mut asset_manager,
        )
        .await;
        let sunbeam_pass =
            SunbeamPass::new(&device_resources, &mut asset_manager).await;
        let custom_world_passes = Vec::<ModelPass>::new();
        let custom_foreground_passes = Vec::<ModelPass>::new();
        let bullet_hole_pass = BulletHolePass::new(
            "/engine_assets/shaders/bullet_hole.wgsl",
            &device_resources,
            &mut asset_manager,
        )
        .await;

        let model_with_holes_pass = ModelPass::new(
            "/engine_assets/shaders/model_with_holes.wgsl",
            &BlendMode::None,
            &device_resources,
            &mut asset_manager,
        )
        .await;

        // GL (the 2D demo's backend) can't run the deferred shadow/lighting
        // pipeline -- see the field comments on gbuffer_pass et al.
        let deferred_supported = device_resources.adapter.get_info().backend != wgpu::Backend::Gl;
        let (gbuffer_pass, shadow_pass, lighting_pass) = if deferred_supported {
            let gbuffer_pass = GBufferPass::new(&device_resources, &mut asset_manager).await;
            let shadow_pass = ShadowPass::new(&device_resources, &mut asset_manager).await;
            let lighting_pass =
                LightingPass::new(&device_resources, &mut asset_manager, &shadow_pass).await;
            (Some(gbuffer_pass), Some(shadow_pass), Some(lighting_pass))
        } else {
            (None, None, None)
        };

        let debug_lines = Vec::<Line>::new();

        let egui_ctx = egui::Context::default();
        let egui_renderer = egui_wgpu::Renderer::new(
            &device_resources.device,
            device_resources.surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        Renderer {
            device_resources,
            egui_texture_ids: HashMap::new(),
            default_sprite_pass,
            custom_sprite_passes: vec![],
            model_pass,
            model_with_holes_pass,
            gbuffer_pass,
            shadow_pass,
            lighting_pass,
            deferred_world_enabled: false,
            postprocess_pass,
            line_pass,
            sunbeam_pass,
            gaussian_splat_pass: None,
            splat_composite_pass: None,
            egui_ctx,
            egui_renderer,
            egui_pass_active: false,
            egui_platform_output: None,
            status_msg: "".to_string(),
            status_msg_color: CgVec4::new(1.0, 0.25, 0.2, 1.0),
            custom_world_passes,
            custom_foreground_passes,

            bullet_hole_pass,
            bullet_hole_actor_index: None,
            bullet_hole_trace: (CG_VEC3_ZERO, CG_VEC3_ZERO),

            asset_manager,
            actor_map: HashMap::<u32, Actor>::new(),
            light_map: HashMap::<u32, Light>::new(),
            particle_map: HashMap::<ParticleHandle, ParticleActor>::new(),
            next_particle_id: INVALID_PARTICLE_HANDLE,
            active_particles: 0,

            debug_lines,

            game_camera: Camera::new(),
            postprocess_mode: PostProcessMode::Passthrough,
            frame_times: Vec::<f32>::new(),
            frame_timer: Instant::now(),
            frame_count: 0,
            window_id: window.id(),
            window,

            game_debug_msg: "".to_string(),
            game_hud_msg: "".to_string(),
            game_topright_msg: "".to_string(),

            allow_debug_text: true,
            display_debug_msg: false,
            debug_text_top_offset: 10.0,
            show_help_hint: true,
            debug_msg_color: CgVec4::new(0.0, 1.0, 0.0, 1.0),
        }
    }

    /// Acquires the next swapchain image.  Returns `None` when there is no
    /// frame to render this tick (occluded/timeout, or the surface needed a
    /// reconfigure) -- the caller should just skip the frame and try again.
    pub fn begin_frame(&mut self) -> Option<(wgpu::SurfaceTexture, wgpu::TextureView)> {
        PERF_SCOPE!("begin_frame())");

        use wgpu::CurrentSurfaceTexture::*;
        let final_texture = match self.device_resources.surface.get_current_texture() {
            Success(texture) | Suboptimal(texture) => texture,
            Timeout | Occluded => return None,
            Outdated | Lost => {
                self.device_resources.surface.configure(
                    &self.device_resources.device,
                    &self.device_resources.surface_config,
                );
                return None;
            }
            Validation => {
                log!("Validation error acquiring the surface texture; skipping frame");
                return None;
            }
        };
        let final_view = final_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        Some((final_texture, final_view))
    }

    pub fn end_frame(&self, final_tex: wgpu::SurfaceTexture) {
        PERF_SCOPE!("end_frame())");

        final_tex.present();
    }

    /// The egui context.  Games and editors build their UI against it any
    /// time between the engine's begin_egui_pass() and render_frame -- in
    /// practice, from inside tick_frame.
    pub fn egui_ctx(&self) -> &egui::Context {
        &self.egui_ctx
    }

    /// Starts this frame's egui pass with the input egui-winit collected.
    /// Called by the engine loop right before ticking the game.
    pub fn begin_egui_pass(&mut self, raw_input: egui::RawInput) {
        // If the previous pass never reached render_frame (frame skipped),
        // retire it first so begin/end stay balanced.
        self.discard_egui_pass();
        self.egui_ctx.begin_pass(raw_input);
        self.egui_pass_active = true;
    }

    /// Ends an unrendered egui pass.  Texture deltas are still applied --
    /// egui only sends them once (fonts especially), so dropping them would
    /// corrupt every later frame.
    fn discard_egui_pass(&mut self) {
        if !self.egui_pass_active {
            return;
        }
        self.egui_pass_active = false;
        let full_output = self.egui_ctx.end_pass();
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &self.device_resources.device,
                &self.device_resources.queue,
                *id,
                delta,
            );
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }

    /// Clipboard/cursor/link requests from the last rendered egui pass; the
    /// host loop forwards them to egui-winit.
    pub fn take_egui_platform_output(&mut self) -> Option<egui::PlatformOutput> {
        self.egui_platform_output.take()
    }

    /// Ends the frame's egui pass and paints it onto `final_view` (the
    /// swapchain).  Runs after every other pass so the GUI is always on top.
    fn render_egui(&mut self, final_view: &wgpu::TextureView, game_config: &Config) {
        if !self.egui_pass_active {
            return;
        }
        self.egui_pass_active = false;
        let full_output = self.egui_ctx.end_pass();

        let device = &self.device_resources.device;
        let queue = &self.device_resources.queue;
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, delta);
        }

        let clipped = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [game_config.window_width, game_config.window_height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui"),
        });
        self.egui_renderer
            .update_buffers(device, queue, &mut encoder, &clipped, &screen);
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: final_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });
            // egui-wgpu wants a pass with no encoder borrow.
            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer.render(&mut render_pass, &clipped, &screen);
        }
        queue.submit(std::iter::once(encoder.finish()));

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        self.egui_platform_output = Some(full_output.platform_output);
    }

    pub fn get_encoder(&mut self, label: &str) -> wgpu::CommandEncoder {
        self.device_resources
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    pub fn submit_encoder(&mut self, command_encoder: wgpu::CommandEncoder) {
        self.device_resources
            .queue
            .submit(std::iter::once(command_encoder.finish()));
    }

    pub fn get_sorted_render_objects(
        &self,
        game_objects: &Vec<GameObject>,
    ) -> (Vec<GameObject>, Vec<GameObject>, Vec<GameObject>) {
        PERF_SCOPE!("sorting render objects");
        let mut skybox_render_objs = Vec::<GameObject>::new();
        let mut cloud_render_objs = Vec::<GameObject>::new();
        let mut game_render_objs = Vec::<GameObject>::new();

        for game_obj in game_objects {
            let new_game_obj = game_obj.clone();
            if matches!(game_obj.object_type, GameObjectType::Skybox) {
                skybox_render_objs.push(new_game_obj);
            } else if matches!(game_obj.object_type, GameObjectType::Cloud) {
                cloud_render_objs.push(new_game_obj.clone());
            } else {
                game_render_objs.push(new_game_obj.clone());
            }
        }

        skybox_render_objs.sort_by(|a, b| a.position.z.partial_cmp(&b.position.z).unwrap());
        cloud_render_objs.sort_by(|a, b| a.position.z.partial_cmp(&b.position.z).unwrap());
        game_render_objs.sort_by(|a, b| a.position.z.partial_cmp(&b.position.z).unwrap());

        (game_render_objs, skybox_render_objs, cloud_render_objs)
    }

    pub fn render_debug_text(
        &mut self,
        command_encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _num_game_objects: u32,
        game_config: &Config,
    ) {
        let device_resources = &mut self.device_resources;

        // Text pass: debug/help text (when enabled) plus the status line (always).
        if self.allow_debug_text || !self.status_msg.is_empty() {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                multiview_mask: None,
                timestamp_writes: None,
            });

            let mut total_frame_times = 0.0;
            for frame_time in &self.frame_times {
                total_frame_times += frame_time;
            }
            let avg_frame_time = total_frame_times / (self.frame_times.len() as f32);
            let frame_rate = 1.0 / avg_frame_time;

            // Optional "Press [H]..." hint prefix (suppressed when a game drives
            // help another way, e.g. a menu button).
            let hint = if self.show_help_hint {
                if self.display_debug_msg {
                    "Press [H] to hide help\n"
                } else {
                    "Press [H] for help.  Press [Space] to cycle scenes\n\n"
                }
            } else {
                ""
            };
            let frame_time_string = {
                if self.display_debug_msg {
                    format!(
                        "{hint}{}\n\
                        FPS: {:.0} \n\
                        Frame time: {:.2} ms\n\
                        Back End: {:?}\n\
                        Graphics: {}\n\n\
                        {}\n",
                        self.game_debug_msg,
                        frame_rate,
                        avg_frame_time * 1000.0,
                        device_resources.adapter.get_info().backend,
                        device_resources.adapter.get_info().name.as_str(),
                        self.game_hud_msg
                    )
                } else {
                    format!("{hint}FPS: {:.0}\n\n {}", frame_rate, self.game_hud_msg)
                }
            };

            let msg_color = [
                self.debug_msg_color.x,
                self.debug_msg_color.y,
                self.debug_msg_color.z,
                self.debug_msg_color.w,
            ];
            let section = TextSection::default()
                .add_text(
                    Text::new(&frame_time_string)
                        .with_color(msg_color)
                        .with_scale(24.0 * 1.0),
                )
                .with_screen_position((10.0, self.debug_text_top_offset));

            // Right-aligned corner text (camera pos, etc.), anchored to the top-right.
            let topright_section = TextSection::default()
                .add_text(
                    Text::new(&self.game_topright_msg)
                        .with_color(msg_color)
                        .with_scale(24.0 * 1.0),
                )
                .with_screen_position((
                    game_config.window_width as f32 - 10.0,
                    self.debug_text_top_offset,
                ))
                .with_layout(Layout::default().h_align(HorizontalAlign::Right));

            // Bottom-center status line, sized generously so it reads on
            // high-DPI touch screens.
            let status_scale = ((game_config.window_width.min(game_config.window_height) as f32)
                * 0.035)
                .clamp(22.0, 44.0);
            let status_section = TextSection::default()
                .add_text(
                    Text::new(&self.status_msg)
                        .with_color([
                            self.status_msg_color.x,
                            self.status_msg_color.y,
                            self.status_msg_color.z,
                            self.status_msg_color.w,
                        ])
                        .with_scale(status_scale),
                )
                .with_screen_position((
                    game_config.window_width as f32 * 0.5,
                    game_config.window_height as f32 - status_scale * 2.5,
                ))
                .with_layout(
                    Layout::default_single_line()
                        .h_align(HorizontalAlign::Center)
                        .v_align(VerticalAlign::Center),
                );

            let mut sections = Vec::new();
            if self.allow_debug_text {
                sections.push(&section);
                if self.display_debug_msg && !self.game_topright_msg.is_empty() {
                    sections.push(&topright_section);
                }
            }
            if !self.status_msg.is_empty() {
                sections.push(&status_section);
            }

            device_resources.brush.resize_view(
                game_config.window_width as f32,
                game_config.window_height as f32,
                &device_resources.queue,
            );
            let _ = &mut device_resources
                .brush
                .queue(
                    &device_resources.device,
                    &device_resources.queue,
                    sections,
                )
                .unwrap();
            device_resources.brush.draw(&mut render_pass);
        }

        // Frame rate update
        self.frame_count += 1;
        if self.frame_count > 16 {
            let elapsed_time = self.frame_timer.elapsed().as_secs_f32();
            let avg_frame_time = elapsed_time / (self.frame_count as f32);
            if self.frame_times.len() > 10 {
                self.frame_times.remove(0);
            }
            self.frame_times.push(avg_frame_time);

            self.frame_timer = Instant::now();
            self.frame_count = 0;
        }
    }

    pub fn render_frame(&mut self, game_objects: &Vec<GameObject>, game_config: &Config) {
        self.update_particles(game_config);
        PERF_SCOPE!("render_frame()");

        // No frame available (occluded window / surface reconfigure): skip,
        // retiring the frame's egui pass so the next begin stays balanced.
        let Some((final_tex, final_view)) = self.begin_frame() else {
            self.discard_egui_pass();
            return;
        };

        // Sorted sprite lists are computed up front (they only read game_objects)
        // so `ctx` can borrow the renderer's fields for the whole pass region
        // without colliding with this &self call.
        let (game_render_objs, skybox_render_objs, cloud_render_objs) =
            self.get_sorted_render_objects(game_objects);

        // Bundle the borrows every pass needs, once, instead of threading
        // device/assets/camera/config through each render call by hand.
        let mut ctx = RenderContext {
            device: &mut self.device_resources,
            assets: &mut self.asset_manager,
            camera: &self.game_camera,
            config: game_config,
        };

        if self.bullet_hole_actor_index.is_some() {
            PERF_SCOPE!("Bullet Holes");

            let actor = self
                .actor_map
                .get_mut(&self.bullet_hole_actor_index.unwrap())
                .unwrap();
            self.bullet_hole_pass
                .render(&mut ctx, actor, &self.bullet_hole_trace);
            self.bullet_hole_actor_index = None;
        }
        // World-layer actors: deferred when the game opted in (and the
        // backend built the passes -- GL never does), otherwise the classic
        // forward path.  Deferred draws actors into the G-buffer, then one
        // fullscreen pass per light composites them (and the clear color)
        // onto the scene target; forward is a single model pass whose World
        // branch clears the scene color/depth targets itself.  Either way,
        // everything below renders forward on top.
        if let (true, Some(gbuffer_pass)) = (self.deferred_world_enabled, self.gbuffer_pass.as_mut())
        {
            {
                PERF_SCOPE!("World GBuffer");
                gbuffer_pass.render(&mut ctx, &self.actor_map);
            }
            {
                PERF_SCOPE!("Deferred Lighting");
                self.lighting_pass.as_mut().unwrap().render(
                    &mut ctx,
                    &self.light_map,
                    &self.actor_map,
                    self.shadow_pass.as_mut().unwrap(),
                );
            }
        } else {
            PERF_SCOPE!("World Opaque");
            self.model_pass
                .render(&mut ctx, &SceneLayer::World, None, &self.actor_map);
        }
        {
            PERF_SCOPE!("World With Holes");
            self.model_with_holes_pass.render(
                &mut ctx,
                &SceneLayer::WorldHole,
                None,
                &self.actor_map,
            );
        }
        if !self.actor_map.is_empty() {
            PERF_SCOPE!("World Custom");
            for i in 0..self.custom_world_passes.len() {
                let pass = &mut self.custom_world_passes[i];
                pass.render(
                    &mut ctx,
                    &SceneLayer::WorldCustom,
                    Some(i),
                    &self.actor_map,
                );
            }
        }

        // Splats run right after the opaque world passes: they depth-test
        // against the depth those passes wrote (without writing it), so 3D
        // geometry occludes them, and everything transparent below (lines,
        // particles, sunbeams) composites on top.  A future PBR opaque pass
        // just needs to render above this point.  The splat pass renders into
        // its own scratch buffer in display space; SplatCompositePass converts
        // the finished composite into the linear HDR scene right after, so the
        // nonlinear sRGB/tonemap-inverse conversion runs once instead of
        // distorting the multi-splat blend (see splat_composite.wgsl).
        let mut splats_rendered = false;
        // Keep the composite pass's tonemap params in lockstep with the
        // postprocess pass (covers the pass being built lazily after settings
        // were set).
        let pp_settings = self.postprocess_pass.settings;
        if let Some(splat_pass) = &mut self.gaussian_splat_pass {
            if splat_pass.has_model() {
                PERF_SCOPE!("Gaussian Splats");
                splat_pass.render(&mut ctx);
                if let Some(composite_pass) = &mut self.splat_composite_pass {
                    composite_pass.settings = pp_settings;
                    composite_pass.render(&mut ctx);
                }
                splats_rendered = true;
            }
        }

        // Shadow-catcher overlay: darkens the just-composited splats where an
        // invisible catcher proxy received a CG object's shadow this frame.
        // Only meaningful in the deferred path (which produced the catcher
        // depth + shadow); a no-op where no catcher was rendered.
        if splats_rendered && self.deferred_world_enabled {
            if let Some(shadow_pass) = self.shadow_pass.as_mut() {
                PERF_SCOPE!("Shadow Catcher Overlay");
                shadow_pass.render_catcher_overlay(&mut ctx);
            }
        }

        {
            PERF_SCOPE!("World Debug");
            self.line_pass.render(&mut ctx, &self.debug_lines);
        }

        if !self.particle_map.is_empty() {
            PERF_SCOPE!("World Transparent");
            self.model_pass.render_particles(
                &mut ctx,
                ParticleBlendMode::AlphaBlend,
                &mut self.particle_map,
            );
            self.model_pass.render_particles(
                &mut ctx,
                ParticleBlendMode::Additive,
                &mut self.particle_map,
            );
        }

        if game_config.sunbeams_enabled {
            self.sunbeam_pass.render(&mut ctx);
        }

        if !self.actor_map.is_empty() {
            PERF_SCOPE!("Foreground Opaque");
            self.model_pass
                .render(&mut ctx, &SceneLayer::Foreground, None, &self.actor_map);
            {
                PERF_SCOPE!("Foreground Custom");
                for i in 0..self.custom_foreground_passes.len() {
                    let pass = &mut self.custom_foreground_passes[i];
                    pass.render(
                        &mut ctx,
                        &SceneLayer::ForegroundCustom,
                        Some(i),
                        &self.actor_map,
                    );
                }
            }
        }

        if !skybox_render_objs.is_empty() {
            PERF_SCOPE!("Sprite Pass Sky");

            self.default_sprite_pass.render(
                &mut ctx,
                SpriteBlend::Opaque,
                &skybox_render_objs,
            );
        }

        if !cloud_render_objs.is_empty() {
            PERF_SCOPE!("Sprite Pass Clouds");

            self.default_sprite_pass.render(
                &mut ctx,
                SpriteBlend::Transparent,
                &cloud_render_objs,
            );
        }

        if !game_render_objs.is_empty() {
            PERF_SCOPE!("2D Game Objects");

            self.default_sprite_pass.render(
                &mut ctx,
                SpriteBlend::Opaque,
                &game_render_objs,
            );
        }

        {
            PERF_SCOPE!("Custom Passes");
            for pass in &mut self.custom_sprite_passes {
                pass.render(&mut ctx, SpriteBlend::Opaque, &game_render_objs);
            }
        }

        {
            PERF_SCOPE!("Postprocess pass");
            self.postprocess_pass.render(
                &mut ctx,
                &final_view,
                Some(self.postprocess_mode.clone()),
            );
        }

        {
            PERF_SCOPE!("Debug text pass");
            let mut command_encoder = self.get_encoder("Debug Text Pass");
            self.render_debug_text(
                &mut command_encoder,
                &final_view,
                game_objects.len() as u32,
                game_config,
            );
            self.submit_encoder(command_encoder);
        }

        {
            PERF_SCOPE!("egui pass");
            self.render_egui(&final_view, game_config);
        }

        self.end_frame(final_tex);

        let cur_time = game_config.start_time.elapsed().as_secs_f32();
        self.debug_lines.retain_mut(|l| cur_time < l.end_time);
    }

    pub fn resize(&mut self, game_config: &Config) {
        log!(
            "Resizing window to {} x {}",
            game_config.window_width,
            game_config.window_height
        );

        self.device_resources.resize(game_config);
        self.postprocess_pass
            .resize(&mut self.device_resources, &self.asset_manager);
        if let Some(shadow_pass) = self.shadow_pass.as_mut() {
            shadow_pass.resize(&self.device_resources);
        }
        if let (Some(lighting_pass), Some(shadow_pass)) =
            (self.lighting_pass.as_mut(), self.shadow_pass.as_ref())
        {
            lighting_pass.resize(&self.device_resources, shadow_pass);
        }
        self.sunbeam_pass.resize(&mut self.device_resources, &self.asset_manager);
        if let Some(composite_pass) = self.splat_composite_pass.as_mut() {
            composite_pass.resize(&self.device_resources);
        }
    }

    pub fn window_id(&self) -> winit::window::WindowId {
        self.window_id
    }

    pub fn add_or_update_actor(&mut self, actor: &Actor) {
        self.actor_map.insert(actor.id, actor.clone());
    }

    pub fn remove_actor(&mut self, actor: &Actor) {
        self.actor_map.remove(&actor.id);
    }

    /// Mirrors a scene light into the renderer so the deferred lighting pass
    /// samples it.  Call again after edits (same id updates in place).
    pub fn add_or_update_light(&mut self, light: &Light) {
        self.light_map.insert(light.id, light.clone());
    }

    pub fn remove_light(&mut self, light: &Light) {
        self.light_map.remove(&light.id);
    }

    /// Removes every light (editor "New Scene" / scene load).
    pub fn clear_lights(&mut self) {
        self.light_map.clear();
    }

    /// Routes World-layer actors through the deferred G-buffer + lighting
    /// pipeline instead of the classic forward model pass.  Only makes sense
    /// for games that register scene lights (add_or_update_light) -- without
    /// any, the world composites to flat clear_color.  Ignored on the GL
    /// backend, which never builds the deferred passes.
    pub fn set_deferred_world_enabled(&mut self, enabled: bool) {
        self.deferred_world_enabled = enabled;
    }

    /// Sets the shadow quality settings (cascade count, tile resolution,
    /// cascade distance); applied at the start of the next frame. A no-op on
    /// the GL backend, which has no shadow pass to configure.
    pub fn set_shadow_settings(&mut self, settings: &ShadowSettings) {
        if let Some(shadow_pass) = self.shadow_pass.as_mut() {
            shadow_pass.request_settings(settings);
        }
    }

    /// Returns `ShadowSettings::default()` on the GL backend, which has no
    /// shadow pass of its own to report settings from.
    pub fn get_shadow_settings(&self) -> ShadowSettings {
        self.shadow_pass
            .as_ref()
            .map_or_else(ShadowSettings::default, |p| p.settings())
    }

    /// This frame's screen-space shadow accumulation texture (the product of
    /// every light's mask; 1 = fully lit).  The Gaussian-splat shadow overlay
    /// will multiply the splats by it. Only the non-GL backends load Gaussian
    /// splats, so this is never called without a shadow pass to back it.
    pub fn shadow_accum_texture(&self) -> &Texture {
        self.shadow_pass
            .as_ref()
            .expect("shadow_accum_texture: no shadow pass (GL backend has none)")
            .shadow_accum()
    }

    pub async fn add_particle_actor(
        &mut self,
        transform: &ActorTransform,
        particle_params: &ParticleParams,
        active: bool,
    ) -> ParticleHandle {
        self.next_particle_id.index = {
            if self.next_particle_id.index == u32::MAX {
                0
            } else {
                self.next_particle_id.index + 1
            }
        };
        let mut particle = ParticleActor::new(
            transform,
            &self.next_particle_id,
            particle_params,
            &self.device_resources,
            &mut self.asset_manager,
        )
        .await;
        particle.set_active(active);
        self.particle_map
            .insert(self.next_particle_id.clone(), particle);

        self.next_particle_id.clone()
    }

    /// Loads a texture up front and returns its (cached) handle, so a particle
    /// using it can later be spawned synchronously with `spawn_particle_actor`
    /// from the non-async frame tick.
    pub async fn preload_texture(&mut self, file_path: &str) -> TextureHandle {
        self.asset_manager
            .load_texture(file_path, &self.device_resources, TextureFilter::Linear)
            .await
    }

    /// egui TextureId for an already-loaded texture, so the in-engine GUI can
    /// draw it (e.g. content-browser thumbnails).  Registered with the egui
    /// renderer on first request and cached; returns None if the handle is
    /// stale.
    pub fn egui_texture_id(&mut self, handle: &TextureHandle) -> Option<egui::TextureId> {
        if let Some(id) = self.egui_texture_ids.get(handle) {
            return Some(*id);
        }
        let texture = self.asset_manager.get_texture(handle);
        let id = self.egui_renderer.register_native_texture(
            &self.device_resources.device,
            &texture.view,
            wgpu::FilterMode::Linear,
        );
        self.egui_texture_ids.insert(*handle, id);
        Some(id)
    }

    /// Synchronous counterpart to `add_particle_actor`: spawns a particle actor
    /// from an already-preloaded texture (see `preload_texture`), so it can run
    /// inside the frame tick.  Returns the new particle's handle.
    pub fn spawn_particle_actor(
        &mut self,
        transform: &ActorTransform,
        particle_params: &ParticleParams,
        texture: &TextureHandle,
        active: bool,
    ) -> ParticleHandle {
        self.next_particle_id.index = {
            if self.next_particle_id.index == u32::MAX {
                0
            } else {
                self.next_particle_id.index + 1
            }
        };
        let mut particle = ParticleActor::from_texture(
            transform,
            &self.next_particle_id,
            particle_params,
            texture,
            &self.device_resources,
            &mut self.asset_manager,
        );
        particle.set_active(active);
        self.particle_map
            .insert(self.next_particle_id.clone(), particle);

        self.next_particle_id.clone()
    }

    /// Removes a particle actor (e.g. when the editor deletes it from the scene).
    pub fn remove_particle_actor(&mut self, handle: &ParticleHandle) {
        self.particle_map.remove(handle);
    }

    pub fn enable_particle_actor(&mut self, handle: &ParticleHandle, enable: bool) {
        let particle = self.particle_map.get_mut(handle).unwrap();
        particle.set_active(enable);
    }

    pub fn update_particle_transform(
        &mut self,
        handle: &ParticleHandle,
        position: &CgVec3,
        scale: &Option<CgVec3>,
    ) {
        let particle = self.particle_map.get_mut(handle).unwrap();
        particle.set_position(position);

        if let Some(s) = scale {
            particle.set_scale(s);
        }
    }
    pub async fn load_model(&mut self, file_path: &str, use_holes: bool) -> ModelHandle {
        self.asset_manager
            .load_model(file_path, &mut self.device_resources, use_holes)
            .await
    }

    /// Registers a model from in-memory glb/gltf or .obj bytes (see
    /// [`AssetManager::add_model_from_bytes`]).  Synchronous, for the web
    /// editor's model import and MuJoCo mesh geoms, both from the frame tick.
    #[cfg(target_arch = "wasm32")]
    pub fn load_model_from_bytes(
        &mut self,
        file_path: &str,
        bytes: &[u8],
        use_holes: bool,
    ) -> ModelHandle {
        self.asset_manager
            .add_model_from_bytes(file_path, bytes, &mut self.device_resources, use_holes)
    }

    /// Every loaded model as (file path, handle), sorted by path -- feeds the
    /// editor's resource list and model dropdowns.
    pub fn get_model_resources(&self) -> Vec<(String, ModelHandle)> {
        self.asset_manager.get_model_resources()
    }

    /// Registers a named material (textures + color/spec constants) actors can
    /// reference; loading the same name again returns the existing handle.
    pub async fn load_material(&mut self, name: &str, desc: &MaterialDesc) -> MaterialHandle {
        self.asset_manager
            .load_material(name, desc, &self.device_resources)
            .await
    }

    /// Every loaded material as (name, handle), sorted by name -- feeds the
    /// editor's material dropdown.
    pub fn get_material_resources(&self) -> Vec<(String, MaterialHandle)> {
        self.asset_manager.get_material_resources()
    }

    /// Synchronously creates a constant-only material (no textures), for the
    /// editor's "new material" button in the frame tick.  Same name returns
    /// the existing handle.
    pub fn create_material(&mut self, name: &str, desc: &MaterialDesc) -> MaterialHandle {
        self.asset_manager
            .create_material(name, desc, &self.device_resources)
    }

    /// Rebuilds an existing material from a new description (textures and/or
    /// constants), keeping its handle valid so actors using it update.  Async
    /// because assigning a texture may need to load it.
    pub async fn reload_material(
        &mut self,
        handle: &MaterialHandle,
        name: &str,
        desc: &MaterialDesc,
    ) {
        self.asset_manager
            .reload_material(handle, name, desc, &self.device_resources)
            .await;
    }

    /// A material's (color constant, metallic/roughness constant), for the
    /// editor's resource inspector.  None if the handle is stale.
    pub fn material_constants(&self, handle: &MaterialHandle) -> Option<(CgVec4, CgVec4)> {
        self.asset_manager
            .get_material(handle)
            .map(|m| (m.color_constant, m.mr_constant))
    }

    /// Overwrites a material's constants; applies immediately (the G-buffer
    /// pass reads them every frame).
    pub fn update_material(
        &mut self,
        handle: &MaterialHandle,
        color_constant: &CgVec4,
        mr_constant: &CgVec4,
    ) {
        self.asset_manager
            .update_material_constants(handle, color_constant, mr_constant);
    }

    /// Replaces a live emitter's particle parameters (editor resource edits).
    /// The emitter keeps its texture -- a changed `texture_file` only affects
    /// future spawns.
    pub fn update_particle_params(&mut self, handle: &ParticleHandle, params: &ParticleParams) {
        if let Some(particle) = self.particle_map.get_mut(handle) {
            particle.params = params.clone();
        }
    }

    /// Loads a splat .ply and appends it as a selectable cloud.  Returns true if
    /// it loaded (a missing/unreadable file is skipped and returns false).  Call
    /// repeatedly to preload several clouds, then cycle with `set_active_gaussian_splat`.
    pub async fn load_gaussian_splat(&mut self, file_path: &str, params: &SplatParams) -> bool {
        self.ensure_gaussian_splat_pipeline().await;
        let splat_pass = self.gaussian_splat_pass.as_mut().unwrap();
        splat_pass.set_params(params);
        splat_pass.load(file_path, &self.device_resources).await
    }

    /// Builds the splat pipeline if it doesn't exist yet. Callers that will
    /// later need `load_gaussian_splat_from_bytes` (which is sync and can't
    /// build the pipeline itself) should call this during their own async
    /// startup, even if no splat is loaded yet -- otherwise the first sync
    /// load attempt fails with "splat renderer not initialized".
    pub async fn ensure_gaussian_splat_pipeline(&mut self) {
        if self.gaussian_splat_pass.is_none() {
            self.gaussian_splat_pass = Some(
                GaussianSplatPass::new(&self.device_resources, &mut self.asset_manager)
                    .await,
            );
            self.splat_composite_pass = Some(
                SplatCompositePass::new(&self.device_resources, &mut self.asset_manager).await,
            );
        }
    }

    /// Parses an in-memory splat .ply (e.g. one the user picked at runtime) and
    /// appends it as a selectable cloud.  Synchronous, so it can be called from
    /// tick_frame -- but the splat pipeline must already exist, i.e. at least one
    /// prior `load_gaussian_splat` call (pipeline creation needs async shader
    /// loads).  On success reports the count and whether the cloud was clamped
    /// to the GPU budget; on failure returns a short user-displayable reason.
    pub fn load_gaussian_splat_from_bytes(
        &mut self,
        bytes: &[u8],
        name: &str,
        params: &SplatParams,
    ) -> Result<SplatLoadInfo, String> {
        let Some(splat_pass) = &mut self.gaussian_splat_pass else {
            log!("load_gaussian_splat_from_bytes called before the splat pipeline exists");
            return Err("splat renderer not initialized".to_string());
        };
        splat_pass.set_params(params);
        splat_pass.load_from_bytes(bytes, name, &self.device_resources)
    }

    /// Number of splat clouds preloaded via `load_gaussian_splat`.
    pub fn num_gaussian_splats(&self) -> usize {
        self.gaussian_splat_pass
            .as_ref()
            .map_or(0, |g| g.num_models())
    }

    /// Selects which preloaded splat cloud to render (out-of-range is ignored).
    pub fn set_active_gaussian_splat(&mut self, index: usize) {
        if let Some(splat_pass) = &mut self.gaussian_splat_pass {
            splat_pass.set_active_model(index);
        }
    }

    /// Unloads every splat cloud (editor "New Scene").  Nothing renders until
    /// the next load.
    pub fn clear_gaussian_splats(&mut self) {
        if let Some(splat_pass) = &mut self.gaussian_splat_pass {
            splat_pass.clear_models();
        }
    }

    /// Unloads a single splat cloud (editor per-splat delete / delete-undo).
    /// Out-of-range indices are ignored.  See
    /// `GaussianSplatPass::remove_model`.
    pub fn remove_gaussian_splat(&mut self, index: usize) {
        if let Some(splat_pass) = &mut self.gaussian_splat_pass {
            splat_pass.remove_model(index);
        }
    }

    /// Number of gaussian splats in the currently active cloud (0 if none).
    pub fn active_gaussian_splat_count(&self) -> u32 {
        self.gaussian_splat_pass
            .as_ref()
            .map_or(0, |g| g.active_splat_count())
    }

    pub fn set_gaussian_splat_params(&mut self, params: &SplatParams) {
        if let Some(splat_pass) = &mut self.gaussian_splat_pass {
            splat_pass.set_params(params);
        }
    }

    /// Sets the world transform of the active splat cloud (editor gizmo drag).
    pub fn set_gaussian_splat_transform(&mut self, transform: &ActorTransform) {
        if let Some(splat_pass) = &mut self.gaussian_splat_pass {
            let rotation: cgmath::Matrix3<f32> = transform.rotation.into();
            let matrix = cgmath::Matrix4::from_translation(transform.position)
                * cgmath::Matrix4::from(rotation)
                * cgmath::Matrix4::from_nonuniform_scale(
                    transform.scale.x,
                    transform.scale.y,
                    transform.scale.z,
                );
            splat_pass.set_transform(matrix);
        }
    }

    pub fn set_camera(&mut self, camera: &Camera) {
        self.game_camera = camera.clone();
    }

    pub fn update_particles(&mut self, game_config: &Config) {
        self.active_particles = 0;
        //  let particle_iter = self.particle_map.iter_mut();
        for particle in &mut self.particle_map {
            if particle.1.is_active() {
                particle.1.tick(game_config);
                self.active_particles += 1;
            }
        }
    }

    pub async fn add_custom_pass(
        &mut self,
        layer: &SceneLayer,
        blend_mode: &BlendMode,
        shader_path: &str,
    ) -> usize {
        let new_pass = ModelPass::new(
            shader_path,
            blend_mode,
            &self.device_resources,
            &mut self.asset_manager,
        )
        .await;

        (match *layer {
            SceneLayer::ForegroundCustom => {
                self.custom_foreground_passes.push(new_pass);
                self.custom_foreground_passes.len()
            }

            SceneLayer::WorldCustom => {
                self.custom_world_passes.push(new_pass);
                self.custom_world_passes.len()
            }

            _ => {
                panic!(
                    "Renderer::add_custom_pass() - Render layer {:?} not supported",
                    layer
                );
            }
        }) - 1
    }

    pub fn add_bullet_hole(&mut self, actor: &Actor, start_trace: &CgVec3, end_trace: &CgVec3) {
        self.bullet_hole_actor_index = Some(actor.id);
        self.bullet_hole_trace = (*start_trace, *end_trace);
    }

    pub fn add_line(
        &mut self,
        start: &CgVec3,
        end: &CgVec3,
        color: &CgVec4,
        thickness: f32,
        duration: f32,
        game_config: &Config,
    ) {
        self.debug_lines.push(Line {
            start: *start,
            end: *end,
            color: *color,
            thickness,
            end_time: game_config.start_time.elapsed().as_secs_f32() + duration,
        });
    }

    pub fn set_allow_debug_text(&mut self, allow: bool) {
        self.allow_debug_text = allow;
    }

    /// Sets the bottom-center status line (empty string hides it).  Unlike the
    /// help text this is always visible, so games use it for things the user
    /// must see -- load errors, clamp warnings, progress.
    pub fn set_status_msg(&mut self, msg: &str, color: &CgVec4) {
        self.status_msg = msg.to_string();
        self.status_msg_color = *color;
    }

    pub fn set_debug_game_msg(&mut self, msg: &str) {
        self.game_debug_msg = msg.to_string();
    }

    /// Right-aligned text drawn in the top-right corner while help is shown.
    pub fn set_debug_topright_msg(&mut self, msg: &str) {
        self.game_topright_msg = msg.to_string();
    }

    pub fn set_hud_msg(&mut self, msg: &str) {
        self.game_hud_msg = msg.to_string();
    }

    pub fn set_debug_font_color(&mut self, color: &CgVec4) {
        self.debug_msg_color = *color;
    }

    pub fn enable_help_text(&mut self) {
        self.display_debug_msg = !self.display_debug_msg;
    }

    /// Sets the Y (physical px) where the top-left debug text and top-right
    /// corner text begin, so a game with a top menu bar can push them clear
    /// of the bar.  Defaults to 10.
    pub fn set_debug_text_top_offset(&mut self, y: f32) {
        self.debug_text_top_offset = y;
    }

    /// Toggles the auto-generated "Press [H] for help" hint line (on by
    /// default).  Turn off when help is reachable another way (e.g. a menu).
    pub fn set_show_help_hint(&mut self, show: bool) {
        self.show_help_hint = show;
    }

    /// Shows/hides the mouse cursor.  Pair with `set_cursor_grabbed` and raw
    /// mouse motion (`InputManager::get_mouse_raw_delta`) for mouse look.
    pub fn set_cursor_visible(&self, visible: bool) {
        self.window.set_cursor_visible(visible);
    }

    /// Grabs the cursor so it can't leave the window/screen while looking
    /// around.  Best effort across platforms: tries hardware Lock (macOS,
    /// Wayland, browser pointer-lock) and falls back to Confine (Windows,
    /// X11).  Errors (e.g. unsupported mode) are ignored.
    pub fn set_cursor_grabbed(&self, grabbed: bool) {
        use winit::window::CursorGrabMode;
        if grabbed {
            if self.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                let _ = self.window.set_cursor_grab(CursorGrabMode::Confined);
            }
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }
    }

    pub fn set_postprocess_mode(&mut self, new_mode: &PostProcessMode) {
        self.postprocess_mode = new_mode.clone();
    }

    pub fn num_active_particles(&self) -> usize {
        self.active_particles
    }

    /// Enables the tonemap applied in the postprocess pass.  Kept in sync
    /// with the splat pass (which inverts the same curve) by `render`.
    pub fn set_tonemap_enabled(&mut self, enabled: bool) {
        self.postprocess_pass.settings.tonemap_enabled = enabled;
    }

    /// Sets the full scene post-process/tonemap settings.  The same curve is
    /// pushed to the gaussian-splat pass each frame in `render` so splats
    /// pre-apply its exact inverse and survive the tonemap unchanged.
    pub fn set_post_process_settings(&mut self, settings: &PostProcessSettings) {
        self.postprocess_pass.settings = *settings;
    }

    /// The current scene post-process/tonemap settings.
    pub fn post_process_settings(&self) -> PostProcessSettings {
        self.postprocess_pass.settings
    }

    pub async fn add_sprite_pass(
        &mut self,
        texture_path: String,
        game_config: &Config,
    ) -> u32 {
        let new_index = self.custom_sprite_passes.len() as u32 + 1;
        let pass = SpritePass::new(
            texture_path,
            new_index,
            &self.device_resources,
            &mut self.asset_manager,
            game_config,
        )
        .await;
        self.custom_sprite_passes.push(pass);
        new_index
    }
}
