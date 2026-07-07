use std::sync::Arc;
use winit::event::StartCause;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
};
// Re-exported so games build their GUI against the exact egui version the
// engine renders with (no version-matching burden in game crates).
pub use egui;

pub mod kb_assets;
pub mod kb_collision;
pub mod kb_config;
pub mod kb_engine;
pub mod kb_game_object;
pub mod kb_input;
pub mod kb_renderer;
pub mod kb_resource;
pub mod kb_utils;
pub mod render_groups {
    pub mod kb_bullet_hole_group;
    pub mod kb_gaussian_splat_group;
    pub mod kb_line_group;
    pub mod kb_model_group;
    pub mod kb_postprocess_group;
    pub mod kb_sprite_group;
    pub mod kb_sunbeam_group;
}

use crate::kb_config::*;
use crate::kb_engine::*;
use crate::kb_input::*;
use crate::kb_renderer::*;
use crate::kb_resource::*;

#[cfg(target_arch = "wasm32")]
const WEBAPP_CANVAS_ID: &str = "target";

pub async fn run_game<T>(mut game_config: KbConfig)
where
    T: KbGameEngine + 'static,
{
    env_logger::init();

    let event_loop: EventLoop<()> = EventLoop::new().unwrap();
    let one_micro = core::time::Duration::from_micros(1);
    let control_flow = ControlFlow::wait_duration(one_micro);
    event_loop.set_control_flow(control_flow);

    #[cfg(target_arch = "wasm32")]
    let window = Arc::new({
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowAttributesExtWebSys;
        let dom_window = web_sys::window().unwrap();
        let dom_document = dom_window.document().unwrap();
        let dom_canvas = dom_document.get_element_by_id(WEBAPP_CANVAS_ID).unwrap();
        let canvas = dom_canvas.dyn_into::<web_sys::HtmlCanvasElement>().ok();
        game_config.window_width = canvas.as_ref().unwrap().width();
        game_config.window_height = canvas.as_ref().unwrap().height();
        let attributes = winit::window::Window::default_attributes().with_canvas(canvas);
        // winit 0.30 wants windows created from ApplicationHandler::resumed;
        // this engine keeps the closure-based loop, where the deprecated
        // create_window is the supported escape hatch.
        #[allow(deprecated)]
        event_loop.create_window(attributes).unwrap()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let window = Arc::new({
        let window_size =
            winit::dpi::PhysicalSize::new(game_config.window_width, game_config.window_height);
        let attributes = winit::window::Window::default_attributes().with_inner_size(window_size);
        #[allow(deprecated)]
        event_loop.create_window(attributes).unwrap()
    });

    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(
        game_config.window_width,
        game_config.window_height,
    ));

    let mut game_engine = T::new(&game_config);
    let mut input_manager = KbInputManager::new();
    let mut game_renderer = KbRenderer::new(window.clone(), &game_config).await;

    // Input side of the in-engine GUI: collects window events into
    // egui::RawInput for the renderer's egui context, and applies egui's
    // platform requests (cursor icons, clipboard, links) back to the window.
    let mut egui_state = egui_winit::State::new(
        game_renderer.egui_ctx().clone(),
        egui::ViewportId::ROOT,
        window.as_ref(),
        None,
        None,
        None,
    );

    game_engine
        .initialize_world(&mut game_renderer, &mut game_config)
        .await;

    let mut frame_timer = instant::Instant::now();
    let mut hack_wait = 0;
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        // Closure-based loop (winit 0.30 prefers ApplicationHandler; see the
        // create_window note above).
        #[allow(deprecated)]
        let _ = event_loop.spawn(move |event, control_flow| {
            let _ = &mut game_renderer;
            let _ = &game_config;
            let _ = &mut frame_timer;
            match event {
                Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                    // The initial 1us timer just kicks us off; from here we render
                    // via requestAnimationFrame (request_redraw + ControlFlow::Wait).
                    // rAF paces to the real frame-production rate, giving the GPU the
                    // back-pressure the browser otherwise lacks -- a timer just floods
                    // it and input shows up frames (seconds) late.
                    window.request_redraw();
                    control_flow.set_control_flow(ControlFlow::Wait);
                }
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == game_renderer.window_id() => {
                    // egui sees every event first; events it consumes (clicks
                    // on widgets, typing into fields) stay out of game input.
                    let egui_consumed = egui_state.on_window_event(&window, event).consumed;
                    match event {
                        WindowEvent::RedrawRequested => {
                            // Driven by requestAnimationFrame (see ResumeTimeReached).
                            hack_wait += 1;
                            if hack_wait > 6 {
                                game_renderer
                                    .begin_egui_pass(egui_state.take_egui_input(&window));
                                game_engine.tick_frame(
                                    &mut game_renderer,
                                    &mut input_manager,
                                    &mut game_config,
                                );
                            }
                            if hack_wait > 8 {
                                game_renderer
                                    .render_frame(game_engine.get_game_objects(), &game_config);
                                if let Some(output) = game_renderer.take_egui_platform_output() {
                                    egui_state.handle_platform_output(&window, output);
                                }
                            }
                            // Schedule the next animation frame.
                            window.request_redraw();
                        }

                        WindowEvent::MouseWheel { delta, .. } => {
                            if let MouseScrollDelta::PixelDelta(pix_delta) = delta {
                                if !egui_consumed {
                                    input_manager.update_mouse_scroll(pix_delta.y as f32 / 150.0);
                                }
                            }
                        }

                        WindowEvent::MouseInput { button, state, .. } => {
                            // Always forward: egui already saw this event above.
                            // Gating on egui_consumed can drop the *release*
                            // (egui often claims it), leaving the button stuck
                            // "down" in the game.  The game itself decides
                            // whether a click acts -- e.g. it only starts a
                            // right-drag when the pointer isn't over egui.
                            input_manager.set_mouse_button_state(button, state);
                        }

                        WindowEvent::CursorMoved { position, .. } => {
                            input_manager.set_mouse_position(position);
                        }

                        WindowEvent::CloseRequested => control_flow.exit(),

                        WindowEvent::Resized(physical_size) => {
                            if physical_size.width > 0 && physical_size.height > 0 {
                                game_config.window_width = physical_size.width;
                                game_config.window_height = physical_size.height;
                                let _ = async {
                                    game_renderer.resize(&game_config);
                                };
                            }
                        }
                        WindowEvent::Touch(winit::event::Touch {
                            phase,
                            location,
                            id,
                            ..
                        }) => {
                            // Always forward: egui already saw this above.  Gating
                            // on egui_consumed dropped touch RELEASES (sticky pads)
                            // and could drop a touch's start while keeping its
                            // moves -- panicking update_touch's lookup.  The game's
                            // pads filter by where the touch started, so touches on
                            // egui widgets don't affect them.
                            input_manager.update_touch(*phase, *id, *location);
                        }
                        WindowEvent::KeyboardInput {
                            device_id: _,
                            event,
                            is_synthetic: _,
                        } => {
                            if !egui_consumed {
                                input_manager.set_key_state(event.physical_key, event.state);

                                if input_manager.get_key_state("h").just_pressed() {
                                    game_renderer.enable_help_text();
                                }

                                if input_manager.get_key_state("v").just_pressed() {
                                    game_config.vsync = !game_config.vsync;

                                    if game_config.vsync {
                                        let one_micro = core::time::Duration::from_micros(1);
                                        control_flow.set_control_flow(ControlFlow::wait_duration(
                                            one_micro,
                                        ));
                                    } else {
                                        control_flow.set_control_flow(ControlFlow::Poll);
                                    }
                                }
                            }
                        }

                        _ => {}
                    }
                }

                // Raw mouse movement (independent of the cursor position) for
                // FPS-style look while the cursor is grabbed/hidden.
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    input_manager.add_mouse_raw_delta(delta.0, delta.1);
                }

                _ => {
                    window.request_redraw();
                }
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        #[allow(deprecated)]
        let _ = event_loop.run(move |event, control_flow| {
            let _ = &mut game_renderer;
            let _ = &game_config;
            match event {
                Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                    hack_wait += 1;
                    if hack_wait > 6 {
                        game_renderer.begin_egui_pass(egui_state.take_egui_input(&window));
                        game_engine.tick_frame(
                            &mut game_renderer,
                            &mut input_manager,
                            &mut game_config,
                        );
                    }
                    if hack_wait > 8 {
                        game_renderer.render_frame(game_engine.get_game_objects(), &game_config);
                        if let Some(output) = game_renderer.take_egui_platform_output() {
                            egui_state.handle_platform_output(&window, output);
                        }
                    }

                    let elapsed_frame_time = frame_timer.elapsed().as_secs_f32();
                    let delay = {
                        if elapsed_frame_time <= 12000.0 {
                            12000.0 - elapsed_frame_time
                        } else {
                            0.0
                        }
                    };
                    frame_timer = instant::Instant::now();
                    let delay = core::time::Duration::from_micros(delay as u64);
                    let new_control_flow = ControlFlow::wait_duration(delay);
                    control_flow.set_control_flow(new_control_flow);
                }
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == game_renderer.window_id() => {
                    // egui sees every event first; events it consumes (clicks
                    // on widgets, typing into fields) stay out of game input.
                    let egui_consumed = egui_state.on_window_event(&window, event).consumed;
                    match event {
                        WindowEvent::RedrawRequested => {
                            if !game_config.vsync {
                                game_renderer
                                    .begin_egui_pass(egui_state.take_egui_input(&window));
                                game_engine.tick_frame(
                                    &mut game_renderer,
                                    &mut input_manager,
                                    &mut game_config,
                                );

                                game_renderer
                                    .render_frame(game_engine.get_game_objects(), &game_config);
                                if let Some(output) = game_renderer.take_egui_platform_output() {
                                    egui_state.handle_platform_output(&window, output);
                                }
                            }
                        }

                        WindowEvent::MouseWheel {
                            delta: MouseScrollDelta::LineDelta(_, y),
                            ..
                        } => {
                            if !egui_consumed {
                                input_manager.update_mouse_scroll(*y);
                            }
                        }

                        WindowEvent::MouseInput { button, state, .. } => {
                            // Always forward: egui already saw this event above.
                            // Gating on egui_consumed can drop the *release*
                            // (egui often claims it), leaving the button stuck
                            // "down" in the game.  The game itself decides
                            // whether a click acts -- e.g. it only starts a
                            // right-drag when the pointer isn't over egui.
                            input_manager.set_mouse_button_state(button, state);
                        }

                        WindowEvent::CursorMoved { position, .. } => {
                            input_manager.set_mouse_position(position);
                        }

                        WindowEvent::CloseRequested => control_flow.exit(),

                        WindowEvent::Resized(physical_size) => {
                            // log!("Resized {} {}", physical_size.width, physical_size.z
                            if physical_size.width > 0 && physical_size.height > 0 {
                                game_config.window_width = physical_size.width;
                                game_config.window_height = physical_size.height;
                                game_renderer.resize(&game_config);
                            }
                        }

                        WindowEvent::Touch(winit::event::Touch {
                            phase,
                            location,
                            id,
                            ..
                        }) => {
                            // Always forward (see the wasm loop's note): gating
                            // touch on egui_consumed made the pads sticky and could
                            // panic update_touch.
                            input_manager.update_touch(*phase, *id, *location);
                        }

                        WindowEvent::KeyboardInput {
                            device_id: _,
                            event,
                            is_synthetic: _,
                        } => {
                            if !egui_consumed {
                                input_manager.set_key_state(event.physical_key, event.state);

                                if input_manager.get_key_state("h").just_pressed() {
                                    game_renderer.enable_help_text();
                                }

                                if input_manager.get_key_state("v").just_pressed() {
                                    game_config.vsync = !game_config.vsync;

                                    if game_config.vsync {
                                        let one_micro = core::time::Duration::from_micros(1);
                                        control_flow.set_control_flow(ControlFlow::wait_duration(
                                            one_micro,
                                        ));
                                    } else {
                                        control_flow.set_control_flow(ControlFlow::Poll);
                                    }
                                }
                            }
                        }

                        _ => {}
                    }
                }

                // Raw mouse movement (independent of the cursor position) for
                // FPS-style look while the cursor is grabbed/hidden.
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    input_manager.add_mouse_raw_delta(delta.0, delta.1);
                }

                _ => {
                    window.request_redraw();
                }
            }
        });
    }
}
