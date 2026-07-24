//! Loads an MJCF scene through the engine's [`black_splat::mujoco::MujocoScene`]
//! and wireframes the live geom poses every frame -- see that module's docs
//! for how the native (mujoco-rs FFI) and wasm (sibling `@mujoco/mujoco`
//! wasm module) paths differ. This example only adds the click-to-kick
//! interaction on top; loading and drawing the scene is all engine-owned.

use cgmath::InnerSpace;

use black_splat::{
    config::Config, engine::GameEngine, fly_camera::FlyCamera, game_object::{Camera, GameObject},
    input::InputManager, log, mujoco::MujocoScene, renderer::Renderer, utils::*,
};

const SCENE_PATH: &str = "game_assets/scene.xml";

// Strength of the angular-velocity kick applied to the hinge on a hit, in
// rad/s. Kick direction depends on which side of the bob you click.
const CLICK_KICK_RAD_PER_SEC: f64 = 6.0;
const BOB_RADIUS: f32 = 0.15;

pub struct MujocoTestGame {
    game_objects: Vec<GameObject>,
    game_camera: Camera,
    fly_camera: FlyCamera,
    scene: Option<MujocoScene>,
}

impl GameEngine for MujocoTestGame {
    fn new(game_config: &Config) -> Self {
        log!("MujocoTestGame::new()");

        let mut game_camera = Camera::new();
        game_camera.set_position(&game_config.start_position);
        game_camera.set_rotation(&game_config.start_rotation);

        let mut fly_camera = FlyCamera::default();
        fly_camera.move_rate = 4.0;
        fly_camera.shift_move_multiplier = 3.0;
        fly_camera.mouse_look_sensitivity = 0.18;

        Self {
            game_objects: Vec::new(),
            game_camera,
            fly_camera,
            scene: None,
        }
    }

    async fn initialize_world(&mut self, renderer: &mut Renderer<'_>, game_config: &mut Config) {
        log!("MujocoTestGame::initialize_world()");
        // Config's default clear_color is (0,0,0,0) -- fully transparent.
        // Native masks that with the opaque window surface, but on the web
        // the canvas's alpha channel lets it through to the page's white
        // background, so an unset clear_color reads as a blank white page.
        game_config.clear_color = CgVec4::new(0.05, 0.05, 0.08, 1.0);
        renderer.set_camera(&self.game_camera);

        let scene = MujocoScene::load(SCENE_PATH)
            .await
            .expect("failed to load MJCF scene");
        // We click-test against the bob and kick the hinge; on wasm these
        // names need a round-trip through JS to resolve (see MujocoScene's
        // docs), so register interest as soon as the scene is loaded.
        scene.watch_geom("bob");
        scene.watch_joint("hinge");
        self.scene = Some(scene);
    }

    fn get_game_objects(&self) -> &Vec<GameObject> {
        &self.game_objects
    }

    fn tick_frame_internal(
        &mut self,
        renderer: &mut Renderer,
        input_manager: &InputManager,
        game_config: &Config,
    ) {
        let delta_time = game_config.delta_time;

        let mut camera_rot = self.game_camera.get_rotation();
        self.fly_camera
            .apply_mouse_look(&mut camera_rot, input_manager, renderer);
        self.fly_camera
            .apply_key_look(&mut camera_rot, input_manager, delta_time);
        self.fly_camera.clamp_pitch(&mut camera_rot);
        self.game_camera.set_rotation(&camera_rot);

        let move_dir = self
            .fly_camera
            .wasd_direction(&self.game_camera, input_manager);
        if move_dir.magnitude2() > 1e-6 {
            let speed = self.fly_camera.move_speed(input_manager);
            let new_pos =
                self.game_camera.get_position() + move_dir.normalize() * speed * delta_time;
            self.game_camera.set_position(&new_pos);
        }
        renderer.set_camera(&self.game_camera);

        if let Some(scene) = &mut self.scene {
            if let Some(bob_world) = scene.geom_world_pos("bob") {
                if let Some(kick) =
                    click_kick_sign(&self.game_camera, input_manager, game_config, bob_world)
                {
                    scene.apply_joint_qvel("hinge", kick);
                }
            }
            scene.tick_and_draw(renderer, game_config);
        }

        renderer.set_debug_game_msg(
            "MuJoCo test: [WASD] move, right-drag look, [click] hit the ball\nHinge pendulum + 2 free-falling boxes",
        );
    }
}

/// Screen-space pick test against the bob's current world position. Returns
/// the hinge angular-velocity kick to apply (sign depends on which side of
/// the bob was clicked) if this frame's left-click landed on it, else None.
fn click_kick_sign(
    game_camera: &Camera,
    input_manager: &InputManager,
    game_config: &Config,
    bob_world: CgVec3,
) -> Option<f64> {
    if !input_manager.get_key_state("mouse_left").just_pressed() {
        return None;
    }

    let (view, _dir, right) = game_camera.calculate_view_matrix();
    let proj = cgmath::perspective(
        cgmath::Deg(game_config.fov),
        game_config.window_width as f32 / game_config.window_height.max(1) as f32,
        0.1,
        10000.0,
    );
    let view_proj = proj * view;

    let to_screen = |world: CgVec3| -> Option<(f32, f32)> {
        let clip = view_proj * CgVec4::new(world.x, world.y, world.z, 1.0);
        if clip.w < 0.01 {
            return None; // Behind the camera.
        }
        Some((
            (clip.x / clip.w + 1.0) * 0.5 * game_config.window_width as f32,
            (1.0 - clip.y / clip.w) * 0.5 * game_config.window_height as f32,
        ))
    };

    let (center_x, center_y) = to_screen(bob_world)?;
    const MIN_PICK_PX: f32 = 16.0;
    let screen_radius = to_screen(bob_world + right * BOB_RADIUS)
        .map_or(MIN_PICK_PX, |(ex, ey)| {
            ((ex - center_x).powi(2) + (ey - center_y).powi(2)).sqrt()
        })
        .max(MIN_PICK_PX);

    let (mouse_x, mouse_y) = input_manager.get_mouse_position();
    let dx = mouse_x as f32 - center_x;
    let dy = mouse_y as f32 - center_y;
    if (dx * dx + dy * dy).sqrt() > screen_radius {
        return None; // Missed.
    }

    // Clicked side of the ball flies away from the click.
    Some(if dx >= 0.0 { -CLICK_KICK_RAD_PER_SEC } else { CLICK_KICK_RAD_PER_SEC })
}
