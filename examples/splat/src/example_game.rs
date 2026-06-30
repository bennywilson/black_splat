use cgmath::InnerSpace;

use kb_engine3::{
    kb_config::*, kb_engine::*, kb_game_object::*, kb_input::*, kb_renderer::*, kb_utils::*, log,
    render_groups::kb_gaussian_splat_group::KbSplatParams,
};

// Drop a 3D gaussian splat .ply here (relative to this example's directory when
// running natively; for the browser build it is fetched from /rust_assets/).
const SPLAT_PLY_PATH: &str = "game_assets/splats/point_cloud.ply";

const CAMERA_MOVE_RATE: f32 = 6.0;
const CAMERA_ROTATION_RATE: f32 = 90.0;

pub struct SplatGame {
    game_objects: Vec<GameObject>,
    game_camera: KbCamera,
}

impl KbGameEngine for SplatGame {
    fn new(_game_config: &KbConfig) -> Self {
        log!("SplatGame::new()");
        let mut game_camera = KbCamera::new();
        game_camera.set_position(&CgVec3::new(0.0, 0.0, -5.0));

        Self {
            game_objects: Vec::<GameObject>::new(),
            game_camera,
        }
    }

    async fn initialize_world(
        &mut self,
        renderer: &mut KbRenderer<'_>,
        game_config: &mut KbConfig,
    ) {
        log!("SplatGame::initialize_world()");
        game_config.clear_color = CgVec4::new(0.02, 0.02, 0.03, 1.0);

        // Hardcoded to match the blk gs_test level. max_sh_degree is the highest
        // the shader evaluates (degree 2), and the loader fills all 8 higher-order
        // coefficients, so unused bands are simply zero for lower-degree plys.
        let params = KbSplatParams {
            falloff: 1.0,
            scale: 3.0,
            contrast: 1.0,
            max_sh_degree: 2.0,
            overall_scale: 1.0,
        };
        renderer.load_gaussian_splat(SPLAT_PLY_PATH, &params).await;

        renderer.set_camera(&self.game_camera);
        renderer.set_hud_msg("Move: [W][A][S][D]   Look: [Arrow Keys]");
    }

    fn get_game_objects(&self) -> &Vec<GameObject> {
        &self.game_objects
    }

    fn tick_frame_internal(
        &mut self,
        renderer: &mut KbRenderer,
        input_manager: &KbInputManager,
        game_config: &KbConfig,
    ) {
        let delta_time = game_config.delta_time;
        let (_view, view_dir, right_dir) = self.game_camera.calculate_view_matrix();
        let forward_dir = CgVec3::new(view_dir.x, view_dir.y, view_dir.z).normalize();

        // Movement
        let mut move_vec = CG_VEC3_ZERO;
        if input_manager.get_key_state("w").is_down() {
            move_vec += forward_dir;
        }
        if input_manager.get_key_state("s").is_down() {
            move_vec += -forward_dir;
        }
        if input_manager.get_key_state("d").is_down() {
            move_vec += right_dir;
        }
        if input_manager.get_key_state("a").is_down() {
            move_vec += -right_dir;
        }
        if move_vec.magnitude2() > 0.001 {
            let speed = if input_manager.get_key_state("left_shift").is_down() {
                CAMERA_MOVE_RATE * 3.0
            } else {
                CAMERA_MOVE_RATE
            };
            let new_pos = self.game_camera.get_position() + move_vec.normalize() * delta_time * speed;
            self.game_camera.set_position(&new_pos);
        }

        // Look
        let mut camera_rot = self.game_camera.get_rotation();
        let rot_amount = delta_time * CAMERA_ROTATION_RATE;
        if input_manager.get_key_state("left_arrow").is_down() {
            camera_rot.x += rot_amount;
        }
        if input_manager.get_key_state("right_arrow").is_down() {
            camera_rot.x -= rot_amount;
        }
        if input_manager.get_key_state("up_arrow").is_down() {
            camera_rot.y -= rot_amount;
        }
        if input_manager.get_key_state("down_arrow").is_down() {
            camera_rot.y += rot_amount;
        }
        camera_rot.y = camera_rot.y.clamp(-89.0, 89.0);
        self.game_camera.set_rotation(&camera_rot);

        renderer.set_camera(&self.game_camera);
    }
}
