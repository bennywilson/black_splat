use cgmath::InnerSpace;

use kb_engine3::{
    kb_config::*, kb_engine::*, kb_game_object::*, kb_input::*, kb_renderer::*, kb_utils::*, log,
    render_groups::kb_gaussian_splat_group::KbSplatParams,
};

// Splat clouds the demo preloads and cycles between with [Space].  Missing files
// are skipped at load, so the demo still runs with whatever is present.
const SPLAT_PLY_PATHS: &[&str] = &[
    "game_assets/splats/church.ply",
    "game_assets/splats/cabin.ply",
    "game_assets/splats/opera.ply",
];

const CAMERA_MOVE_RATE: f32 = 1.0;
const CAMERA_ROTATION_RATE: f32 = 30.0;
const PARAM_RATE: f32 = 1.5;

pub struct SplatGame {
    game_objects: Vec<GameObject>,
    game_camera: KbCamera,
    splat_params: KbSplatParams,
    // Display names of the clouds that actually loaded, aligned with the
    // renderer's splat indices; `active_splat` is the one being shown.
    splat_names: Vec<String>,
    active_splat: usize,
}

impl KbGameEngine for SplatGame {
    fn new(game_config: &KbConfig) -> Self {
        log!("SplatGame::new()");
        let mut game_camera = KbCamera::new();
        game_camera.set_position(&game_config.start_position);
        game_camera.set_rotation(&game_config.start_rotation);
        Self {
            game_objects: Vec::new(),
            game_camera,
            splat_params: KbSplatParams {
                falloff: 4.65,
                scale: 3.0,
                contrast: 1.0,
                max_sh_degree: 2.0,
                overall_scale: 1.0,
            },
            splat_names: Vec::new(),
            active_splat: 0,
        }
    }

    async fn initialize_world(
        &mut self,
        renderer: &mut KbRenderer<'_>,
        game_config: &mut KbConfig,
    ) {
        log!("SplatGame::initialize_world()");
        game_config.clear_color = CgVec4::new(0.02, 0.02, 0.03, 1.0);

        // Preload every available cloud; record the names that loaded so the
        // cycle hotkey and HUD only reference real ones.
        for path in SPLAT_PLY_PATHS {
            if renderer.load_gaussian_splat(path, &self.splat_params).await {
                let name = path
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or(path)
                    .trim_end_matches(".ply")
                    .to_string();
                self.splat_names.push(name);
            }
        }
        renderer.set_active_gaussian_splat(0);
        renderer.set_tonemap_enabled(true);

        renderer.set_camera(&self.game_camera);
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
        if input_manager.get_key_state("w").is_down() { move_vec += forward_dir; }
        if input_manager.get_key_state("s").is_down() { move_vec += -forward_dir; }
        if input_manager.get_key_state("d").is_down() { move_vec += right_dir; }
        if input_manager.get_key_state("a").is_down() { move_vec += -right_dir; }
        if move_vec.magnitude2() > 0.001 {
            let speed = if input_manager.get_key_state("left_shift").is_down() {
                CAMERA_MOVE_RATE * 3.0
            } else {
                CAMERA_MOVE_RATE
            };
            let new_pos = self.game_camera.get_position()
                + move_vec.normalize() * delta_time * speed;
            self.game_camera.set_position(&new_pos);
        }

        // Look
        let mut camera_rot = self.game_camera.get_rotation();
        let rot_amount = delta_time * CAMERA_ROTATION_RATE;
        if input_manager.get_key_state("left_arrow").is_down() { camera_rot.x += rot_amount; }
        if input_manager.get_key_state("right_arrow").is_down() { camera_rot.x -= rot_amount; }
        if input_manager.get_key_state("up_arrow").is_down() { camera_rot.y -= rot_amount; }
        if input_manager.get_key_state("down_arrow").is_down() { camera_rot.y += rot_amount; }
        camera_rot.y = camera_rot.y.clamp(-89.0, 89.0);
        self.game_camera.set_rotation(&camera_rot);

        // Cycle to the next preloaded splat cloud.
        if input_manager.get_key_state("space").just_pressed() && !self.splat_names.is_empty() {
            self.active_splat = (self.active_splat + 1) % self.splat_names.len();
            renderer.set_active_gaussian_splat(self.active_splat);
        }

        // Splat param adjustments
        let adj = delta_time * PARAM_RATE;
        let p = &mut self.splat_params;
        let mut changed = false;

        if input_manager.get_key_state("1").is_down() { p.falloff = (p.falloff - adj).max(0.01); changed = true; }
        if input_manager.get_key_state("2").is_down() { p.falloff = (p.falloff + adj).min(20.0); changed = true; }
        if input_manager.get_key_state("3").is_down() { p.scale = (p.scale - adj).max(0.1); changed = true; }
        if input_manager.get_key_state("4").is_down() { p.scale = (p.scale + adj).min(20.0); changed = true; }
        if input_manager.get_key_state("5").is_down() { p.contrast = (p.contrast - adj).max(0.1); changed = true; }
        if input_manager.get_key_state("6").is_down() { p.contrast = (p.contrast + adj).min(5.0); changed = true; }
        if input_manager.get_key_state("7").is_down() { p.overall_scale = (p.overall_scale - adj).max(0.1); changed = true; }
        if input_manager.get_key_state("8").is_down() { p.overall_scale = (p.overall_scale + adj).min(10.0); changed = true; }
        if input_manager.get_key_state("9").just_pressed() {
            p.max_sh_degree = match p.max_sh_degree as u32 {
                0 => 1.0,
                1 => 2.0,
                2 => 3.0,
                _ => 0.0,
            };
            changed = true;
        }

        if changed {
            renderer.set_gaussian_splat_params(p);
        }

        let pos = self.game_camera.get_position();
        let rot = self.game_camera.get_rotation();
        let active_name = self
            .splat_names
            .get(self.active_splat)
            .map_or("none", |s| s.as_str());
        let splat_count = renderer.active_gaussian_splat_count();
        renderer.set_debug_game_msg(&format!(
            "Move: [W][A][S][D]   [Shift] sprint   Look: [Arrow Keys]\n\
             Camera  pos ({:.2}, {:.2}, {:.2})   rot ({:.1}, {:.1})\n\n\
             [Space]     Cycle Splat  {} ({}/{})   {} splats\n\n\
             Splat Params (hold to adjust):\n\
             [1] / [2]   Falloff      {:.2}\n\
             [3] / [4]   Scale        {:.2}\n\
             [5] / [6]   Contrast     {:.2}\n\
             [7] / [8]   Size         {:.2}\n\
             [9]         SH Degree    {}",
            pos.x, pos.y, pos.z, rot.x, rot.y,
            active_name, self.active_splat + 1, self.splat_names.len().max(1), splat_count,
            p.falloff, p.scale, p.contrast, p.overall_scale, p.max_sh_degree as u32,
        ));

        renderer.set_camera(&self.game_camera);
    }
}
