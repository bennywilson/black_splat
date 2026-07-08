use cgmath::InnerSpace;
use instant::Instant;

use black_splat::{
    assets::*, collision::*, config::*, engine::*, game_object::*, input::*,
    renderer::*, resource::*, utils::*, log,
};

use crate::{game_actors::*, game_vfx::*};

pub const CAMERA_MOVE_RATE: f32 = 10.0;
pub const CAMERA_ROTATION_RATE: f32 = 150.0;
pub const CROSSHAIR_ERROR_RATE: f32 = 10.0;

pub struct Example3DGame {
    player: Option<GamePlayer>,
    mobs: Vec<GameMob>,
    world_actors: Vec<Actor>,
    props: Vec<GameProp>,

    game_objects: Vec<GameObject>,
    game_camera: Camera,

    collision_manager: CollisionManager,
    vfx_manager: GameVfxManager,

    sign_prop: Option<GameProp>,
    barrel_model: ModelHandle,
    shotgun_model: ModelHandle,
    monster_model: ModelHandle,

    monster_pass: usize,
    outline_pass: usize,
    decal_pass: usize,

    monster_spawn_timer: Instant,
    barrel_spawn_timer: Instant,
    shotgun_spawn_timer: Instant,

    score: i32,
    high_score: i32,
    next_harm_time: f32,

    crosshair_error: f32,
    invert_y: bool,
    debug_collision: bool,
    pause_monsters: bool,

    post_process_override: PostProcessMode,
}

impl Example3DGame {
    fn spawn_monster(&mut self, renderer: &mut Renderer<'_>) {
        if self.pause_monsters {
            return;
        }

        if self.mobs.len() > 15 {
            return;
        }

        let pos = [
            CgVec3::new(15.0, 2.0, 15.0),
            CgVec3::new(-15.0, 2.0, 15.0),
            CgVec3::new(-15.0, 2.0, -15.0),
            CgVec3::new(15.0, 2.0, -15.0),
            CgVec3::new(15.0, 2.0, 0.0),
            CgVec3::new(-15.0, 2.0, 0.0),
            CgVec3::new(0.0, 2.0, -15.0),
            CgVec3::new(0.0, 2.0, -15.0),
        ];

        let monster_pos = pos[random_u32(0, 7) as usize];
        let mut monster = GameMob::new(
            &monster_pos,
            &self.monster_model,
            self.monster_pass,
            self.outline_pass,
            &mut self.collision_manager,
        );

        let monster_actors = monster.get_actors();
        renderer.add_or_update_actor(&monster_actors[0]);
        renderer.add_or_update_actor(&monster_actors[1]);

        self.mobs.push(monster);
    }

    fn spawn_barrel(&mut self, renderer: &mut Renderer<'_>) {
        let barrel_pos =
            random_vec3(CgVec3::new(-15.0, 0.0, -15.0), CgVec3::new(15.0, 0.0, 15.0));
        let smoke_pos = barrel_pos + CgVec3::new(0.0, 3.5, 0.0);

        // Smoke
        let (smoke_handle_1, smoke_handle_2) =
            self.vfx_manager.spawn_barrel_smoke(&smoke_pos, renderer);

        let mut barrel = GameProp::new(
            &GamePropType::Barrel,
            &barrel_pos,
            &self.barrel_model,
            self.outline_pass,
            &mut self.collision_manager,
            [smoke_handle_1, smoke_handle_2],
        );

        let barrel_actors = barrel.get_actors();
        for actor in barrel_actors {
            renderer.add_or_update_actor(actor);
        }
        self.props.push(barrel);
    }

    fn spawn_shotgun(&mut self, renderer: &mut Renderer<'_>) {
        let shotgun_pos =
            random_vec3(CgVec3::new(-15.0, 0.0, -15.0), CgVec3::new(15.0, 0.0, 15.0));

        let mut shotgun = GameProp::new(
            &GamePropType::Shotgun,
            &shotgun_pos,
            &self.shotgun_model,
            self.outline_pass,
            &mut self.collision_manager,
            [INVALID_PARTICLE_HANDLE, INVALID_PARTICLE_HANDLE],
        );
        let shotgun_actors = shotgun.get_actors();
        shotgun_actors[1].set_pass(
            &RenderGroupType::WorldCustom,
            &Some(self.outline_pass),
        );

        for actor in shotgun_actors {
            renderer.add_or_update_actor(actor);
        }
        self.props.push(shotgun);
    }

    fn spawn_sign(&mut self, renderer: &mut Renderer<'_>, model_handle: &ModelHandle) {
        {
            let sign_pos = CgVec3::new(0.0, 0.0, 0.0);

            let mut sign = GameProp::new(
                &GamePropType::Sign,
                &sign_pos,
                model_handle,
                self.outline_pass,
                &mut self.collision_manager,
                [INVALID_PARTICLE_HANDLE, INVALID_PARTICLE_HANDLE],
            );
            let sign_actors = sign.get_actors();
            sign_actors[0].set_pass(
                &RenderGroupType::WorldHole,
                &Some(self.outline_pass),
            );
            sign_actors[1].set_pass(
                &RenderGroupType::WorldCustom,
                &Some(self.outline_pass),
            );

            // hack
            renderer.add_bullet_hole(
                &sign_actors[0],
                &CgVec3::new(99999999.0, 9999999.0, 9999999.0),
                &CgVec3::new(0.0, 1.0, 0.0),
            );

            let rotation = cgmath::Quaternion::from(CgMat3::from_angle_y(cgmath::Rad::from(
                cgmath::Deg(90.0),
            )));
            for actor in sign_actors {
                actor.set_rotation(&rotation);
                renderer.add_or_update_actor(actor);
            }

            self.sign_prop = Some(sign);
        }
    }
}

impl GameEngine for Example3DGame {
    fn new(_game_config: &Config) -> Self {
        log!("GameEngine::new() caled...");
        let game_objects = Vec::<GameObject>::new();

        let mut game_camera = Camera::new();
        game_camera.set_position(&CgVec3::new(0.0, 3.5, -5.0));

        Self {
            world_actors: Vec::<Actor>::new(),
            mobs: Vec::<GameMob>::new(),
            props: Vec::<GameProp>::new(),
            game_objects,
            game_camera,
            vfx_manager: GameVfxManager::new(),
            sign_prop: None,
            barrel_model: ModelHandle::make_invalid(),
            shotgun_model: ModelHandle::make_invalid(),
            monster_model: ModelHandle::make_invalid(),
            monster_pass: usize::MAX,
            monster_spawn_timer: Instant::now(),
            shotgun_spawn_timer: Instant::now(),
            barrel_spawn_timer: Instant::now(),
            outline_pass: usize::MAX,
            decal_pass: usize::MAX,
            player: None,
            crosshair_error: 0.0,
            collision_manager: CollisionManager::new(),
            debug_collision: false,
            invert_y: false,
            pause_monsters: false,
            score: 0,
            high_score: 0,
            next_harm_time: -1.0,
            post_process_override: PostProcessMode::Passthrough,
        }
    }

    async fn initialize_world(
        &mut self,
        renderer: &mut Renderer<'_>,
        game_config: &mut Config,
    ) {
        log!("GameEngine::initialize_world()...");
        game_config.clear_color = CgVec4::new(0.87, 0.58, 0.24, 1.0);
        game_config.sun_color = CgVec4::new(0.8 * 0.8, 0.58 * 0.58, 0.24 * 0.24, 0.0);

        // self.game_objects order is hard-coded.  Indices 0-3 contain the cross hair sprites
        for _ in 0..4 {
            self.game_objects.push(GameObject {
                position: CG_VEC3_ZERO,
                scale: CG_VEC3_ONE,
                direction: (1.0, 0.0, 0.0).into(),
                velocity: (0.0, 0.0, 0.0).into(),
                object_type: GameObjectType::Background,
                object_state: GameObjectState::Idle,
                next_attack_time: 0.0,
                texture_index: 1,
                sprite_index: 40,
                anim_frame: 0,
                life_start_time: Instant::now(),
                state_start_time: Instant::now(),
                gravity_scale: 0.0,
                random_val: random_f32(0.0, 1000.0),
                is_enemy: false,
                uv_tiles: (1.0, 1.0),
            });
        }

        // Index 4 and 5 are the virtual thumb sticks
        self.game_objects.push(GameObject {
            position: CgVec3::new(1.45, -0.7, 0.0),
            scale: CgVec3::new(0.2, 0.2, 1.0),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 43,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            is_enemy: false,
            uv_tiles: (2.0, 2.0),
        });

        self.game_objects.push(GameObject {
            position: CgVec3::new(-1.45, -0.7, 0.0),
            scale: CgVec3::new(0.2, 0.2, 1.0),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 45,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            is_enemy: false,
            uv_tiles: (2.0, 2.0),
        });

        self.game_objects.push(GameObject {
            position: CgVec3::new(-1.45, -0.7, 0.0),
            scale: CgVec3::new(0.1, 0.1, 1.0),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 58,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            is_enemy: false,
            uv_tiles: (1.0, 1.0),
        });

        self.game_objects.push(GameObject {
            position: CgVec3::new(1.45, -0.7, 0.0),
            scale: CgVec3::new(0.1, 0.1, 1.0),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 59,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            is_enemy: false,
            uv_tiles: (1.0, 1.0),
        });

        self.shotgun_model = renderer
            .load_model("game_assets/models/shotgun.glb", false)
            .await;
        self.barrel_model = renderer
            .load_model("game_assets/models/barrel.glb", false)
            .await;

        let sign_model = renderer
            .load_model("game_assets/models/sign.glb", true)
            .await;

        if game_config.bullet_holes {
            game_config.sun_beam_pos_scale = [0.0, 300.0, 500.0, 1550.0].into();
            self.spawn_sign(renderer, &sign_model);
        } else {
            game_config.sun_beam_pos_scale = [500.0, 550.0, 500.0, 1550.0].into();
        }

        self.decal_pass = renderer
            .add_custom_pass(
                &RenderGroupType::WorldCustom,
                &BlendMode::Additive,
                "engine_assets/shaders/decal.wgsl",
            )
            .await;

        // First person set up
        let fp_pass = Some(
            renderer
                .add_custom_pass(
                    &RenderGroupType::ForegroundCustom,
                    &BlendMode::None,
                    "game_assets/shaders/first_person.wgsl",
                )
                .await,
        );
        let fp_outline_pass = Some(
            renderer
                .add_custom_pass(
                    &RenderGroupType::ForegroundCustom,
                    &BlendMode::Alpha,
                    "game_assets/shaders/first_person_outline.wgsl",
                )
                .await,
        );
        let hands_model = renderer
            .load_model("game_assets/models/fp_hands.glb", false)
            .await;
        let mut player = GamePlayer::new(&hands_model).await;

        let (hands, hands_outlines) = player.get_actors();
        hands.set_pass(&RenderGroupType::ForegroundCustom, &fp_pass);
        renderer.add_or_update_actor(hands);

        for outline in hands_outlines {
            outline.set_pass(
                &RenderGroupType::ForegroundCustom,
                &fp_outline_pass,
            );
            renderer.add_or_update_actor(outline);
        }
        self.player = Some(player);

        // Monster
        let monster_model = renderer
            .load_model("game_assets/models/monster.glb", false)
            .await;
        let monster_pass = renderer
            .add_custom_pass(
                &RenderGroupType::WorldCustom,
                &BlendMode::Additive,
                "game_assets/shaders/monster.wgsl",
            )
            .await;
        self.monster_pass = monster_pass;
        self.monster_model = monster_model;

        // World objects
        let level_model = renderer
            .load_model("game_assets/models/level.glb", false)
            .await;
        let mut actor = Actor::new();
        actor.set_position(&[0.0, 0.0, 0.0].into());
        actor.set_scale(&(CgVec3::new(10.0, 19.0, 10.0) * GLOBAL_SCALE.x));
        actor.set_model(&level_model);
        renderer.add_or_update_actor(&actor);
        self.world_actors.push(actor);

        let sky_model = renderer
            .load_model("game_assets/models/sky_dome.glb", false)
            .await;
        {
            let sky_pass = Some(
                renderer
                    .add_custom_pass(
                        &RenderGroupType::WorldCustom,
                        &BlendMode::Alpha,
                        "engine_assets/shaders/sky_dome_occlude.wgsl",
                    )
                    .await,
            );
            let mut actor = Actor::new();
            actor.set_position(&[0.0, 0.0, 0.0].into());
            actor.set_scale(&[30.0, 30.0, 30.0].into());
            actor.set_model(&sky_model);
            actor.set_pass(&RenderGroupType::WorldCustom, &sky_pass);
            renderer.add_or_update_actor(&actor);
            self.world_actors.push(actor);
        }
        {
            let sky_pass = Some(
                renderer
                    .add_custom_pass(
                        &RenderGroupType::WorldCustom,
                        &BlendMode::Alpha,
                        "engine_assets/shaders/sky_dome_draw.wgsl",
                    )
                    .await,
            );
            let mut actor = Actor::new();
            actor.set_position(&[0.0, 0.0, 0.0].into());
            actor.set_scale(&[30.0, 30.0, 30.0].into());
            actor.set_model(&sky_model);
            actor.set_pass(&RenderGroupType::WorldCustom, &sky_pass);
            renderer.add_or_update_actor(&actor);
            self.world_actors.push(actor);
        }

        self.outline_pass = renderer
            .add_custom_pass(
                &RenderGroupType::WorldCustom,
                &BlendMode::Alpha,
                "game_assets/shaders/first_person_outline.wgsl",
            )
            .await;
        let pinky_model = renderer
            .load_model("game_assets/models/pinky.glb", false)
            .await;
        let mut actor = Actor::new();
        actor.set_position(&[16.5, 0.5, 6.0].into());
        let pinky_rot_x = cgmath::Rad::from(cgmath::Deg(90.0));
        let pinky_rot_z = cgmath::Rad::from(cgmath::Deg(115.0));
        let pinky_rot = cgmath::Quaternion::from(
            CgMat3::from_angle_z(pinky_rot_z) * CgMat3::from_angle_x(pinky_rot_x),
        );
        actor.set_rotation(&pinky_rot);
        actor.set_scale(&GLOBAL_SCALE);
        actor.set_model(&pinky_model);
        renderer.add_or_update_actor(&actor);
        self.world_actors.push(actor);

        let mut actor = Actor::new();
        actor.set_position(&[16.5, 0.5, 6.0].into());
        actor.set_rotation(&pinky_rot);
        actor.set_scale(&GLOBAL_SCALE);
        actor.set_model(&pinky_model);

        //  #[cfg(target_arch = "wasm32")]
        actor.set_custom_data_1(CgVec4::new(0.25, 0.08, 0.08, 0.08));

        actor.set_pass(
            &RenderGroupType::WorldCustom,
            &Some(self.outline_pass),
        );
        renderer.add_or_update_actor(&actor);
        self.world_actors.push(actor);

        // World Collision
        let collision_box = CollisionShape::AABB(CollisionAABB {
            position: CgVec3::new(0.0, 2.4, 20.0),
            extents: CgVec3::new(20.0, 10.0, 2.0),
            block: true,
        });
        let _ = self.collision_manager.add_collision(&collision_box);

        let collision_box = CollisionShape::AABB(CollisionAABB {
            position: CgVec3::new(0.0, 2.4, -20.0),
            extents: CgVec3::new(-20.0, 10.0, 2.0),
            block: true,
        });
        let _ = self.collision_manager.add_collision(&collision_box);

        let collision_box = CollisionShape::AABB(CollisionAABB {
            position: CgVec3::new(20.0, 2.4, 0.0),
            extents: CgVec3::new(2.0, 10.0, 20.0),
            block: true,
        });
        let _ = self.collision_manager.add_collision(&collision_box);

        let collision_box = CollisionShape::AABB(CollisionAABB {
            position: CgVec3::new(-20.0, 2.4, 0.0),
            extents: CgVec3::new(2.0, 10.0, 20.0),
            block: true,
        });
        let _ = self.collision_manager.add_collision(&collision_box);

        let collision_box = CollisionShape::AABB(CollisionAABB {
            position: CgVec3::new(0.0, -0.5, 0.0),
            extents: CgVec3::new(20.0, 0.0, 20.0),
            block: true,
        });
        let _ = self.collision_manager.add_collision(&collision_box);

        // Trans Flag
        let sun_color = game_config.sun_color;
        let trans_colors = [
            CgVec4::new(
                0.356 * sun_color.x,
                0.807 * sun_color.y,
                0.980 * sun_color.z,
                1.0,
            ),
            CgVec4::new(
                0.96 * sun_color.x,
                0.66 * sun_color.y,
                0.72 * sun_color.z,
                1.0,
            ),
            CgVec4::new(1.0 * sun_color.x, 1.0 * sun_color.y, 1.0 * sun_color.z, 1.0),
        ];

        renderer.add_line(
            &CgVec3::new(5.0, 6.5, 17.4),
            &CgVec3::new(10.0, 6.5, 17.4),
            &trans_colors[0],
            0.25,
            5535.0,
            game_config,
        );
        renderer.add_line(
            &CgVec3::new(5.0, 6.0, 17.4),
            &CgVec3::new(10.0, 6.0, 17.4),
            &trans_colors[1],
            0.25,
            5535.0,
            game_config,
        );
        renderer.add_line(
            &CgVec3::new(5.0, 5.5, 17.4),
            &CgVec3::new(10.0, 5.5, 17.4),
            &trans_colors[2],
            0.25,
            5535.0,
            game_config,
        );
        renderer.add_line(
            &CgVec3::new(5.0, 5.0, 17.4),
            &CgVec3::new(10.0, 5.0, 17.4),
            &trans_colors[1],
            0.25,
            5535.0,
            game_config,
        );
        renderer.add_line(
            &CgVec3::new(5.0, 4.5, 17.4),
            &CgVec3::new(10.0, 4.5, 17.4),
            &trans_colors[0],
            0.25,
            5535.0,
            game_config,
        );

        self.vfx_manager.init(renderer).await;

        self.spawn_shotgun(renderer);
        self.spawn_barrel(renderer);
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
        for game_object in &mut self.game_objects {
            game_object.update(game_config.delta_time);
        }
        let delta_time = game_config.delta_time;
        let (_s, view_dir, right_dir) = self.game_camera.calculate_view_matrix();
        let forward_dir = CgVec3::new(view_dir.x, 0.0, view_dir.z).normalize();
        let camera_pos = self.game_camera.get_position();
        let mut camera_rot = self.game_camera.get_rotation();

        let mut move_vec = CG_VEC3_ZERO;
        let touch_map_iter = input_manager.get_touch_map().iter();
        let mut local_move_vec_move = (0.0, 0.0);
        let mut local_move_vec_look = (0.0, 0.0);
        for touch_pair in touch_map_iter {
            let touch = &touch_pair.1;

            // Left thumb
            if touch.touch_state.is_down() && touch.start_pos.0 < 500.0 {
                local_move_vec_move.0 = (touch.current_pos.0 as f32 - 118.0).clamp(-90.0, 90.0);
                if local_move_vec_move.0 < 0.0 {
                    local_move_vec_move.0 = (local_move_vec_move.0 + 20.0).clamp(-90.0, 0.0);
                } else {
                    local_move_vec_move.0 = (local_move_vec_move.0 - 20.0).clamp(0.0, 90.0);
                }
                local_move_vec_move.1 = (touch.current_pos.1 as f32 - 660.0).clamp(-90.0, 90.0);
                move_vec += right_dir * local_move_vec_move.0;
                move_vec -= forward_dir * local_move_vec_move.1;
            }

            // Right Thumb
            if touch.start_pos.1 > 570.0 && touch.touch_state.is_down() && touch.start_pos.0 > 500.0
            {
                local_move_vec_look.0 = (touch.current_pos.0 as f32 - 1233.0).clamp(-90.0, 90.0);
                local_move_vec_look.1 = (touch.current_pos.1 as f32 - 660.0).clamp(-90.0, 90.0);
                if local_move_vec_look.0 < 0.0 {
                    local_move_vec_look.0 = (local_move_vec_look.0 + 20.0).clamp(-90.0, 0.0);
                } else {
                    local_move_vec_look.0 = (local_move_vec_look.0 - 20.0).clamp(0.0, 90.0);
                }

                camera_rot.x -= 2.0 * delta_time * local_move_vec_look.0;
                camera_rot.y += 1.0 * delta_time * local_move_vec_look.1;
            }

            // Help
            if touch.start_pos.0 < 300.0
                && touch.start_pos.1 < 300.0
                && touch.touch_state.just_pressed()
            {
                renderer.enable_help_text();
            }

            // Help
            if touch.start_pos.0 < 300.0
                && touch.start_pos.1 < 300.0
                && touch.touch_state.just_pressed()
            {
                renderer.enable_help_text();
            }
        }

        if input_manager.get_key_state("w").is_down() {
            move_vec += forward_dir
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

        move_vec = move_vec.normalize();
        if input_manager.get_key_state("left_shift").is_down() {
            move_vec *= 0.45;
        }
        move_vec *= delta_time * CAMERA_MOVE_RATE;

        if move_vec.magnitude2() > 0.001 {
            let trace_start = CgVec3::new(camera_pos.x, 0.25, camera_pos.z);
            let (t, handle, _, _) = self.collision_manager.cast_ray(&trace_start, &move_vec);
            if (0.0..1.0).contains(&t) {
                self.props.retain_mut(|prop| {
                    if prop.get_prop_type() == GamePropType::Shotgun
                        && prop.get_collision_handle() == *handle.as_ref().unwrap()
                    {
                        prop.take_damage(&mut self.collision_manager, renderer);
                        self.player
                            .as_mut()
                            .unwrap()
                            .give_shotgun(&self.shotgun_model);
                        return false;
                    }
                    true
                });
            }
            let mut final_pos = camera_pos + move_vec;
            final_pos.x = final_pos.x.clamp(-17.0, 17.0);
            final_pos.z = final_pos.z.clamp(-17.0, 17.0);

            self.game_camera.set_position(&final_pos);

            self.crosshair_error =
                (self.crosshair_error + delta_time * CROSSHAIR_ERROR_RATE).clamp(0.0, 1.0);
        } else {
            self.crosshair_error =
                (self.crosshair_error - delta_time * CROSSHAIR_ERROR_RATE).clamp(0.0, 1.0);
        }

        let x_radians = delta_time * CAMERA_ROTATION_RATE;
        let y_radians = if self.invert_y {
            -delta_time * CAMERA_ROTATION_RATE
        } else {
            delta_time * CAMERA_ROTATION_RATE
        };

        if input_manager.get_key_state("left_arrow").is_down() {
            camera_rot.x += x_radians;
        }
        if input_manager.get_key_state("right_arrow").is_down() {
            camera_rot.x -= x_radians;
        }
        if input_manager.get_key_state("up_arrow").is_down() {
            camera_rot.y -= y_radians;
        }
        if input_manager.get_key_state("down_arrow").is_down() {
            camera_rot.y += y_radians
        }
        camera_rot.y = camera_rot.y.clamp(-60.0, 75.0);

        self.game_camera.set_rotation(&camera_rot);
        renderer.set_camera(&self.game_camera);

        let player = &mut self.player.as_mut().unwrap();
        let has_shotgun = player.has_shotgun();
        let (cur_state, next_state) = player.tick(input_manager, &self.game_camera, game_config);
        let (hands, hands_outline) = player.get_actors();
        renderer.add_or_update_actor(hands);
        for outline in hands_outline {
            renderer.add_or_update_actor(outline);
        }

        let (_, view_dir, right_dir) = self.game_camera.calculate_view_matrix();
        let muzzle_flash_pos = {
            if has_shotgun {
                hands.get_position()
                    + view_dir * 3.0
                    + right_dir * 0.3
                    + CgVec3::new(0.0, 0.75, 0.0)
            } else {
                hands.get_position() + view_dir * 1.5 + right_dir * 0.3 + CgVec3::new(0.0, 0.5, 0.0)
            }
        };
        self.vfx_manager
            .tick(&muzzle_flash_pos, renderer, game_config);

        if cur_state != GamePlayerState::Shooting && next_state == GamePlayerState::Shooting {
            let (_, view_dir, _) = self.game_camera.calculate_view_matrix();
            let trace_start_pos = self.game_camera.get_position();
            let num_shots = if self.player.as_ref().unwrap().has_shotgun() {
                8
            } else {
                1
            };

            // Muzzle Flash
            let scale = if has_shotgun {
                CgVec3::new(2.0, 2.0, 2.0)
            } else {
                CgVec3::new(1.0, 1.0, 1.0)
            };
            self.vfx_manager
                .spawn_muzzle_flash(&muzzle_flash_pos, &scale, renderer);

            for i in 0..num_shots {
                let trace_end_pos = {
                    let end = trace_start_pos + view_dir * 10000.0;
                    if i > 0 {
                        end + random_vec3(
                            CgVec3::new(-2100.0, -2100.0, -2100.0),
                            CgVec3::new(2100.0, 2100.0, 2100.0),
                        )
                    } else {
                        end
                    }
                };

                let (hit_t, handle, hit_loc, _) = self
                    .collision_manager
                    .cast_ray(&trace_start_pos, &trace_end_pos);
                let found_hit = (0.0..1.0).contains(&hit_t);
                let mut mob_killed = false;

                let color = if found_hit {
                    CgVec4::new(1.0, 0.0, 0.0, 1.0)
                } else {
                    CgVec4::new(0.0, 0.0, 1.0, 1.0)
                };

                if found_hit && self.sign_prop.is_some() {
                    let sign_prop = self.sign_prop.as_mut().unwrap();
                    if sign_prop.collision_handle == handle.unwrap() {
                        let trace_dir = (trace_end_pos - trace_start_pos).normalize();
                        renderer.add_bullet_hole(
                            &sign_prop.get_actors()[0],
                            &hit_loc.unwrap(),
                            &trace_dir,
                        );
                    }
                }
                if found_hit {
                    let hit_loc = hit_loc.unwrap();
                    self.mobs.retain_mut(|mob| {
                        if mob.get_collision_handle() == *handle.as_ref().unwrap() {
                            mob_killed = mob.take_damage(&mut self.collision_manager, renderer);
                            self.score += 1;

                            let mob_pos = mob.get_actors()[0].get_position();
                            self.vfx_manager.spawn_mob_death_fx(
                                &mob_pos,
                                &view_dir,
                                renderer,
                                &mut self.collision_manager,
                                game_config,
                            );

                            !mob_killed
                        } else {
                            true
                        }
                    });

                    let mut barrel_exploded = false;
                    let mut explode_pos = CG_VEC3_ZERO;
                    if !mob_killed {
                        self.props.retain_mut(|prop| {
                            if prop.get_prop_type() == GamePropType::Barrel
                                && prop.get_collision_handle() == *handle.as_ref().unwrap()
                            {
                                explode_pos = prop.get_actors()[0].get_position();

                                prop.take_damage(&mut self.collision_manager, renderer);

                                // Barrel explosion
                                self.vfx_manager.spawn_explosion(&explode_pos, renderer);
                                barrel_exploded = true;
                                return false;
                            }
                            true
                        });
                    };

                    // Radius Damage
                    if barrel_exploded {
                        self.mobs.retain_mut(|mob| {
                            let mob_pos = mob.get_actors()[0].get_position();
                            let magnitude = cgvec3_remove_y(mob_pos - explode_pos).magnitude();
                            if magnitude < 15.0 {
                                mob.take_damage(&mut self.collision_manager, renderer);
                                self.score += 1;

                                self.vfx_manager.spawn_mob_death_fx(
                                    &mob_pos,
                                    &view_dir,
                                    renderer,
                                    &mut self.collision_manager,
                                    game_config,
                                );
                                return false;
                            }

                            true
                        });
                    }

                    if !mob_killed {
                        // Hit a wall, spawn impact
                        self.vfx_manager.spawn_impact(&hit_loc, renderer);
                    }
                }

                if self.debug_collision {
                    renderer.add_line(
                        &trace_start_pos,
                        &trace_end_pos,
                        &color,
                        0.05,
                        0.33,
                        game_config,
                    );
                }
            }
        }

        // Tick monster
        if !self.pause_monsters {
            let speed_multiplier = {
                let t = (self.score as f32 / 20.0).clamp(0.0, 1.0);
                (t * 10.0) + 5.0
            };
            for monster in &mut self.mobs {
                monster.tick(
                    camera_pos,
                    speed_multiplier,
                    &mut self.collision_manager,
                    game_config,
                );
                renderer.add_or_update_actor(&monster.get_actors()[0]);
                renderer.add_or_update_actor(&monster.get_actors()[1]);
            }
        }

        let spawn_timer = {
            let t = 1.0 - (self.score as f32 / 20.0).clamp(0.0, 1.0);
            if self.score == 0 {
                t + game_config.enemy_spawn_delay + 5.0
            } else {
                t + game_config.enemy_spawn_delay + 1.0
            }
        };
        if self.monster_spawn_timer.elapsed().as_secs_f32() > spawn_timer {
            self.monster_spawn_timer = Instant::now();
            self.spawn_monster(renderer);
        }

        let shotgun_count = self
            .props
            .iter()
            .filter(|&p| p.get_prop_type() == GamePropType::Shotgun)
            .count();
        if shotgun_count > 0 {
            self.shotgun_spawn_timer = Instant::now();
        } else if self.shotgun_spawn_timer.elapsed().as_secs_f32() > 20.0 {
            self.spawn_shotgun(renderer);
        }

        let barrel_count = self
            .props
            .iter()
            .filter(|&p| p.get_prop_type() == GamePropType::Barrel)
            .count();
        if barrel_count > 0 {
            self.barrel_spawn_timer = Instant::now();
        } else if self.barrel_spawn_timer.elapsed().as_secs_f32() > 20.0 {
            self.spawn_barrel(renderer);
        }

        let mut num_attacking = 0;
        for monster in &self.mobs {
            if monster.get_state() == GameMobState::Attacking {
                num_attacking += 1;
            }
        }

        if num_attacking > 0 {
            let elapsed_time = game_config.start_time.elapsed().as_secs_f32();
            if self.next_harm_time < 0.0 {
                self.next_harm_time = elapsed_time + 1.0;
            } else if elapsed_time > self.next_harm_time {
                renderer.set_postprocess_mode(&PostProcessMode::ScanLines);
                self.next_harm_time = elapsed_time + 1.0;
                self.score = (self.score - 1).max(0);
            }
        } else {
            renderer.set_postprocess_mode(&PostProcessMode::Passthrough);
        }

        if !matches!(self.post_process_override, PostProcessMode::Passthrough) {
            renderer.set_postprocess_mode(&self.post_process_override);
        }

        if !matches!(self.post_process_override, PostProcessMode::Passthrough) {
            renderer.set_postprocess_mode(&self.post_process_override);
        }

        // UI
        {
            self.high_score = self.high_score.max(self.score);
            renderer.set_hud_msg("-/+ to change post-processes");
            let player = self.player.as_ref().unwrap();
            let (positions, sprites, scale) = {
                if !player.has_shotgun() {
                    (
                        [
                            CgVec3::new(0.0, 0.1, 0.0),
                            CgVec3::new(0.0, -0.1, 0.0),
                            CgVec3::new(0.1, 0.0, 0.0),
                            CgVec3::new(-0.1, 0.0, 0.0),
                        ],
                        [40, 40, 41, 41],
                        CgVec3::new(0.035, 0.035, 1.0),
                    )
                } else {
                    (
                        [
                            CgVec3::new(-0.11, 0.11, 0.0),
                            CgVec3::new(0.11, 0.11, 0.0),
                            CgVec3::new(-0.11, -0.11, 0.0),
                            CgVec3::new(0.11, -0.11, 0.0),
                        ],
                        [48, 49, 56, 57],
                        CgVec3::new(0.065, 0.065, 0.065),
                    )
                }
            };

            let center = (positions[0] + positions[1] + positions[2] + positions[3]) * 0.25;
            for i in 0..4 {
                self.game_objects[i].sprite_index = sprites[i];
                self.game_objects[i].position =
                    positions[i] + (positions[i] - center).normalize() * self.crosshair_error * 0.1;
                self.game_objects[i].scale = scale;
            }
            //self.game_objects.truncate(8);

            let ammo_count = player.get_ammo_count();
            let mut position = CgVec3::new(1.65, 0.0, 0.0);
            let scale = CgVec3::new(0.1, 0.1, 0.1);
            let sprite_index = if player.has_shotgun() { 50 } else { 42 };
            let bullet_spacing = if player.has_shotgun() { 0.1 } else { 0.08 };
            for _ in 0..ammo_count {
                self.game_objects.push(GameObject {
                    position,
                    scale,
                    direction: (1.0, 0.0, 0.0).into(),
                    velocity: (0.0, 0.0, 0.0).into(),
                    object_type: GameObjectType::Background,
                    object_state: GameObjectState::Idle,
                    next_attack_time: 0.0,
                    texture_index: 0,
                    sprite_index,
                    anim_frame: 0,
                    life_start_time: Instant::now(),
                    state_start_time: Instant::now(),
                    gravity_scale: 0.0,
                    random_val: random_f32(0.0, 1000.0),
                    is_enemy: false,
                    uv_tiles: (1.0, 1.0),
                });
                position.x -= bullet_spacing;
            }
        }

        // Virtual sticks
        self.game_objects[6].position.x = (-1.45 + local_move_vec_move.0 * 0.001).clamp(-1.6, -1.4);
        self.game_objects[6].position.y =
            (-0.7 - local_move_vec_move.1 * 0.001).clamp(-0.85, -0.65);

        self.game_objects[7].position.x = (1.45 + local_move_vec_look.0 * 0.001).clamp(1.4, 1.6);
        self.game_objects[7].position.y =
            (-0.7 - local_move_vec_look.1 * 0.001).clamp(-0.85, -0.65);

        // Debug
        if input_manager.get_key_state("i").just_pressed() {
            self.debug_collision = !self.debug_collision;
        }

        if input_manager.get_key_state("y").just_pressed() {
            self.invert_y = !self.invert_y;
        }

        if input_manager.get_key_state("m").just_pressed() {
            self.pause_monsters = !self.pause_monsters;
        }

        if input_manager.get_key_state("+").just_pressed() {
            self.post_process_override = match self.post_process_override {
                PostProcessMode::Passthrough => PostProcessMode::Desaturation,
                PostProcessMode::Desaturation => PostProcessMode::ScanLines,
                PostProcessMode::ScanLines => PostProcessMode::Warp,
                PostProcessMode::Warp => PostProcessMode::Passthrough,
            }
        }

        if self.debug_collision {
            self.collision_manager.debug_draw(renderer, game_config);
        }

        let num_collision_obj = self.collision_manager.num_collision_objects();
        let num_active_particles = renderer.num_active_particles();
        let num_active_decals = self.vfx_manager.num_active_decals();

        let debug_msg = format!("Move: [W][A][S][D] or L Thumb   Look: [Arrow Keys] or R Thumb    Shoot: [Space] or tap bullets\nToggle VSync: [V]   Invert Y: [Y]   Toggle collision: [i]   Pause monsters: [M]\n\
            # collision objs = {}, # active particles {}, # active decals {}", num_collision_obj, num_active_particles, num_active_decals);
        renderer.set_debug_game_msg(&debug_msg);
        renderer.set_debug_font_color(&CgVec4::new(1.0, 0.0, 0.0, 1.0));
    }
}
