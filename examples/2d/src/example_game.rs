use cgmath::{InnerSpace, Vector3};
use instant::Instant;

use black_splat::utils::*;
use black_splat::{
    config::Config,
    engine::GameEngine,
    game_object::{GameObject, GameObjectState, GameObjectType},
    input::InputManager,
    renderer::Renderer,
    log,
};

const SKY_Z: f32 = 0.0;
const SUN_Z: f32 = 15.0;
const CLOUD_Z: f32 = 20.0;
const HILL_Z: f32 = 30.0;
const BUILDING_Z: f32 = 50.0;
const CHARACTER_Z: f32 = 100.0;

pub struct Example2DGame {
    pub game_objects: Vec<GameObject>,
    game_start_time: Instant,
    current_frame_time: Instant,
    next_enemy_spawn_time: f32,
    num_enemies: u32,

    // data
    max_game_objects: usize,
    enemy_spawn_delay: f32,
    enemy_speed: f32,
}

impl Example2DGame {
    fn update_enemies(&mut self) {
        if self.game_objects.len() >= self.max_game_objects {
            return;
        }

        let game_time = self.game_start_time.elapsed().as_secs_f32();
        if game_time > self.next_enemy_spawn_time {
            self.next_enemy_spawn_time = game_time + self.enemy_spawn_delay;
            self.num_enemies += 1;

            let mut start_x = 2.0;
            let mut vel_x = -self.enemy_speed;

            let randnum = random_u32(1, 2);
            if randnum == 2 {
                start_x *= -1.0;
                vel_x *= -1.0;
            }
            let y_pos: f32 = random_f32(0.0, 0.75) - 0.35;

            // Create Enemy
            self.game_objects.push(GameObject {
                position: (start_x, y_pos, CHARACTER_Z).into(),
                scale: (0.1, 0.15, 0.15).into(),
                direction: (1.0, 0.0, 0.0).into(),
                velocity: (vel_x, 0.0, 0.0).into(),
                object_type: GameObjectType::Robot,
                object_state: GameObjectState::Running,
                next_attack_time: 0.0,
                texture_index: 0,
                sprite_index: 8,
                anim_frame: 0,
                life_start_time: Instant::now(),
                state_start_time: Instant::now(),
                gravity_scale: 0.0,
                random_val: random_f32(0.0, 1000.0),
                is_enemy: true,
                uv_tiles: (1.0, 1.0),
            });
        }
    }

    fn update_projectiles(&mut self) {
        let mut i = 0;
        while i < self.game_objects.len() {
            if !matches!(self.game_objects[i].object_type, GameObjectType::Projectile) {
                i += 1;
                continue;
            }

            let mut j = 0;
            while j < self.game_objects.len() {
                // Don't hit other projectiles
                if i == j || matches!(self.game_objects[j].object_type, GameObjectType::Projectile)
                {
                    j += 1;
                    continue;
                }

                // Allegiance test
                if self.game_objects[i].is_enemy == self.game_objects[j].is_enemy {
                    j += 1;
                    continue;
                }

                let dist = cgmath::Vector2::<f32>::new(
                    self.game_objects[i].position.x - self.game_objects[j].position.x,
                    self.game_objects[i].position.y - self.game_objects[j].position.y,
                )
                .magnitude2();
                if dist < 0.05 {
                    if i > j {
                        self.game_objects.remove(i);
                        self.game_objects.remove(j);
                    } else {
                        self.game_objects.remove(j);
                        self.game_objects.remove(i);
                    }
                    break;
                }
                j += 1;
            }

            i += 1;
        }
    }
}

impl GameEngine for Example2DGame {
    fn new(game_config: &Config) -> Self {
        log!("GameEngine::new() caled...");

        let cur_time = Instant::now();

        Self {
            game_objects: Vec::<GameObject>::new(),
            game_start_time: cur_time,
            current_frame_time: cur_time,
            next_enemy_spawn_time: cur_time.elapsed().as_secs_f32() + game_config.enemy_spawn_delay,
            num_enemies: 0,

            max_game_objects: game_config.max_render_instances as usize,
            enemy_spawn_delay: game_config.enemy_spawn_delay,
            enemy_speed: game_config.enemy_move_speed,
        }
    }

    async fn initialize_world(
        &mut self,
        renderer: &mut Renderer<'_>,
        _game_config: &mut Config,
    ) {
        log!("GameEngine::initialize_world() caled...");

        // The sprite art is display-referred, so it must reach the surface
        // untouched.  The filmic tonemap is for the HDR-lit 3D/splat scenes;
        // running it over 2D sprites shifts colors (a pink cast) and makes the
        // anti-aliased sprite edges pop.
        renderer.set_tonemap_enabled(false);

        // Create Player
        self.game_objects.push(GameObject {
            position: (0.0, -0.35, CHARACTER_Z).into(),
            scale: (0.15, 0.15, 0.15).into(),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Character,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 0,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 3.1,
            uv_tiles: (1.0, 1.0),
            random_val: random_f32(0.0, 1000.0),
            is_enemy: false,
        });

        // Sky
        self.game_objects.push(GameObject {
            position: (0.0, -0.35, SKY_Z).into(),
            scale: (2.0, 2.0, 1.0).into(),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Skybox,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 25,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            uv_tiles: (1.0, 1.0),
            is_enemy: false,
        });

        // Sun
        self.game_objects.push(GameObject {
            position: (-0.5, 1.0 - 0.35, SUN_Z).into(),
            scale: (0.15, 0.15, 0.15).into(),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Skybox,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 27,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            uv_tiles: (1.0, 1.0),
            is_enemy: false,
        });

        // Clouds
        let mut i = 0;
        while i < 10 {
            let rand_x = random_f32(-2.0, 2.0);
            let rand_y = random_f32(0.8, 1.1) - 0.35;
            let scale_x = random_f32(0.17, 0.3);
            let scale_y = random_f32(0.13, 0.17);
            let x_speed = random_f32(0.03, 0.05);

            // Cloud
            self.game_objects.push(GameObject {
                position: (rand_x, rand_y, CLOUD_Z).into(),
                scale: (scale_x, scale_y, 0.15).into(),
                direction: (1.0, 0.0, 0.0).into(),
                velocity: (0.0, 0.0, 0.0).into(),
                object_type: GameObjectType::Cloud,
                object_state: GameObjectState::Idle,
                next_attack_time: 0.0,
                texture_index: 0,
                sprite_index: 18 + random_u32(0, 1) as i32,
                uv_tiles: (1.0, 1.0),
                anim_frame: 0,
                life_start_time: Instant::now(),
                state_start_time: Instant::now(),
                gravity_scale: 0.0,
                random_val: random_f32(0.0, 1000.0),
                is_enemy: false,
            });

            if let Some(game_obj) = self.game_objects.last_mut() {
                game_obj.set_velocity(Vector3::<f32>::new(x_speed, 0.0, 0.0));
            }
            i += 1;
        }

        // Hills
        self.game_objects.push(GameObject {
            position: (0.0, 0.75 - 0.35, HILL_Z).into(),
            scale: (2.0, 1.6, 0.15).into(),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 21,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            uv_tiles: (1.0, 1.0),
            is_enemy: false,
        });

        let mut x = -1.9;
        while x < 1.9 {
            let building_width = random_f32(0.12, 0.20);
            let building_height = random_f32(0.25, 0.5);
            self.game_objects.push(GameObject {
                position: (
                    x,
                    building_height - 0.35,
                    BUILDING_Z - (building_height * 3.0),
                )
                    .into(),
                scale: (building_width, building_height, 1.0).into(),
                direction: (1.0, 0.0, 0.0).into(),
                velocity: (0.0, 0.0, 0.0).into(),
                object_type: GameObjectType::Background,
                object_state: GameObjectState::Idle,
                next_attack_time: 0.0,
                texture_index: 0,
                sprite_index: random_u32(32, 39) as i32,
                anim_frame: 0,
                life_start_time: Instant::now(),
                state_start_time: Instant::now(),
                gravity_scale: 0.0,
                random_val: random_f32(0.0, 1000.0),
                uv_tiles: (1.0, 1.0),
                is_enemy: false,
            });
            x += random_f32(0.2, 0.3);
        }

        // Trees
        let mut x = -1.9;
        while x < 1.9 {
            let tree_width = random_f32(0.08, 0.15);
            let tree_height = random_f32(0.16, 0.2);
            self.game_objects.push(GameObject {
                position: (x, tree_height - 0.35, BUILDING_Z + 1.0).into(),
                scale: (tree_width, tree_height, 1.0).into(),
                direction: (1.0, 0.0, 0.0).into(),
                velocity: (0.0, 0.0, 0.0).into(),
                object_type: GameObjectType::Background,
                object_state: GameObjectState::Idle,
                next_attack_time: 0.0,
                texture_index: 0,
                sprite_index: random_u32(23, 24) as i32,
                anim_frame: 0,
                life_start_time: Instant::now(),
                state_start_time: Instant::now(),
                gravity_scale: 0.0,
                random_val: random_f32(0.0, 1000.0),
                uv_tiles: (1.0, 1.0),
                is_enemy: false,
            });
            x += random_f32(0.12, 0.19);
        }
        // Roads
        self.game_objects.push(GameObject {
            position: (1.0, -0.5 - 0.35, BUILDING_Z + 2.0).into(),
            scale: (1.0, 0.5, 1.0).into(),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 22,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            uv_tiles: (1.0, 1.0),
            is_enemy: false,
        });
        self.game_objects.push(GameObject {
            position: (-1.0, -0.5 - 0.35, BUILDING_Z + 2.0).into(),
            scale: (1.0, 0.5, 1.0).into(),
            direction: (1.0, 0.0, 0.0).into(),
            velocity: (0.0, 0.0, 0.0).into(),
            object_type: GameObjectType::Background,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index: 22,
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 0.0,
            random_val: random_f32(0.0, 1000.0),
            uv_tiles: (1.0, 1.0),
            is_enemy: false,
        });
    }

    fn get_game_objects(&self) -> &Vec<GameObject> {
        &self.game_objects
    }

    fn tick_frame_internal(
        &mut self,
        renderer: &mut Renderer,
        input_manager: &InputManager,
        _game_config: &Config,
    ) {
        let _delta_time_secs = self.current_frame_time.elapsed().as_secs_f32();
        self.current_frame_time = Instant::now();

        // Player Movement
        let mut move_vec: cgmath::Vector3<f32> = (0.0, 0.0, 0.0).into();

        let mut swipe_left = false;
        let mut swipe_right = false;
        let mut jump = false;
        let mut shoot = false;

        let touch_map_iter = input_manager.get_touch_map().iter();
        for touch_pair in touch_map_iter {
            let touch = &touch_pair.1;
            if touch.touch_state.is_down() && touch.start_pos.0 < 500.0 {
                if touch.current_pos.0 < touch.start_pos.0 - 8.0 {
                    swipe_left = true;
                } else if touch.current_pos.0 > touch.start_pos.0 + 8.0 {
                    swipe_right = true;
                }

                if touch.current_pos.1 < touch.start_pos.1 - 16.0 {
                    jump = true;
                }
            }

            if touch.touch_state.is_down() && touch.start_pos.0 > 500.0 {
                shoot = true;
            }
        }

        if input_manager.get_key_state("a").is_down() || swipe_left {
            move_vec = Vector3::new(-1.0, 0.0, 0.0);
            self.game_objects[0].direction.x = -1.0;
        }

        if input_manager.get_key_state("d").is_down() || swipe_right {
            move_vec = Vector3::new(1.0, 0.0, 0.0);
            self.game_objects[0].direction.x = 1.0;
        }

        if input_manager.get_key_state("w").is_down() || jump {
            move_vec.y = 1.0;
        }

        self.game_objects[0].set_velocity(move_vec);

        self.update_enemies();
        self.update_projectiles();

        // Player Action
        if (shoot || input_manager.get_key_state("space").is_down())
            && self.game_objects[0].start_attack()
        {
            let direction = self.game_objects[0].direction;
            let velocity = if direction.x > 0.0 {
                (5.0, 0.0, 0.0).into()
            } else {
                (-5.0, 0.0, 0.0).into()
            };
            let new_projectile = GameObject {
                position: self.game_objects[0].position + direction * 0.1,
                scale: (0.035, 0.05, 0.05).into(),
                direction,
                velocity,
                object_type: GameObjectType::Projectile,
                object_state: GameObjectState::Idle,
                next_attack_time: 0.0,
                texture_index: 0,
                sprite_index: 5,
                anim_frame: 0,
                life_start_time: Instant::now(),
                state_start_time: Instant::now(),
                gravity_scale: 0.0,
                random_val: random_f32(0.0, 1000.0),
                uv_tiles: (1.0, 1.0),
                is_enemy: false,
            };

            self.game_objects.push(new_projectile);
        }

        // Update game objects
        let game_object_iter = self.game_objects.iter_mut();
        for game_object in game_object_iter {
            game_object.update(_delta_time_secs);
        }

        let debug_msg = "Move: [W][A][S][D]    Shoot: [Space]\nToggle VSync: [V]".to_string();
        renderer.set_debug_game_msg(&debug_msg);
    }
}
