use instant::Instant;
//use cgmath::InnerSpace;

use crate::{
    assets::*, config::*, resource::*, utils::*, render_groups::model::*,
};

static mut NEXT_ACTOR_ID: u32 = 0;

#[derive(Clone)]
#[allow(dead_code)]
pub struct ActorTransform {
    pub position: CgVec3,
    pub rotation: CgQuat,
    pub scale: CgVec3,
}

#[allow(dead_code)]
impl ActorTransform {
    pub fn new(position: CgVec3, rotation: CgQuat, scale: CgVec3) -> ActorTransform {
        ActorTransform {
            position,
            rotation,
            scale,
        }
    }
    pub fn from_position(position: CgVec3) -> ActorTransform {
        ActorTransform {
            position,
            rotation: CG_QUAT_IDENT,
            scale: CG_VEC3_ONE,
        }
    }

    pub fn from_position_scale(position: CgVec3) -> ActorTransform {
        ActorTransform {
            position,
            rotation: CG_QUAT_IDENT,
            scale: CG_VEC3_ONE,
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ParticleHandle {
    pub index: u32,
}

pub const INVALID_PARTICLE_HANDLE: ParticleHandle = ParticleHandle { index: u32::MAX };

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParticleBlendMode {
    Additive,
    AlphaBlend,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct ParticleParams {
    pub texture_file: String,
    pub blend_mode: ParticleBlendMode,

    pub min_burst_count: u32,
    pub max_burst_count: u32,

    pub min_particle_life: f32,
    pub max_particle_life: f32,

    pub _min_actor_life: f32,
    pub _max_actor_life: f32,

    pub min_start_spawn_rate: f32,
    pub max_start_spawn_rate: f32,

    pub min_start_pos: CgVec3,
    pub max_start_pos: CgVec3,

    pub min_start_scale: CgVec3,
    pub max_start_scale: CgVec3,

    pub min_end_scale: CgVec3,
    pub max_end_scale: CgVec3,

    pub min_start_velocity: CgVec3,
    pub max_start_velocity: CgVec3,

    pub min_start_rotation_rate: f32,
    pub max_start_rotation_rate: f32,

    pub min_start_acceleration: CgVec3,
    pub max_start_acceleration: CgVec3,

    pub min_end_velocity: CgVec3,
    pub max_end_velocity: CgVec3,

    pub start_color_0: CgVec4,
    pub start_color_1: CgVec4,

    pub end_color_0: CgVec4,
    pub _end_color1: CgVec4,
}

#[allow(dead_code)]
pub struct Particle {
    pub position: CgVec3,
    pub acceleration: CgVec3,
    pub velocity: CgVec3,
    pub color: CgVec4,
    pub scale: CgVec3,
    pub rotation: f32,
    pub rotation_rate: f32,
    pub start_time: f32,
    pub start_scale: CgVec3,
    pub end_scale: CgVec3,
    pub life_time: f32,
}

#[allow(dead_code)]
pub struct ParticleActor {
    pub params: ParticleParams,
    pub model: Model,
    pub transform: ActorTransform,
    spawn_rate: f32,
    start_time: Instant,
    next_spawn_time: f32,
    pub particles: Vec<Particle>,
    pub particle_handle: ParticleHandle,

    pub active: bool,
}

impl ParticleActor {
    pub async fn new(
        transform: &ActorTransform,
        particle_handle: &ParticleHandle,
        params: &ParticleParams,
        device_resources: &DeviceResources<'_>,
        asset_manager: &mut AssetManager,
    ) -> Self {
        let model =
            Model::new_particle(&params.texture_file, device_resources, asset_manager).await;
        let spawn_rate = random_f32(params.min_start_spawn_rate, params.max_start_spawn_rate);
        let params = (*params).clone();
        let start_time = instant::Instant::now();
        let next_spawn_time = spawn_rate + start_time.elapsed().as_secs_f32();
        let particles = Vec::<Particle>::new();
        let transform = (*transform).clone();

        ParticleActor {
            params,
            model,
            transform,
            spawn_rate,
            start_time,
            next_spawn_time,
            particles,
            particle_handle: particle_handle.clone(),
            active: true,
        }
    }

    pub fn tick(&mut self, game_config: &Config) {
        let elapsed_time = self.start_time.elapsed().as_secs_f32();
        if self.params._min_actor_life > 0.0 && elapsed_time > self.params._min_actor_life {
            self.set_active(false);
            return;
        }

        if elapsed_time > self.next_spawn_time {
            let params = &self.params;
            self.next_spawn_time = elapsed_time + self.spawn_rate;

            let position = random_vec3(params.min_start_pos, params.max_start_pos);
            let acceleration =
                random_vec3(params.min_start_acceleration, params.max_start_acceleration);
            let velocity = random_vec3(params.min_start_velocity, params.max_start_velocity);
            let color = random_vec4(params.start_color_0, params.start_color_1);
            let life_time = random_f32(params.min_particle_life, params.max_particle_life);
            let start_scale = random_vec3(params.min_start_scale, params.max_start_scale);
            let end_scale = random_vec3(params.min_end_scale, params.max_end_scale);
            let scale = start_scale;
            let rotation_rate =
                random_f32(params.min_start_rotation_rate, params.max_start_spawn_rate);
            let rotation = random_f32(0.0, 100.0);

            let particle = Particle {
                position,
                scale,
                start_scale,
                end_scale,
                acceleration,
                velocity,
                rotation,
                rotation_rate,
                color,
                life_time,
                start_time: elapsed_time,
            };
            self.particles.push(particle);
        }

        let delta_time = game_config.delta_time;

        self.particles.retain_mut(|particle| {
            if elapsed_time > particle.start_time + particle.life_time {
                false
            } else {
                let t = ((elapsed_time - particle.start_time) / particle.life_time).clamp(0.0, 1.0);
                particle.velocity += particle.acceleration * delta_time;
                particle.position += particle.velocity * delta_time;

                particle.rotation += particle.rotation_rate * delta_time;
                particle.scale =
                    particle.start_scale + (particle.end_scale - particle.start_scale) * t;
                particle.color = self.params.start_color_0
                    + (self.params.end_color_0 - self.params.start_color_0) * t;
                particle.color.x = particle.color.x.clamp(0.0, 999999.0);
                particle.color.y = particle.color.y.clamp(0.0, 999999.0);
                particle.color.z = particle.color.z.clamp(0.0, 999999.0);

                true
            }
        });
    }

    pub fn set_position(&mut self, position: &CgVec3) {
        self.transform.position = *position;
    }

    pub fn get_position(&self) -> CgVec3 {
        self.transform.position
    }

    pub fn set_scale(&mut self, scale: &CgVec3) {
        self.transform.scale = *scale;
    }

    pub fn get_scale(&self) -> CgVec3 {
        self.transform.scale
    }

    pub fn set_rotation(&mut self, rotation: &CgQuat) {
        self.transform.rotation = *rotation;
    }

    pub fn get_rotation(&self) -> CgQuat {
        self.transform.rotation
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
        self.particles.clear();
        if active {
            let count = random_u32(self.params.min_burst_count, self.params.max_burst_count);

            self.start_time = Instant::now();
            for _ in 0..count {
                let params = &self.params;
                let position = random_vec3(params.min_start_pos, params.max_start_pos);
                let acceleration =
                    random_vec3(params.min_start_acceleration, params.max_start_acceleration);
                let velocity = random_vec3(params.min_start_velocity, params.max_start_velocity);
                let color = random_vec4(params.start_color_0, params.start_color_1);
                let life_time = random_f32(params.min_particle_life, params.max_particle_life);
                let start_scale = random_vec3(params.min_start_scale, params.max_start_scale);
                let end_scale = random_vec3(params.min_end_scale, params.max_end_scale);
                let scale = start_scale;
                let rotation_rate = random_f32(
                    params.min_start_rotation_rate,
                    params.max_start_rotation_rate,
                );
                let rotation = 0.0;

                let particle = Particle {
                    position,
                    scale,
                    start_scale,
                    end_scale,
                    acceleration,
                    velocity,
                    rotation,
                    rotation_rate,
                    color,
                    life_time,
                    start_time: self.start_time.elapsed().as_secs_f32(),
                };
                self.particles.push(particle);
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

#[derive(Debug, Clone)]
pub struct Actor {
    pub id: u32,
    position: CgVec3,
    rotation: CgQuat,
    scale: CgVec3,
    color: CgVec4,
    custom_data_1: CgVec4,

    render_group: RenderGroupType,
    custom_render_group_handle: Option<usize>,

    model_handle: ModelHandle,
}

impl Default for Actor {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor {
    pub fn new() -> Self {
        unsafe {
            NEXT_ACTOR_ID += 1;
            Actor {
                id: NEXT_ACTOR_ID,
                position: CG_VEC3_ZERO,
                rotation: (0.0, 0.0, 0.0, 1.0).into(),
                scale: CG_VEC3_ONE,
                color: CG_VEC4_ONE,
                custom_data_1: CG_VEC4_ZERO,
                render_group: RenderGroupType::World,
                custom_render_group_handle: None,
                model_handle: ModelHandle::make_invalid(),
            }
        }
    }

    pub fn set_position(&mut self, position: &CgVec3) {
        self.position = *position;
    }

    pub fn get_position(&self) -> CgVec3 {
        self.position
    }

    pub fn set_rotation(&mut self, rotation: &CgQuat) {
        self.rotation = *rotation;
    }

    pub fn get_rotation(&self) -> CgQuat {
        self.rotation
    }

    pub fn set_scale(&mut self, scale: &CgVec3) {
        self.scale = *scale;
    }

    pub fn get_scale(&self) -> CgVec3 {
        self.scale
    }

    pub fn set_model(&mut self, new_model: &ModelHandle) {
        self.model_handle = *new_model;
    }

    pub fn get_model(&self) -> ModelHandle {
        self.model_handle
    }

    pub fn set_render_group(
        &mut self,
        new_render_group: &RenderGroupType,
        custom_render_group_handle: &Option<usize>,
    ) {
        self.render_group = new_render_group.clone();
        self.custom_render_group_handle
            .clone_from(custom_render_group_handle);
    }

    pub fn get_render_group(&self) -> (RenderGroupType, Option<usize>) {
        (self.render_group.clone(), self.custom_render_group_handle)
    }

    pub fn set_color(&mut self, color: CgVec4) {
        self.color = color;
    }

    pub fn get_color(&self) -> CgVec4 {
        self.color
    }

    pub fn set_custom_data_1(&mut self, custom_data: CgVec4) {
        self.custom_data_1 = custom_data;
    }

    pub fn get_custom_data_1(&self) -> CgVec4 {
        self.custom_data_1
    }
}

#[derive(Clone)]
pub struct Camera {
    position: CgVec3,
    rotation: CgVec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

impl Camera {
    pub fn new() -> Self {
        Camera {
            position: CG_VEC3_ZERO,
            rotation: CG_VEC3_ZERO,
        }
    }

    /* pub fn set_look_at(&mut self, new_pos: &CgVec3, target_pos: &CgVec3) {
        self.set_position(new_pos);
        self.set_rotation(&cgmath::Matrix3::look_to_rh((new_pos - target_pos).normalize(), CG_VEC3_UP).into());
    }*/

    pub fn set_position(&mut self, new_pos: &CgVec3) {
        self.position = *new_pos;
    }

    pub fn get_position(&self) -> CgVec3 {
        self.position
    }

    pub fn set_rotation(&mut self, new_rot: &CgVec3) {
        self.rotation = *new_rot;
    }

    pub fn get_rotation(&self) -> CgVec3 {
        self.rotation
    }
    /*
    pub fn set_rotation(&mut self, new_rot: &CgQuat) {
        self.rotation = new_rot.clone();
    }

    pub fn get_rotation(&self) -> CgQuat {
        self.rotation.clone()
    }*/

    pub fn calculate_view_matrix(&self) -> (CgMat4, CgVec3, CgVec3) {
        let cam_pos = self.get_position();
        let eye: CgPoint = CgPoint::new(cam_pos.x, cam_pos.y, cam_pos.z);

        let pitch_rad = cgmath::Rad::from(cgmath::Deg(self.rotation.x));
        let pitch_mat = CgMat4::from_angle_y(pitch_rad);

        let heading_rad = cgmath::Rad::from(cgmath::Deg(self.rotation.y));
        let heading_mat = CgMat4::from_angle_x(heading_rad);
        let view_mat = pitch_mat * heading_mat;
        //let view_mat = cgmath::Matrix4::from(self.get_rotation());
        let right_dir = -CgVec3::new(view_mat.x.x, view_mat.x.y, view_mat.x.z);
        let view_dir = CgVec3::new(view_mat.z.x, view_mat.z.y, view_mat.z.z);
        let target = eye + view_dir;
        let up = cgmath::Vector3::unit_y();
        (CgMat4::look_at_rh(eye, target, up), view_dir, right_dir)
    }
}

// todo: deprecate the below

#[derive(Clone)]
pub enum GameObjectType {
    Character,
    Robot,
    Projectile,
    Background,
    Skybox,
    Cloud,
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum GameObjectState {
    Idle,
    Jumping,
    Running,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct GameObject {
    pub position: CgVec3,
    pub direction: CgVec3,
    pub scale: CgVec3,
    pub velocity: CgVec3,
    pub object_type: GameObjectType,
    pub object_state: GameObjectState,
    pub next_attack_time: f32,
    pub texture_index: u32,
    pub sprite_index: i32,
    pub uv_tiles: (f32, f32),
    pub anim_frame: i32,
    pub life_start_time: Instant,
    pub state_start_time: Instant,
    pub gravity_scale: f32,
    pub is_enemy: bool,
    pub random_val: f32,
}

#[allow(dead_code)]
impl GameObject {
    pub fn new(
        object_type: GameObjectType,
        sprite_index: i32,
        position: CgVec3,
        direction: CgVec3,
        scale: CgVec3,
    ) -> Self {
        GameObject {
            position,
            direction,
            scale,
            velocity: (0.0, 0.0, 0.0).into(),
            object_type,
            object_state: GameObjectState::Idle,
            next_attack_time: 0.0,
            texture_index: 0,
            sprite_index,
            uv_tiles: (1.0, 1.0),
            anim_frame: 0,
            life_start_time: Instant::now(),
            state_start_time: Instant::now(),
            gravity_scale: 3.1,
            random_val: random_f32(0.0, 1000.0),
            is_enemy: false,
        }
    }

    fn set_state(&mut self, next_state: GameObjectState) {
        self.object_state = next_state;
        self.state_start_time = Instant::now();
    }

    fn update_movement(&mut self, delta_time: f32) {
        self.position += self.velocity * delta_time;

        // Apply Gravity
        if f32::abs(self.gravity_scale) > 0.001 {
            if self.position.y > -0.35 {
                self.velocity.y -= delta_time * self.gravity_scale;
            } else if self.position.y < -0.35 {
                self.velocity.y = 0.0;
                self.position.y = -0.35;
                self.set_state(GameObjectState::Idle);
            }
        }

        match self.object_state {
            GameObjectState::Running => {
                let duration = self.state_start_time.elapsed().as_secs_f32() * 5.0;
                self.anim_frame = 1 + (duration as i32) % 4;
            }

            _ => {
                self.anim_frame = 0;
            }
        }

        match self.object_type {
            GameObjectType::Projectile => {
                let duration = self.state_start_time.elapsed().as_secs_f32() * 15.0;
                self.anim_frame = (duration as i32) % 3;
            }

            GameObjectType::Skybox => {
                let duration = self.state_start_time.elapsed().as_secs_f32() * 1.2;
                self.anim_frame = (duration as i32) % 2;
            }

            GameObjectType::Cloud => {
                if self.velocity.x > 0.0 {
                    if self.position.x > 2.1 {
                        self.position.x = -2.1;
                    }
                } else if self.position.x < -2.1 {
                    self.position.x = 2.1;
                }
            }

            GameObjectType::Robot => {
                if self.velocity.x > 0.0 {
                    if self.position.x > 1.9 {
                        self.velocity.x *= -1.0;
                    }
                } else if self.position.x < -1.9 {
                    self.velocity.x *= -1.0;
                }
            }

            GameObjectType::Character => {
                self.position.x = self.position.x.clamp(-1.9, 1.9);
            }
            _ => (),
        }
    }

    pub fn update(&mut self, frame_time: f32) {
        self.update_movement(frame_time);
    }

    pub fn set_velocity(&mut self, move_vec: CgVec3) {
        self.velocity.x = move_vec.x;

        if !matches!(self.object_type, GameObjectType::Character) {
            return;
        }

        let is_jumping = matches!(self.object_state, GameObjectState::Jumping);
        if f32::abs(move_vec.x) < 0.0001 && !is_jumping {
            self.set_state(GameObjectState::Idle);
        } else if !matches!(self.object_state, GameObjectState::Running) && !is_jumping {
            self.set_state(GameObjectState::Running);
        }

        if move_vec.y > 0.0 && !matches!(self.object_state, GameObjectState::Jumping) {
            self.velocity.y = 2.1;
            self.set_state(GameObjectState::Jumping);
        }
    }

    pub fn start_attack(&mut self) -> bool {
        let cur_time = self.life_start_time.elapsed().as_secs_f32();
        if self.next_attack_time > cur_time {
            return false;
        }

        self.next_attack_time = cur_time + 0.04;
        true
    }
}
