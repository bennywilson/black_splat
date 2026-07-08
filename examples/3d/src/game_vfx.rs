use black_splat::{
    assets::*, collision::*, config::*, game_object::*, renderer::*, resource::*,
    utils::*,
};

use crate::game_actors::*;

const MAX_DECALS: usize = 16;

pub struct GameVfxManager {
    pooled_gib_particles: Vec<ParticleHandle>,
    next_pooled_gib: usize,

    pooled_impact_particles: Vec<ParticleHandle>,
    next_pooled_impact: usize,

    pooled_smoke_particles: Vec<ParticleHandle>,
    next_pooled_smoke: usize,

    pooled_muzzle_flashes: Vec<ParticleHandle>,
    next_muzzle_flash: usize,

    pooled_barrel_explosions: Vec<ParticleHandle>,
    next_barrel_explosion: usize,

    decals: Vec<GameDecal>,
    num_active_decals: usize,
    decal_model: Option<ModelHandle>,
    decal_render_group: usize,
}

impl GameVfxManager {
    pub fn new() -> Self {
        GameVfxManager {
            pooled_gib_particles: Vec::<ParticleHandle>::new(),
            next_pooled_gib: 0,

            pooled_impact_particles: Vec::<ParticleHandle>::new(),
            next_pooled_impact: 0,

            pooled_smoke_particles: Vec::<ParticleHandle>::new(),
            next_pooled_smoke: 0,

            pooled_muzzle_flashes: Vec::<ParticleHandle>::new(),
            next_muzzle_flash: 0,

            pooled_barrel_explosions: Vec::<ParticleHandle>::new(),
            next_barrel_explosion: 0,

            decals: Vec::<GameDecal>::new(),
            num_active_decals: 0,
            decal_model: None,
            decal_render_group: usize::MAX,
        }
    }

    pub fn spawn_gibs(&mut self, gibs_position: &CgVec3, renderer: &mut Renderer<'_>) {
        self.next_pooled_gib = (self.next_pooled_gib + 1) % self.pooled_gib_particles.len();
        renderer.enable_particle_actor(&self.pooled_gib_particles[self.next_pooled_gib], true);
        renderer.update_particle_transform(
            &self.pooled_gib_particles[self.next_pooled_gib],
            gibs_position,
            &None,
        );
    }

    pub fn spawn_impact(&mut self, impact_position: &CgVec3, renderer: &mut Renderer<'_>) {
        self.next_pooled_impact =
            (self.next_pooled_impact + 1) % self.pooled_impact_particles.len();
        renderer
            .enable_particle_actor(&self.pooled_impact_particles[self.next_pooled_impact], true);
        renderer.update_particle_transform(
            &self.pooled_impact_particles[self.next_pooled_impact],
            impact_position,
            &None,
        );
    }

    pub fn spawn_barrel_smoke(
        &mut self,
        barrel_pos: &CgVec3,
        renderer: &mut Renderer<'_>,
    ) -> (ParticleHandle, ParticleHandle) {
        self.next_pooled_smoke = (self.next_pooled_smoke + 1) % self.pooled_smoke_particles.len();
        let particle_handle_1 = self.pooled_smoke_particles[self.next_pooled_smoke].clone();
        renderer.enable_particle_actor(&particle_handle_1, true);
        renderer.update_particle_transform(&particle_handle_1, barrel_pos, &None);

        // Ember
        self.next_pooled_smoke = (self.next_pooled_smoke + 1) % self.pooled_smoke_particles.len();
        let particle_handle_2 = self.pooled_smoke_particles[self.next_pooled_smoke].clone();
        renderer.enable_particle_actor(&particle_handle_2, true);
        renderer.update_particle_transform(&particle_handle_2, barrel_pos, &None);

        (particle_handle_1, particle_handle_2)
    }

    pub fn spawn_explosion(&mut self, explosion_position: &CgVec3, renderer: &mut Renderer<'_>) {
        self.next_barrel_explosion =
            (self.next_barrel_explosion + 1) % self.pooled_barrel_explosions.len();
        renderer.enable_particle_actor(
            &self.pooled_barrel_explosions[self.next_barrel_explosion],
            true,
        );
        renderer.update_particle_transform(
            &self.pooled_barrel_explosions[self.next_barrel_explosion],
            explosion_position,
            &None,
        );

        self.next_barrel_explosion =
            (self.next_barrel_explosion + 1) % self.pooled_barrel_explosions.len();
        renderer.enable_particle_actor(
            &self.pooled_barrel_explosions[self.next_barrel_explosion],
            true,
        );
        renderer.update_particle_transform(
            &self.pooled_barrel_explosions[self.next_barrel_explosion],
            explosion_position,
            &None,
        );
    }

    pub fn spawn_muzzle_flash(
        &mut self,
        position: &CgVec3,
        scale: &CgVec3,
        renderer: &mut Renderer<'_>,
    ) {
        self.next_muzzle_flash = (self.next_muzzle_flash + 1) % self.pooled_muzzle_flashes.len();
        renderer.enable_particle_actor(&self.pooled_muzzle_flashes[self.next_muzzle_flash], true);
        renderer.update_particle_transform(
            &self.pooled_muzzle_flashes[self.next_muzzle_flash],
            position,
            &Some(*scale),
        );
    }

    pub fn tick(&mut self, position: &CgVec3, renderer: &mut Renderer, game_config: &Config) {
        for muzzle_flash in &mut self.pooled_muzzle_flashes {
            renderer.update_particle_transform(muzzle_flash, position, &None);
        }

        // Decals
        self.num_active_decals = 0;
        let elapsed_time = game_config.start_time.elapsed().as_secs_f32();
        self.decals.retain_mut(|d| {
            if elapsed_time > d.start_time + 2.0 {
                renderer.remove_actor(&d.actor);
                return false;
            }
            let alpha = 1.0 - (elapsed_time - d.start_time) / 2.0;
            d.actor
                .set_color(0.5 * CgVec4::new(alpha, alpha, alpha, alpha));
            renderer.add_or_update_actor(&d.actor);
            self.num_active_decals += 1;
            true
        });
    }

    pub fn spawn_mob_death_fx(
        &mut self,
        mob_pos: &CgVec3,
        view_dir: &CgVec3,
        renderer: &mut Renderer<'_>,
        collision_manager: &mut CollisionManager,
        game_config: &Config,
    ) {
        self.spawn_gibs(mob_pos, renderer);

        if self.decals.len() > MAX_DECALS {
            return;
        }

        // Floor decals
        let num_floor_decals = random_u32(3, 7);
        for _ in 0..num_floor_decals {
            let mut decal_actor = Actor::new();

            let mut ground_pos =
                mob_pos + random_vec3(CgVec3::new(-3.0, 0.0, -3.0), CgVec3::new(3.0, 0.0, 3.0));
            ground_pos.y = 0.05;
            decal_actor.set_position(&ground_pos);

            let scale = random_f32(1.0, 5.0);
            decal_actor.set_scale(&CgVec3::new(scale, scale, scale));

            let decal_rotation = cgmath::Rad::from(cgmath::Deg(random_f32(0.0, 360.0)));
            let rotation = cgmath::Quaternion::from(CgMat3::from_angle_y(decal_rotation));
            decal_actor.set_rotation(&rotation);

            decal_actor.set_model(self.decal_model.as_ref().unwrap());
            decal_actor.set_render_group(
                &RenderGroupType::WorldCustom,
                &Some(self.decal_render_group),
            );
            renderer.add_or_update_actor(&decal_actor);
            let decal = GameDecal {
                actor: decal_actor,
                start_time: game_config.start_time.elapsed().as_secs_f32()
                    + random_f32(-0.25, 0.25),
            };
            self.decals.push(decal);
        }

        // Wall decals
        let num_wall_decals = random_u32(3, 5);
        let decal_position_range = CgVec3::new(3.0, 3.0, 3.0);
        for _ in 0..num_wall_decals {
            let rot_1 = cgmath::Rad::from(cgmath::Deg(random_f32(-15.0, 15.0)));
            let rot_2 = cgmath::Rad::from(cgmath::Deg(random_f32(-15.0, 15.0)));
            let rot_3 = cgmath::Rad::from(cgmath::Deg(random_f32(-15.0, 15.0)));
            let rotation = cgmath::Quaternion::from(
                CgMat3::from_angle_x(rot_1)
                    * CgMat3::from_angle_y(rot_2)
                    * CgMat3::from_angle_z(rot_3),
            );
            let mut splat_dir = rotation * view_dir * 15.0;

            let (t, _, decal_hit_loc, _) = collision_manager.cast_ray(mob_pos, &splat_dir);
            if (0.0..1.0).contains(&t) {
                splat_dir.y += random_f32(-decal_position_range.x, decal_position_range.x);

                // The world is a cube that extends from approximately (-20.0, -20.0, -20.0) to (20.0, 20.0, 20.0)
                let (pos, rotation) = {
                    let decal_hit_loc = decal_hit_loc.unwrap();
                    if decal_hit_loc.x.abs() > decal_hit_loc.z.abs() {
                        splat_dir.z +=
                            random_f32(-decal_position_range.z, decal_position_range.z);
                        if decal_hit_loc.x < 0.0 {
                            (CgVec3::new(-18.0, splat_dir.y, splat_dir.z), 90.0)
                        } else {
                            (CgVec3::new(18.0, splat_dir.y, splat_dir.z), 90.0)
                        }
                    } else {
                        splat_dir.x +=
                            random_f32(-decal_position_range.x, decal_position_range.x);
                        if decal_hit_loc.z < 0.0 {
                            (CgVec3::new(splat_dir.x, splat_dir.y, -18.0), 180.0)
                        } else {
                            (CgVec3::new(splat_dir.x, splat_dir.y, 18.0), 180.0)
                        }
                    }
                };

                let mut decal_actor = Actor::new();

                decal_actor.set_position(&pos);
                let scale = random_f32(1.0, 5.0);
                decal_actor.set_scale(&CgVec3::new(scale, scale, scale));

                let decal_fix = cgmath::Rad::from(cgmath::Deg(90.0));
                let decal_rotation = cgmath::Rad::from(cgmath::Deg(rotation));
                let spin = cgmath::Rad::from(cgmath::Deg(random_f32(0.0, 360.0)));
                let rotation = cgmath::Quaternion::from(
                    CgMat3::from_angle_y(decal_rotation)
                        * CgMat3::from_angle_x(decal_fix)
                        * CgMat3::from_angle_y(spin),
                );
                decal_actor.set_rotation(&rotation);

                decal_actor.set_model(self.decal_model.as_ref().unwrap());
                decal_actor.set_render_group(
                    &RenderGroupType::WorldCustom,
                    &Some(self.decal_render_group),
                );
                renderer.add_or_update_actor(&decal_actor);

                let decal = GameDecal {
                    actor: decal_actor,
                    start_time: game_config.start_time.elapsed().as_secs_f32()
                        + random_f32(-0.25, 0.25),
                };
                self.decals.push(decal);
            }
        }
    }

    pub async fn init(&mut self, renderer: &mut Renderer<'_>) {
        self.decal_model = Some(
            renderer
                .load_model("game_assets/models/decal.glb", false)
                .await,
        );
        self.decal_render_group = renderer
            .add_custom_render_group(
                &RenderGroupType::WorldCustom,
                &BlendMode::Additive,
                "engine_assets/shaders/decal.wgsl",
            )
            .await;

        // Pooled gibs
        let particle_params = ParticleParams {
            texture_file: "/game_assets/fx/monster_gibs_t.png".to_string(),
            blend_mode: ParticleBlendMode::AlphaBlend,

            min_burst_count: 75,
            max_burst_count: 100,

            min_particle_life: 0.1,
            max_particle_life: 0.75,

            _min_actor_life: 1.5,
            _max_actor_life: 1.5,

            min_start_spawn_rate: 9999.0,
            max_start_spawn_rate: 9999.0,

            min_start_pos: CgVec3::new(-0.5, -0.2, -0.2),
            max_start_pos: CgVec3::new(0.5, 0.2, 0.2),

            min_start_scale: CgVec3::new(0.05, 0.05, 0.05),
            max_start_scale: CgVec3::new(0.45, 0.45, 0.45),

            min_end_scale: CgVec3::new(0.5, 0.5, 0.5),
            max_end_scale: CgVec3::new(2.0, 2.0, 2.0),

            min_start_velocity: CgVec3::new(-10.0, -10.0, -10.0),
            max_start_velocity: CgVec3::new(10.0, 20.0, 10.0),

            min_start_rotation_rate: -100.0,
            max_start_rotation_rate: 100.0,

            min_start_acceleration: CgVec3::new(0.0, -35.0, 0.0),
            max_start_acceleration: CgVec3::new(0.0, -35.0, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(0.9, 0.9, 0.9, 1.0),
            start_color_1: CgVec4::new(1.0, 1.0, 1.0, 1.0),

            end_color_0: CgVec4::new(0.0, 0.0, 0.0, 0.0),
            _end_color1: CgVec4::new(0.0, 0.0, 0.0, 0.0),
        };
        let particle_transform = ActorTransform::from_position(CgVec3::new(3.0, 3.5, 0.0));
        for _ in 0..20 {
            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &particle_params, false)
                .await;
            self.pooled_gib_particles.push(particle_handle);
        }

        // Pooled Impacts
        let particle_params = ParticleParams {
            texture_file: "/game_assets/fx/smoke_t.png".to_string(),
            blend_mode: ParticleBlendMode::AlphaBlend,

            min_burst_count: 100,
            max_burst_count: 100,

            min_particle_life: 0.1,
            max_particle_life: 0.15,

            _min_actor_life: 1.5,
            _max_actor_life: 1.5,

            min_start_spawn_rate: 9999.0,
            max_start_spawn_rate: 9999.0,

            min_start_pos: CgVec3::new(-0.05, -0.05, -0.05),
            max_start_pos: CgVec3::new(0.05, 0.05, 0.05),

            min_start_scale: CgVec3::new(0.05, 0.05, 0.05),
            max_start_scale: CgVec3::new(0.15, 0.15, 0.15),

            min_end_scale: CgVec3::new(0.15, 0.15, 0.15),
            max_end_scale: CgVec3::new(0.3, 0.3, 0.3),

            min_start_velocity: CgVec3::new(-10.0, -10.0, -10.0),
            max_start_velocity: CgVec3::new(10.0, 10.0, 10.0),

            min_start_rotation_rate: -0.03,
            max_start_rotation_rate: 0.03,

            min_start_acceleration: CgVec3::new(0.0, -5.0, 0.0),
            max_start_acceleration: CgVec3::new(0.0, -5.0, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(0.7, 0.7, 0.7, 1.0),
            start_color_1: CgVec4::new(0.9, 0.8, 0.8, 1.0),

            end_color_0: CgVec4::new(0.7, 0.7, 0.7, 0.0),
            _end_color1: CgVec4::new(0.9, 0.8, 0.8, 0.0),
        };
        let particle_transform = ActorTransform::from_position(CgVec3::new(3.0, 3.5, 0.0));
        for _ in 0..20 {
            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &particle_params, false)
                .await;
            self.pooled_impact_particles.push(particle_handle);
        }

        // Pooled smoke
        let particle_smoke_params = ParticleParams {
            texture_file: "/game_assets/fx/smoke_t.png".to_string(),
            blend_mode: ParticleBlendMode::AlphaBlend,

            min_burst_count: 0,
            max_burst_count: 0,

            min_particle_life: 3.0,
            max_particle_life: 5.0,

            _min_actor_life: -1.0,
            _max_actor_life: -1.0,

            min_start_spawn_rate: 0.06,
            max_start_spawn_rate: 0.06,

            min_start_pos: CgVec3::new(-0.5, -0.2, -0.2),
            max_start_pos: CgVec3::new(0.5, 0.2, 0.2),

            min_start_scale: CgVec3::new(0.5, 0.5, 0.5),
            max_start_scale: CgVec3::new(0.8, 0.8, 0.8),

            min_end_scale: CgVec3::new(2.1, 2.1, 2.1),
            max_end_scale: CgVec3::new(3.0, 3.0, 3.0),

            min_start_velocity: CgVec3::new(-0.2, 1.0, -0.2),
            max_start_velocity: CgVec3::new(0.2, 1.0, 0.2),

            min_start_rotation_rate: -0.5,
            max_start_rotation_rate: 0.5,

            min_start_acceleration: CgVec3::new(0.0, -0.1, 0.0),
            max_start_acceleration: CgVec3::new(0.0, -0.1, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(0.4, 0.04, 0.0, 1.0),
            start_color_1: CgVec4::new(0.4, 0.07, 0.0, 1.0),

            end_color_0: CgVec4::new(-0.5, -0.5, -0.5, 0.0),
            _end_color1: CgVec4::new(-0.5, -0.5, -0.5, 1.0),
        };
        let particle_transform = ActorTransform::from_position(CgVec3::new(0.0, 3.5, 0.0));
        let _ = renderer
            .add_particle_actor(&particle_transform, &particle_params, true)
            .await;

        let particle_ember_params = ParticleParams {
            texture_file: "./game_assets/fx/ember_t.png".to_string(),
            blend_mode: ParticleBlendMode::Additive,

            min_burst_count: 0,
            max_burst_count: 0,

            min_particle_life: 1.5,
            max_particle_life: 2.5,

            _min_actor_life: -1.0,
            _max_actor_life: -1.0,

            min_start_spawn_rate: 0.3,
            max_start_spawn_rate: 0.3,

            min_start_pos: CgVec3::new(-0.75, -0.2, -0.75),
            max_start_pos: CgVec3::new(0.75, 0.2, 0.75),

            min_start_scale: CgVec3::new(0.3, 0.3, 0.3),
            max_start_scale: CgVec3::new(0.5, 0.5, 0.5),

            min_end_scale: CgVec3::new(0.0, 0.0, 0.0),
            max_end_scale: CgVec3::new(0.05, 0.05, 0.05),

            min_start_velocity: CgVec3::new(-0.2, 3.0, -0.2),
            max_start_velocity: CgVec3::new(0.2, 3.0, 0.2),

            min_start_rotation_rate: -15.5,
            max_start_rotation_rate: 15.5,

            min_start_acceleration: CgVec3::new(0.0, -0.1, 0.0),
            max_start_acceleration: CgVec3::new(0.0, -0.1, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(2.0, 1.0, 0.2, 1.0),
            start_color_1: CgVec4::new(2.0, 1.0, 0.2, 1.0),

            end_color_0: CgVec4::new(1.0, 0.8, -0.1, 0.0),
            _end_color1: CgVec4::new(1.0, 0.8, -0.1, 1.0),
        };
        let particle_transform = ActorTransform::from_position(CgVec3::new(0.0, 3.5, 0.0));
        let _ = renderer
            .add_particle_actor(&particle_transform, &particle_params, true)
            .await;

        for _ in 0..20 {
            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &particle_smoke_params, false)
                .await;
            self.pooled_smoke_particles.push(particle_handle);

            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &particle_ember_params, false)
                .await;
            self.pooled_smoke_particles.push(particle_handle);
        }

        // Pooled Muzzle Flashes
        let muzzle_flash_params = ParticleParams {
            texture_file: "/game_assets/fx/muzzle_flash_t.png".to_string(),
            blend_mode: ParticleBlendMode::Additive,

            min_burst_count: 1,
            max_burst_count: 1,

            min_particle_life: 0.05,
            max_particle_life: 0.09,

            _min_actor_life: 1.0,
            _max_actor_life: 1.0,

            min_start_spawn_rate: 999.06,
            max_start_spawn_rate: 999.06,

            min_start_pos: CgVec3::new(0.0, 0.0, 0.0),
            max_start_pos: CgVec3::new(0.0, 0.0, 0.0),

            min_start_scale: CgVec3::new(1.0, 1.0, 1.0),
            max_start_scale: CgVec3::new(1.25, 1.25, 1.25),

            min_end_scale: CgVec3::new(0.2, 0.2, 0.2),
            max_end_scale: CgVec3::new(0.3, 0.3, 0.3),

            min_start_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_start_velocity: CgVec3::new(0.0, 0.0, 0.0),

            min_start_rotation_rate: 0.0,
            max_start_rotation_rate: 0.0,

            min_start_acceleration: CgVec3::new(0.0, 0.0, 0.0),
            max_start_acceleration: CgVec3::new(0.0, 0.0, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(1.0, 1.0, 1.0, 1.0),
            start_color_1: CgVec4::new(1.0, 1.0, 1.0, 1.0),

            end_color_0: CgVec4::new(0.8, 0.9, 1.0, 1.0),
            _end_color1: CgVec4::new(1.0, 1.0, 1.5, 1.0),
        };

        for _ in 0..24 {
            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &muzzle_flash_params, false)
                .await;
            self.pooled_muzzle_flashes.push(particle_handle);
        }

        // Barrel Explosions
        let barrel_explosion_fire_params = ParticleParams {
            texture_file: "/game_assets/fx/fire_t.png".to_string(),
            blend_mode: ParticleBlendMode::Additive,

            min_burst_count: 23,
            max_burst_count: 50,

            min_particle_life: 0.15,
            max_particle_life: 0.5,

            _min_actor_life: 5.0,
            _max_actor_life: 5.0,

            min_start_spawn_rate: 99999.06,
            max_start_spawn_rate: 99999.06,

            min_start_pos: CgVec3::new(-0.5, -0.2, -0.2),
            max_start_pos: CgVec3::new(0.5, 0.2, 0.2),

            min_start_scale: CgVec3::new(1.5, 1.5, 1.5),
            max_start_scale: CgVec3::new(2.8, 2.8, 2.8),

            min_end_scale: CgVec3::new(4.5, 4.5, 4.5),
            max_end_scale: CgVec3::new(9.0, 9.0, 9.0),

            min_start_velocity: CgVec3::new(-15.0, 0.1, -15.0),
            max_start_velocity: CgVec3::new(15.0, 30.0, 15.0),

            min_start_rotation_rate: -3.0,
            max_start_rotation_rate: 3.0,

            min_start_acceleration: CgVec3::new(0.0, -12.0, 0.0),
            max_start_acceleration: CgVec3::new(0.0, -12.0, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(3.0, 1.5, 0.35, 1.0),
            start_color_1: CgVec4::new(1.0, 0.5, 0.35, 1.0),

            end_color_0: CgVec4::new(0.0, 0.0, 0.0, 0.0),
            _end_color1: CgVec4::new(-0.5, -0.5, -0.5, 1.0),
        };

        let barrel_explosion_smoke_params = ParticleParams {
            texture_file: "/game_assets/fx/smoke_t.png".to_string(),
            blend_mode: ParticleBlendMode::AlphaBlend,
            min_burst_count: 23,
            max_burst_count: 50,

            min_particle_life: 0.5,
            max_particle_life: 1.5,

            _min_actor_life: 5.0,
            _max_actor_life: 5.0,

            min_start_spawn_rate: 99999.06,
            max_start_spawn_rate: 99999.06,

            min_start_pos: CgVec3::new(-0.5, -0.2, -0.2),
            max_start_pos: CgVec3::new(0.5, 0.2, 0.2),

            min_start_scale: CgVec3::new(1.5, 1.5, 1.5),
            max_start_scale: CgVec3::new(2.8, 2.8, 2.8),

            min_end_scale: CgVec3::new(4.5, 4.5, 4.5),
            max_end_scale: CgVec3::new(9.0, 9.0, 9.0),

            min_start_velocity: CgVec3::new(-10.0, 0.1, -10.0),
            max_start_velocity: CgVec3::new(10.0, 22.0, 10.0),

            min_start_rotation_rate: -3.0,
            max_start_rotation_rate: 3.0,

            min_start_acceleration: CgVec3::new(0.0, -12.0, 0.0),
            max_start_acceleration: CgVec3::new(0.0, -12.0, 0.0),

            min_end_velocity: CgVec3::new(0.0, 0.0, 0.0),
            max_end_velocity: CgVec3::new(0.0, 0.0, 0.0),

            start_color_0: CgVec4::new(0.5, 0.2, 0.1, 1.0),
            start_color_1: CgVec4::new(0.3, 0.3, 0.3, 1.0),

            end_color_0: CgVec4::new(0.0, 0.0, 0.0, 0.0),
            _end_color1: CgVec4::new(-0.5, -0.5, -0.5, 1.0),
        };
        let particle_transform = ActorTransform::from_position(CgVec3::new(0.0, 3.5, 0.0));

        for _ in 0..24 {
            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &barrel_explosion_fire_params, false)
                .await;
            self.pooled_barrel_explosions.push(particle_handle);

            let particle_handle = renderer
                .add_particle_actor(&particle_transform, &barrel_explosion_smoke_params, false)
                .await;
            self.pooled_barrel_explosions.push(particle_handle);
        }
    }

    pub fn num_active_decals(&self) -> usize {
        self.num_active_decals
    }
}
