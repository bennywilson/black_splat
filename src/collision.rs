use std::collections::HashMap;

use crate::{config::*, renderer::*, utils::*, log, make_handle};

make_handle!(CollisionShape, CollisionHandle, CollisionMappings);

#[derive(Clone, Copy)]
pub struct CollisionSphere {
    pub position: CgVec3,
    pub radius: f32,
}

#[derive(Clone, Copy)]
pub struct CollisionAABB {
    pub position: CgVec3,
    pub extents: CgVec3,
    pub block: bool,
}

impl CollisionAABB {
    pub fn max(&self) -> CgVec3 {
        self.position + self.extents
    }

    pub fn min(&self) -> CgVec3 {
        self.position - self.extents
    }
}

#[derive(Clone, Copy)]
pub enum CollisionShape {
    Sphere(CollisionSphere),
    AABB(CollisionAABB),
}

pub struct CollisionManager {
    collision_objects: CollisionMappings,
}

impl Default for CollisionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CollisionManager {
    pub fn new() -> Self {
        log!("Initializing CollisionManager...");
        CollisionManager {
            collision_objects: CollisionMappings::new(),
        }
    }

    pub fn add_collision(&mut self, collision: &CollisionShape) -> CollisionHandle {
        let mappings = &mut self.collision_objects;
        let new_handle = {
            if !mappings.next_handle.is_valid() {
                mappings.next_handle.index = 0;
            }
            let new_handle = mappings.next_handle;
            mappings.next_handle.index += 1;
            new_handle
        };
        self.collision_objects
            .handles_to_assets
            .insert(new_handle, *collision);
        new_handle
    }

    pub fn remove_collision(&mut self, handle: &CollisionHandle) {
        self.collision_objects.handles_to_assets.remove(handle);
    }

    pub fn get_collision(&self, handle: &CollisionHandle) -> CollisionShape {
        *self
            .collision_objects
            .handles_to_assets
            .get(handle)
            .unwrap()
    }

    pub fn update_collision_position(&mut self, handle: &CollisionHandle, new_pos: &CgVec3) {
        let collision = self
            .collision_objects
            .handles_to_assets
            .get_mut(handle)
            .expect("Bad collision handle");

        let new_collision = match collision {
            CollisionShape::Sphere(s) => CollisionShape::Sphere(CollisionSphere {
                position: *new_pos,
                radius: s.radius,
            }),
            CollisionShape::AABB(b) => CollisionShape::AABB(CollisionAABB {
                position: *new_pos,
                extents: b.extents,
                block: b.block,
            }),
        };

        self.collision_objects
            .handles_to_assets
            .insert(*handle, new_collision);
    }

    pub fn cast_ray(
        &mut self,
        start: &CgVec3,
        dir: &CgVec3,
    ) -> (f32, Option<CollisionHandle>, Option<CgVec3>, Option<bool>) {
        let mut closest_hit = f32::MAX;
        let mut closest_handle = CollisionHandle::make_invalid();
        let mut blocks = None;

        for (handle, value) in &mut self.collision_objects.handles_to_assets {
            match value {
                CollisionShape::Sphere(_s) => {}

                CollisionShape::AABB(aabb) => {
                    let mut t_min = aabb.min() - start;
                    t_min.x /= dir.x;
                    t_min.y /= dir.y;
                    t_min.z /= dir.z;

                    let mut t_max = aabb.max() - start;
                    t_max.x /= dir.x;
                    t_max.y /= dir.y;
                    t_max.z /= dir.z;

                    let mut actual_min = CG_VEC3_ZERO;
                    let mut actual_max = CG_VEC3_ZERO;
                    actual_min.x = t_min.x.min(t_max.x);
                    actual_max.x = t_min.x.max(t_max.x);
                    actual_min.y = t_min.y.min(t_max.y);
                    actual_max.y = t_min.y.max(t_max.y);
                    actual_min.z = t_min.z.min(t_max.z);
                    actual_max.z = t_min.z.max(t_max.z);

                    let smallest_max = actual_max.x.min(actual_max.y).min(actual_max.z);
                    let largest_min = actual_min.x.max(actual_min.y).max(actual_min.z);

                    if largest_min > 0.0 && smallest_max >= largest_min && largest_min < closest_hit
                    {
                        closest_hit = largest_min;
                        closest_handle = *handle;
                        blocks = Some(aabb.block);
                    }
                }
            }
        }

        let hit_loc = {
            if closest_handle.is_valid() {
                Some(start + dir * closest_hit)
            } else {
                None
            }
        };
        (closest_hit, Some(closest_handle), hit_loc, blocks)
    }

    pub fn num_collision_objects(&self) -> usize {
        self.collision_objects.handles_to_assets.len()
    }

    pub fn debug_draw(&mut self, renderer: &mut Renderer, config: &Config) {
        for value in &mut self.collision_objects.handles_to_assets.values_mut() {
            match value {
                CollisionShape::Sphere(_s) => {}

                CollisionShape::AABB(aabb) => {
                    let extent_0 = aabb.position
                        + CgVec3::new(-aabb.extents.x, aabb.extents.y, aabb.extents.z);
                    let extent_1 =
                        aabb.position + CgVec3::new(aabb.extents.x, aabb.extents.y, aabb.extents.z);
                    let extent_2 = aabb.position
                        + CgVec3::new(aabb.extents.x, -aabb.extents.y, aabb.extents.z);
                    let extent_3 = aabb.position
                        + CgVec3::new(-aabb.extents.x, -aabb.extents.y, aabb.extents.z);

                    let extent_4 = aabb.position
                        + CgVec3::new(-aabb.extents.x, aabb.extents.y, -aabb.extents.z);
                    let extent_5 = aabb.position
                        + CgVec3::new(aabb.extents.x, aabb.extents.y, -aabb.extents.z);
                    let extent_6 = aabb.position
                        + CgVec3::new(aabb.extents.x, -aabb.extents.y, -aabb.extents.z);
                    let extent_7 = aabb.position
                        + CgVec3::new(-aabb.extents.x, -aabb.extents.y, -aabb.extents.z);

                    let color = CgVec4::new(1.0, 1.0, 0.0, 1.0);

                    renderer.add_line(&extent_0, &extent_1, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_1, &extent_2, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_2, &extent_3, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_3, &extent_0, &color, 0.05, 0.001, config);

                    renderer.add_line(&extent_4, &extent_5, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_5, &extent_6, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_6, &extent_7, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_7, &extent_4, &color, 0.05, 0.001, config);

                    renderer.add_line(&extent_0, &extent_4, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_1, &extent_5, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_2, &extent_6, &color, 0.05, 0.001, config);
                    renderer.add_line(&extent_3, &extent_7, &color, 0.05, 0.001, config);
                }
            }
        }
    }
}
