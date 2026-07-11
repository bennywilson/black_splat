#[allow(unused_macros)]
#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}
#[allow(unused_macros)]
#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! log {
    ( $ ( $t:tt )* ) => {
        println!( $( $t )* );
    };
}

/// Polls `fut` once with a no-op waker, returning its output if it is already
/// ready (or `None` if it would suspend).  For futures that never actually
/// await real async work -- e.g. glb parsing on wasm, which does no I/O -- this
/// drives them to completion synchronously, letting the non-async frame tick
/// register a runtime-imported model without blocking (impossible on wasm).
#[cfg(target_arch = "wasm32")]
pub fn now_or_never<F: std::future::Future>(fut: F) -> Option<F::Output> {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn noop_raw_waker() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            noop_raw_waker()
        }
        let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), vtable)
    }

    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(fut);
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}

pub fn random_f32(min_val: f32, max_val: f32) -> f32 {
    let mut buf: [u8; 4] = [0, 0, 0, 0];
    let _ = getrandom::getrandom(&mut buf);
    let mut t = buf[0] as u32;
    t += (buf[1] as u32) << 8;
    t += (buf[2] as u32) << 16;
    t += (buf[3] as u32) << 24;
    let t = t as f32 / u32::MAX as f32;
    min_val + (max_val - min_val) * t
}

pub fn random_u32(min_val: u32, max_val: u32) -> u32 {
    let mut buf: [u8; 4] = [0, 0, 0, 0];
    let _ = getrandom::getrandom(&mut buf);
    let mut t = buf[0] as u32;
    t += (buf[1] as u32) << 8;
    t += (buf[2] as u32) << 16;
    t += (buf[3] as u32) << 24;
    let dif = (max_val - min_val) + 1;
    min_val + (t % dif)
}

pub fn random_vec3(min_vec: CgVec3, max_vec: CgVec3) -> CgVec3 {
    let x = random_f32(min_vec.x, max_vec.x);
    let y = random_f32(min_vec.y, max_vec.y);
    let z = random_f32(min_vec.z, max_vec.z);
    CgVec3::new(x, y, z)
}

pub fn random_vec4(min_vec: CgVec4, max_vec: CgVec4) -> CgVec4 {
    let x = random_f32(min_vec.x, max_vec.x);
    let y = random_f32(min_vec.y, max_vec.y);
    let z = random_f32(min_vec.z, max_vec.z);
    let w = random_f32(min_vec.w, max_vec.w);
    CgVec4::new(x, y, z, w)
}

#[cfg(target_arch = "wasm32")]
#[macro_export]
macro_rules! PERF_SCOPE {
    ($label:literal) => {};
}

#[cfg(not(target_arch = "wasm32"))]
#[macro_export]
macro_rules! PERF_SCOPE {
    ($label:literal) => {
        tracy_full::zone!($label);
    };
}

#[macro_export]
macro_rules! make_handle {
    ($asset_type:ident, $handle_type:ident, $mapping_type:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $handle_type {
            index: u32,
        }

        #[allow(dead_code)]
        impl $handle_type {
            fn is_valid(&self) -> bool {
                self.index != u32::MAX
            }
            pub fn make_invalid() -> $handle_type {
                $handle_type { index: u32::MAX }
            }
        }

        #[allow(dead_code)]
        pub struct $mapping_type {
            names_to_handles: HashMap<String, $handle_type>,
            handles_to_assets: HashMap<$handle_type, $asset_type>,
            next_handle: $handle_type,
        }

        impl Default for $mapping_type {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $mapping_type {
            pub fn new() -> Self {
                let names_to_handles = HashMap::<String, $handle_type>::new();
                let handles_to_assets = HashMap::<$handle_type, $asset_type>::new();
                let next_handle = $handle_type { index: u32::MAX };
                $mapping_type {
                    names_to_handles,
                    handles_to_assets,
                    next_handle,
                }
            }

            /// Name -> handle view of every loaded asset, for editor
            /// resource lists.
            pub fn get_names_to_handles(&self) -> &HashMap<String, $handle_type> {
                &self.names_to_handles
            }
        }
    };
}

pub type CgVec2 = cgmath::Vector2<f32>;

pub type CgVec3 = cgmath::Vector3<f32>;
pub const CG_VEC3_ZERO: CgVec3 = CgVec3::new(0.0, 0.0, 0.0);
pub const CG_VEC3_ONE: CgVec3 = CgVec3::new(1.0, 1.0, 1.0);
pub const CG_VEC3_UP: CgVec3 = CgVec3::new(0.0, 1.0, 0.0);

pub type CgVec4 = cgmath::Vector4<f32>;
pub const CG_VEC4_ZERO: CgVec4 = CgVec4::new(0.0, 0.0, 0.0, 0.0);
pub const CG_VEC4_ONE: CgVec4 = CgVec4::new(1.0, 1.0, 1.0, 1.0);

pub type CgPoint = cgmath::Point3<f32>;
pub const CG_POINT_ZERO: CgPoint = CgPoint::new(0.0, 0.0, 0.0);

pub type CgQuat = cgmath::Quaternion<f32>;
pub const CG_QUAT_IDENT: CgQuat = CgQuat::new(0.0, 0.0, 0.0, 1.0);

pub type CgMat3 = cgmath::Matrix3<f32>;
pub const CG_MAT3_IDENT: CgMat3 = CgMat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

pub type CgMat4 = cgmath::Matrix4<f32>;
pub const CG_MAT4_IDENT: CgMat4 = CgMat4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
);

pub fn cgmat4_to_cgmat3(mat4: &CgMat4) -> CgMat3 {
    CgMat3::new(
        mat4.x.x, mat4.x.y, mat4.x.z, mat4.y.x, mat4.y.y, mat4.y.z, mat4.z.x, mat4.z.y, mat4.z.z,
    )
}

pub fn cgvec3_remove_y(vec: CgVec3) -> CgVec2 {
    CgVec2::new(vec.x, vec.z)
}

pub fn lerp(op1: f32, op2: f32, time: f32) -> f32 {
    (op2 - op1) * time + op1
}
