// mujoco-rs's build script (a dependency, so it always runs before ours)
// downloads MuJoCo into MUJOCO_DOWNLOAD_DIR and statically links mujoco.lib,
// but on Windows the mujoco.dll it depends on at runtime still has to sit
// next to the exe (or on PATH) for the loader to find it. Copy it into the
// profile dir so `cargo run` works without the user setting PATH by hand.
use std::{env, fs, path::Path, path::PathBuf};

fn find_dll(base: &Path) -> Option<PathBuf> {
    for entry in fs::read_dir(base).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let candidate = path.join("bin").join("mujoco.dll");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let Ok(mujoco_dir) = env::var("MUJOCO_DOWNLOAD_DIR") else {
        return;
    };
    let Some(dll_src) = find_dll(Path::new(&mujoco_dir)) else {
        return;
    };

    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out; the exe lives three
    // levels up, directly in target/<profile>/.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        let _ = fs::copy(&dll_src, profile_dir.join("mujoco.dll"));
    }
}
