To change the preloaded set, edit SPLAT_PLY_PATHS in
examples/splat/src/example_game.rs.  You can also load a .ply at runtime with
[L] (or tapping the top-middle of the screen on touch devices), which opens a
file picker -- on the web build the file is read locally in the browser and
never uploaded anywhere.

It must be a binary-little-endian PLY in the standard 3DGS training format
(properties: x y z, scale_0..2, rot_0..3, opacity, f_dc_0..2, f_rest_0..44).
These are the files produced by inria/gaussian-splatting and most trainers
(the per-iteration point_cloud.ply).

Native run:   from examples/splat, `cargo run --release`
Browser run:  from examples/splat, run build_wasm.bat (the .ply is copied into
              the served rust_assets/ folder automatically).
