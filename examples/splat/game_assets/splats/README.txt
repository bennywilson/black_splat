Drop a 3D Gaussian Splat .ply file here named:

    point_cloud.ply

It must be a binary-little-endian PLY in the standard 3DGS training format
(properties: x y z, scale_0..2, rot_0..3, opacity, f_dc_0..2, f_rest_0..44).
These are the files produced by inria/gaussian-splatting and most trainers
(the per-iteration point_cloud.ply).

To use a different filename or location, edit SPLAT_PLY_PATH in
examples/splat/src/example_game.rs.

Native run:   from examples/splat, `cargo run --release`
Browser run:  from examples/splat, run build_wasm.bat (the .ply is copied into
              the served rust_assets/ folder automatically).
