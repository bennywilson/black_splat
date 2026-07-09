# launcher

A small local web dashboard for building and running the samples in
[`examples/`](../examples).

## Use it

Double-click **`launch.bat`** (or run `python server.py`), then open
<http://localhost:8090/>. `launch.bat` opens the browser for you.

Each example gets four buttons:

| Button       | What it does                                                                 |
|--------------|------------------------------------------------------------------------------|
| **Run native** | `cargo run --release` in the example dir — opens a native window.           |
| **Wasm**       | Builds wasm, runs wasm-bindgen, copies `index.html` + assets, serves on a LAN port (via `examples/serve.py`). Good for the WebGL2 demos (2d/3d). |
| **Tunnel**     | Same wasm build, but served over an HTTPS Cloudflare quick tunnel (via `examples/serve_tunnel.py`). Needed for the **splat** demo's WebGPU on phones. The public URL and a scannable QR image appear under the example. |
| **Stop**       | Kills that example's running job.                                           |

Build output streams live into the console pane on the right; click an
example's name (or its tab) to view its log. One job runs per example at a time
— starting a new one stops the old. The page re-attaches to running jobs after
a reload.

Ports: the launcher itself is `:8090`; examples are served on `:8000`, `:8001`,
`:8002` in discovery order.

## How it works

`server.py` discovers every `examples/<name>/` that is a cargo crate (reading
its binary name from `Cargo.toml`) and drives the same steps the old
`build_wasm*.bat` files did — cargo → wasm-bindgen → copy assets →
`serve.py` / `serve_tunnel.py` — but keeps them in one place. Per-example wasm
asset copying lives in `WASM_ASSETS` in `server.py` (only the splat demo pulls
`.ply`/`.glb` from `/rust_assets/` at runtime).

The old per-example `.bat` files still work and are untouched.

## Requirements

- Python 3.11+ (uses `tomllib`)
- `wasm-bindgen` on PATH for wasm builds (`cargo install wasm-bindgen-cli`)
- `cloudflared` on PATH for the Tunnel action (`scoop install cloudflared`)
