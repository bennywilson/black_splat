# launcher

Tooling for building and running the samples in [`examples/`](../examples):
a web dashboard for point-and-click use, and a CLI for everyone else. All of
it is cross-platform (Windows/Linux/macOS) — the actual logic is plain Python
in `build.py`; the `.bat`/`.sh` files are thin wrappers around it.

## Web dashboard

Double-click **`launch.bat`** (Windows) or run **`./launch.sh`**
(Linux/macOS), then open <http://localhost:8090/> — the launcher opens it
for you.

Each example gets four buttons:

| Button       | What it does                                                                 |
|--------------|------------------------------------------------------------------------------|
| **Run native** | `cargo run --release` in the example dir — opens a native window.           |
| **Wasm**       | Builds wasm, runs wasm-bindgen, copies `index.html` + assets, serves on a LAN port. Good for the WebGL2 demos (2d/3d). |
| **Tunnel**     | Same wasm build, but served over an HTTPS Cloudflare quick tunnel. Needed for the **splat** demo's WebGPU on phones. The public URL and a scannable QR image appear under the example. |
| **Stop**       | Kills that example's running job.                                           |

Build output streams live into the console pane on the right; click an
example's name (or its tab) to view its log. One job runs per example at a time
— starting a new one stops the old. The page re-attaches to running jobs after
a reload.

Ports: the launcher itself is `:8090`; examples are served on `:8000`, `:8001`,
`:8002` in discovery order.

## CLI

No dashboard needed — same logic, driven from a terminal:

```sh
./build.sh list                # or: build.bat list  /  python3 build.py list
./build.sh run 2d wasm         # build + serve; Ctrl+C to stop
./build.sh run splat tunnel    # build + serve over HTTPS; shows a terminal QR
./build.sh run 3d native       # cargo run --release
./build.sh stop                # kill whatever's on :8090 (the dashboard)
./build.sh stop 8000           # kill whatever's on an example's port
```

`stop` is a manual escape hatch for when a server outlives its terminal
(observed in practice: closing a console window doesn't always deliver a
signal Python can catch). The dashboard also cleans up its child jobs on a
normal exit — `stop` is there for when that doesn't happen.

## How it works

`build.py` discovers every `examples/<name>/` that is a cargo crate (reading
its binary name from `Cargo.toml`) and implements the actual steps — cargo →
wasm-bindgen → copy assets → `serve.py` / `serve_tunnel.py` — once. It's
directly importable (the dashboard's `server.py` does this) and directly
runnable (`python3 build.py ...`). Per-example wasm asset copying lives in
`WASM_ASSETS` in `build.py` (only the splat demo pulls `.ply`/`.glb` from
`/rust_assets/` at runtime).

This replaced three near-identical per-example `build_wasm.bat` /
`build_wasm_tunneled.bat` files, which drifted only in the crate name and
splat's asset copy step. `serve.py`/`serve_tunnel.py` (the no-cache dev
server, and the one that also opens an HTTPS Cloudflare tunnel) live here
too, alongside the code that drives them.

## Requirements

- Python 3.11+ (uses `tomllib`)
- `wasm-bindgen` on PATH for wasm builds (`cargo install wasm-bindgen-cli`)
- `cloudflared` on PATH for the Tunnel action (`scoop install cloudflared`)
- Node (`npx`) optional — only used to render the dashboard's QR image; the
  CLI's terminal QR and the tunnel itself work without it
