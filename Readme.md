# kbEngine3

A simple sprite renderer built with **Rust** and **wgpu 0.19.3** (latest at the time of this project).

All sprites are drawn with a single `draw_indexed` call using instancing.

## Setup

1. Install Rust (includes cargo) via [rustup](https://rustup.rs).
2. Verify the toolchain:
   ```sh
   cargo --version
   ```

For browser (wasm) builds, also:

3. Add the wasm target:
   ```sh
   rustup target add wasm32-unknown-unknown
   ```
4. Install the `wasm-bindgen` CLI (must match the `wasm-bindgen` crate version in `Cargo.lock`):
   ```sh
   cargo install wasm-bindgen-cli --version 0.2.126
   ```
5. Install [Python 3](https://www.python.org/) to serve the build locally.

## Building and Debugging

Run the examples from a command prompt:

| Demo | Directory | Command |
| --- | --- | --- |
| 2D demo #1 | this directory | `cargo run` |
| 2D demo #2 | `kbEngine3/Examples/2D/` | `cargo run` |
| 3D demo #3 | `kbEngine3/Examples/3D/` | `cargo run` |

To run a browser build:

1. Run `kbEngine3/Examples/3D/build_wasm.bat` or `kbEngine3/Examples/2D/build_wasm.bat`.
2. Navigate to <http://127.0.0.1:8000> in a web browser.
3. You may need to close the browser or use an incognito window to avoid old cached builds.

> **Note:** Set your working directory to the root of the `kbEngine3` folder when debugging in Visual Studio, running from RenderDoc, etc.

## Config File

Each example uses a config file that controls several parameters. There is an example at `GameAssets/game_config.txt`:

```json
{
    "enemy_spawn_delay": 0.3,
    "enemy_move_speed": 1.0,
    "max_instances": 2000,

    "window_width": 1920,
    "window_height": 1080,

    "_comment": "Valid values for 'graphics_back_end' are default, vulkan, or dx12",
    "graphics_back_end": "default",

    "_comment2": "Valid values for 'graphics_power_pref' are default, low, and high",
    "graphics_power_pref": "default",

    "_comment3": "Valid values for 'vsync' are true and false",
    "vsync": true
}
```

## Resources

- [Project repository](https://github.com/bennywilson/kbEngine3)
- [The Rust Book](https://doc.rust-lang.org/book/ch01-00-getting-started.html)
- [Learn wgpu](https://sotrh.github.io/learn-wgpu/#what-is-wgpu)
- [Vulkan 1.3 specification](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html)
- [Tracy profiler](https://github.com/wolfpld/tracy)

---

Benny Wilson
bennywilson@benny-wilson.com
