cargo build --target wasm32-unknown-unknown --release
if errorlevel 1 ( echo cargo build failed & pause & exit /b 1 )

where wasm-bindgen >nul 2>nul
if errorlevel 1 ( echo wasm-bindgen not found - run: cargo install wasm-bindgen-cli --version 0.2.126 & pause & exit /b 1 )

wasm-bindgen --target web --out-dir target/wasm32-unknown-unknown/release target/wasm32-unknown-unknown/release/black_splat_splat_demo.wasm
if errorlevel 1 ( echo wasm-bindgen failed & pause & exit /b 1 )

powershell cp index.html target/wasm32-unknown-unknown/release

rem The .ply and .glb are fetched at runtime from /rust_assets/ in the browser build.
if not exist target\wasm32-unknown-unknown\release\rust_assets mkdir target\wasm32-unknown-unknown\release\rust_assets
powershell cp game_assets/splats/*.ply target/wasm32-unknown-unknown/release/rust_assets
powershell cp game_assets/models/*.glb target/wasm32-unknown-unknown/release/rust_assets

python3 ..\serve.py target/wasm32-unknown-unknown/release
pause
