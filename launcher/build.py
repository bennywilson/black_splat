#!/usr/bin/env python3
"""Single source of truth for building/running/serving the black_splat samples.

Used two ways:
  * As a library, imported by server.py (the web dashboard).
  * As a CLI, so the dashboard isn't required:
        python3 build.py list
        python3 build.py run <example> <native|wasm|tunnel>
        python3 build.py stop [port]
"""
import argparse
import glob as globmod
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES_DIR = os.path.join(ROOT, "examples")

DEFAULT_DASHBOARD_PORT = 8090
EXAMPLE_BASE_PORT = 8000  # examples get 8000, 8001, 8002, ... by discovery order

# Which extra runtime assets each example's wasm build fetches from /rust_assets/.
# Only the splat demo pulls assets at runtime; the 2d/3d demos need nothing but
# index.html. Keyed by example folder name.
WASM_ASSETS = {
    "splat": [
        ("game_assets/splats/*.ply", "rust_assets"),
        # ** so models nested in per-asset folders (e.g. Barrel/barrel_hq.glb)
        # are found too, matching the native editor's recursive model scan.
        ("game_assets/models/**/*.glb", "rust_assets"),
        # Same per-asset folders also hold the model's own texture maps
        # (e.g. Barrel/barrel_stove_diff_4k.jpg) -- without these the wasm
        # loader's fetch() 404s and the model falls back to the checkerboard.
        ("game_assets/models/**/*.png", "rust_assets"),
        ("game_assets/models/**/*.jpg", "rust_assets"),
        ("game_assets/models/**/*.jpeg", "rust_assets"),
        ("game_assets/fx/*.png", "rust_assets"),
        # DeepMind's official @mujoco/mujoco wasm build, for MujocoActor (see
        # src/mujoco.rs's wasm_bridge module), plus the sample scene it can
        # point at out of the box.
        ("../../node_modules/@mujoco/mujoco/mujoco.js", "mujoco_web"),
        ("../../node_modules/@mujoco/mujoco/mujoco.wasm", "mujoco_web"),
        ("game_assets/*.xml", "rust_assets"),
    ],
    # DeepMind's official @mujoco/mujoco wasm build: a second, independent
    # wasm module index.html loads alongside the app's own (see
    # src/mujoco.rs's wasm_bridge module for the JS<->Rust handoff), plus the
    # MJCF scene MujocoScene::load fetches at runtime.
    "mujoco_test": [
        ("../../node_modules/@mujoco/mujoco/mujoco.js", "mujoco_web"),
        ("../../node_modules/@mujoco/mujoco/mujoco.wasm", "mujoco_web"),
        ("game_assets/*.xml", "rust_assets"),
    ],
}

# black_splat/engine_assets is the shared engine asset location: served for
# EVERY example (paths relative to the repo root, not the example) so any
# project can load an engine asset at runtime on the web, not just the ones the
# engine bakes in via include_bytes!.  Keeps engine assets available everywhere
# without per-project setup.
ENGINE_WASM_ASSETS = [
    ("engine_assets/textures/*.png", "rust_assets"),
    ("engine_assets/textures/*.jpg", "rust_assets"),
]

URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
PY = sys.executable or "python3"

# Every server we start stamps this header on its responses, so before we
# reclaim a busy port we can tell one of our own stale instances (safe to kill)
# apart from an unrelated app that merely happens to use the same port.
IDENT_HEADER = "X-Black-Splat"


def discover_example_projects():
    """Every examples/<name>/ that is a cargo crate, with its binary name."""
    out = []
    for name in sorted(os.listdir(EXAMPLES_DIR)):
        cargo = os.path.join(EXAMPLES_DIR, name, "Cargo.toml")
        if not os.path.isfile(cargo):
            continue

        """with is a context manager which cleans up resources properly"""
        with open(cargo, "rb") as f:
            meta = tomllib.load(f)

        crate = meta.get("package", {}).get("name")
        if crate:
            thumb = next(
                (fname for fname in ("thumbnail.jpg", "thumbnail.png", "thumbnail.svg")
                 if os.path.isfile(os.path.join(EXAMPLES_DIR, name, fname))),
                None,
            )
            out.append({"name": name, "crate": crate, "thumbnail": thumb})
    return out


EXAMPLES = discover_example_projects()
PORTS = {e["name"]: EXAMPLE_BASE_PORT + i for i, e in enumerate(EXAMPLES)}
CRATES = {e["name"]: e["crate"] for e in EXAMPLES}


def _popen(cmd, cwd, env=None):
    """subprocess.Popen preconfigured for line-streamed UTF-8 text, started in
    its own process group/job so kill_tree can take out the whole subtree --
    e.g. cloudflared is a grandchild of the tunnel job."""
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=env, **kwargs,
    )


def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _cargo_env():
    """Env that forces cargo to draw its progress bar even though our stdout is
    a pipe (cargo hides it when it doesn't see a terminal -- which is why a
    build looked dead between the sparse 'Compiling X' milestone lines). Colour
    off so no ANSI junk reaches the web <pre> or a plain CLI."""
    return {
        **os.environ,
        "CARGO_TERM_PROGRESS_WHEN": "always",
        "CARGO_TERM_PROGRESS_WIDTH": "100",
        "CARGO_TERM_COLOR": "never",
    }


def _stream(proc, on_line):
    """Stream a process's merged stdout to on_line(text, transient).

    Reads raw bytes (not text mode's universal-newline translation, which would
    fold '\\r' into '\\n' and erase the distinction) and splits on both:
      * '\\n'  -> a committed line            -> on_line(text, transient=False)
      * '\\r'  -> an in-place redraw (e.g. the cargo progress bar, which repaints
                 with a bare carriage return and no newline) -> transient=True
    A '\\r\\n' pair counts as one committed line. Splitting only ever happens on
    the ASCII bytes 0x0A/0x0D, which never occur inside a UTF-8 multibyte
    sequence, so decoding each segment on its own is safe."""
    stream = getattr(proc.stdout, "buffer", proc.stdout)
    buf = bytearray()
    seg = None  # bytes ended by a lone '\r', held one byte to disambiguate \r\n
    while True:
        chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
        if not chunk:
            break
        for b in chunk:
            if seg is not None:
                if b == 0x0A:  # the held '\r' was really a '\r\n' line ending
                    on_line(seg.decode("utf-8", "replace"), False)
                    seg = None
                    continue
                on_line(seg.decode("utf-8", "replace"), True)  # lone '\r': a redraw
                seg = None
                # fall through to handle b as the start of the next segment
            if b == 0x0A:
                on_line(buf.decode("utf-8", "replace"), False)
                buf.clear()
            elif b == 0x0D:
                seg = bytes(buf)
                buf.clear()
            else:
                buf.append(b)
    if seg is not None:
        on_line(seg.decode("utf-8", "replace"), False)
    if buf:
        on_line(buf.decode("utf-8", "replace"), False)


def stream_step(cmd, cwd, log, env=None):
    """Run a subprocess to completion, streaming merged stdout/stderr to
    log(line[, transient]). Returns the exit code (or 1 if the executable is
    missing)."""
    log(f"$ {' '.join(cmd)}")
    try:
        proc = _popen(cmd, cwd, env=env)
    except FileNotFoundError:
        log(f"error: '{cmd[0]}' not found on PATH")
        return 1
    _stream(proc, log)
    proc.wait()
    return proc.returncode


def build_wasm(example, log):
    """cargo build (wasm) -> wasm-bindgen -> copy index.html + assets.
    Returns the absolute output dir on success, or None on failure."""
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    crate = CRATES[example]
    rel = os.path.join("target", "wasm32-unknown-unknown", "release")
    out_dir = os.path.join(ex_dir, rel)

    if stream_step(["cargo", "build", "--target", "wasm32-unknown-unknown", "--release"],
                   ex_dir, log, env=_cargo_env()):
        log("cargo build failed")
        return None

    if shutil.which("wasm-bindgen") is None:
        log("wasm-bindgen not found - run: cargo install wasm-bindgen-cli --version 0.2.126")
        return None

    wasm_in = os.path.join(rel, f"{crate}.wasm")
    if stream_step(["wasm-bindgen", "--target", "web", "--out-dir", rel, wasm_in], ex_dir, log):
        log("wasm-bindgen failed")
        return None

    shutil.copy(os.path.join(ex_dir, "index.html"), out_dir)
    log("copied index.html")

    # The browser can't enumerate a served directory, so alongside copying the
    # runtime assets we record the models/textures in a manifest the wasm
    # resource browser reads to populate its pickers (see resource_library.rs).
    manifest = {"models": [], "textures": []}
    for pattern, subdir in WASM_ASSETS.get(example, []):
        dest = os.path.join(out_dir, subdir)
        os.makedirs(dest, exist_ok=True)
        # recursive=True so a `**` in the pattern spans subfolders; the copy
        # flattens everything into rust_assets/ (basenames must stay unique).
        matches = globmod.glob(os.path.join(ex_dir, pattern), recursive=True)
        for src in matches:
            shutil.copy(src, dest)
        log(f"copied {len(matches)} file(s): {pattern} -> {subdir}/")

        # Manifest paths are the file's real path relative to the example dir
        # (e.g. game_assets/models/Barrel/barrel_hq.glb) -- the same string the
        # app loads a model by, which its loader reduces to the served basename.
        # Classified by extension, not folder: a model's own texture maps (e.g.
        # Barrel/barrel_stove_diff_4k.jpg) live right next to the .glb that
        # references them, and are listed as browsable textures too, matching
        # native's scan_textures which recurses into game_assets/models/.
        for src in matches:
            rel = os.path.relpath(src, ex_dir).replace("\\", "/")
            ext = os.path.splitext(rel)[1].lower()
            if ext in (".glb", ".gltf"):
                manifest["models"].append(rel)
            elif ext in (".png", ".jpg", ".jpeg"):
                manifest["textures"].append(rel)

    # Shared engine assets, served for every example (source paths are relative
    # to the repo root).  Not added to the manifest -- these are engine
    # internals a project loads by name, not browsable project resources.
    for pattern, subdir in ENGINE_WASM_ASSETS:
        dest = os.path.join(out_dir, subdir)
        os.makedirs(dest, exist_ok=True)
        matches = globmod.glob(os.path.join(ROOT, pattern), recursive=True)
        for src in matches:
            shutil.copy(src, dest)
        if matches:
            log(f"copied {len(matches)} engine asset(s): {pattern} -> {subdir}/")

    if WASM_ASSETS.get(example):
        manifest_dir = os.path.join(out_dir, "rust_assets")
        os.makedirs(manifest_dir, exist_ok=True)
        with open(os.path.join(manifest_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        log(f"wrote manifest.json ({len(manifest['models'])} models, "
            f"{len(manifest['textures'])} textures)")

    return out_dir


def run_native(example, log, on_proc=None):
    """`cargo run --release` in the example dir. Blocks until it exits."""
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    cmd = ["cargo", "run", "--release"]
    log(f"$ {' '.join(cmd)}")
    proc = _popen(cmd, ex_dir, env=_cargo_env())
    if on_proc:
        on_proc(proc)
    _stream(proc, log)
    proc.wait()
    return proc.returncode


def clean_wasm(example, log):
    """`cargo clean` for the wasm target. Returns 0 on success."""
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    cmd = ["cargo", "clean", "--target", "wasm32-unknown-unknown"]
    log(f"$ {' '.join(cmd)}")
    return stream_step(cmd, ex_dir, log, env=_cargo_env())


def run_serve(example, out_dir, tunneled, log, on_proc=None, on_url=None, suppress_terminal_qr=False):
    """Launch serve.py / serve_tunnel.py (this same launcher/ dir) as a
    long-lived process and stream it until it exits or is killed.
    on_url(label, url) fires once for the local URL immediately, and again
    for the public tunnel URL once cloudflared reports it."""
    port = PORTS[example]
    script = os.path.join(HERE, "serve_tunnel.py" if tunneled else "serve.py")
    cmd = [PY, script, out_dir, str(port)]
    log(f"$ {' '.join(cmd)}")
    # PYTHONUNBUFFERED matters most: with stdout piped (not a real terminal),
    # Python block-buffers instead of line-buffering, so a slow-but-fine run
    # looks identical to a silently-dead one until unbuffered.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1"}
    if suppress_terminal_qr:
        # The web dashboard renders its own scannable SVG QR on the card
        # instead (see qr_svg) -- node-qrcode's terminal QR is drawn purely
        # with ANSI colour, which can't render in a browser <pre>. CLI users
        # still get the real terminal QR.
        env["LAUNCHER_QR"] = "0"
    proc = _popen(cmd, HERE, env=env)
    if on_proc:
        on_proc(proc)
    if on_url:
        on_url("Local", f"http://localhost:{port}/")
    public_seen = False
    for line in proc.stdout:
        log(line.rstrip("\n"))
        if tunneled and on_url and not public_seen:
            m = URL_RE.search(line)
            if m:
                public_seen = True
                url = m.group(0)
                host = url.removeprefix("https://")
                log(f"waiting for {host} to resolve publicly (fresh tunnel hostnames take a few seconds) ...")
                if wait_for_dns(host):
                    log("DNS is live; URL is safe to open/scan.")
                else:
                    log("warning: hostname still not resolving after 60s -- the URL may not work yet.")
                on_url("Public (HTTPS)", url)
    proc.wait()
    return proc.returncode


def _dns_query_direct(hostname, server="1.1.1.1"):
    """One A-record lookup sent straight to `server` over UDP:53, bypassing
    the local resolver entirely. Returns True (resolves), False (NXDOMAIN /
    no records yet), or None (couldn't ask -- network blocked the query).
    Hand-rolled wire format because the stdlib's getaddrinfo can only use the
    system resolver, and DoH via urllib trips over missing intermediate certs
    on the Windows Store Python."""
    qid = os.urandom(2)
    # header: id, flags=0x0100 (recursion desired), 1 question, 0 answers/etc.
    packet = qid + b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
    for label in hostname.encode("ascii").split(b"."):
        packet += bytes([len(label)]) + label
    packet += b"\x00\x00\x01\x00\x01"  # root, qtype=A, qclass=IN
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(4)
            s.sendto(packet, (server, 53))
            resp = s.recv(512)
    except OSError:
        return None
    if len(resp) < 12 or resp[:2] != qid:
        return None
    rcode = resp[3] & 0x0F
    ancount = int.from_bytes(resp[6:8], "big")
    return rcode == 0 and ancount > 0


def wait_for_dns(hostname, timeout=60):
    """Wait until `hostname` resolves publicly. Quick-tunnel hostnames are
    minted seconds before use, and a resolver that looks one up too early
    caches the miss -- observed in practice: a Fios gateway holding NXDOMAIN
    for the zone's 30-minute negative TTL, which kills the tunnel URL on every
    device using that router. So don't advertise a URL/QR until the name is
    actually live. Queries 1.1.1.1 directly, which neither consults nor primes
    the local resolver."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = _dns_query_direct(hostname)
        if result:
            return True
        if result is None:
            # Can't reach an outside resolver (e.g. UDP/53 egress blocked):
            # we have no signal either way, so don't sit on the URL for the
            # full timeout -- hand it over unverified.
            return True
        time.sleep(2)
    return False


def qr_svg(url):
    """Render `url` as an SVG QR (white bg, black modules) via node-qrcode.
    Returns the SVG markup, or None if Node/npx isn't available."""
    if shutil.which("npx") is None:
        return None
    out = os.path.join(tempfile.gettempdir(), f"black_splat_qr_{os.getpid()}.svg")
    try:
        cmd = f'npx --yes qrcode -t svg -o "{out}" "{url}"'
        subprocess.run(cmd if sys.platform == "win32" else
                       ["npx", "--yes", "qrcode", "-t", "svg", "-o", out, url],
                       shell=(sys.platform == "win32"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=60)
        with open(out, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def stop_port(port):
    """Kill whatever process is listening on `port`. Cross-platform escape
    hatch for a launcher/job that outlived its terminal -- e.g. the dashboard
    itself, if closing its window didn't take it down cleanly."""
    if sys.platform == "win32":
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        pids = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" and f":{port}" in parts[1] and "LISTENING" in line:
                pids.add(parts[-1])
    else:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True).stdout
        pids = {p for p in out.split() if p}

    if not pids:
        print(f"Nothing was listening on port {port}.")
        return False
    for pid in pids:
        print(f"Stopping PID {pid} on port {port} ...")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", pid],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["kill", "-9", pid])
    return True


def _port_listening(port, host="127.0.0.1"):
    """True if something is accepting TCP connections on host:port right now.
    (A port merely lingering in TIME_WAIT after a restart is NOT listening, so
    this reads as free -- the bind retry rides that window out.)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _served_by_us(port):
    """Whether the HTTP server on `port` is one of ours, told by the
    IDENT_HEADER it stamps on every response (present even on its 404/501).
    True = ours, False = answered but not ours, None = didn't answer as HTTP."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=1.0) as r:
            headers = r.headers
    except urllib.error.HTTPError as e:
        headers = e.headers  # a 404/501 body still carries our header
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return bool(headers.get(IDENT_HEADER))


def free_port_if_ours(port, log=print):
    """Make `port` bindable, reclaiming it *only* from a stale black_splat
    server (one that answers with IDENT_HEADER). Returns True if the port is
    now free (or was), False if it's held by a process that isn't ours -- which
    is left untouched. Never kills an unidentified process."""
    if not _port_listening(port):
        return True  # free, or a TIME_WAIT remnant the bind retry will outlast
    if _served_by_us(port) is not True:
        log(f"port {port} is in use by another process (not this launcher) -- "
            f"leaving it alone.")
        return False
    log(f"port {port} held by a stale launcher/server -- reclaiming it.")
    stop_port(port)
    for _ in range(20):  # wait for the OS to actually drop the listener
        if not _port_listening(port):
            return True
        time.sleep(0.15)
    return not _port_listening(port)


def bind_or_reclaim(make_server, port, log=print):
    """Build a server that binds `port`, first reclaiming the port from a stale
    black_splat instance and riding out the brief TIME_WAIT window left by a
    just-stopped server. `make_server` is a no-arg callable that binds and
    returns the server. Raises the final OSError if the port stays unavailable
    (e.g. it's held by an unrelated app -- reported, not fought over)."""
    last = None
    for attempt in range(12):  # ~2s of retries across a TIME_WAIT window
        try:
            return make_server()
        except OSError as e:
            last = e
            if attempt == 0 and not free_port_if_ours(port, log):
                raise  # not ours -- don't fight over it
            time.sleep(0.15)
    raise last


def _cli():
    # When stdout is redirected/piped (not a real terminal) Python
    # block-buffers by default, so our own print()s could sit unflushed the
    # same way the child processes' did before we fixed that -- force line
    # buffering so `build.py run ... | tee log` etc. shows output live.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Build/run a black_splat sample without the web dashboard.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list discovered examples")

    run_p = sub.add_parser("run", help="build/run one example in the foreground (Ctrl+C to stop)")
    run_p.add_argument("example", choices=sorted(PORTS))
    run_p.add_argument("action", choices=["native", "wasm", "tunnel"])

    stop_p = sub.add_parser("stop", help="kill whatever is listening on a port")
    stop_p.add_argument("port", nargs="?", type=int, default=DEFAULT_DASHBOARD_PORT,
                         help=f"default: {DEFAULT_DASHBOARD_PORT} (the dashboard)")

    args = parser.parse_args()

    if args.cmd == "list":
        for e in EXAMPLES:
            print(f"{e['name']:10} {e['crate']:28} :{PORTS[e['name']]}")
        return

    if args.cmd == "stop":
        stop_port(args.port)
        return

    example, action = args.example, args.action

    # Render a transient (\r) redraw in place; move off it with a newline before
    # the next committed line, so the progress bar behaves like it does in a
    # normal cargo run instead of scrolling every frame.
    pending_cr = [False]

    def clog(line, transient=False):
        if transient:
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            pending_cr[0] = True
        else:
            if pending_cr[0]:
                sys.stdout.write("\n")
                pending_cr[0] = False
            print(line)

    proc_holder = {}
    try:
        if action == "native":
            code = run_native(example, clog, on_proc=lambda p: proc_holder.update(proc=p))
        else:
            out_dir = build_wasm(example, clog)
            if out_dir is None:
                sys.exit(1)
            code = run_serve(
                example, out_dir, tunneled=(action == "tunnel"), log=clog,
                on_proc=lambda p: proc_holder.update(proc=p),
                on_url=lambda label, url: print(f"--- {label}: {url}"),
            )
        sys.exit(code or 0)
    except KeyboardInterrupt:
        kill_tree(proc_holder.get("proc"))
        print("\nstopped.")


if __name__ == "__main__":
    _cli()
