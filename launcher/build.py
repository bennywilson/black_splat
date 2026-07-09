#!/usr/bin/env python3
"""Single source of truth for building/running/serving the black_splat samples.

Used two ways:
  * As a library, imported by server.py (the web dashboard).
  * As a CLI, so the dashboard isn't required:
        python3 build.py list
        python3 build.py run <example> <native|wasm|tunnel>
        python3 build.py stop [port]

This replaces the old per-example build_wasm.bat / build_wasm_tunneled.bat --
they were identical except for the crate name and (for splat) which runtime
assets to copy, so this is one parameterised implementation instead of three
drifting copies. It's plain cross-platform Python (no batch-file-only logic),
so the same code runs on Windows, Linux, and macOS.
"""
import argparse
import glob as globmod
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
        ("game_assets/models/*.glb", "rust_assets"),
    ],
}

URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
PY = sys.executable or "python3"


def discover_examples():
    """Every examples/<name>/ that is a cargo crate, with its binary name."""
    out = []
    for name in sorted(os.listdir(EXAMPLES_DIR)):
        cargo = os.path.join(EXAMPLES_DIR, name, "Cargo.toml")
        if not os.path.isfile(cargo):
            continue
        with open(cargo, "rb") as f:
            meta = tomllib.load(f)
        crate = meta.get("package", {}).get("name")
        if crate:
            out.append({"name": name, "crate": crate})
    return out


EXAMPLES = discover_examples()
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


def stream_step(cmd, cwd, log):
    """Run a subprocess to completion, streaming merged stdout/stderr to
    log(line). Returns the exit code (or 1 if the executable is missing)."""
    log(f"$ {' '.join(cmd)}")
    try:
        proc = _popen(cmd, cwd)
    except FileNotFoundError:
        log(f"error: '{cmd[0]}' not found on PATH")
        return 1
    for line in proc.stdout:
        log(line.rstrip("\n"))
    proc.wait()
    return proc.returncode


def build_wasm(example, log):
    """cargo build (wasm) -> wasm-bindgen -> copy index.html + assets.
    Returns the absolute output dir on success, or None on failure."""
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    crate = CRATES[example]
    rel = os.path.join("target", "wasm32-unknown-unknown", "release")
    out_dir = os.path.join(ex_dir, rel)

    if stream_step(["cargo", "build", "--target", "wasm32-unknown-unknown", "--release"], ex_dir, log):
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

    for pattern, subdir in WASM_ASSETS.get(example, []):
        dest = os.path.join(out_dir, subdir)
        os.makedirs(dest, exist_ok=True)
        matches = globmod.glob(os.path.join(ex_dir, pattern))
        for src in matches:
            shutil.copy(src, dest)
        log(f"copied {len(matches)} file(s): {pattern} -> {subdir}/")

    return out_dir


def run_native(example, log, on_proc=None):
    """`cargo run --release` in the example dir. Blocks until it exits."""
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    cmd = ["cargo", "run", "--release"]
    log(f"$ {' '.join(cmd)}")
    proc = _popen(cmd, ex_dir)
    if on_proc:
        on_proc(proc)
    for line in proc.stdout:
        log(line.rstrip("\n"))
    proc.wait()
    return proc.returncode


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
    proc_holder = {}
    try:
        if action == "native":
            code = run_native(example, print, on_proc=lambda p: proc_holder.update(proc=p))
        else:
            out_dir = build_wasm(example, print)
            if out_dir is None:
                sys.exit(1)
            code = run_serve(
                example, out_dir, tunneled=(action == "tunnel"), log=print,
                on_proc=lambda p: proc_holder.update(proc=p),
                on_url=lambda label, url: print(f"--- {label}: {url}"),
            )
        sys.exit(code or 0)
    except KeyboardInterrupt:
        kill_tree(proc_holder.get("proc"))
        print("\nstopped.")


if __name__ == "__main__":
    _cli()
