#!/usr/bin/env python3
"""Control server for the black_splat sample launcher.

A tiny local web dashboard so you can build/run the examples with buttons
instead of memorising terminal incantations.  It shells out to exactly the same
tools the old build_wasm*.bat files did (cargo -> wasm-bindgen -> copy assets ->
serve.py / serve_tunnel.py), but keeps the steps in one place and streams their
output back to the page over Server-Sent Events.

Start it with launch.bat (or `python server.py`) and open http://localhost:8090.

Per example you get:
  * Native   -> `cargo run --release`            (opens a native window)
  * Wasm     -> build wasm + serve on a LAN port (WebGL2 demos: 2d/3d)
  * Tunnel   -> build wasm + serve over HTTPS via cloudflared (needed for the
                splat demo's WebGPU on phones); surfaces the public URL
  * Stop     -> kill the running job for that example

Only one job runs per example at a time; starting a new one stops the old.
"""
import atexit
import functools
import glob as globmod
import http.server
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES_DIR = os.path.join(ROOT, "examples")

CONTROL_PORT = 8090
EXAMPLE_BASE_PORT = 8000  # examples get 8000, 8001, 8002, ... by discovery order

# Which extra runtime assets each example's wasm build fetches from /rust_assets/.
# Only the splat demo pulls assets at runtime (see its old build_wasm.bat); the
# 2d/3d demos need nothing but index.html.  Keyed by example folder name.
WASM_ASSETS = {
    "splat": [
        ("game_assets/splats/*.ply", "rust_assets"),
        ("game_assets/models/*.glb", "rust_assets"),
    ],
}

URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
# cloudflared colorises its logs; those ANSI/OSC escapes render as junk in the
# browser <pre>, so strip them (CSI ...m colour codes and OSC ...BEL sequences).
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
PY = sys.executable or "python"


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
        if not crate:
            continue
        out.append({"name": name, "crate": crate})
    return out


EXAMPLES = discover_examples()
PORTS = {e["name"]: EXAMPLE_BASE_PORT + i for i, e in enumerate(EXAMPLES)}
CRATES = {e["name"]: e["crate"] for e in EXAMPLES}


class Job:
    """A running build/run for one example, with an append-only event log that
    late SSE subscribers replay in full (so a page reload re-attaches cleanly)."""

    def __init__(self, example, action):
        self.example = example
        self.action = action
        self.events = []            # list of dicts: {type, ...}
        self.subs = []              # list of queue.Queue for live subscribers
        self.lock = threading.Lock()
        self.proc = None            # the long-lived (serve) process, if any
        self.status = "running"     # running | ok | failed | stopped
        self.stop_requested = False

    def emit(self, ev):
        with self.lock:
            self.events.append(ev)
            for q in self.subs:
                q.put(ev)

    def log(self, line):
        self.emit({"type": "log", "line": ANSI_RE.sub("", line).rstrip("\r\n")})

    def set_status(self, status):
        self.status = status
        self.emit({"type": "status", "status": status})

    def url(self, label, url):
        self.emit({"type": "url", "label": label, "url": url})

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            history = list(self.events)
            self.subs.append(q)
        return q, history

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


JOBS = {}          # example name -> Job
JOBS_LOCK = threading.Lock()


def kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        proc.terminate()


def stream_step(job, cmd, cwd):
    """Run a subprocess to completion, streaming merged stdout/stderr into the
    job log.  Returns the exit code (or 1 if the executable is missing)."""
    job.log(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    except FileNotFoundError:
        job.log(f"error: '{cmd[0]}' not found on PATH")
        return 1
    for line in proc.stdout:
        job.log(line)
    proc.wait()
    return proc.returncode


def build_wasm(job, example):
    """cargo build (wasm) -> wasm-bindgen -> copy index.html + assets.
    Returns the absolute output dir on success, or None on failure."""
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    crate = CRATES[example]
    rel = os.path.join("target", "wasm32-unknown-unknown", "release")
    out_dir = os.path.join(ex_dir, rel)

    if stream_step(job, ["cargo", "build", "--target", "wasm32-unknown-unknown", "--release"], ex_dir):
        job.log("cargo build failed")
        return None

    if shutil.which("wasm-bindgen") is None:
        job.log("wasm-bindgen not found - run: cargo install wasm-bindgen-cli --version 0.2.126")
        return None

    wasm_in = os.path.join(rel, f"{crate}.wasm")
    if stream_step(job, ["wasm-bindgen", "--target", "web", "--out-dir", rel, wasm_in], ex_dir):
        job.log("wasm-bindgen failed")
        return None

    shutil.copy(os.path.join(ex_dir, "index.html"), out_dir)
    job.log("copied index.html")

    for pattern, subdir in WASM_ASSETS.get(example, []):
        dest = os.path.join(out_dir, subdir)
        os.makedirs(dest, exist_ok=True)
        matches = globmod.glob(os.path.join(ex_dir, pattern))
        for src in matches:
            shutil.copy(src, dest)
        job.log(f"copied {len(matches)} file(s): {pattern} -> {subdir}/")

    return out_dir


def serve(job, example, out_dir, tunneled):
    """Launch serve.py / serve_tunnel.py as the job's long-lived process and
    stream it until it exits or is stopped."""
    port = PORTS[example]
    script = "serve_tunnel.py" if tunneled else "serve.py"
    cmd = [PY, script, out_dir, str(port)]
    job.log(f"$ {' '.join(cmd)}")
    # Emit UTF-8, and suppress serve_tunnel's terminal QR: node-qrcode draws the
    # QR purely with ANSI colour, which can't render in a browser <pre>, so we
    # render our own scannable SVG on the card instead (see emit_qr below).
    # PYTHONUNBUFFERED matters most: with stdout piped (not a real terminal),
    # Python block-buffers instead of line-buffering, so print()s (and the
    # "waiting on cloudflared" gap) don't reach this log until the buffer
    # fills or the process exits -- a slow-but-fine run looks identical to a
    # silently-dead one. Unbuffered output makes progress show up live.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1", "LAUNCHER_QR": "0"}
    proc = subprocess.Popen(
        cmd, cwd=EXAMPLES_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    job.proc = proc
    job.url("Local", f"http://localhost:{port}/")
    for line in proc.stdout:
        job.log(line)
        if tunneled:
            m = URL_RE.search(line)
            if m:
                url = m.group(0)
                job.url("Public (HTTPS)", url)
                emit_qr(job, url)
    proc.wait()


def emit_qr(job, url):
    """Render a scannable QR for the tunnel URL as an SVG (white background,
    black modules -> theme-independent) and push it to the card."""
    svg = qr_svg(url)
    if svg:
        job.emit({"type": "qr", "url": url, "svg": svg})


def qr_svg(url):
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


def run_native(job, example):
    ex_dir = os.path.join(EXAMPLES_DIR, example)
    cmd = ["cargo", "run", "--release"]
    job.log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=ex_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    job.proc = proc
    for line in proc.stdout:
        job.log(line)
    proc.wait()


def worker(job):
    try:
        if job.action == "native":
            run_native(job, job.example)
        elif job.action in ("wasm", "tunnel"):
            out_dir = build_wasm(job, job.example)
            if out_dir is None:
                job.set_status("failed")
                return
            serve(job, job.example, out_dir, tunneled=(job.action == "tunnel"))
        else:
            job.log(f"unknown action: {job.action}")
            job.set_status("failed")
            return
    except Exception as e:  # surface crashes to the page rather than the console
        job.log(f"launcher error: {e}")
        job.set_status("failed")
        return

    if job.stop_requested:
        job.set_status("stopped")
    else:
        # native/serve processes are meant to run until stopped; a clean exit
        # here just means the window/server closed on its own.
        job.set_status("ok" if job.action == "native" else "stopped")


def start_job(example, action):
    with JOBS_LOCK:
        old = JOBS.get(example)
        if old and old.status == "running":
            old.stop_requested = True
            kill_tree(old.proc)
        job = Job(example, action)
        JOBS[example] = job
    threading.Thread(target=worker, args=(job,), daemon=True).start()
    return job


def stop_job(example):
    with JOBS_LOCK:
        job = JOBS.get(example)
    if job and job.status == "running":
        job.stop_requested = True
        kill_tree(job.proc)
        job.set_status("stopped")
        return True
    return False


class QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # SSE clients (browser tabs) drop connections all the time; those show
        # up as ConnectionReset/BrokenPipe and are not worth a traceback.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the launcher's own console quiet

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or "{}")

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/examples":
            payload = [
                {"name": e["name"], "crate": e["crate"], "port": PORTS[e["name"]]}
                for e in EXAMPLES
            ]
            self._send(200, json.dumps(payload))
        elif self.path == "/api/state":
            with JOBS_LOCK:
                state = {
                    name: {"action": j.action, "status": j.status}
                    for name, j in JOBS.items()
                }
            self._send(200, json.dumps(state))
        elif self.path.startswith("/api/events"):
            self._sse()
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/action":
            data = self._body_json()
            example, action = data.get("example"), data.get("action")
            if example not in PORTS or action not in ("native", "wasm", "tunnel"):
                self._send(400, json.dumps({"error": "bad request"}))
                return
            start_job(example, action)
            self._send(200, json.dumps({"ok": True}))
        elif self.path == "/api/stop":
            data = self._body_json()
            stopped = stop_job(data.get("example"))
            self._send(200, json.dumps({"stopped": stopped}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _sse(self):
        example = self.path.split("job=", 1)[1] if "job=" in self.path else None
        with JOBS_LOCK:
            job = JOBS.get(example)
        if job is None:
            self._send(404, json.dumps({"error": "no job"}))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q, history = job.subscribe()
        try:
            for ev in history:
                self._sse_write(ev)
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep the connection warm
                    self.wfile.flush()
                    continue
                self._sse_write(ev)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            job.unsubscribe(q)

    def _sse_write(self, ev):
        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.flush()


def main():
    if not EXAMPLES:
        print("No examples found under", EXAMPLES_DIR)
        return
    server = QuietServer(("127.0.0.1", CONTROL_PORT), Handler)
    url = f"http://localhost:{CONTROL_PORT}/"
    print(f"black_splat launcher on {url}")
    print("Examples:", ", ".join(f"{e['name']} (:{PORTS[e['name']]})" for e in EXAMPLES))
    print("Press Ctrl+C (or close this window) to stop.")

    def shutdown(*_):
        # Kill every child job (cargo/serve/tunnel) so nothing orphans when the
        # launcher goes away -- otherwise a native run keeps its window open and
        # a serve keeps holding its port after the launcher is gone.
        with JOBS_LOCK:
            for job in JOBS.values():
                kill_tree(job.proc)

    atexit.register(shutdown)
    # SIGBREAK fires on a console-close (the window's X) on Windows; SIGINT is
    # Ctrl+C; SIGTERM is a `taskkill`.  Catch them all so children die with us.
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, lambda *_: sys.exit(0))
            except (ValueError, OSError):
                pass  # not all signals are settable on every platform

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nshutting down...")
        shutdown()


if __name__ == "__main__":
    main()
