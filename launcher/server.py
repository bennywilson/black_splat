#!/usr/bin/env python3
"""Control server for the black_splat sample launcher.

A tiny local web dashboard so you can build/run the examples with buttons
instead of memorising terminal incantations. All the actual build/run/serve
logic lives in build.py (also usable standalone from the CLI); this file is
just the HTTP/SSE layer plus per-example job tracking, and streams build.py's
output back to the page over Server-Sent Events.

Start it with launch.bat / launch.sh (or `python3 server.py`) and open
http://localhost:8090.

Per example you get:
  * Native   -> `cargo run --release`            (opens a native window)
  * Wasm     -> build wasm + serve on a LAN port (WebGL2 demos: 2d/3d)
  * Tunnel   -> build wasm + serve over HTTPS via cloudflared (needed for the
                splat demo's WebGPU on phones); surfaces the public URL + QR
  * Stop     -> kill the running job for that example

Only one job runs per example at a time; starting a new one stops the old.
"""
import atexit
import http.server
import json
import os
import queue
import re
import signal
import sys
import threading

import build
from build import EXAMPLES, PORTS, CRATES

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_PORT = build.DEFAULT_DASHBOARD_PORT

# cloudflared colorises its logs; those ANSI/OSC escapes render as junk in the
# browser <pre>, so strip them (CSI ...m colour codes and OSC ...BEL sequences).
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


class Job:
    """A running build/run for one example, with an append-only event log that
    late SSE subscribers replay in full (so a page reload re-attaches cleanly)."""

    def __init__(self, example, action):
        self.example = example
        self.action = action
        self.events = []            # list of dicts: {type, ...} (replayed in full)
        self.last_transient = None  # newest transient log ev, kept out of events
        self.subs = []              # list of queue.Queue for live subscribers
        self.lock = threading.Lock()
        self.proc = None            # the long-lived (serve) process, if any
        self.status = "running"     # running | ok | failed | stopped
        self.stop_requested = False

    def emit(self, ev, history=True):
        with self.lock:
            if history:
                self.events.append(ev)
            for q in self.subs:
                q.put(ev)

    def log(self, line, transient=False):
        # transient = an in-place progress-bar redraw. Live subscribers get every
        # frame, but we keep only the latest out of `events` so a late/reload
        # subscriber replays one current bar line, not thousands of stale ones.
        ev = {"type": "log", "line": ANSI_RE.sub("", line).rstrip("\r\n"),
              "transient": transient}
        if transient:
            self.last_transient = ev
            self.emit(ev, history=False)
        else:
            # A committed line ends the current bar frame; a still-building step
            # emits a fresh frame right after, so the bar clears on the last
            # committed line of a step (e.g. cargo's "Finished") instead of a
            # partial "[===> ]" lingering under the next step's output.
            self.last_transient = None
            self.emit(ev)

    def set_status(self, status):
        self.status = status
        if status != "running":
            self.last_transient = None  # no stale bar line after the job ends
        self.emit({"type": "status", "status": status})

    def url(self, label, url):
        self.emit({"type": "url", "label": label, "url": url})
        if label.startswith("Public"):
            svg = build.qr_svg(url)
            if svg:
                self.emit({"type": "qr", "url": url, "svg": svg})

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            history = list(self.events)
            if self.last_transient is not None:
                history.append(self.last_transient)  # one current bar frame
            self.subs.append(q)
        return q, history

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)


JOBS = {}          # example name -> Job
JOBS_LOCK = threading.Lock()


def worker(job):
    example = job.example

    def on_proc(p):
        job.proc = p

    try:
        if job.action == "native":
            build.run_native(example, job.log, on_proc=on_proc)
        elif job.action in ("wasm", "tunnel"):
            out_dir = build.build_wasm(example, job.log)
            if out_dir is None:
                job.set_status("failed")
                return
            build.run_serve(
                example, out_dir, tunneled=(job.action == "tunnel"),
                log=job.log, on_proc=on_proc, on_url=job.url,
                suppress_terminal_qr=True,
            )
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
            build.kill_tree(old.proc)
        job = Job(example, action)
        JOBS[example] = job
    threading.Thread(target=worker, args=(job,), daemon=True).start()
    return job


def stop_job(example):
    with JOBS_LOCK:
        job = JOBS.get(example)
    if job and job.status == "running":
        job.stop_requested = True
        build.kill_tree(job.proc)
        job.set_status("stopped")
        return True
    return False


# Subclass ThreadingHTTPServer.  Each request is handled in its own thread
class QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # SSE clients (browser tabs) drop connections all the time, skip printing the traceback
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
                {
                    "name": e["name"], "crate": e["crate"], "port": PORTS[e["name"]],
                    "thumbnail": bool(e["thumbnail"]),
                }
                for e in EXAMPLES
            ]
            self._send(200, json.dumps(payload))
        elif self.path.startswith("/api/thumb/"):
            name = self.path[len("/api/thumb/"):]
            thumb = next((e["thumbnail"] for e in EXAMPLES if e["name"] == name), None)
            if name not in PORTS or not thumb:
                self._send(404, json.dumps({"error": "not found"}))
                return
            ctype = {
                ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
            }[os.path.splitext(thumb)[1]]
            with open(os.path.join(build.EXAMPLES_DIR, name, thumb), "rb") as f:
                self._send(200, f.read(), ctype)
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
        print("No examples found under", build.EXAMPLES_DIR)
        return

    server = QuietServer(("127.0.0.1", CONTROL_PORT), Handler)
    url = f"http://localhost:{CONTROL_PORT}/"
    print(f"black_splat launcher on {url}")
    print("Examples:", ", ".join(f"{e['name']} (:{PORTS[e['name']]})" for e in EXAMPLES))
    print("Press Ctrl+C (or close this window) to stop.")

    def shutdown(*_):
        # Kill every child job  so nothing orphans when the launcher goes away to prevent stale ports
        with JOBS_LOCK:
            for job in JOBS.values():
                build.kill_tree(job.proc)

    atexit.register(shutdown)
    # SIGBREAK fires on a console-close (the window's X) on Windows; SIGINT is
    # Ctrl+C; SIGTERM is a `taskkill` / `kill`. Catch what's available so
    # children die with us. (If none of these fire -- e.g. a hard window-close
    # event Python never sees -- `build.py stop` is the manual escape hatch.)
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
