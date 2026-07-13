#!/usr/bin/env python3
"""Minimal static file server for local wasm iteration.

Identical to `python3 -m http.server` except it sends no-cache headers so the
browser always fetches the freshly built .js/.wasm instead of reusing a stale
cached copy (which otherwise forces you into incognito mode to see changes).

Usage: python3 serve.py <directory> [port]
"""
import functools
import http.server
import socketserver
import sys

import build

PORT = 8000


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # Stamp our identity so a restart can recognise (and reclaim) a stale
        # instance of this server without touching an unrelated app on the port.
        self.send_header(build.IDENT_HEADER, "serve")
        super().end_headers()


class DevServer(socketserver.TCPServer):
    # On Unix, SO_REUSEADDR lets a restart rebind straight through the sockets a
    # just-stopped server left in TIME_WAIT (standard, and what http.server
    # does). NOT on Windows, where SO_REUSEADDR instead lets an unrelated
    # process co-bind a *live* port (a hijack footgun) -- there we reclaim a
    # stale server explicitly (see build.bind_or_reclaim) instead.
    allow_reuse_address = sys.platform != "win32"


def make_server(directory, port):
    """Bind a no-cache dev server on `port`, reclaiming it first if a stale
    black_splat server is squatting on it. Exits with a hint if the port is
    held by some other process."""
    handler = functools.partial(NoCacheHandler, directory=directory)
    try:
        return build.bind_or_reclaim(lambda: DevServer(("", port), handler), port)
    except OSError:
        print(f"error: port {port} is in use by another process. "
              f"Free it or run: python build.py stop {port}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    with make_server(directory, port) as httpd:
        print(f"Serving (no-cache) {directory} on http://127.0.0.1:{port}/ ...")
        httpd.serve_forever()
