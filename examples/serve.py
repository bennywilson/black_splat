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

PORT = 8000


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    handler = functools.partial(NoCacheHandler, directory=directory)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving (no-cache) {directory} on http://127.0.0.1:{port}/ ...")
        httpd.serve_forever()
