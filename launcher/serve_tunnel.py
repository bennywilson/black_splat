#!/usr/bin/env python3
"""Serve a wasm build locally AND expose it over an HTTPS Cloudflare quick
tunnel, then show a QR code for the public URL so a phone can open it with a
camera scan instead of typing a long address.

Why a tunnel: the Gaussian-splat demo needs WebGPU, and browsers only expose
WebGPU (navigator.gpu) in a *secure context*.  Plain http:// on a LAN IP is not
secure, so mobile Safari can't get a WebGPU adapter.  A trycloudflare.com URL is
real HTTPS, which satisfies the secure-context requirement.  (The 2D/3D demos
use WebGL2 and work over plain LAN http via serve.py, so they don't need this.)

Requires cloudflared (`scoop install cloudflared`).  QR rendering uses
`npx qrcode --small` (node-qrcode) when Node is available; otherwise it just
prints the URL.

Usage: python3 serve_tunnel.py <directory> [port]
"""
import functools
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading

from serve import NoCacheHandler  # reuse the no-cache request handler

PORT = 8000
URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def start_server(directory, port):
    handler = functools.partial(NoCacheHandler, directory=directory)
    # ("", port) binds all interfaces, so plain LAN http keeps working too.
    httpd = socketserver.TCPServer(("", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def show_qr(url):
    banner = "=" * 60
    print(f"\n{banner}\n  Open on your phone (HTTPS -> WebGPU works):\n  {url}\n{banner}\n")
    # The web launcher renders its own (scannable) QR image on the dashboard, so
    # it sets LAUNCHER_QR=0 to suppress the terminal QR -- whose ANSI colour
    # blocks would just be garbled noise in a browser <pre>.
    if os.environ.get("LAUNCHER_QR", "1") == "0":
        return
    npx = shutil.which("npx")
    if not npx:
        print("(install Node for a scannable QR, or just type the URL above)\n")
        return
    try:
        # `--small` renders the QR with half-height blocks -- about half the
        # rows (and narrower) than the default, so it fits a normal terminal.
        if sys.platform == "win32":
            # shell=True so Windows resolves npx.cmd; quote the URL as one arg.
            subprocess.run(f'npx --yes qrcode --small "{url}"', shell=True, check=False)
        else:
            subprocess.run([npx, "--yes", "qrcode", "--small", url], check=False)
    except Exception:
        pass


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT

    start_server(directory, port)
    print(f"Serving {directory} on http://localhost:{port}/  (LAN + tunnel)", flush=True)

    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        print(
            "\ncloudflared not found -- install it for the HTTPS tunnel:\n"
            "  scoop install cloudflared\n\n"
            "The LAN URL above still works for the WebGL2 (2D/3D) demos.\n"
            "Press Ctrl+C to stop.\n",
            flush=True,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return

    print("Starting Cloudflare quick tunnel (Ctrl+C to stop)...\n", flush=True)
    proc = subprocess.Popen(
        [cloudflared, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    shown = False
    try:
        # Stream cloudflared's log; when its public URL appears, render the QR
        # once and keep draining output so the tunnel stays alive.
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if not shown:
                match = URL_RE.search(line)
                if match:
                    show_qr(match.group(0))
                    shown = True
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
