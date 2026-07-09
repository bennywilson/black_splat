#!/usr/bin/env bash
# Starts the launcher dashboard and opens it in your browser. Python's
# webbrowser module handles the "open a URL" part on every platform --
# including from Git Bash on Windows, where invoking cmd.exe directly is a
# trap (MSYS path-mangling rewrites /c to C:\, silently breaking `start`).
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="http://localhost:8090/"
(
  sleep 1
  python3 -c "import webbrowser, sys; webbrowser.open(sys.argv[1])" "$URL"
) &
exec python3 "$DIR/server.py"
