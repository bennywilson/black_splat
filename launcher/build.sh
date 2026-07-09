#!/usr/bin/env bash
# Thin wrapper: build.py is the single source of truth (cross-platform Python),
# this just gives you the familiar `./build.sh ...` shell-script invocation.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/build.py" "$@"
