#!/usr/bin/env bash
# Thin wrapper: the dashboard lives in launcher/, this just saves a `cd` so
# you can launch it straight from the repo root.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/launcher/launch.sh"
