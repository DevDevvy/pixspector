#!/usr/bin/env bash
set -euo pipefail

# Ensure venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install pyinstaller

# Build
pyinstaller pixspector_gui.spec

echo "✅ Build complete. Find your app in dist/pixspector"
echo "   On macOS you can bundle into an .app using --windowed or app bundling tools if desired."
