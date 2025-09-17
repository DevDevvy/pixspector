#!/usr/bin/env bash
set -euo pipefail

echo "=== pixspector setup ==="

# Create venv if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install deps
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  pip install -e .
fi

echo "✅ Dependencies installed into .venv"

# Check for c2patool
if ! command -v c2patool >/dev/null 2>&1; then
  echo "⚠️  c2patool not found. C2PA provenance checks will be disabled."
  echo "    Install from: https://github.com/contentauth/c2patool"
else
  echo "✅ c2patool found."
fi

echo ""
echo "Run the tool with:"
echo "  source .venv/bin/activate"
echo "  pixspector analyze examples/sample_images/edited_1.jpg --report out/"
