#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
pixspector analyze examples/sample_images/*.jpg --report out/
