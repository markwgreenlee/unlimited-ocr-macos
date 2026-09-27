#!/bin/bash
# Unlimited-OCR launcher — activates the local venv and runs ocr.py.
# Usage: ./run-ocr.sh <image|pdf|dir> [options]     (see ocr.py --help)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$ROOT/.venv/bin/python" "$ROOT/ocr.py" "$@"
