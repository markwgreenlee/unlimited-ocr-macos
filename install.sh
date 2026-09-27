#!/bin/bash
# Install Unlimited-OCR for Apple Silicon into this directory.
#
# Creates ./.venv, installs the pinned dependency set, and downloads the 6.7 GB
# model weights into ./model. Nothing outside this directory is touched: no sudo,
# no system Python, no Homebrew changes.
#
# Usage:  ./install.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# ---- 1. platform checks -----------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || die "macOS only (this is the Apple Silicon port)."
[ "$(uname -m)" = "arm64" ] || die "Apple Silicon (arm64) required; got $(uname -m)."

OS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
[ "$OS_MAJOR" -ge 14 ] || die "macOS 14+ required for MPS; got $(sw_vers -productVersion)."

# Advisory checks — if the query itself fails, carry on rather than abort.
MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
if [ "$MEM_BYTES" -gt 0 ] && [ "$(( MEM_BYTES / 1073741824 ))" -lt 16 ]; then
    say "WARNING: $(( MEM_BYTES / 1073741824 )) GB unified memory; the bf16 weights alone need ~6.7 GB."
fi

AVAIL_GB="$(df -g . 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)"
if [ -n "$AVAIL_GB" ] && [ "$AVAIL_GB" -gt 0 ] && [ "$AVAIL_GB" -lt 10 ]; then
    die "need ~10 GB free (weights 6.7 GB + venv ~1 GB); ${AVAIL_GB} GB available."
fi

# ---- 2. find a suitable interpreter ----------------------------------------
# torch 2.10 ships macOS arm64 wheels for cp310-cp314; 3.13 is the tested one.
PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        v="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
        case "$v" in
            3.10|3.11|3.12|3.13) PY="$cand"; break ;;
        esac
    fi
done
[ -n "$PY" ] || die "no Python 3.10-3.13 found. Install one (e.g. 'brew install python@3.13' or python.org) and re-run."
say "using $PY ($("$PY" -V 2>&1))"

# ---- 3. virtualenv + dependencies ------------------------------------------
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    say "creating .venv"
    "$PY" -m venv .venv
fi
VENV_PY="$ROOT/.venv/bin/python"

say "installing dependencies (a few hundred MB, several minutes)"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --only-binary=:all: -r requirements.txt

# ---- 4. model weights -------------------------------------------------------
if [ -f "$ROOT/model/model-00001-of-000001.safetensors" ]; then
    say "weights already present, skipping download"
else
    say "downloading weights from Hugging Face (6.7 GB — this is the slow part)"
    "$VENV_PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("baidu/Unlimited-OCR", local_dir="model",
                  allow_patterns=["*.json", "*.py", "LICENSE", "*.safetensors"],
                  max_workers=4)
PY
fi

# ---- 5. verify --------------------------------------------------------------
say "checking device and Metal correctness"
"$VENV_PY" - <<'PY'
import torch
print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print("  note: MPS unavailable — runs will fall back to CPU (correct, ~2.7x slower)")
PY
"$VENV_PY" probe-mps-ops.py || say "probe reported disagreements — see README 'The MPS bug'"

say "generating a test page and running it end to end"
"$VENV_PY" samples/make_sample_page.py
chmod +x run-ocr.sh ocr.py
./run-ocr.sh samples/synthetic_page.png --out out/install-check --mode base

if [ -s "$ROOT/out/install-check/result.md" ]; then
    say "SUCCESS — transcription:"
    cat "$ROOT/out/install-check/result.md"
    cat <<EOF

Install root: $ROOT

Next steps:
  ./run-ocr.sh <image|pdf|directory> [--out DIR]     # see --help
  export UNLIMITED_OCR_ROOT="$ROOT"                  # for the Claude Science skill
EOF
else
    die "run produced an empty result.md — see README 'The MPS bug' and try --device cpu"
fi
