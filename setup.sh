#!/usr/bin/env bash
# setup.sh — one-command environment setup for fightflow
#
# Usage:
#   bash setup.sh
#
# What it does:
#   1. Creates a Python virtual environment in .venv/
#   2. Installs all Python packages from requirements.txt
#   3. Registers the environment as a Jupyter kernel called "fightflow"
#   4. Reminds you to install ffmpeg if it is not already present
#
# After running this script, activate the environment with:
#   source .venv/bin/activate

set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo ""
echo "=== fightflow environment setup ==="
echo ""

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "[1/4] Virtual environment already exists at $VENV_DIR — skipping creation."
else
    echo "[1/4] Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi

# ── 2. Activate ───────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 3. Install packages ───────────────────────────────────────────────────────
echo "[2/4] Upgrading pip ..."
pip install --quiet --upgrade pip

echo "[3/4] Installing packages from requirements.txt ..."
pip install --quiet -r requirements.txt

# ── 4. Register Jupyter kernel ────────────────────────────────────────────────
echo "[4/4] Registering Jupyter kernel as 'fightflow' ..."
python -m ipykernel install --user --name fightflow --display-name "fightflow (Python)"

# ── 5. ffmpeg check ───────────────────────────────────────────────────────────
echo ""
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "⚠  ffmpeg not found."
    echo "   Install it with:  brew install ffmpeg"
    echo "   ffmpeg is required for video frame extraction (Step 4 onwards)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment in future sessions:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "To start Jupyter:"
echo "    jupyter notebook"
echo ""
