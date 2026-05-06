#!/usr/bin/env bash
set -euo pipefail

# Simple setup helper for macOS / Debian-based Linux
# - checks for ffmpeg/ffprobe
# - creates a virtualenv and installs Python deps

PYTHON=${PYTHON:-python3}
VENV_DIR=.venv
REQUIREMENTS=requirements.txt

echo "Checking for ffmpeg and ffprobe..."
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH."
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "Install with Homebrew: brew install ffmpeg"
  else
    echo "Install with apt: sudo apt update && sudo apt install -y ffmpeg"
  fi
else
  echo "ffmpeg found: $(command -v ffmpeg)"
fi

if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffprobe not found in PATH. It is typically provided with ffmpeg."
else
  echo "ffprobe found: $(command -v ffprobe)"
fi

echo "\nCreating virtual environment in ${VENV_DIR} (if missing)..."
${PYTHON} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
python -m pip install --upgrade pip

if [ -f "${REQUIREMENTS}" ]; then
  echo "Installing Python dependencies from ${REQUIREMENTS}..."
  pip install -r ${REQUIREMENTS}
else
  echo "No ${REQUIREMENTS} found — please create it or run 'pip install <packages>' manually."
fi

echo "\nSetup complete. Activate with: source ${VENV_DIR}/bin/activate"
