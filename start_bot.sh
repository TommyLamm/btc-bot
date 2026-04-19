#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/btc-bot"
VENV_DIR="/root/btc-bot-env"

cd "${PROJECT_DIR}"

# Prefer the project virtual environment; fall back to system python only if missing.
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" main.py
