#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/btc-bot"
VENV_DIR="/root/btc-bot-env"

# Optional env files (loaded in order, later files can override earlier values)
ENV_FILES=(
  "/etc/default/btcbot"
  "/etc/btcbot.env"
  "${PROJECT_DIR}/.env"
)

cd "${PROJECT_DIR}"

for env_file in "${ENV_FILES[@]}"; do
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
done

# Prefer the project virtual environment; fall back to system python only if missing.
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" main.py
