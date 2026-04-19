#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SERVICE="btcbot"
ACTION="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  ./run_bot.sh <action> [service_name] [lines]

Actions:
  start       Start service (default: btcbot)
  stop        Stop service
  restart     Restart service
  status      Show service status
  enable      Enable service on boot
  disable     Disable service on boot
  is-active   Print active state
  logs        Show latest logs (default 200 lines)
  follow      Follow logs in real time

Examples:
  ./run_bot.sh start
  ./run_bot.sh status btcbot
  ./run_bot.sh logs btcbot 300
  ./run_bot.sh follow
EOF
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' not found."
    exit 1
  fi
}

run_systemctl() {
  if [[ "${EUID}" -eq 0 ]]; then
    systemctl "$@"
  else
    sudo systemctl "$@"
  fi
}

run_journalctl() {
  if [[ "${EUID}" -eq 0 ]]; then
    journalctl "$@"
  else
    sudo journalctl "$@"
  fi
}

need_cmd systemctl

if [[ -z "${ACTION}" || "${ACTION}" == "-h" || "${ACTION}" == "--help" || "${ACTION}" == "help" ]]; then
  usage
  exit 0
fi

SERVICE="${2:-$DEFAULT_SERVICE}"

case "${ACTION}" in
  start)
    run_systemctl start "${SERVICE}"
    run_systemctl --no-pager status "${SERVICE}" | sed -n '1,25p'
    ;;
  stop)
    run_systemctl stop "${SERVICE}"
    run_systemctl --no-pager status "${SERVICE}" | sed -n '1,25p'
    ;;
  restart)
    run_systemctl restart "${SERVICE}"
    run_systemctl --no-pager status "${SERVICE}" | sed -n '1,25p'
    ;;
  status)
    run_systemctl --no-pager status "${SERVICE}"
    ;;
  enable)
    run_systemctl enable "${SERVICE}"
    run_systemctl is-enabled "${SERVICE}"
    ;;
  disable)
    run_systemctl disable "${SERVICE}"
    run_systemctl is-enabled "${SERVICE}" || true
    ;;
  is-active)
    run_systemctl is-active "${SERVICE}"
    ;;
  logs)
    LINES="${3:-200}"
    run_journalctl -u "${SERVICE}" -n "${LINES}" --no-pager
    ;;
  follow)
    run_journalctl -u "${SERVICE}" -f
    ;;
  *)
    echo "Error: unknown action '${ACTION}'."
    usage
    exit 1
    ;;
esac
