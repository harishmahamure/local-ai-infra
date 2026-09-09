#!/usr/bin/env bash
# Sync control-plane files to GPU box (never model weights).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
# shellcheck disable=SC1091
source "$ROOT/scripts/gpu_paths.sh"

HOST="${REMOTE_HOST:-gpu-box}"
REMOTE_DIR="$(gpu_control_rsync)"

# rsync cannot create parent dirs; ~/ai-inference must exist first.
ssh "${HOST}" "mkdir -p ~/ai-inference/{models/llamacpp,logs,control,config}"

rsync -avz --delete \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '.cursor' \
  --exclude '.pytest_cache' \
  --exclude 'catalog/installed.json' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude 'web/node_modules' \
  --exclude 'web/dist' \
  "${ROOT}/" "${HOST}:${REMOTE_DIR}/"

ssh "${HOST}" "bash -s" <<REMOTE
set -euo pipefail
CTRL="\${REMOTE_AI_DIR:-\$HOME/ai-inference/control}"
chmod +x "\$CTRL/scripts/remote/"*.sh "\$CTRL/bin/ai" 2>/dev/null || true
bash "\$CTRL/scripts/remote/install_systemd_units.sh"
echo "Synced to \$CTRL"
REMOTE

ssh "${HOST}" "systemctl --user restart ai-control.service"

echo "Done. Control API restarted on ${HOST}."
echo "For first-time GPU setup, run: ./bin/ai bootstrap"
