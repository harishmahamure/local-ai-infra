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
  --exclude 'catalog/installed.json' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  "${ROOT}/" "${HOST}:${REMOTE_DIR}/"

# Link ComfyUI extras
ssh "${HOST}" "bash -s" <<REMOTE
set -euo pipefail
CTRL="\${REMOTE_AI_DIR:-\$HOME/ai-inference/control}"
# shellcheck disable=SC1091
source "\$CTRL/scripts/remote/comfy_paths.sh"
mkdir -p "\$COMFYUI_HOME/models"
if [[ -f "\$CTRL/comfyui/extra_model_paths.yaml" ]]; then
  cp -f "\$CTRL/comfyui/extra_model_paths.yaml" "\$COMFYUI_HOME/extra_model_paths.yaml"
fi
chmod +x "\$CTRL/scripts/remote/"*.sh "\$CTRL/bin/ai" 2>/dev/null || true
bash "\$CTRL/scripts/remote/install_systemd_units.sh"
bash "\$CTRL/scripts/remote/install_ltx_workflows.sh" 2>/dev/null || true
echo "Synced to \$CTRL"
REMOTE

# Pick up new routes/modules without a full bootstrap.
ssh "${HOST}" "systemctl --user restart ai-control.service"

echo "Done. Control API restarted on ${HOST}."
echo "For first-time GPU setup (llama.cpp build), run: ./bin/ai bootstrap"
