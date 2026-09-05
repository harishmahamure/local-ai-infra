#!/usr/bin/env bash
# Launch ComfyUI with its project venv (system python lacks sqlalchemy etc.).
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
# shellcheck disable=SC1091
source "${CONTROL}/scripts/remote/comfy_paths.sh"
LAN_IP="${LAN_BIND_IP:-192.168.50.100}"
PORT="${COMFY_PORT:-8188}"
PYTHON="${COMFYUI_HOME}/venv/bin/python3"

if [[ ! -f "${COMFYUI_HOME}/main.py" ]]; then
  echo "ComfyUI not found at ${COMFYUI_HOME}/main.py" >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "ComfyUI venv missing at ${PYTHON}" >&2
  echo "On the GPU box run: cd ${COMFYUI_HOME} && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

cd "$COMFYUI_HOME"
exec "$PYTHON" main.py --listen "$LAN_IP" --port "$PORT"
