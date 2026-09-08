#!/usr/bin/env bash
# GPU box setup for llama.cpp text/vision. Run ON the box or via: ./bin/ai bootstrap
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
LLAMA="${LLAMACPP_ROOT:-$HOME/ai-inference/llama.cpp}"
MODELS="${AI_MODELS:-$HOME/ai-inference/models}"
LAN_IP="${LAN_BIND_IP:-192.168.50.100}"
VENV="${AI_VENV:-$HOME/ai-inference/venv}"
TURBOQUANT="${HOME}/llama-cpp-turboquant"
GGML_LLAMA="${HOME}/llama.cpp"

echo "=== bootstrap ai-inference (llama.cpp only) ==="
mkdir -p "$HOME/ai-inference"/{models/llamacpp,logs,control,config}

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip pyyaml huggingface_hub tqdm fastapi uvicorn httpx

if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

# Reuse the already-built RotorQuant fork for llama-fast (turbo2 KV cache).
if [[ ! -x "$LLAMA/build/bin/llama-server" ]]; then
  if [[ -x "$TURBOQUANT/build/bin/llama-server" ]]; then
    echo "Linking ${LLAMA} -> ${TURBOQUANT}"
    ln -sfn "$TURBOQUANT" "$LLAMA"
  else
    echo "Missing llama-server at ${LLAMA}/build/bin/llama-server" >&2
    echo "Expected existing tree at ${TURBOQUANT}." >&2
    exit 1
  fi
fi

if [[ ! -x "$GGML_LLAMA/build/bin/llama-server" ]]; then
  echo "Warning: Gemma binary missing at ${GGML_LLAMA}/build/bin/llama-server" >&2
  echo "gemma.env pins LLAMACPP_ROOT=${GGML_LLAMA}" >&2
fi

chmod +x "$CONTROL/scripts/remote/"*.sh
bash "$CONTROL/scripts/remote/install_systemd_units.sh"

loginctl enable-linger "$USER" 2>/dev/null || true

systemctl --user disable llama-fast.service gemma.service comfyui.service comfyui-ltx.service ai-download.service 2>/dev/null || true
rm -f "$HOME/.config/systemd/user/comfyui.service" "$HOME/.config/systemd/user/comfyui-ltx.service"
systemctl --user daemon-reload
systemctl --user enable --now ai-control.service

echo "Bootstrap complete."
echo "Control API: http://${LAN_IP}:8090/ ( /docs + /v1 )"
echo "GPU inference stays stopped until: ai start gemma|llama-fast"
echo "Background download: ai download (or POST /v1/downloads)"
