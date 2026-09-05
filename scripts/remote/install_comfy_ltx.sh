#!/usr/bin/env bash
# Second ComfyUI install for LTX-2.5 (>= v0.32.0 native AV nodes). Shares ~/ComfyUI/models.
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
COMFY_LTX="${COMFYUI_LTX_ROOT:-$HOME/ComfyUI-ltx}"
SHARED_MODELS="${COMFYUI_MODELS:-$HOME/ComfyUI/models}"
COMFY_TAG="${COMFY_LTX_TAG:-v0.34.3}"
LAN_IP="${LAN_BIND_IP:-192.168.50.100}"

echo "=== install ComfyUI LTX (${COMFY_TAG}) ==="

if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

if [[ ! -d "${COMFY_LTX}/.git" ]]; then
  git clone --depth 1 --branch "${COMFY_TAG}" https://github.com/comfyanonymous/ComfyUI "${COMFY_LTX}"
else
  echo "ComfyUI-ltx already cloned at ${COMFY_LTX}; fetching ${COMFY_TAG}..."
  git -C "${COMFY_LTX}" fetch --depth 1 origin "refs/tags/${COMFY_TAG}:refs/tags/${COMFY_TAG}" 2>/dev/null || true
  git -C "${COMFY_LTX}" checkout -f "${COMFY_TAG}" 2>/dev/null || git -C "${COMFY_LTX}" pull --ff-only
fi

if [[ -f "${CONTROL}/comfyui/extra_model_paths.yaml" ]]; then
  cp "${CONTROL}/comfyui/extra_model_paths.yaml" "${COMFY_LTX}/extra_model_paths.yaml"
fi

VENV="${COMFY_LTX}/venv"
if [[ ! -x "${VENV}/bin/python3" ]]; then
  echo "Creating venv at ${VENV}..."
  python3 -m venv "${VENV}"
fi

PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python3"

"${PIP}" install -U pip wheel setuptools

echo "Installing PyTorch cu130 (Blackwell sm_120)..."
"${PIP}" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "Installing ComfyUI requirements..."
"${PIP}" install -r "${COMFY_LTX}/requirements.txt"

if ! "${PY}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "Warning: torch CUDA not available in venv (CPU-only torch?). LTX will need GPU." >&2
fi

mkdir -p "${SHARED_MODELS}"/{diffusion_models,text_encoders,vae,checkpoints,latent_upscale_models}

if [[ -f "${CONTROL}/scripts/remote/link_ltx_checkpoints.sh" ]]; then
  bash "${CONTROL}/scripts/remote/link_ltx_checkpoints.sh"
fi

if [[ -f "${CONTROL}/scripts/remote/install_ltx_workflows.sh" ]]; then
  bash "${CONTROL}/scripts/remote/install_ltx_workflows.sh"
fi

# IC-LoRA nodes live in ComfyUI-ltx custom_nodes (not the models-only ~/ComfyUI tree).
CUSTOM_LTX="${COMFY_LTX}/custom_nodes"
mkdir -p "${CUSTOM_LTX}"
_install_ltx_node() {
  local repo="$1" dir="$2"
  if [[ ! -d "${CUSTOM_LTX}/${dir}/.git" ]]; then
    git clone --depth 1 "${repo}" "${CUSTOM_LTX}/${dir}"
  else
    git -C "${CUSTOM_LTX}/${dir}" pull --ff-only || true
  fi
  if [[ -f "${CUSTOM_LTX}/${dir}/requirements.txt" ]]; then
    "${PIP}" install -r "${CUSTOM_LTX}/${dir}/requirements.txt" || true
  fi
}
_install_ltx_node "https://github.com/Lightricks/ComfyUI-LTXVideo" "ComfyUI-LTXVideo"
_install_ltx_node "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything" "ComfyUI-Video-Depth-Anything"

echo "ComfyUI LTX ready at ${COMFY_LTX}"
echo "  listen: http://${LAN_IP}:8189"
echo "  models: ${SHARED_MODELS} (shared with primary ComfyUI)"
