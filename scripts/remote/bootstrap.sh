#!/usr/bin/env bash
# First-time GPU box setup. Run ON the box or via: ssh gpu-box 'bash ~/ai-inference/control/scripts/remote/bootstrap.sh'
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
# shellcheck disable=SC1091
source "${CONTROL}/scripts/remote/comfy_paths.sh"
LLAMA="${LLAMACPP_ROOT:-$HOME/ai-inference/llama.cpp}"
MODELS="${AI_MODELS:-$HOME/ai-inference/models}"
LAN_IP="${LAN_BIND_IP:-192.168.50.100}"
VENV="${AI_VENV:-$HOME/ai-inference/venv}"

echo "=== bootstrap ai-inference ==="
mkdir -p "$HOME/ai-inference"/{models/llamacpp,logs,control,rotorquant}
mkdir -p "$COMFYUI_HOME/models"/{diffusion_models,text_encoders,vae,loras,controlnet,model_patches,checkpoints,latent_upscale_models,Kokorotts}

# Python venv for downloads
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip pyyaml huggingface_hub fastapi uvicorn httpx python-multipart

# CUDA toolkit (nvcc is not on a non-interactive PATH)
if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

need_pkgs=()
for bin in gcc g++ cmake git; do
  command -v "$bin" >/dev/null || need_pkgs+=("$bin")
done
if ((${#need_pkgs[@]} > 0)); then
  echo "Missing build tools: ${need_pkgs[*]}"
  if sudo -n true 2>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev pciutils
  else
    echo "Install once on the GPU box (needs your sudo password):" >&2
    echo "  sudo apt-get update && sudo apt-get install -y build-essential cmake git libcurl4-openssl-dev pciutils" >&2
    exit 1
  fi
fi

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found. Install CUDA toolkit or add /usr/local/cuda/bin to PATH." >&2
  exit 1
fi

# scrya-com/rotorquant is the RotorQuant project (Python/Triton). It has no llama-server.
# Its README builds iso3 llama-server from johndpope/llama-cpp-turboquant.
ROTORQUANT="${ROTORQUANT_ROOT:-$HOME/ai-inference/rotorquant}"
ROTORQUANT_ORIGIN="https://github.com/scrya-com/rotorquant.git"
LLAMA_ORIGIN="https://github.com/johndpope/llama-cpp-turboquant.git"
LLAMA_BRANCH="feature/planarquant-kv-cache"
LLAMA_CMAKE_FLAGS=(
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_CUDA=ON
  -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=native
)
LLAMA_BUILD_STAMP="${LLAMA}/build/.ai-project-build-stamp"
LLAMA_PATCH_ID="gcc13-extern-c-ggml-api"
LLAMA_WANT_STAMP="$(printf '%s\n' "${ROTORQUANT_ORIGIN}" "${LLAMA_ORIGIN}" "${LLAMA_BRANCH}" "${LLAMA_PATCH_ID}" "${LLAMA_CMAKE_FLAGS[@]}" | sha256sum | awk '{print $1}')"

sync_rotorquant() {
  if [[ -e "$ROTORQUANT" && -d "$ROTORQUANT/.git" ]]; then
    local origin
    origin="$(git -C "$ROTORQUANT" remote get-url origin 2>/dev/null || true)"
    if [[ "$origin" != *scrya-com/rotorquant* ]]; then
      echo "Replacing non-scrya RotorQuant tree at ${ROTORQUANT}..."
      rm -rf "$ROTORQUANT"
    fi
  elif [[ -e "$ROTORQUANT" && ! -d "$ROTORQUANT/.git" ]]; then
    rm -rf "$ROTORQUANT"
  fi
  if [[ ! -d "$ROTORQUANT/.git" ]]; then
    git clone --depth 1 "$ROTORQUANT_ORIGIN" "$ROTORQUANT"
  else
    git -C "$ROTORQUANT" pull --ff-only 2>/dev/null || true
  fi
}

llama_origin_ok() {
  [[ -d "$LLAMA/.git" ]] || return 1
  local origin
  origin="$(git -C "$LLAMA" remote get-url origin 2>/dev/null || true)"
  [[ "$origin" == *johndpope/llama-cpp-turboquant* ]]
}

sync_llamacpp_fork() {
  if [[ -e "$LLAMA" ]] && ! llama_origin_ok; then
    echo "Replacing non-RotorQuant llama.cpp at ${LLAMA}..."
    rm -rf "$LLAMA"
  fi
  if [[ ! -d "$LLAMA/.git" ]]; then
    git clone --depth 1 --branch "$LLAMA_BRANCH" "$LLAMA_ORIGIN" "$LLAMA"
  else
    git -C "$LLAMA" fetch --depth 1 origin "$LLAMA_BRANCH"
    git -C "$LLAMA" checkout -B "$LLAMA_BRANCH" FETCH_HEAD
  fi
  # Fork writes `extern "C" GGML_API` (= `extern "C" extern`); GCC 13 rejects it.
  local ops="$LLAMA/ggml/src/ggml-cpu/ops.cpp"
  if [[ -f "$ops" ]] && grep -q 'extern "C" GGML_API ' "$ops"; then
    sed -i 's/extern "C" GGML_API /extern "C" /g' "$ops"
    echo "Patched ${ops} for GCC 13 (${LLAMA_PATCH_ID})"
  fi
}

build_llamacpp() {
  echo "Building iso3 llama-server from ${LLAMA_ORIGIN} (${LLAMA_BRANCH}) with $(nvcc --version | tail -1)..."
  sync_llamacpp_fork
  cmake "$LLAMA" -B "$LLAMA/build" "${LLAMA_CMAKE_FLAGS[@]}"
  cmake --build "$LLAMA/build" --config Release -j "$(nproc)" \
    --target llama-server llama-mtmd-cli llama-cli
  printf '%s\n' "$LLAMA_WANT_STAMP" > "$LLAMA_BUILD_STAMP"
}

sync_rotorquant

if [[ ! -x "$LLAMA/build/bin/llama-server" ]]; then
  build_llamacpp
elif ! llama_origin_ok; then
  echo "Rebuilding llama.cpp (switching to RotorQuant fork)..."
  build_llamacpp
elif [[ ! -f "$LLAMA_BUILD_STAMP" ]] || [[ "$(cat "$LLAMA_BUILD_STAMP")" != "$LLAMA_WANT_STAMP" ]]; then
  echo "Rebuilding llama.cpp (RotorQuant fork / build flags changed)..."
  build_llamacpp
fi

# Install systemd user units (never enable: linger+enable would start all three)
chmod +x "$CONTROL/scripts/remote/"*.sh

# ComfyUI extra paths
if [[ -f "$CONTROL/comfyui/extra_model_paths.yaml" ]]; then
  mkdir -p "$COMFYUI_HOME"
  cp "$CONTROL/comfyui/extra_model_paths.yaml" "$COMFYUI_HOME/extra_model_paths.yaml"
fi

# ComfyUI venv (system python3 lacks sqlalchemy and other deps)
if [[ -f "$COMFYUI_HOME/main.py" ]]; then
  COMFY_VENV="${COMFYUI_HOME}/venv"
  if [[ ! -x "${COMFY_VENV}/bin/python3" ]]; then
    echo "Creating ComfyUI venv at ${COMFY_VENV}..."
    python3 -m venv "$COMFY_VENV"
  fi
  if ! "${COMFY_VENV}/bin/python3" -c "import sqlalchemy" 2>/dev/null; then
    echo "Installing ComfyUI Python dependencies (may take a few minutes)..."
    "${COMFY_VENV}/bin/pip" install -U pip
    "${COMFY_VENV}/bin/pip" install -r "${COMFYUI_HOME}/requirements.txt"
  fi
else
  echo "Warning: ComfyUI not found at ${COMFYUI_HOME}/main.py — comfy profile will fail until installed" >&2
fi

# Custom nodes (controlnet preprocessors, Kokoro TTS, etc.)
if [[ -f "$CONTROL/scripts/remote/install_custom_nodes.sh" ]]; then
  env -u VIRTUAL_ENV -u PYTHONPATH bash "$CONTROL/scripts/remote/install_custom_nodes.sh"
fi

# ComfyUI LTX (>= v0.34.3, port 8189, shares model tree)
if [[ -f "$CONTROL/scripts/remote/install_comfy_ltx.sh" ]]; then
  bash "$CONTROL/scripts/remote/install_comfy_ltx.sh"
fi

# Patch systemd with user paths
bash "$CONTROL/scripts/remote/install_systemd_units.sh"

loginctl enable-linger "$USER" 2>/dev/null || true

# Inference units stay manual; control API always on; download is oneshot only.
systemctl --user disable llama-fast.service llama-slow.service gemma.service comfyui.service comfyui-ltx.service ai-download.service 2>/dev/null || true
systemctl --user enable --now ai-control.service

echo "Bootstrap complete."
echo "Control API: http://${LAN_IP}:8090/ (UI + /api/v1)"
echo "GPU inference stays stopped until: ai start <profile> or PUT /api/v1/profile"
echo "Background download: ai download (or POST /api/v1/downloads)"
