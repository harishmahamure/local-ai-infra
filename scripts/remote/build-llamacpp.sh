#!/usr/bin/env bash
# Build johndpope/llama-cpp-turboquant (iso/RotorQuant KV) into ~/llama-cpp-turboquant
# and point ~/ai-inference/llama.cpp at it. Gemma stays on ~/llama.cpp (ggml-org, no iso).
set -euo pipefail

if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found. Add /usr/local/cuda/bin to PATH." >&2
  exit 1
fi

LLAMA="${LLAMACPP_ROOT:-$HOME/ai-inference/llama.cpp}"
TURBOQUANT="${TURBOQUANT_ROOT:-$HOME/llama-cpp-turboquant}"
LLAMA_ORIGIN="https://github.com/johndpope/llama-cpp-turboquant.git"
LLAMA_BRANCH="feature/planarquant-kv-cache"
LLAMA_PATCH_ID="gcc13-extern-c-ggml-api"
LLAMA_CMAKE_FLAGS=(
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_CUDA=ON
  -DGGML_CUDA_FA_ALL_QUANTS=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES=native
)
LLAMA_BUILD_STAMP="${TURBOQUANT}/build/.ai-project-build-stamp"
LLAMA_WANT_STAMP="$(printf '%s\n' "${LLAMA_ORIGIN}" "${LLAMA_BRANCH}" "${LLAMA_PATCH_ID}" "${LLAMA_CMAKE_FLAGS[@]}" | sha256sum | awk '{print $1}')"

llama_origin_ok() {
  [[ -d "$1/.git" ]] || return 1
  local origin
  origin="$(git -C "$1" remote get-url origin 2>/dev/null || true)"
  [[ "$origin" == *johndpope/llama-cpp-turboquant* ]]
}

sync_fork() {
  if [[ -e "$TURBOQUANT" ]] && ! llama_origin_ok "$TURBOQUANT"; then
    echo "Replacing non-RotorQuant tree at ${TURBOQUANT}..."
    rm -rf "$TURBOQUANT"
  fi
  if [[ ! -d "$TURBOQUANT/.git" ]]; then
    git clone --depth 1 --branch "$LLAMA_BRANCH" "$LLAMA_ORIGIN" "$TURBOQUANT"
  else
    if git -C "$TURBOQUANT" fetch --depth 1 origin "$LLAMA_BRANCH"; then
      git -C "$TURBOQUANT" checkout -B "$LLAMA_BRANCH" FETCH_HEAD
    else
      echo "git fetch failed; building existing ${TURBOQUANT} tree"
    fi
  fi
  local ops="$TURBOQUANT/ggml/src/ggml-cpu/ops.cpp"
  if [[ -f "$ops" ]] && grep -q 'extern "C" GGML_API ' "$ops"; then
    sed -i 's/extern "C" GGML_API /extern "C" /g' "$ops"
    echo "Patched ${ops} for GCC 13 (${LLAMA_PATCH_ID})"
  fi
}

build_fork() {
  echo "Building iso/RotorQuant llama-server from ${LLAMA_ORIGIN} (${LLAMA_BRANCH}) with $(nvcc --version | tail -1)..."
  sync_fork
  cmake "$TURBOQUANT" -B "$TURBOQUANT/build" "${LLAMA_CMAKE_FLAGS[@]}"
  cmake --build "$TURBOQUANT/build" --config Release -j "$(nproc)" \
    --target llama-server llama-mtmd-cli llama-cli
  printf '%s\n' "$LLAMA_WANT_STAMP" > "$LLAMA_BUILD_STAMP"
}

FORCE="${FORCE_LLAMA_REBUILD:-0}"
if [[ "$FORCE" == "1" ]] || [[ ! -x "$TURBOQUANT/build/bin/llama-server" ]]; then
  build_fork
elif [[ ! -f "$LLAMA_BUILD_STAMP" ]] || [[ "$(cat "$LLAMA_BUILD_STAMP")" != "$LLAMA_WANT_STAMP" ]]; then
  echo "Rebuilding llama.cpp (RotorQuant fork / build flags changed)..."
  build_fork
else
  echo "RotorQuant llama-server already built (${LLAMA_WANT_STAMP:0:12}…)"
fi

mkdir -p "$(dirname "$LLAMA")"
ln -sfn "$TURBOQUANT" "$LLAMA"
echo "llama-fast binary: ${LLAMA}/build/bin/llama-server"
"$LLAMA/build/bin/llama-server" --version 2>&1 | tail -5
