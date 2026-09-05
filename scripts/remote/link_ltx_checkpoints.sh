#!/usr/bin/env bash
# Symlink split LTX weights into checkpoints/ for loaders that still expect that folder.
set -euo pipefail

MODELS="${COMFYUI_MODELS:-$HOME/ComfyUI/models}"
CKPT="${MODELS}/checkpoints"
mkdir -p "${CKPT}"

link_if_target() {
  local name="$1"
  local target="$2"
  if [[ ! -f "${target}" ]]; then
    return 0
  fi
  if [[ -e "${CKPT}/${name}" && ! -L "${CKPT}/${name}" ]]; then
    echo "Skip ${name}: ${CKPT}/${name} exists and is not a symlink" >&2
    return 0
  fi
  ln -sfn "${target}" "${CKPT}/${name}"
  echo "Linked ${CKPT}/${name} -> ${target}"
}

TRANSFORMER="${MODELS}/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
AUDIO_VAE="${MODELS}/vae/ltx-2.5-audio-vae-bf16.safetensors"

link_if_target "ltx-2.5-distilled-transformer-comfy-int8-convrot.safetensors" "${TRANSFORMER}"
link_if_target "ltx-2.5-audio-vae-bf16.safetensors" "${AUDIO_VAE}"
