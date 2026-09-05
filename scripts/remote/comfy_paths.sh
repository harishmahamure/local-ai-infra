#!/usr/bin/env bash
# Resolve ComfyUI install dir vs model weights dir.
# download.env uses COMFYUI_ROOT for ~/ComfyUI/models; older scripts used it for ~/ComfyUI.
set -euo pipefail

if [[ -n "${COMFYUI_HOME:-}" ]]; then
  :
elif [[ "${COMFYUI_ROOT:-}" == */models ]]; then
  COMFYUI_HOME="${COMFYUI_ROOT%/models}"
elif [[ -n "${COMFYUI_ROOT:-}" && -f "${COMFYUI_ROOT}/main.py" ]]; then
  COMFYUI_HOME="${COMFYUI_ROOT}"
else
  COMFYUI_HOME="${HOME}/ComfyUI"
fi

if [[ -n "${COMFYUI_MODELS:-}" ]]; then
  :
elif [[ "${COMFYUI_ROOT:-}" == */models ]]; then
  COMFYUI_MODELS="${COMFYUI_ROOT}"
else
  COMFYUI_MODELS="${COMFYUI_HOME}/models"
fi

# Legacy alias used by launch/install scripts.
COMFY="${COMFYUI_HOME}"
