#!/usr/bin/env bash
# Copy project LTX ComfyUI workflows into ~/ComfyUI-ltx/blueprints for easy discovery.
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
COMFY_LTX="${COMFYUI_LTX_ROOT:-$HOME/ComfyUI-ltx}"
SRC="${CONTROL}/comfyui/workflows"
DEST="${COMFY_LTX}/blueprints"

if [[ ! -d "${SRC}" ]]; then
  echo "No workflows at ${SRC}; skip."
  exit 0
fi

mkdir -p "${DEST}"
count=0
for f in "${SRC}"/*.json; do
  [[ -f "$f" ]] || continue
  cp -f "$f" "${DEST}/$(basename "$f")"
  count=$((count + 1))
done
echo "Installed ${count} LTX workflow(s) to ${DEST}"
