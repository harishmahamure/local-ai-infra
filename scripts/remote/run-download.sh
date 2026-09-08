#!/usr/bin/env bash
# Run catalog download as a systemd oneshot (survives Mac SSH disconnect).
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
VENV="${AI_VENV:-$HOME/ai-inference/venv}"
CONFIG="${AI_CONFIG:-$HOME/ai-inference/config}"
LOGS="${AI_LOGS:-$HOME/ai-inference/logs}"
REQUEST="${CONFIG}/download-request.json"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ -f "${CONFIG}/download.env" ]]; then
  # shellcheck disable=SC1090
  set -a && source "${CONFIG}/download.env" && set +a
fi

_expand_home() {
  local val="${1:-}"
  val="${val//\$HOME/$HOME}"
  printf '%s' "$val"
}

_sanitize_path_var() {
  local name="$1"
  local val="${!name:-}"
  [[ -z "$val" ]] && return
  val="$(_expand_home "$val")"
  if [[ "$val" == *'${'* ]] || [[ "$val" == /Users/* ]] || [[ "$val" == /Volumes/* ]]; then
    unset "$name"
    return
  fi
  printf -v "$name" '%s' "$val"
}

_sanitize_path_var LLAMACPP_MODELS
_sanitize_path_var HF_HOME

export LLAMACPP_MODELS="${LLAMACPP_MODELS:-$HOME/ai-inference/models/llamacpp}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export INSTALLED_JSON="${INSTALLED_JSON:-$CONTROL/catalog/installed.json}"
export DOWNLOAD_STATE="${LOGS}/download-state.json"

mkdir -p "$LOGS" "$CONFIG" "$LLAMACPP_MODELS" "$HF_HOME"
cd "$CONTROL"

extra=()
if [[ -f "$REQUEST" ]]; then
  while IFS= read -r id; do
    [[ -n "$id" ]] && extra+=(--id "$id")
  done < <(python3 -c "
import json
from pathlib import Path
p = Path('${REQUEST}')
if p.exists():
    for i in json.loads(p.read_text()).get('ids') or []:
        print(i)
")
fi

exec python3 scripts/download_models.py "${extra[@]}"
