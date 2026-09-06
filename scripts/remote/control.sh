#!/usr/bin/env bash
# GPU box lifecycle: exactly one active profile (llama-fast | gemma | comfy | comfy-ltx | tts).
set -euo pipefail

LAN_IP="${LAN_BIND_IP:-192.168.50.100}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_LTX_PORT="${COMFY_LTX_PORT:-8189}"
CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
EXCLUSIVE="${CONTROL}/scripts/remote/ensure-exclusive.sh"

UNITS=(llama-fast gemma comfyui comfyui-ltx)

usage() {
  cat <<EOF
Usage: control.sh <status|start|stop|switch|models> [llama-fast|gemma|comfy|comfy-ltx|tts]
EOF
  exit 1
}

systemctl_user() {
  systemctl --user "$@"
}

active_units_list() {
  local u
  for u in "${UNITS[@]}"; do
    if systemctl_user is-active --quiet "${u}.service" 2>/dev/null; then
      echo "$u"
    fi
  done
}

active_unit() {
  local units
  units="$(active_units_list)"
  if [[ -z "$units" ]]; then
    echo "none"
    return 0
  fi
  local count
  count="$(printf '%s\n' "$units" | grep -c .)"
  if [[ "$count" -gt 1 ]]; then
    echo "CONFLICT:${units//$'\n'/,}"
    return 0
  fi
  printf '%s\n' "$units"
}

unit_state() {
  local u="$1"
  systemctl_user is-active "${u}.service" 2>/dev/null || echo "inactive"
}

stop_all() {
  if [[ -f "$EXCLUSIVE" ]]; then
    bash "$EXCLUSIVE"
  else
    local u
    for u in "${UNITS[@]}"; do
      systemctl_user stop "${u}.service" 2>/dev/null || true
    done
  fi
}

wait_http() {
  local url="$1"
  local tries="${2:-60}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

unit_failed() {
  local unit="$1"
  local state
  state="$(systemctl_user is-failed "${unit}.service" 2>/dev/null || true)"
  [[ "$state" == "failed" ]]
}

unit_log_path() {
  local unit="$1"
  case "$unit" in
    comfyui) echo "${HOME}/ai-inference/logs/comfyui.log" ;;
    comfyui-ltx) echo "${HOME}/ai-inference/logs/comfyui-ltx.log" ;;
    llama-fast) echo "${HOME}/ai-inference/logs/llama-fast.log" ;;
    gemma) echo "${HOME}/ai-inference/logs/gemma.log" ;;
    *) echo "" ;;
  esac
}

print_unit_diagnostics() {
  local unit="$1"
  local log_path
  log_path="$(unit_log_path "$unit")"
  echo "--- systemctl status ${unit}.service ---" >&2
  systemctl_user status "${unit}.service" --no-pager -n 20 >&2 || true
  if [[ -n "$log_path" && -f "$log_path" ]]; then
    echo "--- tail ${log_path} ---" >&2
    tail -n 30 "$log_path" >&2 || true
  fi
}

wait_http_or_unit() {
  local url="$1"
  local unit="$2"
  local tries="${3:-60}"
  local i
  for i in $(seq 1 "$tries"); do
    if unit_failed "$unit"; then
      echo "${unit} entered failed state after start" >&2
      print_unit_diagnostics "$unit"
      return 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Service did not become ready at ${url}" >&2
  print_unit_diagnostics "$unit"
  return 1
}

http_ok() {
  curl -sf "$1" >/dev/null 2>&1
}

print_urls() {
  local active="${1:-}"
  case "$active" in
    llama-fast|gemma)
      echo "Active URL: http://${LAN_IP}:${LLAMA_PORT}/v1"
      ;;
    comfyui|comfy)
      echo "Active URL: http://${LAN_IP}:${COMFY_PORT}"
      ;;
    comfyui-ltx|comfy-ltx)
      echo "Active URL: http://${LAN_IP}:${COMFY_LTX_PORT}"
      ;;
    none|"")
      echo "Active URL: none (GPU idle)"
      echo "  LLM:       http://${LAN_IP}:${LLAMA_PORT}/v1   (start llama-fast|gemma)"
      echo "  Comfy:     http://${LAN_IP}:${COMFY_PORT}      (start comfy)"
      echo "  Comfy LTX: http://${LAN_IP}:${COMFY_LTX_PORT} (start comfy-ltx)"
      ;;
    *)
      echo "LLM:       http://${LAN_IP}:${LLAMA_PORT}/v1"
      echo "Comfy:     http://${LAN_IP}:${COMFY_PORT}"
      echo "Comfy LTX: http://${LAN_IP}:${COMFY_LTX_PORT}"
      ;;
  esac
}

loaded_model_hint() {
  local active="$1"
  case "$active" in
    llama-fast)
      echo "Qwen3.6-35B-A3B RotorQuant Q4 + mmproj (262K ctx)"
      ;;
    gemma)
      echo "Gemma 4 E4B Q4_K_M + mmproj (text + vision, 128K ctx)"
      ;;
    comfyui)
      echo "ComfyUI (Qwen-Image dynamic generation)"
      ;;
    comfyui-ltx|comfy-ltx)
      echo "ComfyUI LTX-2.5 (text/image to video + audio)"
      ;;
    none)
      echo "none — GPU idle"
      ;;
    CONFLICT*)
      echo "MORE THAN ONE SERVICE — run: ai stop && ai start <profile>"
      ;;
  esac
}

cmd_models() {
  if [[ -f "${CONTROL}/scripts/model_status.py" ]]; then
    source "${AI_VENV:-$HOME/ai-inference/venv}/bin/activate" 2>/dev/null || true
    export COMFYUI_ROOT="${COMFYUI_ROOT:-$HOME/ComfyUI/models}"
    export LLAMACPP_MODELS="${LLAMACPP_MODELS:-$HOME/ai-inference/models/llamacpp}"
    export INSTALLED_JSON="${INSTALLED_JSON:-$CONTROL/catalog/installed.json}"
    python "${CONTROL}/scripts/model_status.py" "$@"
  else
    echo "model_status.py not found under ${CONTROL}" >&2
    exit 1
  fi
}

cmd_status() {
  local active api_state load_state model_hint
  active="$(active_unit)"

  if [[ "$active" == "none" ]]; then
    load_state="STOPPED"
    api_state="down"
  elif [[ "$active" == CONFLICT* ]]; then
    load_state="CONFLICT"
    api_state="multiple services — exclusive GPU violated"
  elif [[ "$active" == "comfyui" ]]; then
    if http_ok "http://${LAN_IP}:${COMFY_PORT}/"; then
      load_state="LOADED"
      api_state="ready"
    else
      load_state="STARTING"
      api_state="not responding"
    fi
  elif [[ "$active" == "comfyui-ltx" ]]; then
    if http_ok "http://${LAN_IP}:${COMFY_LTX_PORT}/"; then
      load_state="LOADED"
      api_state="ready"
    else
      load_state="STARTING"
      api_state="not responding"
    fi
  else
    if http_ok "http://${LAN_IP}:${LLAMA_PORT}/v1/models" || http_ok "http://${LAN_IP}:${LLAMA_PORT}/health"; then
      load_state="LOADED"
      api_state="ready"
    else
      load_state="STARTING"
      api_state="not responding"
    fi
  fi

  model_hint="$(loaded_model_hint "$active")"

  echo "=== Runtime ==="
  printf "Exclusive:  one GPU profile at a time\n"
  printf "Load state: %s\n" "$load_state"
  printf "Profile:    %s\n" "$active"
  printf "Model:      %s\n" "$model_hint"
  printf "API:        %s\n" "$api_state"
  print_urls "$active"
  echo
  echo "--- services ---"
  for u in "${UNITS[@]}"; do
    printf "  %-14s %s\n" "${u}:" "$(unit_state "$u")"
  done
  echo "--- GPU ---"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
  echo "--- processes ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "  (no GPU processes)"
  echo
  cmd_models
}

cmd_start() {
  local target="${1:-}"
  [[ -n "$target" ]] || usage
  case "$target" in
    llama-fast|gemma|comfy|comfy-ltx|tts) ;;
    comfyui) target="comfy" ;;
    comfyui-ltx) target="comfy-ltx" ;;
    *) echo "Unknown profile: $target" >&2; exit 1 ;;
  esac

  echo "Stopping other GPU profiles (exclusive: one model)..."
  stop_all
  if [[ "$target" == "tts" ]]; then
    echo "Ready (LOADED): TTS — GPU idle for in-process Chatterbox"
    return
  fi
  local unit="$target"
  [[ "$target" == "comfy" ]] && unit="comfyui"
  [[ "$target" == "comfy-ltx" ]] && unit="comfyui-ltx"

  echo "Starting ${unit} (others remain stopped)..."
  systemctl_user start "${unit}.service"

  local leftover
  leftover="$(active_units_list | grep -v "^${unit}$" || true)"
  if [[ -n "$leftover" ]]; then
    echo "Exclusive GPU violated; extra units still active: ${leftover}" >&2
    stop_all
    exit 1
  fi

  if [[ "$target" == "comfy" || "$target" == "comfyui" ]]; then
    wait_http_or_unit "http://${LAN_IP}:${COMFY_PORT}/" "$unit" || exit 1
    echo "Ready (LOADED): http://${LAN_IP}:${COMFY_PORT}"
  elif [[ "$target" == "comfy-ltx" || "$target" == "comfyui-ltx" ]]; then
    wait_http_or_unit "http://${LAN_IP}:${COMFY_LTX_PORT}/" "$unit" || exit 1
    echo "Ready (LOADED): http://${LAN_IP}:${COMFY_LTX_PORT}"
  else
    wait_http_or_unit "http://${LAN_IP}:${LLAMA_PORT}/health" "$unit" 30 || \
    wait_http_or_unit "http://${LAN_IP}:${LLAMA_PORT}/v1/models" "$unit" 30 || {
      exit 1
    }
    echo "Ready (LOADED): http://${LAN_IP}:${LLAMA_PORT}/v1"
  fi
}

cmd_stop() {
  stop_all
  echo "All GPU services stopped (UNLOADED)."
}

ACTION="${1:-status}"
shift || true

case "$ACTION" in
  status) cmd_status ;;
  models) cmd_models "$@" ;;
  start|switch) cmd_start "${1:-}" ;;
  stop) cmd_stop ;;
  url) print_urls "$(active_unit)" ;;
  *) usage ;;
esac
