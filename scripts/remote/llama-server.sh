#!/usr/bin/env bash
# Wrapper: sources profile env and launches llama-server.
set -euo pipefail

PROFILE="${1:?profile required: llama-fast|gemma}"
HOME_DIR="${HOME}"
ENV_FILE="${HOME_DIR}/ai-inference/config/${PROFILE}.env"
LAN_IP="${LAN_BIND_IP:-192.168.50.100}"
PORT="${LLAMA_PORT:-8080}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

# Gemma 4 needs ggml-org llama.cpp (iso3 fork does not know architecture 'gemma4').
LLAMA="${LLAMACPP_ROOT:-${HOME_DIR}/ai-inference/llama.cpp}"
SERVER_BIN="${LLAMA}/build/bin/llama-server"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing $SERVER_BIN — run bootstrap on the GPU box" >&2
  exit 1
fi

MODEL="${MODEL_PATH:?MODEL_PATH not set}"
MMPROJ="${MMPROJ_PATH:?MMPROJ_PATH not set}"

llama_help() {
  "$SERVER_BIN" --help 2>&1 || true
}

LLAMA_HELP_TEXT=""
llama_help_text() {
  if [[ -z "$LLAMA_HELP_TEXT" ]]; then
    LLAMA_HELP_TEXT="$(llama_help)"
  fi
  printf '%s' "$LLAMA_HELP_TEXT"
}

llama_supports_flag() {
  local flag="$1"
  llama_help_text >/dev/null
  [[ "$LLAMA_HELP_TEXT" == *"$flag"* ]]
}

llama_supports_cache_type() {
  local cache_type="$1"
  llama_help_text >/dev/null
  [[ "$LLAMA_HELP_TEXT" == *"$cache_type"* ]]
}

resolve_cache_type() {
  local requested="${1:-}"
  local fallback="${2:-q8_0}"

  [[ -n "$requested" ]] || return 0

  if llama_supports_cache_type "$requested"; then
    printf '%s\n' "$requested"
    return 0
  fi

  if [[ "$requested" != "$fallback" ]] && llama_supports_cache_type "$fallback"; then
    echo "llama-server: cache type '${requested}' not supported; using '${fallback}'" >&2
    printf '%s\n' "$fallback"
    return 0
  fi

  echo "llama-server: cache type '${requested}' not supported (no fallback)" >&2
  return 1
}

CACHE_K=""
CACHE_V=""
if [[ -n "${CACHE_TYPE_K:-}" || -n "${CACHE_TYPE_V:-}" ]]; then
  # Trust the profile env. Help-sniffing used to drop iso3 (grep -q + ARG_MAX)
  # and silently fall back to q8_0, which OOMs a 1M context on 32GB.
  CACHE_K="${CACHE_TYPE_K:-iso3}"
  CACHE_V="${CACHE_TYPE_V:-iso3}"
fi

ARGS=(
  -m "$MODEL"
  --mmproj "$MMPROJ"
  --host "$LAN_IP"
  --port "$PORT"
  -ngl "${NGL:-99}"
  -c "${CTX:-8192}"
  --temp "${TEMP:-0.7}"
  --top-p "${TOP_P:-0.95}"
  --top-k "${TOP_K:-20}"
)

if [[ -n "${FLASH_ATTN:-}" ]]; then
  ARGS+=(--flash-attn "${FLASH_ATTN}")
fi

ARGS+=(--jinja)

if [[ -n "$CACHE_K" && -n "$CACHE_V" ]]; then
  ARGS+=(--cache-type-k "$CACHE_K" --cache-type-v "$CACHE_V")
fi

if [[ -n "${PARALLEL:-}" ]]; then
  ARGS+=(--parallel "$PARALLEL")
fi

if [[ -n "${ROPE_SCALING:-}" ]]; then
  ARGS+=(--rope-scaling "$ROPE_SCALING")
  if [[ -n "${YARN_ORIG_CTX:-}" ]]; then
    ARGS+=(--yarn-orig-ctx "$YARN_ORIG_CTX")
  fi
  if [[ -n "${ROPE_SCALE:-}" ]]; then
    ARGS+=(--rope-scale "$ROPE_SCALE")
  fi
fi

[[ -n "${REASONING_EFFORT:-}" ]] && \
  ARGS+=(--chat-template-kwargs "{\"reasoning_effort\":\"${REASONING_EFFORT}\"}")

# Draft MTP is a second GGUF on the GPU. Off unless ENABLE_MTP=1 (one model at a time).
if [[ "${ENABLE_MTP:-0}" == "1" && -n "${MTP_PATH:-}" && -f "$MTP_PATH" ]]; then
  ARGS+=(-md "$MTP_PATH" --spec-type draft-mtp)
fi

echo "llama-server argv: ${ARGS[*]}" >&2
exec "$SERVER_BIN" "${ARGS[@]}"
