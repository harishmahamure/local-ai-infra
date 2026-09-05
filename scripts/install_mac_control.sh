#!/usr/bin/env bash
# Install Mac launchd proxy for AI control API (port 8090).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"

VENV="${ROOT}/.venv"
GPU_URL="${GPU_CONTROL_URL:-http://192.168.50.100:8090}"
PLIST_SRC="$ROOT/deploy/launchd/com.ai-project.control.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.ai-project.control.plist"
DOMAIN="gui/$(id -u)"
LABEL="com.ai-project.control"
SERVICE="${DOMAIN}/${LABEL}"

mkdir -p "$ROOT/logs" "$HOME/Library/LaunchAgents"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PIP_NO_CACHE_DIR=1
pip install -q -U pip fastapi uvicorn httpx pyyaml huggingface_hub

tmp_plist="$(mktemp "${TMPDIR:-/tmp}/com.ai-project.control.XXXXXX.plist")"
trap 'rm -f "$tmp_plist"' EXIT

sed -e "s|@ROOT@|$ROOT|g" \
    -e "s|@VENV@|$VENV|g" \
    -e "s|@GPU_CONTROL_URL@|$GPU_URL|g" \
    "$PLIST_SRC" > "$tmp_plist"

if ! plutil -lint "$tmp_plist"; then
  echo "Invalid launchd plist generated from $PLIST_SRC" >&2
  exit 1
fi

mv "$tmp_plist" "$PLIST_DST"
trap - EXIT

# Unload any previous registration (domain + plist path is the supported form).
launchctl bootout "$DOMAIN" "$PLIST_DST" 2>/dev/null || true
launchctl bootout "$SERVICE" 2>/dev/null || true
sleep 0.5

if launchctl print "$SERVICE" &>/dev/null; then
  echo "Mac control proxy already loaded; restarting..."
  launchctl kickstart -k "$SERVICE"
elif launchctl bootstrap "$DOMAIN" "$PLIST_DST"; then
  launchctl enable "$SERVICE" 2>/dev/null || true
else
  echo "launchctl bootstrap failed; try manually:" >&2
  echo "  launchctl bootout $DOMAIN $PLIST_DST" >&2
  echo "  launchctl bootstrap $DOMAIN $PLIST_DST" >&2
  echo "Or run without launchd:" >&2
  echo "  cd $ROOT && $VENV/bin/uvicorn control-api.mac_proxy:app --host 0.0.0.0 --port 8090" >&2
  exit 1
fi

echo "Mac control proxy installed."
echo "  UI/API: http://127.0.0.1:8090"
echo "  Upstream GPU: $GPU_URL"
echo "  Logs: $ROOT/logs/mac-control.log"
