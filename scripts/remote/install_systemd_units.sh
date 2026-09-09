#!/usr/bin/env bash
# Install or refresh user systemd units from deploy/systemd/ (no llama.cpp rebuild).
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
LLAMA="${LLAMACPP_ROOT:-$HOME/ai-inference/llama.cpp}"
LAN_IP="${LAN_BIND_IP:-192.168.50.100}"
VENV="${AI_VENV:-$HOME/ai-inference/venv}"

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

if [[ ! -d "$CONTROL/deploy/systemd" ]]; then
  echo "Missing $CONTROL/deploy/systemd" >&2
  exit 1
fi

cp "$CONTROL/deploy/systemd/"*.service "$UNIT_DIR/"
systemctl --user disable --now llama-slow.service comfyui-ltx.service 2>/dev/null || true
rm -f "$UNIT_DIR/llama-slow.service" "$UNIT_DIR/comfyui-ltx.service"

for svc in llama-fast gemma comfyui ai-control ai-download; do
  f="$UNIT_DIR/${svc}.service"
  [[ -f "$f" ]] || continue
  sed -i "s|@HOME@|$HOME|g" "$f"
  sed -i "s|@LLAMA@|$LLAMA|g" "$f"
  sed -i "s|@LAN_IP@|$LAN_IP|g" "$f"
  sed -i "s|@VENV@|$VENV|g" "$f"
done

if [[ -f "$CONTROL/scripts/remote/seed-llama-env.sh" ]]; then
  bash "$CONTROL/scripts/remote/seed-llama-env.sh"
fi

systemctl --user daemon-reload
echo "Systemd units installed in $UNIT_DIR (daemon-reload done)."
