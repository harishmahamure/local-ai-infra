#!/usr/bin/env bash
# Hardware + software inventory on GPU box.
set -euo pipefail

HOST="${REMOTE_HOST:-gpu-box}"

ssh "${HOST}" 'bash -s' <<'REMOTE'
set -euo pipefail
echo "=== HOST ==="
hostname
uname -a
echo "=== OS ==="
cat /etc/os-release 2>/dev/null | head -5 || true
echo "=== CPU/MEM ==="
lscpu 2>/dev/null | sed -n "1,12p" || true
free -h
echo "=== DISK ==="
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null || true
df -hT
echo "=== NVIDIA ==="
nvidia-smi || true
echo "=== TOOLS ==="
for c in python3 pip3 git cmake nvcc docker; do
  printf "%-8s " "$c"
  command -v "$c" >/dev/null && $c --version 2>/dev/null | head -1 || echo "missing"
done
echo "=== PATHS ==="
ls -la ~/ComfyUI 2>/dev/null | head -5 || echo "ComfyUI not found"
ls -la ~/ai-inference 2>/dev/null | head -5 || echo "ai-inference not found"
REMOTE
