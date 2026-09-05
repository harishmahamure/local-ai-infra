#!/usr/bin/env bash
# One-time: install Mac pubkey on GPU box (you type password once).
set -euo pipefail

KEY="${HOME}/.ssh/id_ed25519_gpu"
HOST="${REMOTE_HOST:-gpu-box}"

if [[ ! -f "${KEY}.pub" ]]; then
  ssh-keygen -t ed25519 -f "${KEY}" -N "" -C "ai-project-gpu-box"
fi

echo "Installing pubkey on ${HOST}..."
ssh-copy-id -i "${KEY}.pub" "${HOST}"

echo "Testing..."
ssh "${HOST}" 'hostname && echo SSH OK'

echo "Add to your shell (optional):"
echo "  export PATH=\"$(cd "$(dirname "$0")/.." && pwd)/bin:\$PATH\""
