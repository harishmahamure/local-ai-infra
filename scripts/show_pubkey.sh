#!/usr/bin/env bash
# Print pubkey to add on GPU box if ssh-copy-id is unavailable.
set -euo pipefail
KEY="${HOME}/.ssh/id_ed25519_gpu.pub"
echo "Add this line to harishmahamure@192.168.50.100:~/.ssh/authorized_keys"
echo ""
cat "$KEY"
echo ""
echo "Then test: ssh gpu-box hostname"
