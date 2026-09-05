# GPU-box path helpers. Source from Mac scripts after loading .env.
# `~` in .env expands to THIS machine's home; do not use that over SSH.

gpu_is_local_home_path() {
  local p="${1:-}"
  [[ -z "$p" || "$p" == /Users/* || "$p" == /Volumes/* ]]
}

# Path for ssh remote shells (evaluated on the GPU box).
gpu_control_ssh() {
  if gpu_is_local_home_path "${REMOTE_AI_DIR:-}"; then
    printf '%s' '$HOME/ai-inference/control'
  else
    printf '%s' "$REMOTE_AI_DIR"
  fi
}

# Path for rsync dest (remote ~ is expanded by ssh/rsync, not this Mac).
gpu_control_rsync() {
  if gpu_is_local_home_path "${REMOTE_AI_DIR:-}"; then
    printf '%s' '~/ai-inference/control'
  else
    printf '%s' "$REMOTE_AI_DIR"
  fi
}
