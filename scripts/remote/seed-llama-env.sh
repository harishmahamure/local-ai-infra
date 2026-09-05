#!/usr/bin/env bash
# Seed or merge llama profile env from repo examples.
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
MODELS="${AI_MODELS:-$HOME/ai-inference/models}"
CONFIG_DIR="${HOME}/ai-inference/config"
RESEED="${RESEED_LLAMA_ENV:-0}"

# Always synced from example on bootstrap (repo defaults).
SYNC_KEYS=(CTX CACHE_TYPE_K CACHE_TYPE_V FLASH_ATTN NGL ROPE_SCALING YARN_ORIG_CTX ROPE_SCALE LLAMACPP_ROOT)

set_env_key() {
  local file="$1"
  local key="$2"
  local value="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

seed_profile() {
  local profile="$1"
  local example="${CONTROL}/llamacpp/${profile}.env.example"
  local dst="${CONFIG_DIR}/${profile}.env"

  [[ -f "$example" ]] || {
    echo "Missing example: $example" >&2
    return 1
  }

  mkdir -p "$CONFIG_DIR"
  local rendered
  rendered="$(mktemp)"
  sed -e "s|@MODELS@|${MODELS}/llamacpp|g" -e "s|@HOME@|${HOME}|g" "$example" > "$rendered"

  if [[ ! -f "$dst" ]] || [[ "$RESEED" == "1" ]]; then
    cp "$rendered" "$dst"
    if [[ "$RESEED" == "1" ]]; then
      echo "Re-seeded ${dst} from example"
    else
      echo "Created ${dst}"
    fi
    rm -f "$rendered"
    return 0
  fi

  local key value
  for key in "${SYNC_KEYS[@]}"; do
    value="$(grep -E "^${key}=" "$rendered" | tail -1 | cut -d= -f2- || true)"
    [[ -n "$value" ]] || continue
    set_env_key "$dst" "$key" "$value"
    echo "Synced ${key}=${value} in ${dst}"
  done

  while IFS= read -r line; do
    [[ "$line" =~ ^# ]] && continue
    [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if ! grep -q "^${key}=" "$dst" 2>/dev/null; then
      printf '%s=%s\n' "$key" "$value" >> "$dst"
      echo "Added ${key} to ${dst}"
    fi
  done < "$rendered"

  rm -f "$rendered"
}

for profile in llama-fast gemma; do
  seed_profile "$profile"
done
