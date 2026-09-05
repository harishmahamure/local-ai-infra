#!/usr/bin/env bash
# Move models downloaded to a broken COMFYUI_ROOT path into ~/ComfyUI/models.
set -euo pipefail

WRONG_DIR="${1:-$HOME/ai-inference/control/\$\{COMFYUI_ROOT:-\$HOME/ComfyUI/models\}}"
RIGHT_ROOT="${COMFYUI_ROOT:-$HOME/ComfyUI/models}"

if [[ ! -d "$WRONG_DIR" ]]; then
  echo "Nothing to recover (missing: $WRONG_DIR)"
  exit 0
fi

echo "Recovering from: $WRONG_DIR"
echo "Into: $RIGHT_ROOT"

moved=0
skipped=0

while IFS= read -r -d '' src; do
  rel="${src#"$WRONG_DIR"/}"
  dest="$RIGHT_ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then
    if [[ "$(stat -c%s "$dest")" -ge "$(stat -c%s "$src")" ]]; then
      echo "  skip (exists): $rel"
      skipped=$((skipped + 1))
      continue
    fi
    echo "  replace smaller: $rel"
  else
    echo "  move: $rel"
  fi
  mv -f "$src" "$dest"
  moved=$((moved + 1))
done < <(find "$WRONG_DIR" -type f -print0)

find "$WRONG_DIR" -type d -empty -delete 2>/dev/null || true
rmdir "$WRONG_DIR" 2>/dev/null || true

echo "Done. moved=$moved skipped=$skipped"
