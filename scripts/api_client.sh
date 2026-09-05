#!/usr/bin/env bash
# HTTP client helpers for ai CLI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"

CONTROL_URL="${CONTROL_URL:-http://127.0.0.1:8090}"
GPU_CONTROL_URL="${GPU_CONTROL_URL:-http://192.168.50.100:8090}"

api_base() {
  if curl -sf --max-time 2 "${CONTROL_URL}/health" >/dev/null 2>&1; then
    echo "$CONTROL_URL"
    return 0
  fi
  if curl -sf --max-time 2 "${GPU_CONTROL_URL}/health" >/dev/null 2>&1; then
    echo "$GPU_CONTROL_URL"
    return 0
  fi
  return 1
}

api_get() {
  local path="$1"
  local base
  base="$(api_base)" || {
    echo "Control API unreachable at ${CONTROL_URL} or ${GPU_CONTROL_URL}" >&2
    return 1
  }
  curl -sf "${base}${path}"
}

api_mutate() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  local base
  base="$(api_base)" || {
    echo "Control API unreachable at ${CONTROL_URL} or ${GPU_CONTROL_URL}" >&2
    return 1
  }
  local tmp
  tmp="$(mktemp)"
  local code
  if [[ -n "$body" ]]; then
    code="$(curl -s -o "$tmp" -w "%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$body" "${base}${path}")"
  else
    code="$(curl -s -o "$tmp" -w "%{http_code}" -X "$method" "${base}${path}")"
  fi
  if [[ "$code" =~ ^2 ]]; then
    cat "$tmp"
    rm -f "$tmp"
    return 0
  fi
  cat "$tmp" >&2
  rm -f "$tmp"
  return 1
}

print_status_text() {
  python3 - "$1" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
print("=== Runtime ===")
print(f"Exclusive:  one GPU profile at a time")
print(f"Load state: {s.get('loadState')}")
print(f"Profile:    {s.get('profile')}")
print(f"Model:      {s.get('model')}")
print(f"API:        {s.get('apiState')}")
url = s.get('activeUrl')
if url:
    print(f"Active URL: {url}")
else:
    eps = s.get('endpoints', {})
    print("Active URL: none (GPU idle)")
    print(f"  LLM:   {eps.get('llm')}   (start llama-fast|gemma)")
    print(f"  Comfy: {eps.get('comfy')}      (start comfy)")
print()
print("--- services ---")
for k, v in (s.get('services') or {}).items():
    print(f"  {k + ':':<12} {v}")
gpu = s.get('gpu') or {}
print("--- GPU ---")
if gpu.get('name'):
    print(f"  {gpu['name']}, {gpu.get('memoryUsed')} / {gpu.get('memoryTotal')}, util {gpu.get('utilization')}")
else:
    print("  nvidia-smi unavailable")
print("--- processes ---")
procs = s.get('gpuProcesses') or []
if procs:
    for p in procs:
        print(f"  {p.get('pid')}, {p.get('name')}, {p.get('memory')}")
else:
    print("  (no GPU processes)")
PY
}

print_models_text() {
  python3 - "$1" <<'PY'
import json, sys
data = json.loads(sys.argv[1])
bundles = data.get('bundles', [])
installed = data.get('installed_manifest', {})
print("=== Download status ===")
complete = sum(1 for b in bundles if b['status'] == 'complete')
print(f"Catalog: {complete}/{len(bundles)} complete\n")
icons = {'complete': 'OK', 'partial': 'PARTIAL', 'missing': 'MISSING'}
for b in bundles:
    icon = icons[b['status']]
    verified = 'verified' if installed.get(b['id'], {}).get('files') else 'unverified'
    print(f"[{icon:7}] {b['id']:<28} {b['files_ok']}/{b['files_required']} files  {b['size_human']:>8}  ({verified})")
    if b['status'] != 'complete':
        for f in b.get('files', []):
            if f['state'] != 'ok' and not f.get('optional'):
                print(f"          - missing: {f['name']}")
PY
}
