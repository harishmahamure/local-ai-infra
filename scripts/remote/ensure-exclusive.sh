#!/usr/bin/env bash
# GPU mutex: at most one inference process (llama-fast | gemma).
#
#   ensure-exclusive.sh              stop every managed unit + stray processes
#   ensure-exclusive.sh UNIT.service ExecStartPre: kill strays only
set -u

SELF_UNIT="${1:-}"
SELF_NAME="${SELF_UNIT%.service}"
UNITS=(llama-fast gemma)
PORTS=(8080)
LOCK_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
LOCK_FILE="${LOCK_DIR}/ai-inference.profile"

systemctl_user() {
  systemctl --user "$@"
}

stop_managed_units() {
  local u
  for u in "${UNITS[@]}"; do
    if [[ -n "$SELF_NAME" && "$u" == "$SELF_NAME" ]]; then
      continue
    fi
    systemctl_user stop "${u}.service" 2>/dev/null || true
  done
}

pids_on_port() {
  local port="$1"
  ss -lntpH "sport = :${port}" 2>/dev/null \
    | grep -oE 'pid=[0-9]+' \
    | cut -d= -f2 \
    | sort -u
}

kill_pid() {
  local pid="$1"
  [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || return 0
  kill "$pid" 2>/dev/null || true
}

kill_stray_inference() {
  local pid port
  local -a extra=()

  for port in "${PORTS[@]}"; do
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      extra+=("$pid")
    done < <(pids_on_port "$port")
  done

  while read -r pid; do
    [[ -n "$pid" ]] || continue
    extra+=("$pid")
  done < <(pgrep -f '/llama-server( |$)' 2>/dev/null || true)

  if ((${#extra[@]} == 0)); then
    return 0
  fi

  local -A seen=()
  for pid in "${extra[@]}"; do
    [[ -z "${seen[$pid]:-}" ]] || continue
    seen[$pid]=1
    kill_pid "$pid"
  done

  sleep 1
  for pid in "${!seen[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
}

wait_gpu_compute_idle() {
  local tries="${1:-20}"
  local i apps
  for i in $(seq 1 "$tries"); do
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
    if [[ "${apps:-0}" -eq 0 ]]; then
      return 0
    fi
    sleep 1
  done
  echo "warning: GPU still has ${apps:-?} compute process(es) after unload wait" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null >&2 || true
  return 0
}

mkdir -p "$LOCK_DIR" 2>/dev/null || true

if [[ -z "$SELF_UNIT" ]]; then
  stop_managed_units
fi

kill_stray_inference
wait_gpu_compute_idle 20

if [[ -n "$SELF_NAME" ]]; then
  printf '%s\n' "$SELF_NAME" >"$LOCK_FILE" 2>/dev/null || true
else
  rm -f "$LOCK_FILE" 2>/dev/null || true
fi

exit 0
