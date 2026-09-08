#!/usr/bin/env bash
# Install pinned ComfyUI custom nodes from comfyui/custom_nodes.yaml
set -euo pipefail

CONTROL="${AI_CONTROL:-$HOME/ai-inference/control}"
# shellcheck disable=SC1091
source "${CONTROL}/scripts/remote/comfy_paths.sh"
# Image/ControlNet nodes must live on the comfy-profile runtime (ComfyUI-ltx on :8188).
# ~/ComfyUI is the shared weights tree and is not launched as a ComfyUI process.
RUNTIME="${COMFYUI_LTX_ROOT:-$HOME/ComfyUI-ltx}"
MANIFEST="${CONTROL}/comfyui/custom_nodes.yaml"
CUSTOM="${RUNTIME}/custom_nodes"
COMFY_VENV="${RUNTIME}/venv"

if [[ ! -f "$MANIFEST" ]]; then
  echo "No custom_nodes.yaml at $MANIFEST — skipping"
  exit 0
fi

if [[ ! -d "$RUNTIME" ]]; then
  echo "ComfyUI runtime not found at $RUNTIME — skipping custom nodes"
  exit 0
fi

mkdir -p "$CUSTOM"

if [[ ! -x "${COMFY_VENV}/bin/pip" ]]; then
  if [[ -f "${RUNTIME}/main.py" ]]; then
    echo "Creating ComfyUI venv at ${COMFY_VENV}..."
    python3 -m venv "${COMFY_VENV}"
    "${COMFY_VENV}/bin/pip" install -U pip wheel setuptools
    echo "Installing ComfyUI base requirements (includes torch)..."
    "${COMFY_VENV}/bin/pip" install -r "${RUNTIME}/requirements.txt"
  else
    echo "ComfyUI venv missing at ${COMFY_VENV} — skipping custom-node pip deps" >&2
    exit 0
  fi
fi

PIP="${COMFY_VENV}/bin/pip"
export CONTROL CUSTOM PIP

python3 << 'PY'
import os
import shutil
import subprocess
from pathlib import Path

import yaml

control = Path(os.environ["CONTROL"])
custom = Path(os.environ["CUSTOM"])
manifest_path = control / "comfyui" / "custom_nodes.yaml"
pip = os.environ["PIP"]

SKIP_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
}


def _filter_requirements(req_path: Path) -> list[str]:
    kept: list[str] = []
    for raw in req_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pkg_part = line.split(";", 1)[0].strip()
        base = pkg_part.split("[", 1)[0]
        for sep in ("==", ">=", "<=", "!=", "~=", ">", "<"):
            if sep in base:
                base = base.split(sep, 1)[0]
                break
        if base.strip().lower() in SKIP_PACKAGES:
            print(f"    skip torch stack dep: {pkg_part}")
            continue
        kept.append(raw)
    return kept


def _pip_install_requirements(req_path: Path) -> None:
    filtered = _filter_requirements(req_path)
    if not filtered:
        print("    (no extra pip deps after filtering torch stack)")
        return
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(filtered) + "\n")
        tmp = handle.name
    try:
        subprocess.run([pip, "install", "-r", tmp], check=True)
    finally:
        os.unlink(tmp)

data = yaml.safe_load(manifest_path.read_text())
nodes = data.get("nodes") or []


def _git_update(dest: Path, pin: str | None) -> None:
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin"], check=False)
    if pin:
        subprocess.run(["git", "-C", str(dest), "checkout", pin], check=True)
    else:
        branch = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "origin", branch], check=False)


def _git_clone(repo: str, dest: Path, pin: str | None) -> None:
    if pin:
        subprocess.run(["git", "clone", repo, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", pin], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)


for entry in nodes:
    node_id = entry["id"]
    repo = entry["repo"]
    dirname = entry["dir"]
    pin = entry.get("pin")
    dest = custom / dirname

    if dest.is_dir() and (dest / ".git").is_dir():
        print(f"  update {node_id} ({dirname})")
        _git_update(dest, pin)
    elif dest.is_dir():
        print(f"  replace non-git {node_id} ({dirname})")
        shutil.rmtree(dest)
        print(f"  clone {node_id} -> {dirname}")
        _git_clone(repo, dest, pin)
    else:
        print(f"  clone {node_id} -> {dirname}")
        _git_clone(repo, dest, pin)

    req = entry.get("pip")
    if req and (dest / req).is_file():
        print(f"    pip install -r {dirname}/{req} (into ComfyUI venv, torch excluded)")
        _pip_install_requirements(dest / req)

print("Custom nodes install complete.")
PY

echo "Custom nodes ready under ${CUSTOM}"
