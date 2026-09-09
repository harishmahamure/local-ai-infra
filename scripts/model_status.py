#!/usr/bin/env python3
"""Report catalog download status (disk) and optional runtime load state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Install: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "models.yaml"
LLAMA_ROOT = Path(os.environ.get("LLAMACPP_MODELS", os.path.expanduser("~/ai-inference/models/llamacpp")))
COMFY_ROOT = Path(os.environ.get("COMFYUI_MODELS", os.path.expanduser("~/model_backup/comfyui")))
INSTALLED = Path(os.environ.get("INSTALLED_JSON", ROOT / "catalog" / "installed.json"))


def human_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def file_state(path: Path) -> dict:
    if not path.exists():
        return {"state": "missing", "bytes": 0}
    size = path.stat().st_size
    if size == 0:
        return {"state": "empty", "bytes": 0}
    return {"state": "ok", "bytes": size}


def dest_root(dest: str) -> Path:
    if dest == "comfyui":
        return COMFY_ROOT
    return LLAMA_ROOT


def bundle_status(model: dict) -> dict:
    root = dest_root(model["dest"])
    files_out = []
    ok = 0
    total_bytes = 0
    required = 0

    for spec in model.get("files", []):
        optional = bool(spec.get("optional"))
        if not optional:
            required += 1
        local = root / spec["local"]
        st = file_state(local)
        if st["state"] == "ok":
            if not optional:
                ok += 1
            total_bytes += st["bytes"]
        files_out.append(
            {
                "local": str(local),
                "name": spec["local"],
                "remote": f"{spec['repo']}/{spec['path']}",
                "optional": optional,
                **st,
            }
        )

    optional_ok = sum(1 for f in files_out if f["optional"] and f["state"] == "ok")
    if ok == required:
        status = "complete"
    elif ok == 0 and optional_ok == 0:
        status = "missing"
    else:
        status = "partial"

    return {
        "id": model["id"],
        "dest": model["dest"],
        "status": status,
        "gated": bool(model.get("gated")),
        "hfRepo": model.get("hf_repo"),
        "files_ok": ok,
        "files_required": required,
        "size_bytes": total_bytes,
        "size_human": human_size(total_bytes),
        "files": files_out,
    }


def load_catalog() -> dict:
    with open(CATALOG) as f:
        return yaml.safe_load(f)


def print_text(bundles: list[dict], installed: dict) -> None:
    print("=== Download status ===")
    print(f"llama.cpp: {LLAMA_ROOT}")
    print(f"ComfyUI:   {COMFY_ROOT}")
    print()

    complete = sum(1 for b in bundles if b["status"] == "complete")
    print(f"Catalog: {complete}/{len(bundles)} complete\n")

    for b in bundles:
        icon = {"complete": "OK", "partial": "PARTIAL", "missing": "MISSING"}[b["status"]]
        rec = installed.get(b["id"], {})
        verified = "verified" if rec.get("files") else "unverified"
        print(
            f"[{icon:7}] {b['id']:<28} {b['files_ok']}/{b['files_required']} files  "
            f"{b['size_human']:>8}  ({verified})"
        )
        if b["status"] != "complete":
            for f in b["files"]:
                if f["state"] != "ok" and not f["optional"]:
                    print(f"          - missing: {f['name']}")
                elif f["optional"] and f["state"] != "ok":
                    print(f"          - optional missing: {f['name']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalog download status on disk")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--id", action="append", dest="ids", help="Filter to model id")
    args = parser.parse_args()

    catalog = load_catalog()
    installed: dict = {}
    if INSTALLED.exists():
        installed = json.loads(INSTALLED.read_text())

    bundles = []
    for model in catalog["models"]:
        if args.ids and model["id"] not in args.ids:
            continue
        bundles.append(bundle_status(model))

    if args.json:
        print(json.dumps({"bundles": bundles, "installed_manifest": installed}, indent=2))
        return 0

    print_text(bundles, installed)
    missing = [b for b in bundles if b["status"] != "complete"]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
