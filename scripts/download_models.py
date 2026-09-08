#!/usr/bin/env python3
"""Download models from catalog/models.yaml with license gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    from tqdm.auto import tqdm
except ImportError:
    print("Install: pip install pyyaml huggingface_hub tqdm", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog" / "models.yaml"
ALLOWLIST = {"Apache-2.0", "MIT", "BSD-3-Clause", "Gemma", "LTX-2-Community"}

def _safe_models_root(env_name: str, default: str) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        raw = os.path.expandvars(os.path.expanduser(raw.strip()))
        if raw.startswith("/Users/") or raw.startswith("/Volumes/") or "${" in raw:
            raw = ""
    if not raw:
        raw = os.path.expanduser(default)
    path = Path(raw)
    if str(path).startswith(("/Users/", "/Volumes/")):
        path = Path(os.path.expanduser(default))
    return path


LLAMA_ROOT = _safe_models_root("LLAMACPP_MODELS", "~/ai-inference/models/llamacpp")
INSTALLED = Path(os.environ.get("INSTALLED_JSON", ROOT / "catalog" / "installed.json"))
DOWNLOAD_STATE = Path(
    os.environ.get("DOWNLOAD_STATE", os.path.expanduser("~/ai-inference/logs/download-state.json"))
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(**fields) -> None:
    state: dict = {
        "status": "idle",
        "current": None,
        "currentBundle": None,
        "currentFile": None,
        "currentBytes": None,
        "currentTotal": None,
        "currentPercent": None,
        "startedAt": None,
        "finishedAt": None,
        "error": None,
    }
    if DOWNLOAD_STATE.exists():
        try:
            state.update(json.loads(DOWNLOAD_STATE.read_text()))
        except json.JSONDecodeError:
            pass
    state.update(fields)
    DOWNLOAD_STATE.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_STATE.write_text(json.dumps(state, indent=2) + "\n")


class _StateTqdm(tqdm):
    """Write live byte progress into download-state.json for the control UI."""

    def __init__(self, *args, **kwargs):
        self._last_state_write = 0.0
        super().__init__(*args, **kwargs)

    def update(self, n: float | int = 1):
        result = super().update(n)
        self._emit_state(force=self.n == self.total)
        return result

    def _emit_state(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_state_write < 0.5:
            return
        self._last_state_write = now
        total = int(self.total or 0)
        n = int(self.n or 0)
        write_state(
            currentBytes=n,
            currentTotal=total or None,
            currentPercent=round(100.0 * n / total, 1) if total else None,
        )


def load_catalog() -> dict:
    with open(CATALOG) as f:
        return yaml.safe_load(f)


def dest_root(dest: str) -> Path:
    if dest != "llamacpp":
        raise RuntimeError(f"Unsupported dest {dest!r}; this downloader only fetches llamacpp GGUFs")
    return LLAMA_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gated_hint(repo: str) -> str:
    return (
        f"Gated HuggingFace repo {repo}: accept the license at "
        f"https://huggingface.co/{repo} then set HF_TOKEN in Mac .env and run "
        f"`./bin/ai download` for that bundle (syncs token to GPU)."
    )


def _http_download_error(repo: str, remote_path: str, exc: Exception, *, gated: bool) -> Exception:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code in (401, 403):
        return RuntimeError(_gated_hint(repo))
    if code == 404:
        return RuntimeError(f"HuggingFace file not found: {repo}/{remote_path}")
    if gated and code is None:
        return RuntimeError(_gated_hint(repo))
    return exc if isinstance(exc, Exception) else RuntimeError(str(exc))


def _require_token(model: dict, token: str | None) -> None:
    if model.get("gated") and not token:
        repo = model.get("hf_repo") or (model.get("files") or [{}])[0].get("repo", "")
        raise RuntimeError(
            f"HF_TOKEN required for gated bundle {model['id']}. "
            f"{_gated_hint(repo) if repo else 'Add HF_TOKEN to .env on your Mac.'}"
        )


def download_file(
    repo: str,
    remote_path: str,
    local_path: Path,
    token: str | None,
    *,
    bundle_id: str,
    gated: bool = False,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"  skip (exists): {local_path.name}")
        return
    write_state(
        status="running",
        current=f"{repo}/{remote_path}",
        currentFile=local_path.name,
        currentBundle=bundle_id,
        currentBytes=0,
        currentTotal=None,
        currentPercent=0.0,
    )
    print(f"  download: {repo}/{remote_path}")
    try:
        cached = hf_hub_download(repo_id=repo, filename=remote_path, token=token, tqdm_class=_StateTqdm)
    except GatedRepoError:
        raise RuntimeError(_gated_hint(repo)) from None
    except HfHubHTTPError as exc:
        mapped = _http_download_error(repo, remote_path, exc, gated=gated)
        if mapped is exc:
            raise
        raise mapped from exc
    shutil.copy2(cached, local_path)
    if os.environ.get("DOWNLOAD_PRUNE_CACHE") == "1":
        try:
            Path(cached).unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Download commercial-use models from catalog")
    parser.add_argument("--id", action="append", dest="ids", help="Model id (repeatable). Default: all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prune-cache", action="store_true", help="Remove HF cache blob after copy")
    args = parser.parse_args()

    if not args.dry_run:
        write_state(
            status="running",
            startedAt=_now(),
            finishedAt=None,
            error=None,
            current=None,
            failedFiles=[],
        )

    if args.prune_cache:
        os.environ["DOWNLOAD_PRUNE_CACHE"] = "1"

    try:
        catalog = load_catalog()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        installed: dict = {}
        if INSTALLED.exists():
            installed = json.loads(INSTALLED.read_text())
        failures: list[dict] = []

        for model in catalog["models"]:
            mid = model["id"]
            if args.ids and mid not in args.ids:
                continue
            license_id = model.get("license", "")
            if license_id not in ALLOWLIST:
                raise RuntimeError(f"BLOCKED {mid}: license {license_id!r} not in allowlist")
            if not model.get("commercial"):
                raise RuntimeError(f"BLOCKED {mid}: commercial flag false")

            if not args.dry_run:
                try:
                    _require_token(model, token)
                except RuntimeError as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    failures.append({"bundleId": mid, "name": mid, "error": str(exc)})
                    write_state(failedFiles=failures, lastError=str(exc))
                    continue

            if not args.dry_run:
                write_state(status="running", currentBundle=mid, current=mid)

            print(f"\n=== {mid} ({license_id}) ===")
            root = dest_root(model["dest"])
            files_meta = []
            bundle_failed = False

            for spec in model.get("files", []):
                if spec.get("optional") and args.dry_run:
                    continue
                local = root / spec["local"]
                if args.dry_run:
                    print(f"  would fetch: {spec['repo']}/{spec['path']} -> {local}")
                    continue
                try:
                    download_file(
                        spec["repo"],
                        spec["path"],
                        local,
                        token,
                        bundle_id=mid,
                        gated=bool(model.get("gated")),
                    )
                    files_meta.append({"local": str(local), "sha256": sha256_file(local)})
                except Exception as e:
                    print(f"  {'optional skip' if spec.get('optional') else 'ERROR'}: {e}", file=sys.stderr)
                    if spec.get("optional"):
                        continue
                    bundle_failed = True
                    failures.append(
                        {
                            "bundleId": mid,
                            "name": spec["local"],
                            "error": str(e),
                        }
                    )
                    write_state(failedFiles=failures, lastError=str(e))

            if not args.dry_run and not bundle_failed:
                installed[mid] = {
                    "license": license_id,
                    "dest": model["dest"],
                    "files": files_meta,
                }

        if not args.dry_run:
            INSTALLED.parent.mkdir(parents=True, exist_ok=True)
            INSTALLED.write_text(json.dumps(installed, indent=2) + "\n")
            print(f"\nWrote {INSTALLED}")
            if failures:
                summary = "; ".join(f"{f['bundleId']}: {f['error']}" for f in failures)
                write_state(
                    status="failed",
                    current=None,
                    currentFile=None,
                    currentBundle=None,
                    currentBytes=None,
                    currentTotal=None,
                    currentPercent=None,
                    finishedAt=_now(),
                    error=summary,
                    failedFiles=failures,
                )
                print(f"ERROR: {len(failures)} file(s) failed. Other bundles finished.", file=sys.stderr)
                return 1
            write_state(
                status="completed",
                current=None,
                currentFile=None,
                currentBundle=None,
                currentBytes=None,
                currentTotal=None,
                currentPercent=None,
                finishedAt=_now(),
                error=None,
                failedFiles=[],
            )
        return 0
    except Exception as exc:
        if not args.dry_run:
            write_state(status="failed", finishedAt=_now(), error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
