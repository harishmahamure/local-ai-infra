#!/usr/bin/env bash
# End-to-end smoke for the 10 Qwen image operations. Run against the GPU control API.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
BASE="${GPU_CONTROL_URL:-${CONTROL_URL:-http://192.168.50.100:8090}}"

python3 - "$BASE" <<'PY'
import json, sys, time, urllib.request, urllib.error, io, struct, zlib

base = sys.argv[1].rstrip("/")


def png(w=64, h=64, rgb=(40, 80, 120)):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def req(method, path, data=None, headers=None, timeout=60):
    body = None
    hdrs = dict(headers or {})
    if data is not None and not isinstance(data, (bytes, bytearray)):
        body = json.dumps(data).encode()
        hdrs.setdefault("Content-Type", "application/json")
    else:
        body = data
    r = urllib.request.Request(base + path, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype or raw.startswith(b"{") or raw.startswith(b"["):
                return resp.status, json.loads(raw.decode() or "{}")
            return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw.decode())
        except Exception:
            payload = {"error": {"message": raw.decode(errors="replace")}}
        return exc.code, payload


def upload(name, blob):
    boundary = "----smokeBoundary"
    buf = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n".encode()
        + blob
        + f"\r\n--{boundary}--\r\n".encode()
    )
    status, data = req("POST", "/v1/assets", data=buf, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    if status >= 300:
        raise SystemExit(f"upload failed {status}: {data}")
    return data["asset_id"]


def wait_job(job_id, timeout=600):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        status, data = req("GET", f"/v1/jobs/{job_id}")
        last = data
        st = data.get("status")
        print(f"  {job_id} {st} {data.get('phase')} {data.get('progress')}")
        if st in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return data
        time.sleep(3)
    raise SystemExit(f"timeout waiting for {job_id}: {last}")


def run(operation, body, label=None):
    label = label or operation
    t0 = time.time()
    status, data = req("POST", "/v1/image/jobs", data={"operation": operation, **body})
    if status >= 300:
        print(f"FAIL {label}: submit {status} {data}")
        return False, None
    job = wait_job(data["job_id"])
    elapsed = time.time() - t0
    ok = job.get("status") == "SUCCEEDED"
    print(f"{'PASS' if ok else 'FAIL'} {label} {elapsed:.1f}s assets={job.get('asset_ids')} error={job.get('error')}")
    return ok, job


print("API", base)
status, health = req("GET", "/health")
print("health", health)
blob = png()
mask = png(rgb=(255, 255, 255))
image_id = upload("seed.png", blob)
mask_id = upload("mask.png", mask)

results = []
ok, job = run("generate_prop", {"prompt": "a brass lantern, studio product shot", "fast": True, "width": 1024, "height": 1024})
results.append(ok)
prop = (job or {}).get("asset_ids") or [image_id]
prop_id = prop[0]

ok, job = run("generate_location", {"prompt": "misty pine forest at dawn", "fast": True, "width": 1280, "height": 720})
results.append(ok)
loc_id = ((job or {}).get("asset_ids") or [image_id])[0]

ok, job = run("generate_character", {"prompt": "young ranger, leather coat, full body", "fast": True, "width": 1024, "height": 1024})
results.append(ok)
char_id = ((job or {}).get("asset_ids") or [image_id])[0]

ok, _ = run("generate_character", {"prompt": "same person in rain", "image": {"asset_id": char_id}, "fast": True})
results.append(ok)

ok, job = run("generate_character_turnaround", {"prompt": "the same ranger", "character": {"asset_id": char_id}, "fast": True})
results.append(ok)

ok, _ = run("generate_attire", {"prompt": "crimson wool cloak with brass clasp", "fast": True})
results.append(ok)

ok, _ = run("generate_keyframe", {"prompt": "the ranger at the forest edge at dusk", "character": {"asset_id": char_id}, "location": {"asset_id": loc_id}})
results.append(ok)

ok, _ = run("generate_shot_reference", {"prompt": "the ranger looking off-frame", "character": {"asset_id": char_id}, "location": {"asset_id": loc_id}, "framing": "close-up", "lens": "85", "camera_height": "eye level"})
results.append(ok)

ok, _ = run("inpaint_asset", {"prompt": "replace the object with mossy stone", "image": {"asset_id": prop_id}, "mask": {"asset_id": mask_id}})
results.append(ok)

ok, _ = run("outpaint_asset", {"prompt": "continue the scene", "image": {"asset_id": loc_id}, "left": 64, "right": 64})
results.append(ok)

ok, _ = run("upscale_asset", {"image": {"asset_id": prop_id}, "scale": 2})
results.append(ok)

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} passed")
sys.exit(1 if failed else 0)
PY
