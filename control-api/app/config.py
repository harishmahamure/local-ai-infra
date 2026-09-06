from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AI_CONTROL", Path(__file__).resolve().parents[2]))
HOME = Path(os.environ.get("HOME", Path.home()))
LOGS = Path(os.environ.get("AI_LOGS", HOME / "ai-inference" / "logs"))
CONFIG = Path(os.environ.get("AI_CONFIG", HOME / "ai-inference" / "config"))

CONTROL_SH = ROOT / "scripts" / "remote" / "control.sh"
LINK_LTX_CHECKPOINTS_SH = ROOT / "scripts" / "remote" / "link_ltx_checkpoints.sh"
DOWNLOAD_STATE = Path(os.environ.get("DOWNLOAD_STATE", LOGS / "download-state.json"))
DOWNLOAD_REQUEST = Path(os.environ.get("DOWNLOAD_REQUEST", CONFIG / "download-request.json"))
DOWNLOAD_LOG = Path(os.environ.get("DOWNLOAD_LOG", LOGS / "download.log"))
DOWNLOAD_ENV = Path(os.environ.get("DOWNLOAD_ENV", CONFIG / "download.env"))

LAN_IP = os.environ.get("LAN_BIND_IP", "192.168.50.100")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_LTX_PORT = int(os.environ.get("COMFY_LTX_PORT", "8189"))
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8090"))
AI_VENV = Path(os.environ.get("AI_VENV", HOME / "ai-inference" / "venv"))
PYTHON = AI_VENV / "bin" / "python"

VALID_PROFILES = {"llama-fast", "comfy", "comfy-ltx", "gemma", "tts"}
CHATTERBOX_MODELS = Path(os.environ.get("CHATTERBOX_MODELS", HOME / "ai-inference" / "models" / "chatterbox"))

ENGINE_DB = Path(os.environ.get("AI_ENGINE_DB", LOGS / "engine.sqlite"))
ARTIFACT_ROOT = Path(os.environ.get("AI_ARTIFACT_ROOT", LOGS / "artifacts"))
ARTIFACT_TTL_HOURS = int(os.environ.get("AI_ARTIFACT_TTL_HOURS", "24"))
COMMERCIAL_MODE = os.environ.get("AI_COMMERCIAL_MODE", "true").lower() in {"1", "true", "yes"}
CATALOG_DIR = Path(os.environ.get("AI_CATALOG_DIR", ROOT / "catalog"))
MAX_UPLOAD_BYTES = int(os.environ.get("AI_MAX_UPLOAD_BYTES", str(80 * 1024 * 1024)))
MAX_PROMPT_CHARS = int(os.environ.get("AI_MAX_PROMPT_CHARS", "8000"))

# Planner routing: Gemma handles text and vision (no mid-job profile switch).
PLANNER_PROFILE_TEXT = "gemma"
PLANNER_PROFILE_VISION = "gemma"
PLANNER_MODEL_TEXT = "gemma-4-e4b"
PLANNER_MODEL_VISION = "gemma-4-e4b"
