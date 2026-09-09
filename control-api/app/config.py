from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("AI_CONTROL", Path(__file__).resolve().parents[2]))
HOME = Path(os.environ.get("HOME", Path.home()))
LOGS = Path(os.environ.get("AI_LOGS", HOME / "ai-inference" / "logs"))
CONFIG = Path(os.environ.get("AI_CONFIG", HOME / "ai-inference" / "config"))

CONTROL_SH = ROOT / "scripts" / "remote" / "control.sh"
DOWNLOAD_STATE = Path(os.environ.get("DOWNLOAD_STATE", LOGS / "download-state.json"))
DOWNLOAD_REQUEST = Path(os.environ.get("DOWNLOAD_REQUEST", CONFIG / "download-request.json"))
DOWNLOAD_LOG = Path(os.environ.get("DOWNLOAD_LOG", LOGS / "download.log"))
DOWNLOAD_ENV = Path(os.environ.get("DOWNLOAD_ENV", CONFIG / "download.env"))

LAN_IP = os.environ.get("LAN_BIND_IP", "192.168.50.100")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "8090"))
AI_VENV = Path(os.environ.get("AI_VENV", HOME / "ai-inference" / "venv"))
PYTHON = AI_VENV / "bin" / "python"
STATE_DIR = Path(os.environ.get("AI_STATE", HOME / "ai-inference" / "state"))
ASSET_DIR = Path(os.environ.get("AI_ASSETS", HOME / "ai-inference" / "assets"))
JOBS_DB = Path(os.environ.get("AI_JOBS_DB", STATE_DIR / "jobs.db"))
COMFY_URL = os.environ.get("COMFY_URL", f"http://{LAN_IP}:{COMFY_PORT}")
COMFYUI_MODELS = Path(os.environ.get("COMFYUI_MODELS", HOME / "model_backup" / "comfyui"))

VALID_PROFILES = {"llama-fast", "gemma", "comfyui"}
COMMERCIAL_MODE = os.environ.get("AI_COMMERCIAL_MODE", "true").lower() in {"1", "true", "yes"}
CATALOG_DIR = Path(os.environ.get("AI_CATALOG_DIR", ROOT / "catalog"))
