"""ComfyUI API-format graph for ACE-Step 1.5 text-to-audio."""

from __future__ import annotations

from typing import Any

UNET = "acestep_v1.5_turbo.safetensors"
CLIP = "qwen_1.7b_ace15.safetensors"
VAE = "ace_1.5_vae.safetensors"


def build(plan: dict[str, Any]) -> dict[str, Any]:
    prompt = str(plan.get("prompt") or "")
    duration = max(0.25, min(180.0, float(plan.get("duration_seconds") or 8)))
    seed = int(plan.get("seed") or 0)
    steps = int(plan.get("steps") or 8)
    cfg = float(plan.get("cfg") or 4.0)
    prefix = str(plan.get("filename_prefix") or "engine_ace")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "stable_audio"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptyLatentAudio", "inputs": {"seconds": duration, "batch_size": 1}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
            },
        },
        "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveAudio", "inputs": {"audio": ["7", 0], "filename_prefix": prefix}},
    }
