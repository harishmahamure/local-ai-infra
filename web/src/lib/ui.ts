import type { Operation, StylePreset } from "../api/types";

export const UNLOAD_MODEL = "gemma-4-e4b";
export const GENERATE_OPS = [
  "generate_character",
  "generate_character_turnaround",
  "generate_attire",
  "generate_location",
  "generate_prop",
  "generate_keyframe",
  "generate_shot_reference",
];
export const EDIT_OPS = ["inpaint_asset", "outpaint_asset", "upscale_asset"];
export const STYLED_OPS = [
  "generate_character",
  "generate_character_turnaround",
  "generate_keyframe",
  "generate_shot_reference",
];
export const FALLBACK_STYLE_PRESETS: StylePreset[] = [
  { id: "cinematic_naturalism", label: "Cinematic Naturalism" },
  { id: "painterly_concept", label: "Painterly concept" },
  { id: "graphic_still", label: "Graphic still" },
  { id: "photoreal_cinematic", label: "Photoreal cinematic" },
  { id: "documentary", label: "Documentary" },
  { id: "off", label: "Off" },
];
export const FALLBACK_ASPECTS = [
  { id: "1:1", width: 1328, height: 1328 },
  { id: "16:9", width: 1664, height: 928 },
  { id: "9:16", width: 928, height: 1664 },
  { id: "4:3", width: 1472, height: 1104 },
  { id: "3:4", width: 1104, height: 1472 },
  { id: "3:2", width: 1584, height: 1056 },
  { id: "2:3", width: 1056, height: 1584 },
];

export const CHAT_CURL = `curl -X POST /v1/text/chat \\
  -H 'Content-Type: application/json' \\
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`;

export const IMAGE_CURL = `curl -X POST /v1/image/jobs \\
  -H 'Content-Type: application/json' \\
  -d '{
    "operation": "generate_prop",
    "prompt": "a brass lantern"
  }'`;

export function badgeClass(state: string | undefined): string {
  const s = (state || "STOPPED").toUpperCase();
  if (s === "LOADED") return "loaded";
  if (s === "STARTING") return "starting";
  if (s.startsWith("CONFLICT")) return "conflict";
  return "stopped";
}

export function jobBadgeClass(status: string | undefined): string {
  const s = (status || "").toUpperCase();
  if (s === "SUCCEEDED") return "loaded";
  if (s === "RUNNING" || s === "QUEUED") return "starting";
  if (s === "FAILED") return "conflict";
  return "stopped";
}

export function parseMib(value: string | null | undefined): number | null {
  if (!value) return null;
  const n = parseFloat(String(value).replace(/[^\d.]/g, ""));
  return Number.isFinite(n) ? n : null;
}

export function vramPercent(used?: string | null, total?: string | null): number {
  const u = parseMib(used);
  const t = parseMib(total);
  if (!u || !t) return 0;
  return Math.min(100, (u / t) * 100);
}

export type FieldFlags = {
  prompt: boolean;
  fast: boolean;
  image: boolean;
  mask: boolean;
  character: boolean;
  attire: boolean;
  location: boolean;
  padding: boolean;
  scale: boolean;
  shot: boolean;
  style: boolean;
  controlStrength: boolean;
  feathering: boolean;
  advanced: boolean;
};

export function fieldsFor(op: Operation): FieldFlags {
  const slots = op.referenceSlots || [];
  return {
    prompt: op.promptRequired !== false,
    fast: op.id !== "upscale_asset",
    image: slots.includes("image"),
    mask: Boolean(op.requiresMask) || slots.includes("mask"),
    character: slots.includes("character"),
    attire: slots.includes("attire"),
    location: slots.includes("location"),
    padding: op.id === "outpaint_asset",
    scale: op.id === "upscale_asset",
    shot: op.id === "generate_shot_reference",
    style: STYLED_OPS.includes(op.id),
    controlStrength: op.id === "inpaint_asset" || op.id === "outpaint_asset",
    feathering: op.id === "outpaint_asset",
    advanced: op.id !== "upscale_asset",
  };
}
