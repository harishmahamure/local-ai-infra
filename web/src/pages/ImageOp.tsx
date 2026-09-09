import { type FormEvent, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, uploadAsset } from "../api/client";
import { useStore } from "../api/store";
import type { Job, StylePreset } from "../api/types";
import { AdvancedPanel } from "../components/AdvancedPanel";
import { FALLBACK_STYLE_PRESETS, fieldsFor } from "../lib/ui";

type JobBody = {
  operation: string;
  prompt?: string;
  negative_prompt?: string;
  fast?: boolean;
  style_preset?: string;
  seed?: number;
  steps?: number;
  cfg?: number;
  sampler_name?: string;
  scheduler?: string;
  shift?: number;
  lora_strength?: number;
  control_strength?: number;
  feathering?: number;
  width?: number;
  height?: number;
  scale?: number;
  left?: number;
  right?: number;
  top?: number;
  bottom?: number;
  framing?: string;
  lens?: string;
  camera_height?: string;
  image?: { asset_id: string };
  mask?: { asset_id: string };
  character?: { asset_id: string };
  attire?: { asset_id: string };
  location?: { asset_id: string };
};

const DEFAULT_STYLE = "cinematic_naturalism";

function fileList(form: HTMLFormElement, name: string): FileList | null {
  const el = form.elements.namedItem(name);
  return el instanceof HTMLInputElement ? el.files : null;
}

function optionalNumber(data: FormData, name: string): number | undefined {
  const raw = String(data.get(name) || "").trim();
  if (!raw) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function optionalString(data: FormData, name: string): string | undefined {
  const raw = String(data.get(name) || "").trim();
  return raw || undefined;
}

function styleStorageKey(opId: string): string {
  return `imageop:style:${opId}`;
}

function loadStyle(opId: string): string {
  try {
    return window.localStorage.getItem(styleStorageKey(opId)) || DEFAULT_STYLE;
  } catch {
    return DEFAULT_STYLE;
  }
}

function uniquePresets(items: StylePreset[]): StylePreset[] {
  const seen = new Set<string>();
  const out: StylePreset[] = [];
  for (const item of items) {
    if (!item.id || seen.has(item.id)) continue;
    seen.add(item.id);
    out.push(item);
  }
  return out;
}

async function slot(files: FileList | null, field: keyof JobBody, body: JobBody) {
  const file = files?.[0];
  if (!file) return;
  body[field] = { asset_id: await uploadAsset(file) } as never;
}

function StyleSelect({ opId, presets }: { opId: string; presets: StylePreset[] }) {
  const [value, setValue] = useState(() => loadStyle(opId));

  function onChange(next: string) {
    setValue(next);
    window.localStorage.setItem(styleStorageKey(opId), next);
  }

  return (
    <label>
      Style
      <select name="style_preset" value={value} onChange={(event) => onChange(event.target.value)}>
        {presets.map((preset) => (
          <option key={preset.id} value={preset.id}>{preset.label}</option>
        ))}
      </select>
    </label>
  );
}

export function ImageOp() {
  const { op: opId } = useParams();
  const navigate = useNavigate();
  const { operations, capabilities, busy, run } = useStore();
  const [localError, setLocalError] = useState<string | null>(null);
  const [fast, setFast] = useState(false);
  const op = operations.find((item) => item.id === opId);
  const stylePresets = useMemo(
    () => uniquePresets(capabilities?.stylePresets?.length ? capabilities.stylePresets : FALLBACK_STYLE_PRESETS),
    [capabilities],
  );

  if (!opId) {
    return <section className="card"><p className="muted">Unknown operation. <Link to="/images">Back to images</Link></p></section>;
  }
  if (!operations.length) {
    return <section className="card"><p className="muted">Loading {opId}…</p></section>;
  }
  if (!op) {
    return <section className="card"><p className="muted">Unknown operation. <Link to="/images">Back to images</Link></p></section>;
  }

  const f = fieldsFor(op);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setLocalError(null);
    await run(async () => {
      const body: JobBody = { operation: opId || "" };
      const prompt = optionalString(data, "prompt");
      if (prompt) body.prompt = prompt;
      const negative = optionalString(data, "negative_prompt");
      if (negative) body.negative_prompt = negative;
      if (f.fast) body.fast = data.get("fast") === "on";
      if (f.style) body.style_preset = optionalString(data, "style_preset") || DEFAULT_STYLE;
      const seed = optionalNumber(data, "seed");
      if (seed !== undefined) body.seed = seed;
      const steps = optionalNumber(data, "steps");
      if (steps !== undefined) body.steps = steps;
      const cfg = optionalNumber(data, "cfg");
      if (cfg !== undefined) body.cfg = cfg;
      const sampler = optionalString(data, "sampler_name");
      if (sampler) body.sampler_name = sampler;
      const scheduler = optionalString(data, "scheduler");
      if (scheduler) body.scheduler = scheduler;
      const shift = optionalNumber(data, "shift");
      if (shift !== undefined) body.shift = shift;
      const lora = optionalNumber(data, "lora_strength");
      if (lora !== undefined) body.lora_strength = lora;
      const control = optionalNumber(data, "control_strength");
      if (control !== undefined) body.control_strength = control;
      const feathering = optionalNumber(data, "feathering");
      if (feathering !== undefined) body.feathering = feathering;
      const width = optionalNumber(data, "width");
      if (width !== undefined) body.width = width;
      const height = optionalNumber(data, "height");
      if (height !== undefined) body.height = height;
      if (f.scale) body.scale = Number(data.get("scale") || 4);
      if (f.padding) {
        body.left = Number(data.get("left") || 0);
        body.right = Number(data.get("right") || 0);
        body.top = Number(data.get("top") || 0);
        body.bottom = Number(data.get("bottom") || 0);
      }
      const framing = optionalString(data, "framing");
      if (framing) body.framing = framing;
      const lens = optionalString(data, "lens");
      if (lens) body.lens = lens;
      const cameraHeight = optionalString(data, "camera_height");
      if (cameraHeight) body.camera_height = cameraHeight;
      await slot(fileList(form, "image"), "image", body);
      await slot(fileList(form, "mask"), "mask", body);
      await slot(fileList(form, "character"), "character", body);
      await slot(fileList(form, "attire"), "attire", body);
      await slot(fileList(form, "location"), "location", body);
      const created = await api<Job>("/v1/image/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      navigate(`/jobs/${encodeURIComponent(created.job_id)}`);
    }).catch((err: unknown) => {
      setLocalError(err instanceof Error ? err.message : String(err));
    });
  }

  return (
    <section className="card">
      <h2>{op.label}</h2>
      <p className="muted">{op.description}</p>
      {!op.available ? <p className="missing">Missing models: {(op.missingBundles || []).join(", ")}</p> : null}
      {localError ? <p className="error">{localError}</p> : null}
      <form className="job-form" onSubmit={(event) => void onSubmit(event)}>
        {f.prompt ? <label>Prompt <textarea name="prompt" rows={3} placeholder="Describe the result" /></label> : null}
        {f.prompt ? <label>Negative <textarea name="negative_prompt" rows={2} placeholder="Optional" /></label> : null}
        <div className="job-grid">
          {f.image ? <label>Image <input type="file" name="image" accept="image/*" /></label> : null}
          {f.mask ? <label>Mask <input type="file" name="mask" accept="image/*" /></label> : null}
          {f.character ? <label>Character <input type="file" name="character" accept="image/*" /></label> : null}
          {f.attire ? <label>Attire <input type="file" name="attire" accept="image/*" /></label> : null}
          {f.location ? <label>Location <input type="file" name="location" accept="image/*" /></label> : null}
          {f.scale ? (
            <label>Scale
              <select name="scale" defaultValue="4">
                <option value="4">4x</option>
                <option value="2">2x</option>
              </select>
            </label>
          ) : null}
        </div>
        {f.padding ? (
          <div className="job-grid">
            <label>Left <input type="number" name="left" min={0} step={8} defaultValue={0} /></label>
            <label>Right <input type="number" name="right" min={0} step={8} defaultValue={0} /></label>
            <label>Top <input type="number" name="top" min={0} step={8} defaultValue={0} /></label>
            <label>Bottom <input type="number" name="bottom" min={0} step={8} defaultValue={0} /></label>
          </div>
        ) : null}
        {f.shot ? (
          <div className="job-grid">
            <label>Framing <input name="framing" placeholder="medium close-up" /></label>
            <label>Lens <input name="lens" placeholder="35" /></label>
            <label>Camera height <input name="camera_height" placeholder="eye level" /></label>
          </div>
        ) : null}
        {f.fast ? (
          <label className="check">
            <input type="checkbox" name="fast" checked={fast} onChange={(event) => setFast(event.target.checked)} />
            Fast (Lightning 4-step)
          </label>
        ) : null}
        {f.style ? <StyleSelect key={`style:${op.id}`} opId={op.id} presets={stylePresets} /> : null}
        {f.advanced ? <AdvancedPanel key={`adv:${op.id}`} op={op} fields={f} capabilities={capabilities} fast={fast} /> : null}
        <div className="actions">
          <button type="submit" disabled={busy || !op.available}>Queue job</button>
          <Link className="btn" to="/images">All operations</Link>
        </div>
      </form>
    </section>
  );
}
