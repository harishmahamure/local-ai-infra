import { type FormEvent, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, uploadAsset } from "../api/client";
import { useStore } from "../api/store";
import type { Job } from "../api/types";
import { AdvancedPanel } from "../components/AdvancedPanel";
import { VideoAdvancedPanel } from "../components/VideoAdvancedPanel";
import { fieldsFor, LIVE_WALLPAPER_OPS, LTX_VIDEO_OPS, WALLPAPER_OPS } from "../lib/ui";

type JobBody = {
  operation: string;
  prompt?: string;
  negative_prompt?: string;
  target?: string;
  upscale?: boolean;
  refine?: boolean;
  fast?: boolean;
  seed?: number;
  steps?: number;
  cfg?: number;
  video_cfg?: number;
  duration?: number;
  fps?: number;
  sampler_name?: string;
  scheduler?: string;
  shift?: number;
  lora_strength?: number;
  width?: number;
  height?: number;
  image?: { asset_id: string };
};

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

function fileList(form: HTMLFormElement, name: string): FileList | null {
  const el = form.elements.namedItem(name);
  return el instanceof HTMLInputElement ? el.files : null;
}

export function WallpaperOp() {
  const { op: opId } = useParams();
  const navigate = useNavigate();
  const { operations, videoOperations, capabilities, busy, run } = useStore();
  const [localError, setLocalError] = useState<string | null>(null);
  const [fast, setFast] = useState(false);
  const [upscale, setUpscale] = useState(true);
  const [refine, setRefine] = useState(true);
  const [target, setTarget] = useState("");

  const still = operations.find((item) => item.id === opId);
  const live = videoOperations.find((item) => item.id === opId);
  const op = still || live;
  const isLive = Boolean(opId && LIVE_WALLPAPER_OPS.includes(opId));
  const isT2V = Boolean(opId && LTX_VIDEO_OPS.includes(opId));
  const isVideoJob = isLive || isT2V;
  const isStill = Boolean(opId && WALLPAPER_OPS.includes(opId));
  const targets = isVideoJob ? ["mobile", "video"] : ["mobile", "desktop"];
  const selectedTarget = target || (isT2V ? "video" : isLive ? "mobile" : "desktop");
  const sizedOp = useMemo(() => {
    if (!op) return op;
    if (isVideoJob) {
      return {
        ...op,
        defaultWidth: selectedTarget === "video" ? 1216 : 704,
        defaultHeight: selectedTarget === "video" ? 704 : 1216,
      };
    }
    return {
      ...op,
      defaultWidth: selectedTarget === "mobile" ? 928 : 1664,
      defaultHeight: selectedTarget === "mobile" ? 1664 : 928,
    };
  }, [op, isVideoJob, selectedTarget]);

  if (!opId) {
    return <section className="card"><p className="muted">Unknown operation. <Link to="/wallpapers">Back to wallpapers</Link></p></section>;
  }
  if (!operations.length && !videoOperations.length) {
    return <section className="card"><p className="muted">Loading {opId}…</p></section>;
  }
  if (!op || (!isVideoJob && !isStill)) {
    return <section className="card"><p className="muted">Unknown operation. <Link to="/wallpapers">Back to wallpapers</Link></p></section>;
  }

  const f = fieldsFor(op);
  const endpoint = isVideoJob ? "/v1/video/jobs" : "/v1/image/jobs";

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setLocalError(null);
    await run(async () => {
      const body: JobBody = { operation: opId || "", target: selectedTarget };
      const prompt = optionalString(data, "prompt");
      if (prompt) body.prompt = prompt;
      const negative = optionalString(data, "negative_prompt");
      if (negative) body.negative_prompt = negative;
      const seed = optionalNumber(data, "seed");
      if (seed !== undefined) body.seed = seed;
      const steps = optionalNumber(data, "steps");
      if (steps !== undefined) body.steps = steps;
      const width = optionalNumber(data, "width");
      if (width !== undefined) body.width = width;
      const height = optionalNumber(data, "height");
      if (height !== undefined) body.height = height;
      if (isStill) {
        body.upscale = upscale;
        if (fast) body.fast = true;
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
      } else {
        body.refine = refine;
        const duration = optionalNumber(data, "duration");
        if (duration !== undefined) body.duration = duration;
        const fps = optionalNumber(data, "fps");
        if (fps !== undefined) body.fps = fps;
        const videoCfg = optionalNumber(data, "video_cfg");
        if (videoCfg !== undefined) body.video_cfg = videoCfg;
        if (isLive) {
          const files = fileList(form, "image");
          const file = files?.[0];
          if (file) body.image = { asset_id: await uploadAsset(file) };
        }
      }
      const created = await api<Job>(endpoint, {
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
        <label>
          Target
          <select name="target" value={selectedTarget} onChange={(event) => setTarget(event.target.value)}>
            {targets.map((item) => (
              <option key={item} value={item}>{item}</option>
            ))}
          </select>
        </label>
        <label>Prompt <textarea name="prompt" rows={3} placeholder={isT2V ? "Describe the scene, motion, camera, and lighting" : "Describe the devotion scene or motion"} required /></label>
        <label>Negative <textarea name="negative_prompt" rows={2} placeholder="Optional. Leave blank for defaults." /></label>
        {isLive ? <label>Image <input type="file" name="image" accept="image/*" required /></label> : null}
        {isStill ? (
          <>
            <label className="check">
              <input type="checkbox" name="upscale" checked={upscale} onChange={(event) => setUpscale(event.target.checked)} />
              High quality 2x upscale
            </label>
            <label className="check">
              <input type="checkbox" name="fast" checked={fast} onChange={(event) => setFast(event.target.checked)} />
              Fast (Lightning 4-step)
            </label>
          </>
        ) : (
          <label className="check">
            <input type="checkbox" name="refine" checked={refine} onChange={(event) => setRefine(event.target.checked)} />
            Quality refine (studio)
          </label>
        )}
        {isStill && sizedOp ? (
          <AdvancedPanel key={`adv:${op.id}:${selectedTarget}`} op={sizedOp} fields={f} capabilities={capabilities} fast={fast} />
        ) : null}
        {isVideoJob && sizedOp ? <VideoAdvancedPanel key={`vadv:${op.id}`} op={sizedOp} /> : null}
        <div className="actions">
          <button type="submit" disabled={busy || !op.available}>Queue job</button>
          <Link className="btn" to="/wallpapers">All wallpapers</Link>
        </div>
      </form>
    </section>
  );
}
