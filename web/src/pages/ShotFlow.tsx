import { type FormEvent, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, uploadAsset } from "../api/client";
import { useStore } from "../api/store";
import type { Job } from "../api/types";
import { VideoAdvancedPanel } from "../components/VideoAdvancedPanel";

type JobBody = Record<string, unknown>;

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

function filesOf(form: HTMLFormElement, name: string): File[] {
  const el = form.elements.namedItem(name);
  if (!(el instanceof HTMLInputElement) || !el.files) return [];
  return Array.from(el.files);
}

export function ShotFlow() {
  const { op: opId } = useParams();
  const navigate = useNavigate();
  const { shotFlows, shotCapabilities, busy, run } = useStore();
  const [localError, setLocalError] = useState<string | null>(null);
  const [refine, setRefine] = useState(true);
  const [target, setTarget] = useState("video");
  const op = shotFlows.find((item) => item.id === opId);

  if (!opId) {
    return <section className="card"><p className="muted">Unknown flow. <Link to="/shots">Back to shots</Link></p></section>;
  }
  if (!shotFlows.length) {
    return <section className="card"><p className="muted">Loading {opId}…</p></section>;
  }
  if (!op) {
    return <section className="card"><p className="muted">Unknown flow. <Link to="/shots">Back to shots</Link></p></section>;
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setLocalError(null);
    await run(async () => {
      const body: JobBody = { operation: opId, target, refine };
      const prompt = optionalString(data, "prompt");
      if (prompt) body.prompt = prompt;
      const negative = optionalString(data, "negative_prompt");
      if (negative) body.negative_prompt = negative;
      for (const name of ["seed", "steps", "duration", "fps", "video_cfg", "width", "height", "control_strength"]) {
        const value = optionalNumber(data, name);
        if (value !== undefined) body[name] = value;
      }
      const camera = optionalString(data, "camera");
      if (camera) body.camera = camera;
      const control = optionalString(data, "control");
      if (control) body.control = control;
      const action = optionalString(data, "action");
      if (action) body.action = action;
      const parent = optionalString(data, "parent_job_id");
      if (parent) body.parent_job_id = parent;
      if (data.get("prefer_nvfp4") === "on") body.prefer_nvfp4 = true;
      if (data.get("quality_lora") === "on") body.quality_lora = true;

      for (const slot of op?.slots || []) {
        if (slot.kind === "job") continue;
        const uploaded = [];
        for (const file of filesOf(form, slot.id)) {
          uploaded.push({ asset_id: await uploadAsset(file) });
        }
        if (!uploaded.length) continue;
        if (slot.id === "keyframes") {
          body.keyframes = uploaded.map((item, index) => ({
            ...item,
            frame_idx: optionalNumber(data, `keyframe_idx_${index}`) ?? (index + 1) * 8,
          }));
        } else if (slot.id === "clips" || slot.multiple) {
          body[slot.id] = uploaded;
        } else {
          body[slot.id] = uploaded[0];
        }
      }

      const created = await api<Job>("/v1/shots/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      navigate(`/jobs/${encodeURIComponent(created.job_id)}`);
    }).catch((err: unknown) => {
      setLocalError(err instanceof Error ? err.message : String(err));
    });
  }

  const cameras = shotCapabilities?.cameras || [];
  const controls = shotCapabilities?.controls || ["canny", "motion_track"];
  const actions = shotCapabilities?.actions || ["battle", "destruction", "magic", "particles", "weather"];

  return (
    <section className="card">
      <h2>{op.label}</h2>
      <p className="muted">{op.description}</p>
      {!op.available ? <p className="missing">Missing models: {(op.missingBundles || []).join(", ")}</p> : null}
      {localError ? <p className="error">{localError}</p> : null}
      <form className="job-form" onSubmit={(event) => void onSubmit(event)}>
        <label>
          Target
          <select name="target" value={target} onChange={(event) => setTarget(event.target.value)}>
            <option value="video">video</option>
            <option value="mobile">mobile</option>
          </select>
        </label>
        {op.promptRequired !== false ? (
          <label>Prompt <textarea name="prompt" rows={3} placeholder="Describe the shot, motion, and lighting" required /></label>
        ) : (
          <label>Prompt <textarea name="prompt" rows={2} placeholder="Optional override" /></label>
        )}
        <label>Negative <textarea name="negative_prompt" rows={2} placeholder="Optional. Leave blank for defaults." /></label>
        {(op.slots || []).map((slot) => {
          if (slot.kind === "job") {
            return (
              <label key={slot.id}>
                {slot.label}
                <input name="parent_job_id" placeholder="job_…" required={slot.required} />
              </label>
            );
          }
          const accept = slot.kind.startsWith("video") ? "video/*" : "image/*";
          return (
            <label key={slot.id}>
              {slot.label}
              <input type="file" name={slot.id} accept={accept} multiple={Boolean(slot.multiple)} required={Boolean(slot.required)} />
            </label>
          );
        })}
        {op.camera ? (
          <label>
            Camera
            <select name="camera" required>
              {cameras.map((item) => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
          </label>
        ) : null}
        {op.control ? (
          <label>
            Control
            <select name="control" defaultValue="canny">
              {controls.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>
        ) : null}
        {op.action ? (
          <label>
            Action
            <select name="action" defaultValue="battle">
              {actions.map((item) => (
                <option key={item} value={item}>{item}</option>
              ))}
            </select>
          </label>
        ) : null}
        <label className="check">
          <input type="checkbox" name="refine" checked={refine} onChange={(event) => setRefine(event.target.checked)} />
          Quality refine (spatial + temporal)
        </label>
        <label className="check">
          <input type="checkbox" name="prefer_nvfp4" defaultChecked />
          Prefer nvfp4 checkpoint
        </label>
        <label className="check">
          <input type="checkbox" name="quality_lora" />
          Quality LoRA (int8 path)
        </label>
        <VideoAdvancedPanel key={`vadv:${op.id}`} op={op} />
        <div className="actions">
          <button type="submit" disabled={busy || !op.available}>Queue shot</button>
          <Link className="btn" to="/shots">All shots</Link>
        </div>
      </form>
    </section>
  );
}
