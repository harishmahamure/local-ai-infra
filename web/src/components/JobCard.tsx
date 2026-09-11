import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useStore } from "../api/store";
import type { Job } from "../api/types";
import { jobBadgeClass } from "../lib/ui";

function isVideoAsset(job: Job, assetId: string): boolean {
  const mime = job.assets?.find((item) => item.asset_id === assetId)?.mime_type || "";
  if (mime.startsWith("video/")) return true;
  return job.operation === "generate_live_wallpaper" || job.operation === "generate_video" || job.operation.startsWith("shot_");
}

export function JobCard({ job, detail = false }: { job: Job; detail?: boolean }) {
  const { busy, run } = useStore();
  const navigate = useNavigate();
  const pct = Math.round((job.progress || 0) * 100);
  const canCancel = job.status === "QUEUED" || job.status === "RUNNING";
  const canDelete = job.status !== "RUNNING";
  const hasVideo = (job.asset_ids || []).some((id) => isVideoAsset(job, id));

  function cancel() {
    void run(() => api(`/v1/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: "POST" }).then(() => undefined));
  }

  function removeJob() {
    if (!window.confirm("Delete this job and its generated files from the GPU box?")) return;
    void run(async () => {
      await api(`/v1/jobs/${encodeURIComponent(job.job_id)}`, { method: "DELETE" });
      if (detail) navigate("/jobs");
    });
  }

  function removeAsset(assetId: string) {
    if (!window.confirm("Delete this file from the GPU box?")) return;
    void run(() => api(`/v1/assets/${encodeURIComponent(assetId)}`, { method: "DELETE" }).then(() => undefined));
  }

  function removeAllImages() {
    if (!window.confirm("Delete all outputs for this job from the GPU box?")) return;
    void run(() => api(`/v1/jobs/${encodeURIComponent(job.job_id)}/assets`, { method: "DELETE" }).then(() => undefined));
  }

  return (
    <article className="job-row">
      <div className="job-head">
        <h3>{job.operation}</h3>
        <span className={`badge ${jobBadgeClass(job.status)}`}>{job.status}</span>
      </div>
      <p className="meta">
        {job.job_id} · {job.phase || ""} · {pct}%
        {job.queue_position ? ` · queue #${job.queue_position}` : ""}
      </p>
      <div className="job-bar" aria-hidden="true"><div style={{ width: `${pct}%` }} /></div>
      {job.error?.message ? <p className="error">{job.error.message}</p> : null}
      {job.asset_ids?.length ? (
        <div className={`thumbs${detail ? " large" : ""}`}>
          {job.asset_ids.map((id) => {
            const video = isVideoAsset(job, id);
            const src = `/v1/assets/${encodeURIComponent(id)}/content`;
            return (
              <div className="thumb" key={id}>
                <a href={src} target="_blank" rel="noreferrer">
                  {video ? (
                    <video src={src} controls playsInline muted loop />
                  ) : (
                    <img src={src} alt="" />
                  )}
                </a>
                <button type="button" className="danger thumb-del" disabled={busy} onClick={() => removeAsset(id)}>
                  {video ? "Delete video" : "Delete image"}
                </button>
              </div>
            );
          })}
        </div>
      ) : null}
      <div className="actions">
        {canCancel ? <button type="button" disabled={busy} onClick={cancel}>Cancel</button> : null}
        {canDelete ? (
          <button type="button" className="danger" disabled={busy} onClick={removeJob}>
            Delete job
          </button>
        ) : null}
        {job.asset_ids?.length ? (
          <button type="button" className="danger" disabled={busy} onClick={removeAllImages}>
            {hasVideo ? "Delete all outputs" : "Delete all images"}
          </button>
        ) : null}
        {!detail ? <Link className="btn" to={`/jobs/${encodeURIComponent(job.job_id)}`}>Open</Link> : null}
      </div>
    </article>
  );
}
