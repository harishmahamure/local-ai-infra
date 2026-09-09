import { Link } from "react-router-dom";
import { api } from "../api/client";
import { useStore } from "../api/store";
import type { Job } from "../api/types";
import { jobBadgeClass } from "../lib/ui";

export function JobCard({ job, detail = false }: { job: Job; detail?: boolean }) {
  const { busy, run } = useStore();
  const pct = Math.round((job.progress || 0) * 100);
  const canCancel = job.status === "QUEUED" || job.status === "RUNNING";

  function cancel() {
    void run(() => api(`/v1/jobs/${encodeURIComponent(job.job_id)}/cancel`, { method: "POST" }).then(() => undefined));
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
          {job.asset_ids.map((id) => (
            <a key={id} href={`/v1/assets/${encodeURIComponent(id)}/content`} target="_blank" rel="noreferrer">
              <img src={`/v1/assets/${encodeURIComponent(id)}/content`} alt="" />
            </a>
          ))}
        </div>
      ) : null}
      <div className="actions">
        {canCancel ? <button type="button" disabled={busy} onClick={cancel}>Cancel</button> : null}
        {!detail ? <Link className="btn" to={`/jobs/${encodeURIComponent(job.job_id)}`}>Open</Link> : null}
      </div>
    </article>
  );
}
