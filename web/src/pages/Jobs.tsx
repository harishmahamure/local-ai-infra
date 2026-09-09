import { useState } from "react";
import { api } from "../api/client";
import { useStore } from "../api/store";
import { JobCard } from "../components/JobCard";

const FILTERS = ["", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"];

export function Jobs() {
  const { jobs, busy, run } = useStore();
  const [filter, setFilter] = useState("");
  const visible = jobs.filter((job) => !filter || job.status === filter);
  const canDeleteAll = filter !== "RUNNING" && visible.some((job) => job.status !== "RUNNING");

  function deleteAllJobs() {
    const label = filter ? `${filter.toLowerCase()} jobs` : "all jobs that are not running";
    if (!window.confirm(`Delete ${label} and their generated images from the GPU box?`)) return;
    const query = filter ? `?status=${encodeURIComponent(filter)}` : "";
    void run(() => api(`/v1/jobs${query}`, { method: "DELETE" }).then(() => undefined));
  }

  function deleteAllImages() {
    if (!window.confirm("Delete all images on the GPU box except those on a running job?")) return;
    void run(() => api("/v1/assets", { method: "DELETE" }).then(() => undefined));
  }

  return (
    <section className="card">
      <div className="dl-head">
        <h2>Queue</h2>
        <div className="actions">
          <button type="button" className="danger" disabled={busy || !canDeleteAll} onClick={deleteAllJobs}>
            Delete all jobs
          </button>
          <button type="button" className="danger" disabled={busy} onClick={deleteAllImages}>
            Delete all images
          </button>
        </div>
      </div>
      <div className="filters">
        {FILTERS.map((status) => (
          <button
            key={status || "all"}
            type="button"
            className={`chip${filter === status ? " active" : ""}`}
            onClick={() => setFilter(status)}
          >
            {status || "All"}
          </button>
        ))}
      </div>
      <div className="job-list">
        {visible.length ? visible.map((job) => <JobCard key={job.job_id} job={job} />) : <p className="muted">No jobs in this filter.</p>}
      </div>
    </section>
  );
}
