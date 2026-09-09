import { useState } from "react";
import { useStore } from "../api/store";
import { JobCard } from "../components/JobCard";

const FILTERS = ["", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"];

export function Jobs() {
  const { jobs } = useStore();
  const [filter, setFilter] = useState("");
  const visible = jobs.filter((job) => !filter || job.status === filter);

  return (
    <section className="card">
      <h2>Queue</h2>
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
