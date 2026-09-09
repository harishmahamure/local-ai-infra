import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { useStore } from "../api/store";
import type { Job } from "../api/types";
import { JobCard } from "../components/JobCard";

export function JobDetail() {
  const { id } = useParams();
  const { jobs } = useStore();
  const [live, setLive] = useState<Job | null>(null);
  const listed = jobs.find((job) => job.job_id === id);
  const job = live || listed;

  useEffect(() => {
    if (!id) return;
    let active = true;
    void api<Job>(`/v1/jobs/${encodeURIComponent(id)}`)
      .then((payload) => {
        if (active) setLive(payload);
      })
      .catch(() => undefined);
    const source = new EventSource(`/v1/jobs/${encodeURIComponent(id)}/events`);
    source.addEventListener("job", (event) => {
      try {
        setLive(JSON.parse((event as MessageEvent).data) as Job);
      } catch {
        /* ignore */
      }
    });
    return () => {
      active = false;
      source.close();
    };
  }, [id]);

  if (!id) {
    return <section className="card"><p className="muted">Missing job id.</p></section>;
  }
  if (!job) {
    return <section className="card"><p className="muted">Loading job {id}…</p></section>;
  }

  const prompt = typeof job.inputs?.prompt === "string" ? job.inputs.prompt : "";

  return (
    <section className="card">
      <h2>Detail</h2>
      <JobCard job={job} detail />
      <dl className="facts">
        <div><dt>Created</dt><dd>{job.created_at || "—"}</dd></div>
        <div><dt>Started</dt><dd>{job.started_at || "—"}</dd></div>
        <div><dt>Finished</dt><dd>{job.finished_at || "—"}</dd></div>
        <div><dt>Seed</dt><dd>{job.seed ?? "—"}</dd></div>
      </dl>
      {prompt ? <p className="muted">{prompt}</p> : null}
      <div className="actions"><Link className="btn" to="/jobs">All jobs</Link></div>
    </section>
  );
}
