import { Link } from "react-router-dom";
import { useStore } from "../api/store";
import { JobCard } from "../components/JobCard";
import { RuntimeFacts } from "../components/RuntimeFacts";
import { RuntimeActions } from "../components/RuntimeActions";

export function Overview() {
  const { status, jobs } = useStore();
  const recent = jobs.slice(0, 5);
  const chatReady = status?.loadState === "LOADED" && status.profile !== "comfyui";

  return (
    <>
      <section className="card">
        <h2>Runtime</h2>
        <RuntimeFacts status={status} />
        <RuntimeActions />
      </section>
      <section className="card">
        <h2>Chat API</h2>
        <p className="muted">
          Text + vision is <code>POST /v1/text/chat</code>
          {chatReady ? "." : " after a llama.cpp profile is loaded."}
          {" "}See the copyable request on <Link to="/runtime">Runtime</Link>.
        </p>
      </section>
      <section className="card">
        <h2>Images</h2>
        <p className="muted">
          Queue stills from <Link to="/images">Images</Link>, or <code>POST /v1/image/jobs</code>.
          {" "}Results stay on <Link to="/jobs">Jobs</Link>.
        </p>
      </section>
      <section className="card">
        <h2>Recent jobs</h2>
        <div className="quick-jobs">
          {recent.length ? recent.map((job) => <JobCard key={job.job_id} job={job} />) : <p className="muted">No image jobs yet.</p>}
        </div>
      </section>
    </>
  );
}
