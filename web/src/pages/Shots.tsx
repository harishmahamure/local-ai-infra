import { Link } from "react-router-dom";
import { useStore } from "../api/store";
import type { Operation } from "../api/types";

function OpCard({ op }: { op: Operation }) {
  const missing = (op.missingBundles || []).join(", ");
  return (
    <Link
      className={`op-card${op.available ? "" : " disabled"}`}
      to={`/shots/${op.id}`}
      aria-disabled={!op.available}
    >
      <h3>{op.label}</h3>
      <p className="meta">{op.description}</p>
      {!op.available && missing ? <p className="missing">Missing: {missing}</p> : null}
    </Link>
  );
}

export function Shots() {
  const { shotFlows } = useStore();
  return (
    <section className="card">
      <h2>Shot flows</h2>
      <p className="muted">
        Guide-driven LTX 2.5 shots: start/end frames, multi-keyframe, continuation, motion transfer, camera, action, repair, extend, and stitch.
      </p>
      <div className="op-grid">
        {shotFlows.length ? shotFlows.map((op) => <OpCard key={op.id} op={op} />) : <p className="muted">Loading shot flows…</p>}
      </div>
    </section>
  );
}
