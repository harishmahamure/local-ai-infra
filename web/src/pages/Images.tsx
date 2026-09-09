import { Link } from "react-router-dom";
import { useStore } from "../api/store";
import { EDIT_OPS, GENERATE_OPS } from "../lib/ui";
import type { Operation } from "../api/types";

function OpCard({ op }: { op: Operation }) {
  const missing = (op.missingBundles || []).join(", ");
  return (
    <Link
      className={`op-card${op.available ? "" : " disabled"}`}
      to={`/images/${op.id}`}
      aria-disabled={!op.available}
    >
      <h3>{op.label}</h3>
      <p className="meta">{op.description}</p>
      {!op.available && missing ? <p className="missing">Missing: {missing}</p> : null}
    </Link>
  );
}

export function Images() {
  const { operations } = useStore();
  const byId = Object.fromEntries(operations.map((op) => [op.id, op]));
  const generate = GENERATE_OPS.map((id) => byId[id]).filter(Boolean);
  const edit = EDIT_OPS.map((id) => byId[id]).filter(Boolean);

  return (
    <>
      <section className="card">
        <h2>Generate</h2>
        <div className="op-grid">
          {generate.length ? generate.map((op) => <OpCard key={op.id} op={op} />) : <p className="muted">Loading operations…</p>}
        </div>
      </section>
      <section className="card">
        <h2>Edit</h2>
        <div className="op-grid">{edit.map((op) => <OpCard key={op.id} op={op} />)}</div>
      </section>
    </>
  );
}
