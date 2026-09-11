import { Link } from "react-router-dom";
import { useStore } from "../api/store";
import { LIVE_WALLPAPER_OPS, LTX_VIDEO_OPS, WALLPAPER_OPS } from "../lib/ui";
import type { Operation } from "../api/types";

function OpCard({ op, base }: { op: Operation; base: string }) {
  const missing = (op.missingBundles || []).join(", ");
  return (
    <Link
      className={`op-card${op.available ? "" : " disabled"}`}
      to={`${base}/${op.id}`}
      aria-disabled={!op.available}
    >
      <h3>{op.label}</h3>
      <p className="meta">{op.description}</p>
      {!op.available && missing ? <p className="missing">Missing: {missing}</p> : null}
    </Link>
  );
}

export function Wallpapers() {
  const { operations, videoOperations } = useStore();
  const stillById = Object.fromEntries(operations.map((op) => [op.id, op]));
  const videoById = Object.fromEntries(videoOperations.map((op) => [op.id, op]));
  const stills = WALLPAPER_OPS.map((id) => stillById[id]).filter(Boolean);
  const live = LIVE_WALLPAPER_OPS.map((id) => videoById[id]).filter(Boolean);
  const t2v = LTX_VIDEO_OPS.map((id) => videoById[id]).filter(Boolean);

  return (
    <>
      <section className="card">
        <h2>Devotion stills</h2>
        <p className="muted">High-quality mobile and desktop wallpapers. Prompt, negative, and advanced sampling are all exposed.</p>
        <div className="op-grid">
          {stills.length ? stills.map((op) => <OpCard key={op.id} op={op} base="/wallpapers" />) : <p className="muted">Loading operations…</p>}
        </div>
      </section>
      <section className="card">
        <h2>LTX video</h2>
        <p className="muted">Text-to-video. Describe the scene and motion. No start image.</p>
        <div className="op-grid">
          {t2v.length ? t2v.map((op) => <OpCard key={op.id} op={op} base="/wallpapers" />) : <p className="muted">Loading operations…</p>}
        </div>
      </section>
      <section className="card">
        <h2>Live wallpapers</h2>
        <p className="muted">LTX image-to-video. Upload a still (including a devotion wallpaper) and animate it for mobile or landscape video.</p>
        <div className="op-grid">
          {live.length ? live.map((op) => <OpCard key={op.id} op={op} base="/wallpapers" />) : <p className="muted">Loading operations…</p>}
        </div>
      </section>
    </>
  );
}
