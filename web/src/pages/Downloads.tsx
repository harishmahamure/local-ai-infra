import { api } from "../api/client";
import { useStore } from "../api/store";

export function Downloads() {
  const { downloads, busy, run } = useStore();
  const progress = downloads?.progress || {};
  const pct = progress.percentComplete ?? 0;
  const running = Boolean(downloads?.running) || downloads?.status === "running";
  const summary = running
    ? `Downloading ${pct}%`
    : `${downloads?.status || "idle"} · ${progress.bundlesComplete || 0}/${progress.bundlesTotal || 0} complete`;
  const current = [downloads?.currentBundle, downloads?.currentFile].filter(Boolean).join(" / ");

  function startAll() {
    void run(() => api("/v1/downloads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [] }),
    }).then(() => undefined));
  }

  return (
    <section className="card">
      <h2>Downloads</h2>
      <div className="dl-head">
        <p>{summary}</p>
        <button type="button" disabled={busy} onClick={startAll}>Start all</button>
      </div>
      <div className="progress" aria-hidden="true"><div style={{ width: `${Math.min(100, pct)}%` }} /></div>
      <p className="muted">{current}</p>
      <pre className="log">{(downloads?.logTail || []).join("\n") || "(no log)"}</pre>
    </section>
  );
}
