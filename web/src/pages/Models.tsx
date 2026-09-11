import { api } from "../api/client";
import { useStore } from "../api/store";
import type { Bundle, CatalogModel } from "../api/types";

function BundleRow({ bundle, rec }: { bundle: Bundle; rec: CatalogModel | undefined }) {
  const { busy, run } = useStore();
  const missing = (bundle.files || []).filter((f) => f.state !== "ok" && !f.optional).map((f) => f.name);

  function load() {
    void run(() => api(`/v1/models/${encodeURIComponent(bundle.id)}/load`, { method: "POST" }).then(() => undefined));
  }

  function download() {
    void run(() => api("/v1/downloads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [bundle.id] }),
    }).then(() => undefined));
  }

  return (
    <article className="bundle">
      <div>
        <h3>{bundle.id}</h3>
        <p className="meta">
          {(bundle.status || "").toUpperCase()} · {bundle.files_ok}/{bundle.files_required} files · {bundle.size_human || "0"} · {rec?.state || ""}
        </p>
        {missing.length ? <p className="missing">Missing: {missing.join(", ")}</p> : null}
      </div>
      <div className="bundle-actions">
        <button type="button" disabled={busy || bundle.status !== "complete"} onClick={load}>Load</button>
        <button type="button" disabled={busy} onClick={download}>Download</button>
      </div>
    </article>
  );
}

export function Models() {
  const { models } = useStore();
  const catalog = models?.models || [];
  const byId = Object.fromEntries(catalog.map((m) => [m.id, m]));
  const bundles = models?.bundles || [];
  const chat = bundles.filter((b) => {
    const runtime = byId[b.id]?.runtime || b.dest;
    return runtime !== "comfyui" && runtime !== "comfy-ltx";
  });
  const image = bundles.filter((b) => (byId[b.id]?.runtime || b.dest) === "comfyui");
  const video = bundles.filter((b) => byId[b.id]?.runtime === "comfy-ltx");

  return (
    <>
      <section className="card">
        <h2>Chat</h2>
        <div className="catalog">
          {chat.length ? chat.map((b) => <BundleRow key={b.id} bundle={b} rec={byId[b.id]} />) : <p className="muted">No chat bundles.</p>}
        </div>
      </section>
      <section className="card">
        <h2>Image</h2>
        <div className="catalog">
          {image.length ? image.map((b) => <BundleRow key={b.id} bundle={b} rec={byId[b.id]} />) : <p className="muted">No image bundles.</p>}
        </div>
      </section>
      <section className="card">
        <h2>Video</h2>
        <div className="catalog">
          {video.length ? video.map((b) => <BundleRow key={b.id} bundle={b} rec={byId[b.id]} />) : <p className="muted">No video bundles.</p>}
        </div>
      </section>
    </>
  );
}
