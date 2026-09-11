import { useState } from "react";
import { Link } from "react-router-dom";
import { useStore } from "../api/store";
import { RuntimeActions } from "../components/RuntimeActions";
import { RuntimeFacts } from "../components/RuntimeFacts";
import { CHAT_CURL, IMAGE_CURL, WALLPAPER_CURL } from "../lib/ui";

export function Runtime() {
  const { status } = useStore();
  const [copied, setCopied] = useState("Copy curl");
  const busyGpu = (status?.profile === "comfyui" || status?.profile === "comfy-ltx") && status.loadState === "LOADED";
  const procs = status?.gpuProcesses || [];

  async function copy() {
    try {
      await navigator.clipboard.writeText(CHAT_CURL);
      setCopied("Copied");
    } catch {
      setCopied("Copy failed");
    }
  }

  return (
    <>
      <section className="card">
        <h2>Profile</h2>
        <RuntimeFacts status={status} />
        <ul className="procs">
          {procs.length ? procs.map((p) => (
            <li key={p.pid}>{p.pid}  {p.name}  {p.memory}</li>
          )) : <li>No GPU processes</li>}
        </ul>
        <RuntimeActions />
      </section>
      <section className="card endpoint">
        <h2>Text + vision chat</h2>
        <p className="muted">API only. Call this against a loaded llama.cpp profile (Gemma or Qwen3.6). No chat window.</p>
        <dl>
          <dt>Endpoint</dt>
          <dd>POST /v1/text/chat</dd>
          <dt>Models</dt>
          <dd>gemma-4-e4b · qwen36-35b-a3b-rq</dd>
          <dt>Stream</dt>
          <dd>Set <code>stream=true</code> for SSE</dd>
        </dl>
        {busyGpu ? (
          <p className="error">A ComfyUI profile is loaded. Chat returns 409 GPU_BUSY until you switch to a chat profile.</p>
        ) : null}
        <pre>{CHAT_CURL}</pre>
        <div className="actions">
          <button type="button" onClick={() => void copy()}>{copied}</button>
          <a className="btn" href="/docs">Open /docs</a>
          <Link className="btn" to="/">Overview</Link>
        </div>
      </section>
      <section className="card endpoint">
        <h2>Image jobs</h2>
        <p className="muted">API only. Queue one job at a time. Watch progress and outputs on Jobs.</p>
        <dl>
          <dt>Endpoint</dt>
          <dd>POST /v1/image/jobs</dd>
          <dt>Catalog</dt>
          <dd>GET /v1/image/operations</dd>
        </dl>
        <pre>{IMAGE_CURL}</pre>
        <div className="actions">
          <Link className="btn" to="/jobs">Open Jobs</Link>
          <a className="btn" href="/docs">Open /docs</a>
        </div>
      </section>
      <section className="card endpoint">
        <h2>Wallpaper jobs</h2>
        <p className="muted">Devotion stills use Qwen. LTX text-to-video and live wallpapers auto-switch the GPU profile.</p>
        <dl>
          <dt>Still</dt>
          <dd>POST /v1/image/jobs · generate_devotion_wallpaper</dd>
          <dt>Video</dt>
          <dd>POST /v1/video/jobs · generate_video</dd>
          <dt>Live</dt>
          <dd>POST /v1/video/jobs · generate_live_wallpaper</dd>
        </dl>
        <pre>{WALLPAPER_CURL}</pre>
        <div className="actions">
          <Link className="btn" to="/wallpapers">Open Wallpapers</Link>
          <Link className="btn" to="/jobs">Open Jobs</Link>
        </div>
      </section>
    </>
  );
}
