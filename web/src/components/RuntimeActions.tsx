import { api } from "../api/client";
import { useStore } from "../api/store";
import { UNLOAD_MODEL } from "../lib/ui";

export function RuntimeActions() {
  const { busy, run } = useStore();

  function load(modelId: string) {
    void run(() => api(`/v1/models/${encodeURIComponent(modelId)}/load`, { method: "POST" }).then(() => undefined));
  }

  function stop() {
    void run(() => api(`/v1/models/${encodeURIComponent(UNLOAD_MODEL)}/unload`, { method: "POST" }).then(() => undefined));
  }

  return (
    <div className="actions">
      <button type="button" disabled={busy} onClick={() => load("gemma-4-e4b")}>Load Gemma</button>
      <button type="button" disabled={busy} onClick={() => load("qwen36-35b-a3b-rq")}>Load Qwen3.6</button>
      <button type="button" disabled={busy} onClick={() => load("qwen-image-2512-fp8")}>Load ComfyUI</button>
      <button type="button" disabled={busy} onClick={() => load("ltx-2.5-distilled")}>Load LTX</button>
      <button type="button" className="danger" disabled={busy} onClick={stop}>Stop all</button>
    </div>
  );
}
