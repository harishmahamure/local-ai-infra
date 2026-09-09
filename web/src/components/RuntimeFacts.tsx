import type { Status } from "../api/types";
import { vramPercent } from "../lib/ui";

export function RuntimeFacts({ status }: { status: Status | null }) {
  const gpu = status?.gpu || {};
  const queue = status?.queue || {};
  const vram = gpu.memoryUsed && gpu.memoryTotal ? `${gpu.memoryUsed} / ${gpu.memoryTotal}` : "—";
  const pct = vramPercent(gpu.memoryUsed, gpu.memoryTotal);
  return (
    <>
      <dl className="facts">
        <div><dt>Profile</dt><dd>{status?.profile || "none"}</dd></div>
        <div><dt>Model</dt><dd>{status?.model || "—"}</dd></div>
        <div><dt>API</dt><dd>{status?.apiState || "—"}</dd></div>
        <div><dt>VRAM</dt><dd>{vram}</dd></div>
        <div><dt>Util</dt><dd>{gpu.utilization || "—"}</dd></div>
        <div><dt>Queue</dt><dd>{queue.queueDepth ?? 0}</dd></div>
        <div><dt>Active job</dt><dd>{queue.activeJobId || "none"}</dd></div>
      </dl>
      <div className="vram" aria-hidden="true"><div style={{ width: `${pct}%` }} /></div>
    </>
  );
}
