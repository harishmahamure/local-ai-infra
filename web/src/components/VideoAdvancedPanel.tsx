import { useEffect, useState } from "react";
import type { Operation } from "../api/types";

type VideoAdvState = {
  duration: string;
  fps: string;
  steps: string;
  video_cfg: string;
  seed: string;
};

const EMPTY: VideoAdvState = {
  duration: "",
  fps: "",
  steps: "",
  video_cfg: "",
  seed: "",
};

function storageKey(opId: string): string {
  return `videoop:adv:${opId}`;
}

function loadState(opId: string): VideoAdvState {
  try {
    const raw = window.localStorage.getItem(storageKey(opId));
    if (!raw) return { ...EMPTY };
    const parsed = JSON.parse(raw) as Partial<VideoAdvState>;
    return { ...EMPTY, ...parsed };
  } catch {
    return { ...EMPTY };
  }
}

export function VideoAdvancedPanel({ op }: { op: Operation }) {
  const [state, setState] = useState<VideoAdvState>(() => loadState(op.id));

  useEffect(() => {
    window.localStorage.setItem(storageKey(op.id), JSON.stringify(state));
  }, [op.id, state]);

  function patch(next: Partial<VideoAdvState>) {
    setState((prev) => ({ ...prev, ...next }));
  }

  function reset() {
    window.localStorage.removeItem(storageKey(op.id));
    setState({ ...EMPTY });
  }

  return (
    <details className="advanced">
      <summary>Advanced</summary>
      <div className="job-grid">
        <label>
          Duration (sec)
          <input
            type="number"
            name="duration"
            min={1}
            max={30}
            step={0.5}
            placeholder={String(op.defaultDuration ?? 4)}
            value={state.duration}
            onChange={(event) => patch({ duration: event.target.value })}
          />
        </label>
        <label>
          FPS
          <input
            type="number"
            name="fps"
            min={8}
            max={60}
            placeholder={String(op.defaultFps ?? 24)}
            value={state.fps}
            onChange={(event) => patch({ fps: event.target.value })}
          />
        </label>
        <label>
          Steps
          <input
            type="number"
            name="steps"
            min={1}
            max={40}
            placeholder={String(op.defaultSteps ?? 8)}
            value={state.steps}
            onChange={(event) => patch({ steps: event.target.value })}
          />
        </label>
        <label>
          Video CFG
          <input
            type="number"
            name="video_cfg"
            min={0}
            max={15}
            step={0.1}
            placeholder={String(op.defaultCfg ?? 1)}
            value={state.video_cfg}
            onChange={(event) => patch({ video_cfg: event.target.value })}
          />
        </label>
        <label>
          Seed
          <input
            type="number"
            name="seed"
            placeholder="random"
            value={state.seed}
            onChange={(event) => patch({ seed: event.target.value })}
          />
        </label>
      </div>
      <div className="actions">
        <button type="button" className="btn" onClick={reset}>Reset to defaults</button>
      </div>
    </details>
  );
}
