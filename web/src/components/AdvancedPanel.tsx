import { useEffect, useMemo, useState } from "react";
import type { Capabilities, Operation } from "../api/types";
import { FALLBACK_ASPECTS, type FieldFlags } from "../lib/ui";

type AdvState = {
  steps: string;
  cfg: string;
  sampler_name: string;
  scheduler: string;
  shift: string;
  lora_strength: string;
  aspect: string;
  width: string;
  height: string;
  seed: string;
  control_strength: string;
  feathering: string;
};

const EMPTY: AdvState = {
  steps: "",
  cfg: "",
  sampler_name: "",
  scheduler: "",
  shift: "",
  lora_strength: "",
  aspect: "",
  width: "",
  height: "",
  seed: "",
  control_strength: "",
  feathering: "",
};

function storageKey(opId: string): string {
  return `imageop:adv:${opId}`;
}

function loadState(opId: string): AdvState {
  try {
    const raw = window.localStorage.getItem(storageKey(opId));
    if (!raw) return { ...EMPTY };
    const parsed = JSON.parse(raw) as Partial<AdvState>;
    return { ...EMPTY, ...parsed };
  } catch {
    return { ...EMPTY };
  }
}

function aspectId(width: number, height: number, aspects: { id: string; width: number; height: number }[]): string {
  const match = aspects.find((item) => item.width === width && item.height === height);
  return match?.id || "custom";
}

export function AdvancedPanel({
  op,
  fields,
  capabilities,
  fast,
}: {
  op: Operation;
  fields: FieldFlags;
  capabilities: Capabilities | null;
  fast: boolean;
}) {
  const aspects = capabilities?.aspects?.length ? capabilities.aspects : FALLBACK_ASPECTS;
  const samplers = capabilities?.samplers?.length ? capabilities.samplers : ["euler"];
  const schedulers = capabilities?.schedulers?.length ? capabilities.schedulers : ["simple"];
  const [state, setState] = useState<AdvState>(() => loadState(op.id));

  useEffect(() => {
    window.localStorage.setItem(storageKey(op.id), JSON.stringify(state));
  }, [op.id, state]);

  const defaultAspect = useMemo(
    () => aspectId(op.defaultWidth || 1328, op.defaultHeight || 1328, aspects),
    [op.defaultWidth, op.defaultHeight, aspects],
  );

  function patch(next: Partial<AdvState>) {
    setState((prev) => ({ ...prev, ...next }));
  }

  function onAspect(value: string) {
    if (value === "custom") {
      patch({ aspect: "custom" });
      return;
    }
    const found = aspects.find((item) => item.id === value);
    if (!found) {
      patch({ aspect: value });
      return;
    }
    patch({ aspect: value, width: String(found.width), height: String(found.height) });
  }

  function reset() {
    window.localStorage.removeItem(storageKey(op.id));
    setState({ ...EMPTY });
  }

  const selectedAspect = state.aspect || defaultAspect;
  const custom = selectedAspect === "custom";

  return (
    <details className="advanced">
      <summary>Advanced</summary>
      <div className="job-grid">
        <label>
          Steps
          <input
            type="number"
            name="steps"
            min={1}
            max={80}
            placeholder={String(op.defaultSteps ?? 50)}
            value={state.steps}
            onChange={(event) => patch({ steps: event.target.value })}
          />
        </label>
        <label>
          CFG
          <input
            type="number"
            name="cfg"
            min={0}
            max={10}
            step={0.1}
            placeholder={String(op.defaultCfg ?? 4)}
            value={state.cfg}
            onChange={(event) => patch({ cfg: event.target.value })}
          />
        </label>
        <label>
          Sampler
          <select
            name="sampler_name"
            value={state.sampler_name}
            onChange={(event) => patch({ sampler_name: event.target.value })}
          >
            <option value="">euler (default)</option>
            {samplers.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>
        <label>
          Scheduler
          <select
            name="scheduler"
            value={state.scheduler}
            onChange={(event) => patch({ scheduler: event.target.value })}
          >
            <option value="">simple (default)</option>
            {schedulers.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>
        <label>
          AuraFlow shift
          <input
            type="number"
            name="shift"
            min={0}
            max={100}
            step={0.1}
            placeholder={String(op.defaultShift ?? 3.1)}
            value={state.shift}
            onChange={(event) => patch({ shift: event.target.value })}
          />
        </label>
        {fast ? (
          <label>
            LoRA strength
            <input
              type="number"
              name="lora_strength"
              min={0}
              max={2}
              step={0.05}
              placeholder="1.0"
              value={state.lora_strength}
              onChange={(event) => patch({ lora_strength: event.target.value })}
            />
          </label>
        ) : null}
        <label>
          Aspect
          <select value={selectedAspect} onChange={(event) => onAspect(event.target.value)}>
            {aspects.map((item) => (
              <option key={item.id} value={item.id}>{item.id} ({item.width}x{item.height})</option>
            ))}
            <option value="custom">Custom</option>
          </select>
        </label>
        <label>
          Width
          <input
            type="number"
            name="width"
            min={64}
            max={4096}
            step={8}
            placeholder={String(op.defaultWidth ?? 1328)}
            value={state.width}
            onChange={(event) => patch({ width: event.target.value, aspect: "custom" })}
            disabled={!custom && !state.width}
          />
        </label>
        <label>
          Height
          <input
            type="number"
            name="height"
            min={64}
            max={4096}
            step={8}
            placeholder={String(op.defaultHeight ?? 1328)}
            value={state.height}
            onChange={(event) => patch({ height: event.target.value, aspect: "custom" })}
            disabled={!custom && !state.height}
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
        {fields.controlStrength ? (
          <label>
            Control strength
            <input
              type="number"
              name="control_strength"
              min={0}
              max={2}
              step={0.05}
              placeholder="1.0"
              value={state.control_strength}
              onChange={(event) => patch({ control_strength: event.target.value })}
            />
          </label>
        ) : null}
        {fields.feathering ? (
          <label>
            Feathering
            <input
              type="number"
              name="feathering"
              min={0}
              step={1}
              placeholder="40"
              value={state.feathering}
              onChange={(event) => patch({ feathering: event.target.value })}
            />
          </label>
        ) : null}
      </div>
      <div className="actions">
        <button type="button" className="btn" onClick={reset}>Reset to defaults</button>
      </div>
    </details>
  );
}
