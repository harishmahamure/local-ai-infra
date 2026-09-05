"""Job stage progress for GET job responses and the control UI."""

from __future__ import annotations

from typing import Any


LTX_STAGES: list[dict[str, Any]] = [
    {"id": "switching", "label": "Load ComfyUI LTX", "weight": 8},
    {"id": "load_models", "label": "Load LTX models", "weight": 10},
    {"id": "prompt_enhance", "label": "Enhance prompt (optional)", "weight": 6},
    {"id": "encode", "label": "Encode text (+ image)", "weight": 8},
    {"id": "ic_guide", "label": "IC-LoRA annotate (optional)", "weight": 8},
    {"id": "sample", "label": "Stage 1 sample", "weight": 30},
    {"id": "refine", "label": "Latent upscale + refine", "weight": 18},
    {"id": "decode", "label": "VAE decode + audio", "weight": 8},
    {"id": "export", "label": "Write MP4", "weight": 4},
]

GENERATE_STAGES: list[dict[str, Any]] = [
    {"id": "planning", "label": "Plan with LLM", "weight": 20},
    {"id": "switching", "label": "Load ComfyUI", "weight": 10},
    {"id": "running", "label": "Generate images", "weight": 70},
]

UPSCALE_STAGES: list[dict[str, Any]] = [
    {"id": "switching", "label": "Load ComfyUI", "weight": 15},
    {"id": "running", "label": "RealESRGAN upscale", "weight": 85},
]

_PIPELINE_ORDER = {
    "ltx": ["switching", "load_models", "prompt_enhance", "encode", "ic_guide", "sample", "refine", "decode", "export"],
    "generate": ["planning", "switching", "running"],
    "character_master": ["planning", "switching", "running"],
    "upscale": ["switching", "running"],
}


def _blank_stage(defn: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": defn["id"],
        "label": defn["label"],
        "percent": 0,
        "state": "pending",
        "value": None,
        "max": None,
    }


def _weighted_percent(defs: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> int:
    total_w = sum(int(s["weight"]) for s in defs) or 1
    acc = 0.0
    for s in defs:
        acc += int(s["weight"]) * float(by_id[s["id"]].get("percent") or 0) / 100.0
    return int(min(100, round(acc / total_w * 100)))


def _current_stage(stages: list[dict[str, Any]]) -> str | None:
    running = next((s["id"] for s in stages if s.get("state") == "running"), None)
    if running:
        return running
    failed = next((s["id"] for s in stages if s.get("state") == "failed"), None)
    if failed:
        return failed
    if stages and all(s.get("state") == "done" for s in stages):
        return stages[-1]["id"]
    pending = next((s["id"] for s in stages if s.get("state") == "pending"), None)
    return pending or (stages[0]["id"] if stages else None)


def snapshot(
    defs: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    *,
    current_node: str | None = None,
    detail: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stages = [by_id[s["id"]] for s in defs]
    out: dict[str, Any] = {
        "percentComplete": _weighted_percent(defs, by_id),
        "currentStage": _current_stage(stages),
        "currentNode": current_node,
        "detail": detail,
        "stages": stages,
    }
    if source:
        out["source"] = source
    return out


def source_meta(endpoint: str, job_id: str) -> dict[str, Any]:
    path = endpoint.replace("{jobId}", job_id)
    return {
        "endpoint": path,
        "fields": [
            "progress.percentComplete",
            "progress.currentStage",
            "progress.currentNode",
            "progress.detail",
            "progress.stages[].id",
            "progress.stages[].percent",
            "progress.stages[].state",
            "phase",
            "status",
        ],
    }


class ProgressTracker:
    def __init__(self, defs: list[dict[str, Any]]) -> None:
        self.defs = defs
        self.by_id = {s["id"]: _blank_stage(s) for s in defs}
        self.node_meta: dict[str, dict[str, str]] = {}
        self._nodes_by_stage: dict[str, list[str]] = {}
        self._node_pct: dict[str, int] = {}
        self._done_nodes: set[str] = set()
        self.current_node: str | None = None
        self.detail: str | None = None

    def bind_nodes(self, node_meta: dict[str, dict[str, str]]) -> None:
        self.node_meta = {str(k): v for k, v in (node_meta or {}).items()}
        self._nodes_by_stage = {}
        for nid, meta in self.node_meta.items():
            stage = meta.get("stage")
            if stage:
                self._nodes_by_stage.setdefault(stage, []).append(nid)

    def skip_optional_stage(self, stage_id: str) -> None:
        """Mark a pipeline stage complete when the graph has no nodes for it."""
        row = self.by_id.get(stage_id)
        if row and stage_id not in self._nodes_by_stage:
            row["state"] = "done"
            row["percent"] = 100

    def set_pipeline_stage(
        self,
        stage_id: str,
        *,
        percent: int | None = None,
        state: str = "running",
        detail: str | None = None,
    ) -> None:
        if stage_id not in self.by_id:
            return
        found = False
        for defn in self.defs:
            sid = defn["id"]
            row = self.by_id[sid]
            if sid == stage_id:
                found = True
                row["state"] = state
                if percent is not None:
                    row["percent"] = max(0, min(100, int(percent)))
                elif state == "done":
                    row["percent"] = 100
                elif state == "running" and row["percent"] <= 0:
                    row["percent"] = 1
            elif not found:
                row["state"] = "done"
                row["percent"] = 100
        if detail is not None:
            self.detail = detail
        if state == "failed":
            self.by_id[stage_id]["state"] = "failed"

    def complete_all(self) -> None:
        for row in self.by_id.values():
            row["state"] = "done"
            row["percent"] = 100
        self.detail = None
        self.current_node = None

    def fail_current(self, message: str | None = None) -> None:
        current = _current_stage(list(self.by_id.values()))
        if current:
            self.by_id[current]["state"] = "failed"
        if message:
            self.detail = message

    def apply_ws(self, msg: dict[str, Any]) -> None:
        typ = msg.get("type")
        data = msg.get("data") or {}
        if typ == "progress_state":
            for nid, st in (data.get("nodes") or {}).items():
                if isinstance(st, dict):
                    self._apply_node(str(nid), st)
        elif typ == "progress":
            nid = data.get("node")
            if nid is not None:
                self._apply_node(
                    str(nid),
                    {"value": data.get("value"), "max": data.get("max"), "state": "running"},
                )
        elif typ == "executing":
            nid = data.get("node")
            if nid:
                self._apply_node(str(nid), {"state": "running"})
            else:
                self.complete_all()
        elif typ == "executed":
            nid = data.get("node")
            if nid:
                self._mark_node_done(str(nid))
        elif typ == "execution_cached":
            for nid in data.get("nodes") or []:
                self._mark_node_done(str(nid))
        elif typ == "execution_error":
            self.fail_current(str(data.get("exception_message") or data.get("error") or "ComfyUI error"))

    def snapshot(self, source: dict[str, Any] | None = None) -> dict[str, Any]:
        return snapshot(
            self.defs,
            self.by_id,
            current_node=self.current_node,
            detail=self.detail,
            source=source,
        )

    def _mark_node_done(self, nid: str) -> None:
        self._done_nodes.add(nid)
        self._node_pct[nid] = 100
        meta = self.node_meta.get(nid)
        if not meta:
            return
        self._refresh_stage(meta["stage"])

    def _apply_node(self, nid: str, st: dict[str, Any]) -> None:
        meta = self.node_meta.get(nid)
        if not meta:
            return
        nstate = str(st.get("state") or "running").lower()
        value = st.get("value")
        maxv = st.get("max")
        label = meta.get("label") or nid
        self.current_node = label
        if nstate in {"finished", "complete", "success"}:
            node_pct = 100
            self._done_nodes.add(nid)
        elif maxv not in (None, 0) and value is not None:
            node_pct = int(min(100, round(float(value) / float(maxv) * 100)))
            self.detail = f"{label} {int(value)}/{int(maxv)}"
        else:
            node_pct = max(self._node_pct.get(nid, 0), 1)
            self.detail = label
        self._node_pct[nid] = node_pct
        if node_pct >= 100:
            self._done_nodes.add(nid)
        self._refresh_stage(meta["stage"])
        row = self.by_id.get(meta["stage"])
        if row is not None and maxv not in (None, 0) and value is not None:
            row["value"] = int(value)
            row["max"] = int(maxv)

    def _refresh_stage(self, stage_id: str) -> None:
        nodes = self._nodes_by_stage.get(stage_id) or []
        if not nodes:
            self.set_pipeline_stage(stage_id, percent=self._node_pct.get(stage_id, 1), state="running")
            return
        scores = [self._node_pct.get(n, 100 if n in self._done_nodes else 0) for n in nodes]
        avg = int(round(sum(scores) / len(scores)))
        done = all(n in self._done_nodes or self._node_pct.get(n, 0) >= 100 for n in nodes)
        self.set_pipeline_stage(
            stage_id,
            percent=100 if done else avg,
            state="done" if done else "running",
        )


def synthesize(
    job: dict[str, Any],
    defs: list[dict[str, Any]],
    pipeline: str,
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build stage percents from status/phase when live Comfy progress is absent."""
    existing = job.get("progress") if isinstance(job.get("progress"), dict) else None
    if existing and existing.get("stages"):
        out = dict(existing)
        if source:
            out["source"] = source
        return out

    phase = str(job.get("phase") or job.get("status") or "")
    status = str(job.get("status") or "")
    by_id = {s["id"]: _blank_stage(s) for s in defs}
    order = [s["id"] for s in defs]
    pipeline_ids = _PIPELINE_ORDER.get(pipeline, order)

    if status == "completed" or phase in {"done", "completed"}:
        for row in by_id.values():
            row["state"] = "done"
            row["percent"] = 100
        return snapshot(defs, by_id, source=source)

    active = None
    if phase.startswith("switching") or status == "switching":
        active = "switching" if "switching" in by_id else pipeline_ids[0]
    elif phase.startswith("running") or status == "running":
        active = next((s for s in order if s not in pipeline_ids[:1] or s == "running"), order[-1])
        if "load_models" in by_id and phase == "running":
            active = "load_models"
        elif "running" in by_id:
            active = "running"
    elif status == "failed" or phase == "failed":
        if "load_models" in by_id and job.get("promptId"):
            active = "load_models"
        elif "switching" in by_id and job.get("plan"):
            active = "switching"
        else:
            active = order[0]
        for sid in order:
            if sid == active:
                by_id[sid]["state"] = "failed"
                by_id[sid]["percent"] = max(1, by_id[sid]["percent"])
                break
            by_id[sid]["state"] = "done"
            by_id[sid]["percent"] = 100
        detail = job.get("error")
        return snapshot(defs, by_id, detail=str(detail) if detail else None, source=source)

    if active not in by_id:
        active = order[0]

    found = False
    for sid in order:
        row = by_id[sid]
        if sid == active:
            found = True
            row["state"] = "running"
            row["percent"] = _partial_percent(job, sid)
        elif not found:
            row["state"] = "done"
            row["percent"] = 100
    detail = _detail_from_job(job)
    return snapshot(defs, by_id, detail=detail, source=source)


def _partial_percent(job: dict[str, Any], stage_id: str) -> int:
    if stage_id != "running":
        return 15
    total = int(job.get("total") or job.get("count") or 0)
    done = int(job.get("completedCount") or 0)
    if total > 0:
        return max(1, min(99, int(round(done / total * 100))))
    phase = str(job.get("phase") or "")
    if "/" in phase:
        try:
            frac = phase.split()[-1]
            cur, tot = frac.split("/", 1)
            return max(1, min(99, int(round(int(cur) / max(int(tot), 1) * 100))))
        except ValueError:
            return 20
    return 20


def _detail_from_job(job: dict[str, Any]) -> str | None:
    phase = job.get("phase")
    if phase:
        return str(phase)
    return None


def attach(job: dict[str, Any], kind: str, endpoint: str) -> dict[str, Any]:
    job_id = str(job.get("jobId") or "")
    src = source_meta(endpoint, job_id)
    defs = {
        "ltx": LTX_STAGES,
        "generate": GENERATE_STAGES,
        "character_master": GENERATE_STAGES,
        "upscale": UPSCALE_STAGES,
    }[kind]
    progress = synthesize(job, defs, kind, source=src)
    out = dict(job)
    out["progress"] = progress
    return out

