const POLL_MS = 2000;
const UNLOAD_MODEL = "gemma-4-e4b";

const els = {
  badge: document.getElementById("load-badge"),
  gpuName: document.getElementById("gpu-name"),
  profile: document.getElementById("fact-profile"),
  model: document.getElementById("fact-model"),
  api: document.getElementById("fact-api"),
  vram: document.getElementById("fact-vram"),
  util: document.getElementById("fact-util"),
  vramBar: document.getElementById("vram-bar"),
  procs: document.getElementById("proc-list"),
  runtimeError: document.getElementById("runtime-error"),
  catalog: document.getElementById("catalog"),
  dlSummary: document.getElementById("dl-summary"),
  dlBar: document.getElementById("dl-bar"),
  dlCurrent: document.getElementById("dl-current"),
  dlLog: document.getElementById("dl-log"),
};

let busy = false;
let lastStatus = null;
let lastModels = null;

function setBusy(on, source) {
  busy = on;
  document.querySelectorAll("button").forEach((btn) => {
    btn.disabled = on;
    btn.classList.toggle("busy", on && btn === source);
  });
}

function badgeClass(state) {
  const s = (state || "STOPPED").toUpperCase();
  if (s === "LOADED") return "loaded";
  if (s === "STARTING") return "starting";
  if (s.startsWith("CONFLICT")) return "conflict";
  return "stopped";
}

function parseMib(value) {
  if (!value) return null;
  const n = parseFloat(String(value).replace(/[^\d.]/g, ""));
  return Number.isFinite(n) ? n : null;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: { message: text || res.statusText } };
  }
  if (!res.ok) {
    const msg = data?.error?.message || data?.detail || res.statusText;
    throw new Error(msg);
  }
  return data;
}

function renderStatus(s) {
  lastStatus = s;
  const state = s.loadState || "STOPPED";
  els.badge.textContent = state;
  els.badge.className = `badge ${badgeClass(state)}`;
  const gpu = s.gpu || {};
  els.gpuName.textContent = gpu.name || "GPU box";
  els.profile.textContent = s.profile || "none";
  els.model.textContent = s.model || "—";
  els.api.textContent = s.apiState || "—";
  els.vram.textContent = gpu.memoryUsed && gpu.memoryTotal
    ? `${gpu.memoryUsed} / ${gpu.memoryTotal}`
    : "—";
  els.util.textContent = gpu.utilization || "—";
  const used = parseMib(gpu.memoryUsed);
  const total = parseMib(gpu.memoryTotal);
  els.vramBar.style.width = used && total ? `${Math.min(100, (used / total) * 100)}%` : "0%";
  const procs = s.gpuProcesses || [];
  els.procs.innerHTML = "";
  if (!procs.length) {
    const li = document.createElement("li");
    li.textContent = "No GPU processes";
    els.procs.appendChild(li);
  } else {
    for (const p of procs) {
      const li = document.createElement("li");
      li.textContent = `${p.pid}  ${p.name}  ${p.memory}`;
      els.procs.appendChild(li);
    }
  }
}

function renderCatalog(data) {
  lastModels = data;
  const bundles = data.bundles || [];
  const models = data.models || [];
  const byId = Object.fromEntries(models.map((m) => [m.id, m]));
  if (!bundles.length) {
    els.catalog.textContent = "No catalog bundles.";
    return;
  }
  els.catalog.innerHTML = "";
  for (const b of bundles) {
    const rec = byId[b.id] || {};
    const row = document.createElement("article");
    row.className = "bundle";
    const missing = (b.files || [])
      .filter((f) => f.state !== "ok" && !f.optional)
      .map((f) => f.name);
    const info = document.createElement("div");
    info.innerHTML = `<h3>${b.id}</h3>
      <p class="meta">${(b.status || "").toUpperCase()} · ${b.files_ok}/${b.files_required} files · ${b.size_human || "0"} · ${rec.state || ""}</p>`;
    if (missing.length) {
      const miss = document.createElement("p");
      miss.className = "missing";
      miss.textContent = `Missing: ${missing.join(", ")}`;
      info.appendChild(miss);
    }
    const actions = document.createElement("div");
    actions.className = "bundle-actions";
    const load = document.createElement("button");
    load.type = "button";
    load.textContent = "Load";
    load.disabled = b.status !== "complete";
    load.addEventListener("click", () => runAction("load", b.id, load));
    const dl = document.createElement("button");
    dl.type = "button";
    dl.textContent = "Download";
    dl.addEventListener("click", () => runAction("download", b.id, dl));
    actions.append(load, dl);
    row.append(info, actions);
    els.catalog.appendChild(row);
  }
}

function renderDownloads(d) {
  const progress = d.progress || {};
  const pct = progress.percentComplete ?? 0;
  const running = Boolean(d.running) || d.status === "running";
  els.dlSummary.textContent = running
    ? `Downloading ${pct}%`
    : `${d.status || "idle"} · ${progress.bundlesComplete || 0}/${progress.bundlesTotal || 0} complete`;
  els.dlBar.style.width = `${Math.min(100, pct)}%`;
  const current = [d.currentBundle, d.currentFile].filter(Boolean).join(" / ");
  els.dlCurrent.textContent = current || "";
  const tail = (d.logTail || []).join("\n");
  els.dlLog.textContent = tail || "(no log)";
}

function showError(err) {
  els.runtimeError.hidden = !err;
  els.runtimeError.textContent = err ? String(err.message || err) : "";
}

async function refresh() {
  try {
    const [status, models, downloads] = await Promise.all([
      api("/v1/status"),
      api("/v1/models"),
      api("/v1/downloads"),
    ]);
    renderStatus(status);
    renderCatalog(models);
    renderDownloads(downloads);
    if (!busy) showError(null);
  } catch (err) {
    showError(err);
  }
}

async function runAction(kind, modelId, source) {
  if (busy) return;
  setBusy(true, source);
  showError(null);
  try {
    if (kind === "load") {
      await api(`/v1/models/${encodeURIComponent(modelId)}/load`, { method: "POST" });
    } else if (kind === "stop") {
      await api(`/v1/models/${encodeURIComponent(UNLOAD_MODEL)}/unload`, { method: "POST" });
    } else if (kind === "download") {
      await api("/v1/downloads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: modelId ? [modelId] : [] }),
      });
    } else if (kind === "download-all") {
      await api("/v1/downloads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: [] }),
      });
    }
    await refresh();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false, source);
  }
}

document.querySelectorAll("[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const action = btn.dataset.action;
    if (action === "load") runAction("load", btn.dataset.model, btn);
    else if (action === "stop") runAction("stop", null, btn);
    else if (action === "download-all") runAction("download-all", null, btn);
  });
});

refresh();
setInterval(() => {
  if (!busy) refresh();
}, POLL_MS);
