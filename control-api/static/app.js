const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const toast = (msg) => {
  const el = $("#toast");
  el.textContent = msg;
  el.style.display = "block";
  setTimeout(() => { el.style.display = "none"; }, 3500);
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error?.message || data?.detail || res.statusText;
    throw new Error(msg);
  }
  return data;
}

function badgeClass(kind) {
  return ({ ok: "ok", warn: "warn", bad: "bad", idle: "idle" })[kind] || "idle";
}

function setBadge(el, text, kind) {
  el.textContent = text;
  el.className = `badge ${badgeClass(kind)}`;
}

function loadKind(state) {
  if (state === "LOADED") return "ok";
  if (state === "STOPPED") return "idle";
  if (state === "CONFLICT") return "bad";
  return "warn";
}

function dlKind(state) {
  if (state === "completed") return "ok";
  if (state === "running") return "warn";
  if (state === "failed") return "bad";
  return "idle";
}

function fileStateBadge(state) {
  if (state === "ok") return ["ok", "Downloaded"];
  if (state === "missing") return ["idle", "Missing"];
  if (state === "empty") return ["bad", "Empty"];
  return ["idle", state];
}

const routes = {
  "/": "dashboard",
  "/generate": "generate",
  "/upscale": "upscale",
  "/ltx-video": "ltx-video",
  "/downloads": "downloads",
  "/runtime": "runtime",
  "/models": "models",
};

function navigate() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const page = routes[hash] || "dashboard";
  $$(".page").forEach(p => p.classList.toggle("active", p.dataset.page === page));
  $$("nav a").forEach(a => a.classList.toggle("active", a.dataset.route === hash));
  document.title = `AI Control — ${page.charAt(0).toUpperCase() + page.slice(1)}`;
}

let cache = {};
let genPollTimer = null;
let upPollTimer = null;
let ltxPollTimer = null;

const LTX_BUNDLES = ["ltx-2.5-distilled", "ltx-2.5-studio", "ltx-2.5-prompt-enhancer"];

const IMAGE_BUNDLES = [
  "qwen-image-2512-fp8",
  "qwen-image-2512-lightning-lora",
  "qwen-lora-advertisement",
  "qwen-lora-poster",
  "qwen-lora-flat-cartoon",
  "qwen-lora-eligen-poster",
  "qwen-lora-realism",
  "qwen-controlnet-2512-fun-union",
  "qwen-controlnet-diffsynth",
  "qwen-image-edit-2511-fp8",
  "chroma1-hd",
  "upscalers-esrgan",
  "gemma-4-e4b",
];

function optionalNum(raw) {
  const v = String(raw ?? "").trim();
  if (!v) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function populatePresetSelect(selectEl, presets, filterMode) {
  if (!selectEl) return;
  const list = (presets || []).filter(p => !filterMode || p.mode === filterMode || !p.mode);
  selectEl.innerHTML = `<option value="">Default preset</option>` +
    list.map(p => `<option value="${p.id}">${p.label}</option>`).join("");
}

async function refresh() {
  const endpoints = [
    ["status", "/api/v1/status"],
    ["models", "/api/v1/models"],
    ["downloads", "/api/v1/downloads"],
    ["capabilities", "/api/v1/capabilities"],
  ];
  const results = await Promise.allSettled(endpoints.map(([, path]) => api(path)));
  const errors = [];
  results.forEach((result, i) => {
    const [key] = endpoints[i];
    if (result.status === "fulfilled") {
      cache[key] = result.value;
    } else {
      errors.push(`${key}: ${result.reason?.message || "failed"}`);
    }
  });
  renderDashboard();
  renderDownloads();
  renderRuntime();
  renderModels();
  renderCapabilities();
  renderUpscaleCapabilities();
  renderLtxCapabilities();
  $("#lastRefresh").textContent = "Updated " + new Date().toLocaleTimeString();
  if (errors.length) toast(errors.join(" · "));
}

function renderDashboard() {
  const { status, downloads } = cache;
  const p = downloads.progress || {};
  setBadge($("#dashLoad"), status.loadState, loadKind(status.loadState));
  setBadge($("#dashDl"), downloads.status || "idle", dlKind(downloads.status));
  $("#dashProfile").textContent = status.profile;
  $("#dashBytes").textContent = p.bytesOnDiskHuman || "0 B";
  $("#dashFiles").textContent = `${p.filesComplete || 0} / ${p.filesRequired || 0} files`;
  $("#dashBundles").textContent = `${p.bundlesComplete || 0} / ${p.bundlesTotal || 0} bundles`;
  const pct = p.percentComplete || 0;
  $("#dashProgressBar").style.width = `${pct}%`;
  $("#dashProgressLabel").textContent = `${pct}% required files on disk`;
  if (downloads.running && downloads.currentFile) {
    $("#dashCurrent").textContent = `Downloading: ${downloads.currentFile}`;
  } else if (downloads.currentBundle) {
    $("#dashCurrent").textContent = `Bundle: ${downloads.currentBundle}`;
  } else {
    $("#dashCurrent").textContent = downloads.status === "completed" ? "All catalog downloads finished" : "—";
  }
}

function renderCapabilities() {
  const caps = cache.capabilities || {};
  const presets = caps.presets || [];
  const ready = presets.filter(p => p.ready).length;
  const fastDefault = caps.fastGenModeDefault ?? "auto";
  const defaultRes = caps.defaultWidth && caps.defaultHeight
    ? `${caps.defaultWidth}×${caps.defaultHeight}`
    : "1920×1080";
  $("#capSummary").textContent =
    `${ready}/${presets.length} presets ready · modes: ${(caps.modes || []).join(", ")} · default: ${defaultRes} · fast gen: ${fastDefault}`;
  $("#capPresets").innerHTML = presets.map(p => {
    const badge = p.ready ? `<span class="badge ok">ready</span>` : `<span class="badge warn">missing</span>`;
    return `<div class="cap-preset"><strong>${p.id}</strong> ${badge}
      <div class="meta">${p.label} · ${p.mode}</div>
      <div class="meta">${p.when_to_use}</div></div>`;
  }).join("");
}

function renderUpscaleCapabilities() {
  const up = cache.capabilities?.upscale || {};
  const badge = up.ready ? `<span class="badge ok">ready</span>` : `<span class="badge warn">models missing</span>`;
  const summaryEl = $("#upCapSummary");
  if (summaryEl) {
    summaryEl.innerHTML = `${badge} · RealESRGAN · scales: ${(up.scales || [2, 4]).join("×, ")}×`;
  }
}

function renderLtxCapabilities() {
  const ltx = cache.capabilities?.ltxVideo || {};
  const presets = ltx.presets || [];
  const ready = presets.filter(p => p.ready).length;
  const badge = ltx.ready ? `<span class="badge ok">ready</span>` : `<span class="badge warn">models missing</span>`;
  const summaryEl = $("#ltxCapSummary");
  if (summaryEl) {
    const orientations = (ltx.orientations || []).map(o => o.id).join(", ");
    const studio = ltx.studioReady ? "studio ✓" : "studio missing";
    const enh = ltx.promptEnhanceReady ? "enhancer ✓" : "enhancer optional";
    const cam = ltx.cameraLorasReady ? "camera LoRAs ✓" : "camera LoRAs optional";
    const ic = ltx.icLoraReady ? "IC-LoRA ✓" : "IC-LoRA optional";
    const lip = ltx.lipdubReady ? "LipDub ✓" : "LipDub optional";
    const motion = ltx.motionTrackReady ? "Motion Track ✓" : "Motion Track optional";
    const wf = ltx.workflow?.label || "LTX Studio quality";
    summaryEl.innerHTML = `${badge} · LTX-2.5 · ${ready}/${presets.length} presets · ${studio} · ${enh} · ${cam} · ${ic} · ${lip} · ${motion} · ${wf} · max ${ltx.maxDurationSeconds || 10}s`;
  }
  const root = $("#ltxCapPresets");
  if (!root) return;
  root.innerHTML = presets.map(p => {
    const pBadge = p.ready ? `<span class="badge ok">ready</span>` : `<span class="badge warn">missing</span>`;
    const d = p.defaults || {};
    return `<div class="cap-preset"><strong>${p.id}</strong> ${pBadge}
      <div class="meta">${p.label}</div>
      <div class="meta">${p.when_to_use} · ${d.width}×${d.height} · ${d.length}f @ ${d.fps}fps · refine: ${d.refine ? "yes" : "no"}</div></div>`;
  }).join("");
  populatePresetSelect($("#ltxPreset"), presets);
  applyLtxFormDefaults();

  const hint = $("#ltxComfyHint");
  if (hint) {
    const profile = normalizeProfile(cache.status?.profile);
    const loaded = cache.status?.loadState === "LOADED";
    const ltxUrl = cache.status?.endpoints?.comfyLtx || cache.status?.activeUrl;
    if (profile === "comfy-ltx" && loaded && ltxUrl) {
      hint.innerHTML =
        `ComfyUI LTX is running. <a href="${ltxUrl}" target="_blank" rel="noopener">Open ComfyUI (:8189)</a> → ` +
        `<strong>Workflow → Open</strong> → <code>LTX-2.5-Quality-T2V.json</code> (in blueprints). ` +
        `Or use this page with <strong>Speed: Quality</strong> — same two-stage refine pipeline.`;
    } else {
      hint.textContent =
        "Start comfy-ltx on Runtime. High-quality ComfyUI workflow: LTX-2.5-Quality-T2V.json (synced to ~/ComfyUI-ltx/blueprints/).";
    }
  }
}

function applyLtxFormDefaults() {
  const ltx = cache.capabilities?.ltxVideo || {};
  const orientations = ltx.orientations || [];
  const speedModes = ltx.speedModes || [];
  const orientEl = $("#ltxOrientation");
  const speedEl = $("#ltxSpeed");
  if (orientEl && orientations.length && !orientEl.dataset.bound) {
    orientEl.innerHTML = orientations
      .map(o => `<option value="${o.id}">${o.label} (${o.width}×${o.height})</option>`)
      .join("");
    orientEl.dataset.bound = "1";
  }
  if (speedEl && speedModes.length && !speedEl.dataset.bound) {
    speedEl.innerHTML = speedModes
      .map(m => `<option value="${m.id}">${m.label}${m.refine ? " · refine" : ""}</option>`)
      .join("");
    speedEl.dataset.bound = "1";
  }
  syncLtxAdvancedFromMain();
}

function syncLtxAdvancedFromMain() {
  const ltx = cache.capabilities?.ltxVideo || {};
  const orient = ltx.orientations?.find(o => o.id === ($("#ltxOrientation")?.value || "portrait"));
  const speed = ltx.speedModes?.find(m => m.id === ($("#ltxSpeed")?.value || "standard"));
  if (orient && $("#ltxWidth") && !$("#ltxWidth").value) {
    $("#ltxWidth").placeholder = String(orient.width);
    $("#ltxHeight").placeholder = String(orient.height);
  }
  if (speed && $("#ltxSteps") && !$("#ltxSteps").value) {
    $("#ltxSteps").placeholder = String(speed.steps);
    $("#ltxVideoCfg").placeholder = String(speed.video_cfg);
  }
  if ($("#ltxRefine") && speed && !$("#ltxRefine").dataset.userTouched) {
    $("#ltxRefine").checked = !!speed.refine;
  }
  if ($("#ltxTiledDecode") && speed && !$("#ltxTiledDecode").dataset.userTouched) {
    $("#ltxTiledDecode").checked = !!speed.tiled_decode;
  }
  const enhField = $("#ltxPromptEnhanceField");
  const enhInput = $("#ltxPromptEnhance");
  if (enhField && enhInput) {
    const ready = !!ltx.promptEnhanceReady;
    enhInput.disabled = !ready;
    enhField.title = ready
      ? "Same as ComfyUI prompt_enhance — expands short prompts before generation"
      : "Download ltx-2.5-prompt-enhancer to enable";
  }
  const camEl = $("#ltxCameraMotion");
  if (camEl && (ltx.cameraMotions || []).length && !camEl.dataset.bound) {
    camEl.innerHTML = ltx.cameraMotions
      .map(m => `<option value="${m.id}">${m.label}</option>`)
      .join("");
    camEl.dataset.bound = "1";
  }
  updateLtxIcUi();
}

function renderDownloads() {
  const { downloads } = cache;
  const p = downloads.progress || {};
  setBadge($("#dlBadge"), downloads.status || "idle", dlKind(downloads.status));
  $("#dlSummary").textContent =
    `${p.bytesOnDiskHuman || "0 B"} on disk · ${p.filesComplete || 0}/${p.filesRequired || 0} required files · ${p.bundlesComplete || 0}/${p.bundlesTotal || 0} bundles`;
  $("#dlProgressBar").style.width = `${p.percentComplete || 0}%`;
  $("#dlProgressLabel").textContent = `${p.percentComplete || 0}% complete`;
  $("#dlCurrent").textContent = downloads.currentFile
    ? `Current file: ${downloads.currentFile}`
    : downloads.current
      ? `Current: ${downloads.current}`
      : "";
  $("#dlLog").textContent = (downloads.logTail || []).join("\n") || "(no log yet)";
  $("#startDownload").disabled = downloads.running === true;

  const tbody = $("#filesBody");
  tbody.innerHTML = (p.files || []).map(f => {
    const [kind, label] = fileStateBadge(f.state);
    const rowClass = f.isCurrent ? "current-row" : f.state === "ok" ? "ok-row" : "missing-row";
    return `<tr class="${rowClass}">
      <td><span class="badge ${badgeClass(kind)}">${label}</span></td>
      <td>${f.bundleId}</td>
      <td class="file-name">${f.name}</td>
      <td>${f.sizeHuman}</td>
      <td>${f.optional ? "yes" : ""}</td>
    </tr>`;
  }).join("");
}

function normalizeProfile(profile) {
  if (profile === "comfyui") return "comfy";
  if (profile === "comfyui-ltx") return "comfy-ltx";
  if (profile?.startsWith("CONFLICT")) return "conflict";
  return profile || "none";
}

function bundleStatus(bundleId) {
  const b = (cache.models?.bundles || []).find(x => x.id === bundleId);
  if (!b) return { text: "Catalog bundle unknown", ok: false };
  if (b.status === "complete") return { text: "Models on disk: ready", ok: true };
  if (b.status === "partial") return { text: `Models on disk: partial (${b.files_ok}/${b.files_required} files)`, ok: false };
  return { text: "Models on disk: missing — download first", ok: false };
}

function renderRuntime() {
  const { status } = cache;
  const active = normalizeProfile(status.profile);
  setBadge($("#loadBadge"), status.loadState, loadKind(status.loadState));
  $("#profile").textContent = status.profile;
  $("#model").textContent = status.model;
  $("#apiState").textContent = status.apiState;
  $("#activeUrl").textContent = status.activeUrl || "GPU idle";

  const eps = status.endpoints || {};
  const links = $("#endpointLinks");
  links.innerHTML = "";
  if (status.loadState === "LOADED" && status.activeUrl) {
    const primary = document.createElement("a");
    primary.href = status.activeUrl;
    primary.target = "_blank";
    primary.rel = "noopener";
    primary.textContent = active === "comfy" ? "Open ComfyUI →"
      : active === "comfy-ltx" ? "Open ComfyUI LTX →"
      : "Open LLM API →";
    links.appendChild(primary);
  }
  if (eps.control) {
    const ctrl = document.createElement("a");
    ctrl.href = eps.control;
    ctrl.target = "_blank";
    ctrl.rel = "noopener";
    ctrl.textContent = "GPU control API";
    links.appendChild(ctrl);
  }

  $$("[data-profile]").forEach(btn => {
    btn.classList.toggle("secondary", btn.dataset.profile !== active || status.loadState !== "LOADED");
  });

  $$("[data-profile-card]").forEach(card => {
    card.classList.toggle("active", card.dataset.profileCard === active && status.loadState === "LOADED");
  });

  const fast = bundleStatus("qwen36-35b-a3b-rq");
  const gemma = bundleStatus("gemma-4-e4b");
  const comfyBundles = IMAGE_BUNDLES.map(id => bundleStatus(id));
  const comfyReadyCount = comfyBundles.filter(b => b.ok).length;
  $("#llamaFastReady").textContent = fast.text;
  $("#gemmaReady").textContent = gemma.text;
  $("#comfyReady").textContent = `${comfyReadyCount}/${IMAGE_BUNDLES.length} image bundles ready`;
  const ltxBundles = LTX_BUNDLES.map(id => bundleStatus(id));
  const ltxReadyCount = ltxBundles.filter(b => b.ok).length;
  const ltxEl = $("#comfyLtxReady");
  if (ltxEl) {
    ltxEl.textContent = `${ltxReadyCount}/${LTX_BUNDLES.length} LTX bundles ready (distilled · studio · prompt enhancer)`;
  }

  const gpu = status.gpu || {};
  $("#gpuName").textContent = gpu.name || "nvidia-smi unavailable";
  $("#gpuMem").textContent = gpu.memoryUsed ? `VRAM: ${gpu.memoryUsed} / ${gpu.memoryTotal}` : "—";
  $("#gpuUtil").textContent = gpu.utilization ? `Utilization: ${gpu.utilization}` : "—";
  $("#gpuProcs").textContent = (status.gpuProcesses || [])
    .map(p => `${p.pid} ${p.name} ${p.memory}`).join("\n") || "(no GPU processes)";

  const llamaPanel = $("#llamaPanel");
  llamaPanel.style.display = active === "llama-fast" || active === "gemma" ? "block" : "none";

  const llamaActions = $("#llamaActions");
  llamaActions.innerHTML = "";
  if ((active === "llama-fast" || active === "gemma") && eps.llm) {
    const open = document.createElement("button");
    open.type = "button";
    open.textContent = "Open /v1/models";
    open.addEventListener("click", () => window.open(`${eps.llm}/models`, "_blank", "noopener"));
    llamaActions.appendChild(open);
  }
}

function renderModels() {
  const { models } = cache;
  const bundles = models.bundles || [];
  const complete = bundles.filter(b => b.status === "complete").length;
  $("#catalogSummary").textContent = `${complete}/${bundles.length} bundles complete`;
  const root = $("#catalogRoot");
  root.innerHTML = bundles.map((b, i) => {
    const filesRows = (b.files || []).map(f => {
      const [kind, label] = fileStateBadge(f.state);
      return `<tr>
        <td><span class="badge ${badgeClass(kind)}">${label}</span></td>
        <td class="file-name">${f.name}</td>
        <td>${f.state === "ok" ? humanFromBytes(f.bytes) : "—"}</td>
        <td>${f.optional ? "optional" : ""}</td>
      </tr>`;
    }).join("");
    return `<section class="card" style="margin-bottom:.75rem">
      <div class="bundle-head" data-idx="${i}">
        <strong>${b.id}</strong>
        <span class="badge ${badgeClass(b.status === "complete" ? "ok" : b.status === "partial" ? "warn" : "idle")}">${b.status}</span>
        <span class="meta"> · ${b.files_ok}/${b.files_required} files · ${b.size_human}</span>
      </div>
      <table style="margin-top:.5rem">
        <thead><tr><th>Status</th><th>File</th><th>Size</th><th></th></tr></thead>
        <tbody>${filesRows}</tbody>
      </table>
    </section>`;
  }).join("");
}

function humanFromBytes(n) {
  if (!n) return "0 B";
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(1) + " GB";
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(1) + " MB";
  if (n >= 1024) return (n / 1024).toFixed(1) + " KB";
  return n + " B";
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderGenImages(images) {
  $("#genOutput").innerHTML = images.map(img => {
    const src = img.data || img.url;
    const label = img.filename || img.mime || "output";
    return `<figure><img src="${src}" alt="${label}" loading="lazy" /><figcaption class="meta">${label}</figcaption></figure>`;
  }).join("");
}

function renderJobProgress(el, job, fallbackEndpoint) {
  if (!el) return;
  const p = job?.progress;
  if (!p?.stages?.length) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  const src = p.source || {};
  const endpoint = src.endpoint || fallbackEndpoint;
  const fields = (src.fields || [
    "progress.percentComplete",
    "progress.stages[].percent",
    "phase",
  ]).join(", ");
  const pct = p.percentComplete ?? 0;
  const extra = [p.detail, p.currentNode].filter(Boolean).join(" · ");
  el.innerHTML = `
    <div class="progress"><span style="width:${pct}%"></span></div>
    <p class="meta">${pct}% overall${extra ? ` · ${extra}` : ""}</p>
    <p class="meta job-progress-api">Already in API: <code>${endpoint}</code> → <code>${fields}</code></p>
    <div class="job-stages">${p.stages.map(s => {
      const state = s.state || "pending";
      const sp = s.percent ?? 0;
      const counts = (s.value != null && s.max != null) ? ` ${s.value}/${s.max}` : "";
      return `<div class="job-stage ${state}">
        <div class="job-stage-head"><span>${s.label || s.id}</span><span>${sp}% · ${state}${counts}</span></div>
        <div class="progress"><span style="width:${sp}%"></span></div>
      </div>`;
    }).join("")}</div>`;
}

function genProgressLabel(job) {
  const total = job.count || 1;
  const done = job.completedCount || (job.images?.length || 0);
  const phase = job.phase ? ` (${job.phase})` : "";
  if (job.status === "running" && total > 1) {
    return `Status: ${job.status} — ${done}/${total}${phase}`;
  }
  return `Status: ${job.status}${phase}`;
}

async function pollGenerateJob(jobId) {
  try {
    const job = await api(`/api/v1/generate/${jobId}`);
    $("#genStatus").textContent = genProgressLabel(job);
    renderJobProgress($("#genProgress"), job, `GET /api/v1/generate/${jobId}`);
    if (job.plan) {
      $("#genPlan").textContent = JSON.stringify(job.plan, null, 2);
    }
    if (job.images?.length) {
      renderGenImages(job.images);
    }
    if (job.status === "completed") {
      clearInterval(genPollTimer);
      genPollTimer = null;
      $("#genSubmit").disabled = false;
      toast(`Generation complete (${job.images?.length || 0} image(s))`);
      return;
    }
    if (job.status === "failed") {
      clearInterval(genPollTimer);
      genPollTimer = null;
      $("#genSubmit").disabled = false;
      if (!job.images?.length) {
        $("#genOutput").textContent = job.error || "Generation failed";
      }
      toast(job.error || "Generation failed");
    }
  } catch (err) {
    const done = $("#genStatus").textContent;
    if (!done.includes("Reconnecting")) {
      $("#genStatus").textContent = `Reconnecting… (${err.message}) — images saved on server under jobs/${jobId}/images/`;
    }
  }
}

async function submitGenerate() {
  const prompt = $("#genPrompt").value.trim();
  if (!prompt) {
    toast("Enter a prompt");
    return;
  }
  const count = Math.min(10, Math.max(1, parseInt($("#genCount").value, 10) || 1));
  const fileInput = $("#genImage");
  let image = null;
  if (fileInput.files?.[0]) {
    image = await fileToBase64(fileInput.files[0]);
  }

  $("#genSubmit").disabled = true;
  $("#genOutput").textContent = "Starting job…";
  $("#genPlan").textContent = "";
  $("#genStatus").textContent = "Submitting…";

  try {
    const body = { prompt, count };
    const fastRaw = $("#genFastMode").value;
    if (fastRaw === "true") body.fastGenMode = true;
    else if (fastRaw === "false") body.fastGenMode = false;
    const stepsRaw = $("#genSteps").value;
    if (stepsRaw) body.steps = parseInt(stepsRaw, 10);
    const upscaleRaw = $("#genUpscale").value;
    if (upscaleRaw === "true") body.upscale = true;
    else if (upscaleRaw === "false") body.upscale = false;
    if (image) body.image = image;
    const job = await api("/api/v1/generate", { method: "POST", body: JSON.stringify(body) });
    $("#genStatus").textContent = `Job ${job.jobId} — ${job.status} (0/${job.count || count})`;
    if (genPollTimer) clearInterval(genPollTimer);
    genPollTimer = setInterval(() => pollGenerateJob(job.jobId), 2000);
    await pollGenerateJob(job.jobId);
  } catch (err) {
    $("#genSubmit").disabled = false;
    toast(err.message);
  }
}

function renderUpImages(images) {
  $("#upOutput").innerHTML = images.map(img => {
    const src = img.data || img.url;
    const label = img.filename || img.mime || "output";
    return `<figure><img src="${src}" alt="${label}" loading="lazy" /><figcaption class="meta">${label}</figcaption></figure>`;
  }).join("");
}

async function pollUpscaleJob(jobId) {
  try {
    const job = await api(`/api/v1/upscale/${jobId}`);
    $("#upStatus").textContent = `Status: ${job.status}${job.phase ? ` (${job.phase})` : ""}`;
    renderJobProgress($("#upProgress"), job, `GET /api/v1/upscale/${jobId}`);
    if (job.plan) {
      $("#upPlan").textContent = JSON.stringify(job.plan, null, 2);
    }
    if (job.images?.length) {
      renderUpImages(job.images);
    }
    if (job.status === "completed") {
      clearInterval(upPollTimer);
      upPollTimer = null;
      $("#upSubmit").disabled = false;
      toast(`Upscale complete (${job.images?.length || 0} image(s))`);
      return;
    }
    if (job.status === "failed") {
      clearInterval(upPollTimer);
      upPollTimer = null;
      $("#upSubmit").disabled = false;
      if (!job.images?.length) {
        $("#upOutput").textContent = job.error || "Upscale failed";
      }
      toast(job.error || "Upscale failed");
    }
  } catch (err) {
    const done = $("#upStatus").textContent;
    if (!done.includes("Reconnecting")) {
      $("#upStatus").textContent = `Reconnecting… (${err.message}) — images saved on server under jobs/${jobId}/images/`;
    }
  }
}

async function submitUpscale() {
  const fileInput = $("#upImage");
  const sourceJobId = $("#upSourceJobId").value.trim();
  const sourceFilename = $("#upSourceFilename").value.trim();
  const scale = parseInt($("#upScale").value, 10) || 4;

  if (!fileInput.files?.[0] && !(sourceJobId && sourceFilename)) {
    toast("Upload an image or provide source job ID + filename");
    return;
  }

  $("#upSubmit").disabled = true;
  $("#upOutput").textContent = "Starting job…";
  $("#upPlan").textContent = "";
  $("#upStatus").textContent = "Submitting…";

  try {
    const body = { scale };
    if (fileInput.files?.[0]) {
      body.image = await fileToBase64(fileInput.files[0]);
    }
    if (sourceJobId) body.sourceJobId = sourceJobId;
    if (sourceFilename) body.sourceFilename = sourceFilename;
    const job = await api("/api/v1/upscale", { method: "POST", body: JSON.stringify(body) });
    $("#upStatus").textContent = `Job ${job.jobId} — ${job.status}`;
    if (upPollTimer) clearInterval(upPollTimer);
    upPollTimer = setInterval(() => pollUpscaleJob(job.jobId), 2000);
    await pollUpscaleJob(job.jobId);
  } catch (err) {
    $("#upSubmit").disabled = false;
    toast(err.message);
  }
}

function renderLtxVideos(videos) {
  $("#ltxOutput").innerHTML = videos.map(v => {
    const src = v.url;
    const label = v.filename || "output";
    return `<figure><video src="${src}" controls playsinline style="max-width:100%"></video><figcaption class="meta">${label} (with audio)</figcaption></figure>`;
  }).join("");
}

async function pollLtxVideoJob(jobId) {
  try {
    const job = await api(`/api/v1/ltx-video/${jobId}`);
    $("#ltxStatus").textContent = `Status: ${job.status}${job.phase ? ` (${job.phase})` : ""}`;
    renderJobProgress($("#ltxProgress"), job, `GET /api/v1/ltx-video/${jobId}`);
    if (job.plan) {
      const selected = job.plan.selected || {};
      const pick = [
        selected.camera ? `camera: ${selected.camera}` : null,
        (selected.ic || []).length ? `ic: ${(selected.ic || []).join(", ")}` : null,
        selected.controlType ? `control: ${selected.controlType}` : null,
        job.plan.camera_skip ? `camera skipped: ${job.plan.camera_skip}` : null,
        job.plan.ic_skip ? `ic skipped: ${job.plan.ic_skip}` : null,
      ].filter(Boolean);
      $("#ltxPlan").textContent =
        (pick.length ? `LoRAs: ${pick.join(" · ")}\n\n` : "") + JSON.stringify(job.plan, null, 2);
    }
    if (job.videos?.length) {
      renderLtxVideos(job.videos);
    }
    if (job.status === "completed") {
      clearInterval(ltxPollTimer);
      ltxPollTimer = null;
      $("#ltxSubmit").disabled = false;
      toast(`LTX video complete (${job.videos?.length || 0} video(s))`);
      return;
    }
    if (job.status === "failed") {
      clearInterval(ltxPollTimer);
      ltxPollTimer = null;
      $("#ltxSubmit").disabled = false;
      if (!job.videos?.length) {
        $("#ltxOutput").textContent = job.error || "LTX generation failed";
      }
      toast(job.error || "LTX generation failed");
    }
  } catch (err) {
    const done = $("#ltxStatus").textContent;
    if (!done.includes("Reconnecting")) {
      $("#ltxStatus").textContent = `Reconnecting… (${err.message}) — videos saved on server under jobs/${jobId}/videos/`;
    }
  }
}

function ltxSubmitMode(mode) {
  return mode.startsWith("lipsync") ? "lipsync" : mode;
}

function updateLtxModeUi() {
  const mode = $("#ltxMode")?.value || "t2v";
  const showStart = ["i2v", "flf2v", "lipsync", "lipsync_flf", "lipsync_fml", "a2v", "motion_transfer"].includes(mode);
  const showEnd = ["flf2v", "lipsync", "lipsync_flf", "lipsync_fml", "a2v"].includes(mode);
  const showMid = ["lipsync", "lipsync_fml"].includes(mode);
  const showAudio = ["a2v", "lipsync", "lipsync_flf", "lipsync_fml"].includes(mode);
  const startReq = ["i2v", "flf2v", "lipsync", "lipsync_flf", "lipsync_fml"].includes(mode);
  const audioReq = showAudio;
  const set = (id, on, required) => {
    const el = $(id);
    if (!el) return;
    el.style.display = on ? "block" : "none";
    const label = el.querySelector("span");
    if (label && required != null) label.dataset.required = required ? "1" : "0";
  };
  set("#ltxImageField", showStart, startReq);
  set("#ltxEndImageField", showEnd, mode === "flf2v" || mode === "lipsync_flf" || mode === "lipsync_fml");
  set("#ltxMiddleImageField", showMid, mode === "lipsync_fml");
  set("#ltxAudioField", showAudio, audioReq);
  const showRef = mode === "motion_transfer" || !mode.startsWith("lipsync");
  set("#ltxRefVideoField", showRef, mode === "motion_transfer");
  const audioPrompt = $("#ltxAudio")?.closest("label");
  if (audioPrompt) audioPrompt.style.display = showAudio ? "none" : "block";
  const durationInput = $("#ltxDuration");
  const durationLabel = durationInput?.closest("label");
  if (durationInput) {
    durationInput.disabled = showAudio;
    durationInput.placeholder = showAudio ? "from audio (max 10s)" : "4";
  }
  if (durationLabel) durationLabel.style.opacity = showAudio ? "0.6" : "1";
  updateLtxIcUi();
}

function updateLtxIcUi() {
  const mode = $("#ltxMode")?.value || "t2v";
  const explicitIc = mode.startsWith("lipsync") || mode === "motion_transfer";
  const hasVideo = !!$("#ltxRefVideo")?.files?.length;
  const box = $("#ltxIcFields");
  if (box) box.style.display = !explicitIc && hasVideo ? "block" : "none";
}

async function submitLtxVideo() {
  const uiMode = $("#ltxMode").value;
  const mode = ltxSubmitMode(uiMode);
  const prompt = $("#ltxPrompt").value.trim();
  const audioPrompt = $("#ltxAudio").value.trim();
  const durationRaw = $("#ltxDuration").value.trim();
  const fileInput = $("#ltxImage");

  if (["i2v", "flf2v", "lipsync"].includes(mode) && !fileInput.files?.[0]) {
    toast("Upload a start image");
    return;
  }
  if ((uiMode === "flf2v" || uiMode === "lipsync_flf" || uiMode === "lipsync_fml") && !$("#ltxEndImage")?.files?.[0]) {
    toast("Upload a last-frame image");
    return;
  }
  if (uiMode === "lipsync_fml" && !$("#ltxMiddleImage")?.files?.[0]) {
    toast("Upload a middle-frame image");
    return;
  }
  if (["a2v", "lipsync"].includes(mode) && !$("#ltxAudioFile")?.files?.[0]) {
    toast("Upload an audio file");
    return;
  }
  if (mode === "motion_transfer" && !$("#ltxRefVideo")?.files?.[0]) {
    toast("Upload a reference video for motion transfer");
    return;
  }
  if (mode === "t2v" && !prompt && !audioPrompt) {
    toast("Enter a prompt or audio description");
    return;
  }

  $("#ltxSubmit").disabled = true;
  $("#ltxOutput").textContent = "Starting job…";
  $("#ltxPlan").textContent = "";
  $("#ltxStatus").textContent = "Submitting…";

  try {
    const body = { mode, prompt };
    body.orientation = $("#ltxOrientation")?.value || "portrait";
    body.speed = $("#ltxSpeed")?.value || "standard";
    if (audioPrompt && !["a2v", "lipsync"].includes(mode)) body.audioPrompt = audioPrompt;
    if (durationRaw && !["a2v", "lipsync"].includes(mode)) body.duration = parseFloat(durationRaw);
    if ($("#ltxPreset")?.value) body.preset = $("#ltxPreset").value;
    const seed = optionalNum($("#ltxSeed")?.value);
    if (seed !== undefined) body.seed = seed;
    const width = optionalNum($("#ltxWidth")?.value);
    const height = optionalNum($("#ltxHeight")?.value);
    const steps = optionalNum($("#ltxSteps")?.value);
    const videoCfg = optionalNum($("#ltxVideoCfg")?.value);
    const audioCfg = optionalNum($("#ltxAudioCfg")?.value);
    const strength = optionalNum($("#ltxStrength")?.value);
    const refineSteps = optionalNum($("#ltxRefineSteps")?.value);
    const refineDenoise = optionalNum($("#ltxRefineDenoise")?.value);
    if (width !== undefined) body.width = width;
    if (height !== undefined) body.height = height;
    if (steps !== undefined) body.steps = steps;
    if (videoCfg !== undefined) body.videoCfg = videoCfg;
    if (audioCfg !== undefined) body.audioCfg = audioCfg;
    if (strength !== undefined) body.strength = strength;
    if (refineSteps !== undefined) body.refineSteps = refineSteps;
    if (refineDenoise !== undefined) body.refineDenoise = refineDenoise;
    if ($("#ltxRefine")?.checked) body.refine = true;
    if ($("#ltxTiledDecode")?.checked) body.tiledDecode = true;
    if ($("#ltxPromptEnhance")?.checked) body.promptEnhance = true;
    if ($("#ltxCameraMotion")?.value) body.cameraMotion = $("#ltxCameraMotion").value;
    if (!mode.startsWith("lipsync") && mode !== "motion_transfer") {
      if ($("#ltxIcLora")?.value) body.icLora = $("#ltxIcLora").value;
      if ($("#ltxControlType")?.value) body.controlType = $("#ltxControlType").value;
      if ($("#ltxDetailer")?.checked) body.detailer = true;
    }
    const neg = $("#ltxNegative")?.value.trim();
    if (neg) body.negativePrompt = neg;
    if (fileInput?.files?.[0] && ["i2v", "flf2v", "lipsync", "a2v", "motion_transfer"].includes(mode)) {
      body.image = await fileToBase64(fileInput.files[0]);
    }
    if ($("#ltxEndImage")?.files?.[0]) body.endImage = await fileToBase64($("#ltxEndImage").files[0]);
    if ($("#ltxMiddleImage")?.files?.[0]) body.middleImage = await fileToBase64($("#ltxMiddleImage").files[0]);
    if ($("#ltxAudioFile")?.files?.[0]) body.audio = await fileToBase64($("#ltxAudioFile").files[0]);
    const refVideo = $("#ltxRefVideo")?.files?.[0];
    if (refVideo) body.referenceVideo = await fileToBase64(refVideo);
    const job = await api("/api/v1/ltx-video", { method: "POST", body: JSON.stringify(body) });
    $("#ltxStatus").textContent = `Job ${job.jobId} — ${job.status}`;
    if (ltxPollTimer) clearInterval(ltxPollTimer);
    ltxPollTimer = setInterval(() => pollLtxVideoJob(job.jobId), 2000);
    await pollLtxVideoJob(job.jobId);
  } catch (err) {
    $("#ltxSubmit").disabled = false;
    toast(err.message);
  }
}

function bindActions() {
  $$("[data-profile]").forEach(btn => {
    btn.addEventListener("click", async () => {
      try {
        await api("/api/v1/profile", { method: "PUT", body: JSON.stringify({ id: btn.dataset.profile }) });
        toast(`Started ${btn.dataset.profile}`);
        await refresh();
      } catch (err) { toast(err.message); }
    });
  });
  $("#stopBtn")?.addEventListener("click", async () => {
    try {
      await api("/api/v1/profile", { method: "DELETE" });
      toast("Stopped all profiles");
      await refresh();
    } catch (err) { toast(err.message); }
  });
  $("#startDownload")?.addEventListener("click", async () => {
    try {
      await api("/api/v1/downloads", { method: "POST", body: JSON.stringify({ ids: [] }) });
      toast("Download started in background");
      await refresh();
    } catch (err) { toast(err.message); }
  });
  $("#refreshBtn")?.addEventListener("click", refresh);
  $("#genSubmit")?.addEventListener("click", submitGenerate);
  $("#upSubmit")?.addEventListener("click", submitUpscale);
  $("#upClear")?.addEventListener("click", () => {
    $("#upImage").value = "";
    $("#upSourceJobId").value = "";
    $("#upSourceFilename").value = "";
    $("#upScale").value = "4";
    $("#upOutput").textContent = "Upload an image or set source job fields.";
    $("#upPlan").textContent = "";
    $("#upStatus").textContent = "—";
    renderJobProgress($("#upProgress"), null);
  });
  $("#genClear")?.addEventListener("click", () => {
    $("#genPrompt").value = "";
    $("#genImage").value = "";
    $("#genFastMode").value = "true";
    $("#genSteps").value = "";
    $("#genUpscale").value = "";
    $("#genOutput").textContent = "Submit a prompt to start.";
    $("#genPlan").textContent = "";
    $("#genStatus").textContent = "—";
    renderJobProgress($("#genProgress"), null);
  });
  $("#ltxMode")?.addEventListener("change", updateLtxModeUi);
  $("#ltxRefVideo")?.addEventListener("change", updateLtxIcUi);
  $("#ltxOrientation")?.addEventListener("change", syncLtxAdvancedFromMain);
  $("#ltxSpeed")?.addEventListener("change", syncLtxAdvancedFromMain);
  $("#ltxRefine")?.addEventListener("change", () => { $("#ltxRefine").dataset.userTouched = "1"; });
  $("#ltxTiledDecode")?.addEventListener("change", () => { $("#ltxTiledDecode").dataset.userTouched = "1"; });
  $("#ltxSubmit")?.addEventListener("click", submitLtxVideo);
  $("#ltxClear")?.addEventListener("click", () => {
    $("#ltxPrompt").value = "";
    $("#ltxAudio").value = "";
    $("#ltxImage").value = "";
    if ($("#ltxEndImage")) $("#ltxEndImage").value = "";
    if ($("#ltxMiddleImage")) $("#ltxMiddleImage").value = "";
    if ($("#ltxAudioFile")) $("#ltxAudioFile").value = "";
    if ($("#ltxRefVideo")) $("#ltxRefVideo").value = "";
    if ($("#ltxCameraMotion")) $("#ltxCameraMotion").value = "auto";
    if ($("#ltxIcLora")) $("#ltxIcLora").value = "auto";
    if ($("#ltxControlType")) $("#ltxControlType").value = "auto";
    if ($("#ltxDetailer")) $("#ltxDetailer").checked = false;
    $("#ltxDuration").value = "";
    updateLtxModeUi();
    $("#ltxOutput").textContent = "Submit a prompt to start.";
    $("#ltxPlan").textContent = "";
    $("#ltxStatus").textContent = "—";
    renderJobProgress($("#ltxProgress"), null);
  });
  updateLtxModeUi();
}

window.addEventListener("hashchange", navigate);
navigate();
bindActions();
refresh();
setInterval(refresh, 5000);
