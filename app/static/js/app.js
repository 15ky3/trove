/* Trove web interface — vanilla JS, no build step.
   Live updates arrive over /ws; REST is only for actions and the first load. */

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    view: "library",
    jobs: [],
    library: [],
    libType: "",
    libFilter: "",
    updates: {},        // "type:repo_id" -> result of the last update check
    updatesAt: 0,
    hubType: "model",
    hubResults: [],
    settings: {},
    sheet: null,
    ws: null,
    wsRetry: 0,
  };

  /* -------------------------------------------------------------- Formatting */

  const UNITS = ["B", "KB", "MB", "GB", "TB", "PB"];

  function fmtBytes(n, digits) {
    n = Number(n) || 0;
    let i = 0;
    while (n >= 1024 && i < UNITS.length - 1) { n /= 1024; i++; }
    const d = digits ?? (i === 0 ? 0 : n < 10 ? 2 : 1);
    return `${n.toFixed(d)} ${UNITS[i]}`;
  }

  const fmtSpeed = (n) => (n > 0 ? `${fmtBytes(n, 1)}/s` : "—");

  function fmtEta(seconds) {
    if (!isFinite(seconds) || seconds <= 0 || seconds > 86400 * 2) return "";
    const s = Math.round(seconds);
    if (s < 60) return `${s} s left`;
    if (s < 3600) return `${Math.round(s / 60)} min left`;
    return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min left`;
  }

  function fmtRel(ts) {
    if (!ts) return "—";
    const diff = (Date.now() - ts * 1000) / 1000;
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
    if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
    if (diff < 86400 * 30) return `${Math.round(diff / 86400)} d ago`;
    return new Date(ts * 1000).toLocaleDateString("en-GB");
  }

  const fmtNum = (n) => new Intl.NumberFormat("en-GB").format(Number(n) || 0);
  const shortSha = (sha) => String(sha || "").slice(0, 7);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function splitId(repoId) {
    const i = String(repoId).indexOf("/");
    return i < 0 ? { org: "", name: repoId } : { org: repoId.slice(0, i + 1), name: repoId.slice(i + 1) };
  }

  const TYPE_LABEL = { model: "Model", dataset: "Dataset", space: "Space" };
  const STATUS_LABEL = {
    queued: "Waiting", running: "Running", done: "Done", error: "Failed", cancelled: "Cancelled",
  };

  /* --------------------------------------------------------------------- API */

  async function api(path, options = {}) {
    const opts = { headers: { "Content-Type": "application/json" }, ...options };
    if (opts.body && typeof opts.body !== "string") opts.body = JSON.stringify(opts.body);
    const res = await fetch(path, opts);
    if (res.status === 401) { showGate(); throw new Error("Not signed in"); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
    return data;
  }

  function toast(message, kind = "info", title = "") {
    const node = document.createElement("div");
    node.className = `toast is-${kind}`;
    node.innerHTML = (title ? `<b>${esc(title)}</b>` : "") + esc(message);
    $("#toasts").appendChild(node);
    setTimeout(() => {
      node.style.transition = "opacity .25s ease, transform .25s ease";
      node.style.opacity = "0";
      node.style.transform = "translateY(6px)";
      setTimeout(() => node.remove(), 260);
    }, kind === "err" ? 7000 : 3800);
  }

  const fail = (err) => toast(err.message || String(err), "err", "Didn't work");

  /* --------------------------------------------------------------- Signing in */

  function showGate() {
    $("#gate").hidden = false;
    $("#shell").hidden = true;
    setTimeout(() => $("#gate-pass").focus(), 50);
  }

  async function boot() {
    const session = await api("/api/session");
    if (session.version) $("#app-version").textContent = session.version;
    if (session.auth_required && !session.authenticated) { showGate(); return; }
    $("#gate").hidden = true;
    $("#shell").hidden = false;
    connect();
    await Promise.all([loadJobs(), loadLibrary(), loadSettings(), loadDisk()]);
    setView(viewFromHash() || state.view, true);
  }

  $("#gate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const error = $("#gate-error");
    error.hidden = true;
    try {
      await api("/api/login", { method: "POST", body: { password: $("#gate-pass").value } });
      $("#gate-pass").value = "";
      await boot();
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
    }
  });

  /* ---------------------------------------------------------------- WebSocket */

  function connect() {
    if (state.ws && state.ws.readyState <= 1) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;

    ws.onopen = () => { state.wsRetry = 0; };
    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }
      if (msg.type === "jobs") state.jobs = msg.jobs;
      else if (msg.type === "job") upsertJob(msg.job);
      else if (msg.type === "removed") state.jobs = state.jobs.filter((j) => j.id !== msg.id);
      else if (msg.type === "library") { loadLibrary(); loadDisk(); return; }
      afterJobs();
    };
    ws.onclose = () => {
      state.ws = null;
      state.wsRetry = Math.min(state.wsRetry + 1, 6);
      setTimeout(connect, 500 * 2 ** state.wsRetry);
    };
    ws.onerror = () => ws.close();

    clearInterval(connect._ping);
    connect._ping = setInterval(() => {
      if (state.ws && state.ws.readyState === 1) state.ws.send("ping");
    }, 25000);
  }

  function upsertJob(job) {
    const index = state.jobs.findIndex((j) => j.id === job.id);
    const wasActive = index >= 0 && ["queued", "running"].includes(state.jobs[index].status);
    if (index >= 0) state.jobs[index] = job; else state.jobs.push(job);
    // A finished download changes the library, so pull it again.
    if (wasActive && job.status === "done") { loadLibrary(); loadDisk(); }
    if (state.sheet?.kind === "job" && state.sheet.id === job.id) renderJobSheet(job);
  }

  function afterJobs() {
    renderQueue();
    renderNavCounts();
    renderThroughput();
    if (state.view === "queue") renderViewMeta();
  }

  /* ----------------------------------------------------------------- Loading */

  async function loadJobs() {
    const data = await api("/api/jobs");
    state.jobs = data.jobs;
    afterJobs();
  }

  async function loadLibrary(refresh = false) {
    const data = await api(`/api/library?refresh=${refresh ? 1 : 0}`);
    state.library = data.repos;
    state.dataDir = data.data_dir;
    $("#up-datadir").textContent = data.data_dir;
    $("#disk-path").textContent = data.data_dir;
    renderLibrary();
    renderNavCounts();
    renderSrcOptions();
    if (state.view === "library") renderViewMeta();
  }

  async function loadDisk() {
    const disk = await api("/api/disk");
    if (!disk.total) return;
    const used = ((disk.total - disk.free) / disk.total) * 100;
    $("#disk-free").textContent = `${fmtBytes(disk.free)} free`;
    const fill = $("#disk-fill");
    fill.style.width = `${used.toFixed(1)}%`;
    fill.classList.toggle("is-full", used > 92);
  }

  async function loadSettings() {
    state.settings = await api("/api/settings");
    const s = state.settings;
    $("#set-endpoint").value = s.endpoint || "";
    $("#set-concurrent").value = s.max_concurrent;
    $("#set-workers").value = s.max_workers;
    $("#set-autoclear").checked = !!s.auto_clear_done;
    $("#set-token").placeholder = s.token_set ? `stored (${s.token_hint})` : "hf_…";
    $("#set-token-state").textContent = s.token_set
      ? "A token is stored. Leave the field empty to keep it."
      : "No token — private repos and uploads need one.";
    $("#set-token-state").className = `hint ${s.token_set ? "is-ok" : ""}`;
    $("#up-token-hint").textContent = s.token_set ? "" : "Uploads fail without a token.";
    $("#up-token-hint").className = `hint ${s.token_set ? "" : "is-err"}`;
  }

  /* ------------------------------------------------------------------ Views */

  const VIEW_TITLES = {
    library: "Library",
    hub: "Hub",
    queue: "Queue",
    upload: "Upload",
    settings: "Settings",
  };

  function viewFromHash() {
    const view = location.hash.replace(/^#/, "");
    return VIEW_TITLES[view] ? view : "";
  }

  function setView(view, fromHash = false) {
    state.view = view;
    if (!fromHash && viewFromHash() !== view) location.hash = view;
    $$("#nav .nav-item").forEach((b) => b.classList.toggle("is-active", b.dataset.view === view));
    $$(".view").forEach((v) => { v.hidden = v.id !== `view-${view}`; });
    $("#view-title").textContent = VIEW_TITLES[view];
    renderViewMeta();
    if (view === "hub" && !state.hubResults.length) runSearch();
    if (view === "settings") loadSettings();
  }

  function renderViewMeta() {
    const sub = $("#view-sub");
    if (state.view === "library") {
      const total = state.library.reduce((sum, r) => sum + r.size, 0);
      sub.textContent = state.library.length
        ? `${state.library.length} repos · ${fmtBytes(total)} in ${state.dataDir || "/data"}`
        : "Nothing downloaded yet";
    } else if (state.view === "queue") {
      const active = state.jobs.filter((j) => ["queued", "running"].includes(j.status)).length;
      sub.textContent = active ? `${active} active · ${state.jobs.length} total` : `${state.jobs.length} entries`;
    } else if (state.view === "hub") {
      sub.textContent = state.hubResults.length ? `${state.hubResults.length} results` : "Search the Hugging Face Hub";
    } else if (state.view === "upload") {
      sub.textContent = "Push a local folder to a Hub repo";
    } else {
      sub.textContent = "Token, concurrency and behaviour";
    }
  }

  function renderNavCounts() {
    const active = state.jobs.filter((j) => ["queued", "running"].includes(j.status)).length;
    const queue = $('[data-count="queue"]');
    queue.textContent = active || "";
    queue.classList.toggle("is-live", active > 0);
    $('[data-count="library"]').textContent = state.library.length || "";
  }

  function renderThroughput() {
    const running = state.jobs.filter((j) => j.status === "running");
    const speed = running.reduce((sum, j) => sum + (j.speed || 0), 0);
    const box = $("#throughput");
    box.hidden = running.length === 0;
    $("#tp-text").textContent = `${fmtSpeed(speed)} · ${running.length} active`;
  }

  /* --------------------------------------------------------- Rendering tapes */

  function progressOf(job) {
    if (job.status === "done") return 1;
    if (job.kind === "upload" && job.total_files) return Math.min(1, job.done_files / job.total_files);
    if (job.total_bytes) return Math.min(1, job.done_bytes / job.total_bytes);
    return 0;
  }

  // The numbers change every second, the structure (buttons!) almost never.
  // Rendering them apart keeps click targets stable while the figures keep
  // moving — otherwise an update swaps out the button under the cursor.
  function jobMetaHTML(job) {
    const eta = job.status === "running" && job.speed > 0 && job.total_bytes
      ? fmtEta((job.total_bytes - job.done_bytes) / job.speed) : "";
    const meta = [];

    if (job.total_bytes) {
      meta.push(`<span class="${job.status === "running" ? "is-live" : ""}">${fmtBytes(job.done_bytes)} / ${fmtBytes(job.total_bytes)}</span>`);
    }
    if (job.status === "running" && job.stalled) {
      meta.push(`<span class="is-err">stalled — no data for several minutes</span>`);
    } else if (job.status === "running") {
      meta.push(`<span class="is-live">${fmtSpeed(job.speed)}</span>`);
      if (eta) meta.push(`<span>${eta}</span>`);
    }
    if (job.status === "queued") meta.push("<span>waiting for a free slot</span>");
    if (job.status === "error" && job.error) meta.push(`<span class="is-err">${esc(job.error)}</span>`);
    if (job.status === "done" && job.finished_at) meta.push(`<span>${fmtRel(job.finished_at)}</span>`);
    meta.push(`<span class="path">${esc(job.kind === "upload" ? job.src : job.dest)}</span>`);
    return meta.join("");
  }

  function jobNumHTML(job) {
    const pct = Math.round(progressOf(job) * 100);
    const head = ["running", "queued"].includes(job.status) ? `${pct} %` : STATUS_LABEL[job.status];
    const sub = job.total_files ? `${fmtNum(job.done_files)}/${fmtNum(job.total_files)} files` : "—";
    return `<b>${head}</b><span>${sub}</span>`;
  }

  function jobRowHTML(job) {
    const { org, name } = splitId(job.repo_id);
    const actions = ["queued", "running"].includes(job.status)
      ? `<button class="btn btn-sm btn-danger" data-act="cancel">Cancel</button>`
      : `<button class="btn btn-sm" data-act="retry">Try again</button>
         <button class="btn btn-sm btn-danger" data-act="forget" title="Remove from the list">Remove</button>`;

    return `
      <div class="tape-id">
        <i class="dot is-${job.status}"></i>
        <span class="tape-name"><span class="org">${esc(org)}</span><b>${esc(name)}</b></span>
        <span class="tag ${job.kind === "upload" ? "is-accent" : ""}">${job.kind === "upload" ? "Upload" : "Download"}</span>
        ${job.revision ? `<span class="tag">${esc(job.revision)}</span>` : ""}
      </div>
      <div class="tape-meta">${jobMetaHTML(job)}</div>
      <div class="tape-side">
        <div class="tape-num ${job.status === "running" ? "is-accent" : ""}">${jobNumHTML(job)}</div>
        <div class="tape-actions">
          <button class="btn btn-sm" data-act="log">Log</button>
          ${actions}
        </div>
      </div>`;
  }

  function renderQueue() {
    const list = $("#queue-list");
    const stats = $("#queue-stats");
    const counts = state.jobs.reduce((acc, j) => ({ ...acc, [j.status]: (acc[j.status] || 0) + 1 }), {});
    stats.textContent = state.jobs.length
      ? `${counts.running || 0} running · ${counts.queued || 0} waiting · ${counts.done || 0} done · ${counts.error || 0} failed`
      : "No transfers";

    if (!state.jobs.length) {
      list.innerHTML = `<div class="empty"><b>The queue is empty</b>Start a download from the Hub — several at once is fine.</div>`;
      list.dataset.empty = "1";
      return;
    }
    if (list.dataset.empty) { list.innerHTML = ""; delete list.dataset.empty; }

    const existing = new Map($$(".tape", list).map((n) => [n.dataset.id, n]));
    state.jobs.forEach((job, index) => {
      // Only rebuild the structure when something about it actually changed.
      const sig = `${job.status}|${job.kind}|${job.revision}`;
      let node = existing.get(job.id);
      if (!node) {
        node = document.createElement("div");
        node.dataset.id = job.id;
      } else {
        existing.delete(job.id);
      }
      if (node.dataset.sig !== sig) {
        node.dataset.sig = sig;
        node.className = `tape is-${job.status}`;
        node.innerHTML = jobRowHTML(job);
      } else {
        $(".tape-meta", node).innerHTML = jobMetaHTML(job);
        $(".tape-num", node).innerHTML = jobNumHTML(job);
      }
      node.style.setProperty("--p", `${(progressOf(job) * 100).toFixed(1)}%`);
      if (list.children[index] !== node) list.insertBefore(node, list.children[index] || null);
    });
    existing.forEach((node) => node.remove());
  }

  $("#queue-list").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-act]");
    if (!button) return;
    const id = button.closest(".tape").dataset.id;
    const job = state.jobs.find((j) => j.id === id);
    try {
      if (button.dataset.act === "cancel") await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      else if (button.dataset.act === "retry") await api(`/api/jobs/${id}/retry`, { method: "POST" });
      else if (button.dataset.act === "forget") await api(`/api/jobs/${id}`, { method: "DELETE" });
      else if (button.dataset.act === "log" && job) openJobSheet(job);
    } catch (err) { fail(err); }
  });

  $("#queue-clear").addEventListener("click", async () => {
    try {
      const res = await api("/api/jobs/clear", { method: "POST" });
      await loadJobs();
      toast(`Removed ${res.removed} entries.`, "ok");
    } catch (err) { fail(err); }
  });

  /* ----------------------------------------------------------------- Library */

  const updateKey = (repo) => `${repo.repo_type}:${repo.repo_id}`;
  const updateFor = (repo) => state.updates[updateKey(repo)];

  // One Hub request per repo, so this is deliberately a button rather than
  // something that runs on every library load.
  async function checkUpdates() {
    const data = await api("/api/library/updates");
    state.updates = {};
    data.repos.forEach((entry) => { state.updates[updateKey(entry)] = entry; });
    state.updatesAt = data.checked_at;
    renderLibrary();

    const failed = data.repos.filter((e) => e.error).length;
    const note = failed ? ` ${failed} could not be checked.` : "";
    if (data.outdated) {
      toast(`${data.outdated} of ${data.repos.length} out of date.${note}`, "ok", "Updates available");
    } else {
      toast(`Everything is current.${note}`, "ok");
    }
    return data;
  }

  function renderLibrary() {
    const outdated = Object.values(state.updates).filter((e) => e.outdated).length;
    const all = $("#lib-update-all");
    all.hidden = !outdated;
    all.textContent = `Update all (${outdated})`;

    const list = $("#lib-list");
    const filter = state.libFilter.toLowerCase();
    const repos = state.library.filter(
      (r) => (!state.libType || r.repo_type === state.libType) && (!filter || r.repo_id.toLowerCase().includes(filter))
    );

    if (!repos.length) {
      list.innerHTML = state.library.length
        ? `<div class="empty"><b>No match</b>Try a different filter.</div>`
        : `<div class="empty"><b>Nothing downloaded yet</b>Find a model in the Hub — it lands in ${esc(state.dataDir || "/data")}.</div>`;
      return;
    }

    list.innerHTML = repos.map((repo) => {
      const { org, name } = splitId(repo.repo_id);
      const update = updateFor(repo);
      return `
      <div class="tape is-clickable" data-repo="${esc(repo.repo_id)}" data-type="${esc(repo.repo_type)}" style="--p:0%">
        <div class="tape-id">
          <i class="dot is-model"></i>
          <span class="tape-name"><span class="org">${esc(org)}</span><b>${esc(name)}</b></span>
          <span class="tag">${esc(TYPE_LABEL[repo.repo_type] || repo.repo_type)}</span>
          ${repo.revision && repo.revision !== "main" ? `<span class="tag">${esc(repo.revision)}</span>` : ""}
          ${repo.partial ? `<span class="tag">selected files</span>` : ""}
          ${repo.complete ? "" : `<span class="tag is-err">incomplete</span>`}
          ${update?.outdated ? `<span class="tag is-accent">update available</span>` : ""}
        </div>
        <div class="tape-meta">
          <span>${fmtNum(repo.files)} files</span>
          ${repo.commit ? `<span>@${esc(shortSha(repo.commit))}</span>` : ""}
          ${update?.outdated ? `<span class="is-accent">→ @${esc(shortSha(update.remote_commit))}</span>` : ""}
          <span>${fmtRel(repo.downloaded_at)}</span>
          <span class="path">${esc(repo.path)}</span>
        </div>
        <div class="tape-side">
          <div class="tape-num"><b>${fmtBytes(repo.size)}</b><span>on disk</span></div>
          <div class="tape-actions">
            ${update?.outdated ? `<button class="btn btn-sm btn-accent" data-act="update">Update</button>` : ""}
            <button class="btn btn-sm" data-act="files">Files</button>
            <button class="btn btn-sm" data-act="upload">Upload</button>
            <button class="btn btn-sm btn-danger" data-act="delete">Delete</button>
          </div>
        </div>
      </div>`;
    }).join("");
  }

  $("#lib-list").addEventListener("click", async (event) => {
    const row = event.target.closest(".tape");
    if (!row) return;
    const repoId = row.dataset.repo;
    const repoType = row.dataset.type;
    const button = event.target.closest("[data-act]");
    const action = button ? button.dataset.act : "files";

    if (action === "files") { openLocalSheet(repoId, repoType); return; }
    if (action === "update") { await runUpdate({ repo_id: repoId, repo_type: repoType }); return; }
    if (action === "upload") {
      const repo = state.library.find((r) => r.repo_id === repoId && r.repo_type === repoType);
      prefillUpload(repo);
      return;
    }
    if (action === "delete") {
      const repo = state.library.find((r) => r.repo_id === repoId && r.repo_type === repoType);
      if (!confirm(`Delete ${repoId}?\n\nThis frees ${repo ? fmtBytes(repo.size) : "the folder"} and cannot be undone.`)) return;
      try {
        const res = await api("/api/library/delete", { method: "POST", body: { repo_id: repoId, repo_type: repoType } });
        toast(`Deleted ${repoId} — ${fmtBytes(res.freed)} freed.`, "ok");
        await Promise.all([loadLibrary(), loadDisk()]);
      } catch (err) { fail(err); }
    }
  });

  $("#lib-filter").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    $$("#lib-filter button").forEach((b) => b.classList.toggle("is-active", b === button));
    state.libType = button.dataset.type;
    renderLibrary();
  });

  $("#lib-search").addEventListener("input", (event) => {
    state.libFilter = event.target.value.trim();
    renderLibrary();
  });

  async function runUpdate(body) {
    try {
      const res = await api("/api/library/update", { method: "POST", body });
      if (!res.queued.length) {
        toast(res.skipped.length ? "Already in the queue." : "Nothing to update.");
        return;
      }
      // Queued repos are being replaced, so the old check result is stale.
      res.queued.forEach((id) => {
        Object.keys(state.updates).forEach((key) => {
          if (state.updates[key].repo_id === id) state.updates[key].outdated = false;
        });
      });
      renderLibrary();
      const extra = res.skipped.length ? ` ${res.skipped.length} already queued.` : "";
      toast(`${res.queued.length} queued.${extra}`, "ok", "Update started");
      closeSheet();
      setView("queue");
    } catch (err) { fail(err); }
  }

  $("#lib-check").addEventListener("click", async (event) => {
    event.target.disabled = true;
    const label = event.target.textContent;
    event.target.textContent = "Checking…";
    try { await checkUpdates(); }
    catch (err) { fail(err); }
    finally { event.target.disabled = false; event.target.textContent = label; }
  });

  $("#lib-update-all").addEventListener("click", async (event) => {
    event.target.disabled = true;
    try { await runUpdate({ all_outdated: true }); }
    finally { event.target.disabled = false; }
  });

  $("#lib-refresh").addEventListener("click", async (event) => {
    event.target.disabled = true;
    try { await loadLibrary(true); toast("Sizes recalculated.", "ok"); }
    catch (err) { fail(err); }
    finally { event.target.disabled = false; }
  });

  /* --------------------------------------------------------------------- Hub */

  const REPO_ID_RE = /^[A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*$/;
  let searchTimer = null;

  async function runSearch() {
    const query = $("#hub-search").value.trim();
    const list = $("#hub-list");
    list.innerHTML = `<div class="empty">Searching…</div>`;
    try {
      const data = await api(`/api/search?q=${encodeURIComponent(query)}&repo_type=${state.hubType}&limit=40`);
      state.hubResults = data.results;
      renderHub();
      renderViewMeta();
    } catch (err) {
      list.innerHTML = `<div class="empty"><b>Search failed</b>${esc(err.message)}</div>`;
    }
  }

  function renderHub() {
    const list = $("#hub-list");
    if (!state.hubResults.length) {
      list.innerHTML = `<div class="empty"><b>Nothing found</b>Try another term, or paste a full repo ID.</div>`;
      return;
    }
    const local = new Set(state.library.map((r) => `${r.repo_type}:${r.repo_id}`));

    list.innerHTML = state.hubResults.map((item) => {
      const { org, name } = splitId(item.repo_id);
      const isLocal = local.has(`${item.repo_type}:${item.repo_id}`);
      return `
      <div class="tape is-clickable" data-repo="${esc(item.repo_id)}" data-type="${esc(item.repo_type)}" style="--p:0%">
        <div class="tape-id">
          <i class="dot ${isLocal ? "is-done" : ""}"></i>
          <span class="tape-name"><span class="org">${esc(org)}</span><b>${esc(name)}</b></span>
          ${item.pipeline_tag ? `<span class="tag">${esc(item.pipeline_tag)}</span>` : ""}
          ${item.gated ? `<span class="tag is-err">gated</span>` : ""}
          ${item.private ? `<span class="tag is-accent">private</span>` : ""}
          ${isLocal ? `<span class="tag">local</span>` : ""}
        </div>
        <div class="tape-meta">
          <span>${fmtNum(item.downloads)} downloads</span>
          <span>${fmtNum(item.likes)} likes</span>
          <span>${fmtRel(item.updated_at)}</span>
          ${item.tags.length ? `<span class="path">${esc(item.tags.join(" · "))}</span>` : ""}
        </div>
        <div class="tape-side">
          <div class="tape-actions">
            <button class="btn btn-sm" data-act="detail">Details</button>
            <button class="btn btn-sm btn-accent" data-act="get">Download</button>
          </div>
        </div>
      </div>`;
    }).join("");
  }

  $("#hub-list").addEventListener("click", (event) => {
    const row = event.target.closest(".tape");
    if (!row) return;
    const button = event.target.closest("[data-act]");
    if (button?.dataset.act === "get") { startDownload(row.dataset.repo, row.dataset.type); return; }
    openHubSheet(row.dataset.repo, row.dataset.type);
  });

  $("#hub-type").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    $$("#hub-type button").forEach((b) => b.classList.toggle("is-active", b === button));
    state.hubType = button.dataset.type;
    runSearch();
  });

  $("#hub-search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(runSearch, 350);
  });

  $("#hub-search").addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    const value = $("#hub-search").value.trim();
    if (REPO_ID_RE.test(value)) { clearTimeout(searchTimer); openHubSheet(value, state.hubType); }
    else runSearch();
  });

  $("#hub-direct").addEventListener("click", () => {
    const value = $("#hub-search").value.trim();
    if (!REPO_ID_RE.test(value)) {
      toast("Enter a full repo ID first, for example Qwen/Qwen3-8B.", "err", "Not a repo ID");
      return;
    }
    startDownload(value, state.hubType);
  });

  async function startDownload(repoId, repoType, extra = {}) {
    try {
      await api("/api/jobs/download", { method: "POST", body: { repo_id: repoId, repo_type: repoType, ...extra } });
      toast(`${repoId} queued.`, "ok", "Download started");
      closeSheet();
      setView("queue");
    } catch (err) { fail(err); }
  }

  /* ------------------------------------------------------------------- Sheet */

  function openSheet(eyebrow, title, body, footer) {
    $("#sheet-eyebrow").textContent = eyebrow;
    $("#sheet-title").textContent = title;
    $("#sheet-body").innerHTML = body;
    $("#sheet-foot").innerHTML = footer;
    $("#sheet").hidden = false;
  }

  function closeSheet() {
    $("#sheet").hidden = true;
    state.sheet = null;
  }

  $("#sheet").addEventListener("click", (event) => {
    if (event.target.closest("[data-close]")) closeSheet();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#sheet").hidden) closeSheet();
  });

  const statHTML = (label, value) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`;

  // Set path and filename separately: in "onnx/model_qint8_arm64.onnx" the name
  // has to stay readable, the folder may be trimmed.
  function fileLabelHTML(name) {
    const cut = String(name).lastIndexOf("/");
    let dir = cut < 0 ? "" : name.slice(0, cut + 1);
    const base = cut < 0 ? name : name.slice(cut + 1);
    // Split GGUF quants sit in a folder named exactly like the file inside it.
    // Dropping that path costs no information and gives the name the full width.
    if (dir && base.startsWith(dir.slice(0, -1))) dir = "";
    // Full path as a tooltip, in case the folder got trimmed away.
    return `<span class="file-name"${dir ? ` title="${esc(name)}"` : ""}>${
      dir ? `<span class="dir">${esc(dir)}</span>` : ""}<span class="base">${esc(base)}</span></span>`;
  }

  const fileListHTML = (files, limit = 400) => `
    <div class="file-list">
      ${files.slice(0, limit).map((f) => `
        <div class="file-row">${fileLabelHTML(f.name)}<span class="file-size">${fmtBytes(f.size)}</span></div>`).join("")}
      ${files.length > limit ? `<div class="file-row"><span>… ${fmtNum(files.length - limit)} more</span><span></span></div>` : ""}
    </div>`;

  /* ------------------------------------------------------------ Picking files */

  // Files split across parts (model-00001-of-00004.safetensors, a GGUF quant in
  // several pieces) belong together — ticked individually they would be useless.
  const SHARD_RE = /^(.*?)[-_]?(\d{5})-of-(\d{5})(\.[A-Za-z0-9]+)$/;

  function groupFiles(files) {
    const groups = new Map();
    for (const file of files) {
      const match = file.name.match(SHARD_RE);
      const key = match ? `${match[1]}|${match[3]}|${match[4]}` : file.name;
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          label: match ? `${match[1]}${match[4]}` : file.name,
          names: [],
          size: 0,
        });
      }
      const group = groups.get(key);
      group.names.push(file.name);
      group.size += file.size;
    }
    return [...groups.values()];
  }

  const picker = { groups: [], selected: new Set(), total: 0 };

  function pickerRowsHTML(filter = "") {
    const needle = filter.trim().toLowerCase();
    const rows = picker.groups.filter((g) => !needle || g.label.toLowerCase().includes(needle));
    if (!rows.length) return `<div class="file-row"><span>No file matches that filter.</span><span></span></div>`;
    return rows.map((group) => `
      <label class="file-row is-pick ${picker.selected.has(group.key) ? "is-on" : ""}" data-key="${esc(group.key)}">
        <input type="checkbox" ${picker.selected.has(group.key) ? "checked" : ""}>
        ${fileLabelHTML(group.label)}
        ${group.names.length > 1 ? `<span class="parts">${group.names.length} parts</span>` : ""}
        <span class="file-size">${fmtBytes(group.size)}</span>
      </label>`).join("");
  }

  function selectedFiles() {
    return picker.groups.filter((g) => picker.selected.has(g.key)).flatMap((g) => g.names);
  }

  function selectedSize() {
    return picker.groups.filter((g) => picker.selected.has(g.key)).reduce((sum, g) => sum + g.size, 0);
  }

  function renderPickerState() {
    const count = selectedFiles().length;
    const size = count ? selectedSize() : picker.total;
    const button = $("#sheet-download");
    if (button) {
      button.textContent = count
        ? `Download ${fmtNum(count)} file${count === 1 ? "" : "s"} · ${fmtBytes(size)}`
        : `Download everything · ${fmtBytes(picker.total)}`;
    }
    const summary = $("#pick-summary");
    if (summary) {
      summary.textContent = count
        ? `${fmtNum(count)} of ${fmtNum(picker.groups.reduce((n, g) => n + g.names.length, 0))} files · ${fmtBytes(size)}`
        : "Nothing picked — the whole repo comes down.";
    }
    const all = $("#pick-all");
    if (all) {
      all.checked = picker.selected.size === picker.groups.length && picker.groups.length > 0;
      all.indeterminate = picker.selected.size > 0 && picker.selected.size < picker.groups.length;
    }
  }

  function filePickerHTML(files) {
    picker.groups = groupFiles(files);
    picker.selected = new Set();
    picker.total = files.reduce((sum, f) => sum + f.size, 0);
    return `
      <div>
        <div class="picker-head">
          <label class="check"><input type="checkbox" id="pick-all"> <span>Select all</span></label>
          <input class="field field-sm" id="pick-filter" placeholder="Filter, e.g. Q4_K_M" autocomplete="off">
        </div>
        <div class="file-list" id="pick-list">${pickerRowsHTML()}</div>
        <p class="hint" id="pick-summary"></p>
      </div>`;
  }

  function wirePicker() {
    const list = $("#pick-list");
    if (!list) return;

    list.addEventListener("change", (event) => {
      const row = event.target.closest(".file-row[data-key]");
      if (!row) return;
      const key = row.dataset.key;
      if (event.target.checked) picker.selected.add(key); else picker.selected.delete(key);
      row.classList.toggle("is-on", event.target.checked);
      renderPickerState();
    });

    $("#pick-all").addEventListener("change", (event) => {
      picker.selected = event.target.checked ? new Set(picker.groups.map((g) => g.key)) : new Set();
      list.innerHTML = pickerRowsHTML($("#pick-filter").value);
      renderPickerState();
    });

    $("#pick-filter").addEventListener("input", (event) => {
      list.innerHTML = pickerRowsHTML(event.target.value);
    });

    renderPickerState();
  }

  async function openHubSheet(repoId, repoType) {
    state.sheet = { kind: "hub", id: repoId };
    openSheet(TYPE_LABEL[repoType] || repoType, repoId, `<p class="hint">Loading repo details…</p>`, "");
    let info;
    try {
      info = await api(`/api/repo?repo_id=${encodeURIComponent(repoId)}&repo_type=${repoType}`);
    } catch (err) {
      $("#sheet-body").innerHTML = `<p class="hint is-err">${esc(err.message)}</p>`;
      return;
    }
    if (state.sheet?.id !== repoId) return;

    $("#sheet-body").innerHTML = `
      <div class="stat-grid">
        ${statHTML("Size", fmtBytes(info.total_size))}
        ${statHTML("Files", fmtNum(info.files.length))}
        ${statHTML("Downloads", fmtNum(info.downloads))}
        ${statHTML("Likes", fmtNum(info.likes))}
      </div>
      <div class="sheet-section">
        <h3>Target folder</h3>
        <p class="mono hint">${esc(info.local_dir)}</p>
        ${info.local ? `<p class="hint is-ok">Already here — running again only fetches what changed.</p>` : ""}
      </div>
      <div class="sheet-section">
        <h3>Pick files</h3>
        ${filePickerHTML(info.files)}
      </div>
      <div class="sheet-section">
        <h3>Options</h3>
        <div class="form-control">
          <input class="field mono" id="sheet-rev" placeholder="Revision (empty = main)" value="${esc(info.revision === "main" ? "" : info.revision)}">
          <input class="field mono" id="sheet-allow" placeholder="Or by pattern: *.safetensors, *.json">
          <input class="field mono" id="sheet-ignore" placeholder="Skip these: *.bin, *.msgpack">
          <p class="hint">Patterns are comma-separated and add to whatever you ticked above.</p>
        </div>
      </div>`;

    $("#sheet-foot").innerHTML = `
      <button class="btn btn-accent" id="sheet-download">Download everything · ${fmtBytes(info.total_size)}</button>
      <span class="spacer"></span>
      <button class="btn" data-close>Close</button>`;

    wirePicker();

    $("#sheet-download").addEventListener("click", () => {
      const patterns = (id) => $(`#${id}`).value.split(",").map((s) => s.trim()).filter(Boolean);
      startDownload(repoId, repoType, {
        revision: $("#sheet-rev").value.trim(),
        files: selectedFiles(),
        allow_patterns: patterns("sheet-allow"),
        ignore_patterns: patterns("sheet-ignore"),
      });
    });
  }

  async function openLocalSheet(repoId, repoType) {
    state.sheet = { kind: "local", id: repoId };
    const repo = state.library.find((r) => r.repo_id === repoId && r.repo_type === repoType);
    const update = state.updates[`${repoType}:${repoId}`];
    openSheet("Local", repoId, `<p class="hint">Reading the folder…</p>`, "");
    let data;
    try {
      data = await api(`/api/library/files?repo_id=${encodeURIComponent(repoId)}&repo_type=${repoType}`);
    } catch (err) {
      $("#sheet-body").innerHTML = `<p class="hint is-err">${esc(err.message)}</p>`;
      return;
    }
    if (state.sheet?.id !== repoId) return;

    $("#sheet-body").innerHTML = `
      <div class="stat-grid">
        ${statHTML("Size", fmtBytes(repo?.size || 0))}
        ${statHTML("Files", fmtNum(data.files.length))}
        ${statHTML("Revision", esc(repo?.revision || "main"))}
        ${statHTML("Commit", esc(shortSha(repo?.commit) || "—"))}
      </div>
      <div class="sheet-section">
        <h3>Path</h3>
        <p class="mono hint">${esc(data.path)}</p>
        ${repo?.partial ? `<p class="hint">Partial copy — refreshing keeps the same selection, and it is left out of update checks.</p>` : ""}
        ${update?.outdated ? `<p class="hint is-accent">The Hub is at @${esc(shortSha(update.remote_commit))} — updating replaces this copy.</p>` : ""}
        ${update && !update.outdated && !update.skipped && !update.error ? `<p class="hint is-ok">Up to date as of the last check.</p>` : ""}
        ${update?.error ? `<p class="hint is-err">${esc(update.error)}</p>` : ""}
      </div>
      <div class="sheet-section">
        <h3>Files</h3>
        ${fileListHTML(data.files)}
      </div>`;

    $("#sheet-foot").innerHTML = `
      <button class="btn ${update?.outdated ? "btn-accent" : ""}" id="sheet-update">${
        update?.outdated ? `Update to @${esc(shortSha(update.remote_commit))}` : "Refresh from Hub"
      }</button>
      <button class="btn" id="sheet-upload">Upload</button>
      <span class="spacer"></span>
      <button class="btn btn-danger" id="sheet-delete">Delete</button>`;

    $("#sheet-update").addEventListener("click", () => startDownload(repoId, repoType, {
      revision: repo?.revision || "",
      files: repo?.files_selected || [],
      allow_patterns: repo?.allow_patterns || [],
      ignore_patterns: repo?.ignore_patterns || [],
    }));
    $("#sheet-upload").addEventListener("click", () => { prefillUpload(repo); closeSheet(); });
    $("#sheet-delete").addEventListener("click", async () => {
      if (!confirm(`Delete ${repoId}? This cannot be undone.`)) return;
      try {
        const res = await api("/api/library/delete", { method: "POST", body: { repo_id: repoId, repo_type: repoType } });
        toast(`Deleted ${repoId} — ${fmtBytes(res.freed)} freed.`, "ok");
        closeSheet();
        await Promise.all([loadLibrary(), loadDisk()]);
      } catch (err) { fail(err); }
    });
  }

  function openJobSheet(job) {
    state.sheet = { kind: "job", id: job.id };
    openSheet(job.kind === "upload" ? "Upload" : "Download", job.repo_id, "", `
      <span class="spacer"></span><button class="btn" data-close>Close</button>`);
    renderJobSheet(job);
  }

  function renderJobSheet(job) {
    const pct = Math.round(progressOf(job) * 100);
    $("#sheet-body").innerHTML = `
      <div class="stat-grid">
        ${statHTML("Status", STATUS_LABEL[job.status])}
        ${statHTML("Progress", `${pct} %`)}
        ${statHTML("Transferred", fmtBytes(job.done_bytes))}
        ${statHTML("Rate", job.status === "running" ? fmtSpeed(job.speed) : "—")}
      </div>
      ${job.error ? `<p class="hint is-err">${esc(job.error)}</p>` : ""}
      ${job.stalled ? `<p class="hint is-err">Nothing has arrived for several minutes. Check that the container reaches huggingface.co — and on a NAS, that the system has enough entropy (see Troubleshooting in the README).</p>` : ""}
      <div class="sheet-section">
        <h3>${job.kind === "upload" ? "Source" : "Target"}</h3>
        <p class="mono hint">${esc(job.kind === "upload" ? job.src : job.dest)}</p>
      </div>
      <div class="sheet-section">
        <h3>History</h3>
        <div class="log">
          ${(job.logs || []).map((line) => `
            <div><time>${new Date(line.t * 1000).toLocaleTimeString("en-GB")}</time><span class="is-${esc(line.level)}">${esc(line.msg)}</span>${line.n > 1 ? `<em class="repeat">× ${fmtNum(line.n)}</em>` : ""}</div>`).join("")
            || `<div><span>No messages yet.</span></div>`}
        </div>
      </div>`;
    const log = $("#sheet-body .log");
    if (log) log.scrollTop = log.scrollHeight;
  }

  /* ------------------------------------------------------------------ Upload */

  function renderSrcOptions() {
    $("#up-src-list").innerHTML = state.library
      .map((r) => `<option value="${esc(r.path)}">${esc(r.repo_id)}</option>`)
      .join("");
  }

  function prefillUpload(repo) {
    if (repo) {
      $("#up-src").value = repo.path;
      $("#up-repo").value = repo.repo_id;
      $$("#up-type button").forEach((b) => b.classList.toggle("is-active", b.dataset.type === repo.repo_type));
    }
    setView("upload");
    $("#up-repo").focus();
  }

  $("#up-type").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    $$("#up-type button").forEach((b) => b.classList.toggle("is-active", b === button));
  });

  $("#upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {
      repo_id: $("#up-repo").value.trim(),
      path: $("#up-src").value.trim(),
      repo_type: $("#up-type .is-active").dataset.type,
      private: $("#up-private").checked,
      create_repo: $("#up-create").checked,
      commit_message: $("#up-msg").value.trim(),
      ignore_patterns: $("#up-ignore").value.split(",").map((s) => s.trim()).filter(Boolean),
    };
    try {
      await api("/api/jobs/upload", { method: "POST", body });
      toast(`${body.repo_id} queued.`, "ok", "Upload started");
      setView("queue");
    } catch (err) { fail(err); }
  });

  /* ---------------------------------------------------------------- Settings */

  $("#settings-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {
      endpoint: $("#set-endpoint").value.trim(),
      max_concurrent: Number($("#set-concurrent").value),
      max_workers: Number($("#set-workers").value),
      auto_clear_done: $("#set-autoclear").checked,
    };
    const token = $("#set-token").value.trim();
    if (token) body.hf_token = token;
    try {
      await api("/api/settings", { method: "PUT", body });
      $("#set-token").value = "";
      await loadSettings();
      toast("Settings saved.", "ok");
    } catch (err) { fail(err); }
  });

  $("#set-token-test").addEventListener("click", async (event) => {
    const button = event.target;
    button.disabled = true;
    const status = $("#set-token-state");
    try {
      const who = await api("/api/settings/test-token", { method: "POST", body: { token: $("#set-token").value.trim() } });
      status.textContent = `Signed in as ${who.name}${who.orgs.length ? ` · orgs: ${who.orgs.join(", ")}` : ""}`;
      status.className = "hint is-ok";
    } catch (err) {
      status.textContent = err.message;
      status.className = "hint is-err";
    } finally {
      button.disabled = false;
    }
  });

  $("#btn-logout").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" }).catch(() => {});
    location.reload();
  });

  /* --------------------------------------------------------------- Navigation */

  window.addEventListener("hashchange", () => {
    const view = viewFromHash();
    if (view && view !== state.view) setView(view, true);
  });

  $("#nav").addEventListener("click", (event) => {
    const button = event.target.closest(".nav-item");
    if (button) setView(button.dataset.view);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, textarea") || event.metaKey || event.ctrlKey) return;
    const map = { 1: "library", 2: "hub", 3: "queue", 4: "upload", 5: "settings" };
    if (map[event.key]) setView(map[event.key]);
  });

  // Let relative timestamps age without asking the server again.
  setInterval(() => {
    if (state.view === "library") renderLibrary();
  }, 60000);

  boot().catch((err) => {
    console.error(err);
    // A 401 has already routed us to the gate. Anything else is a failed load,
    // and throwing the user back to a login form would misrepresent it.
    if (!$("#shell").hidden) {
      toast(err.message || String(err), "err", "Could not load");
      return;
    }
    showGate();
    const box = $("#gate-error");
    box.textContent = err.message || String(err);
    box.hidden = false;
  });
})();
