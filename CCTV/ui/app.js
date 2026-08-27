/* NETRA dashboard.
 *
 * The organising idea: an operator should be able to answer four questions in
 * about five seconds -- what happened, why does the system think so, how bad is
 * it, and what do I do now. Everything on this screen serves one of those, and
 * anything that served none of them was left out.
 *
 * No framework and no build step. That is a reproducibility decision: a
 * reviewer clones the repo, runs one command, and the UI works.
 */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

const STATE = {
  incidents: [], selected: null, cameras: [], health: null,
  typeFilter: "traffic", incidentSignature: "", cameraSignature: "",
  problemVideos: [], uploadJobs: [], activeJob: null, jobTimer: null,
  uploadPreviewUrl: null,
};

const WORKFLOW = ["detected", "verified", "assigned", "responding", "resolved", "closed"];

const eventLabel = (event) => event.event_type === "collision_candidate"
  ? "Accident candidate" : (event.label || event.event_type || "Incident");

const fmt = {
  t: (s) => (s === null || s === undefined) ? "—" : `${Number(s).toFixed(1)}s`,
  pct: (v) => (v === null || v === undefined) ? "—" : `${(Number(v) * 100).toFixed(0)}%`,
  num: (v, d = 2) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d),
  clock: (ms) => new Date(ms * 1000).toLocaleTimeString([], { hour12: false }),
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.text();
}

function toast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

/* ---------------- header ---------------- */

function tickClock() {
  const el = document.getElementById("clock");
  if (!el) return;
  const d = new Date();
  el.textContent = d.toLocaleDateString("en-IN", {
    day: "2-digit", month: "short", year: "numeric" }) + "  " +
    d.toLocaleTimeString("en-IN", { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

function renderKpis(s) {
  const clips = STATE.problemVideos.length;
  const accidents = STATE.problemVideos.filter(v => Number(v.events_total || 0) > 0).length;
  const wait = s.awaiting_verification ?? 0;
  $("#kpis").innerHTML = `
    <span class="seg"><b>${s.open_incidents ?? 0}</b> active</span>
    <span class="seg accident"><b>${accidents}/${clips}</b> accident candidates</span>
    <span class="seg"><b>${wait}</b> need review</span>`;
  $("#tab-accidents").textContent = clips;
}

/* ---------------- uploaded video analysis ---------------- */

function showJob(job) {
  STATE.uploadJobs = [job, ...STATE.uploadJobs.filter(j => j.id !== job.id)];
  if (STATE.typeFilter === "upload") renderUploadJobs();
  const tray = $("#job-tray");
  tray.hidden = false;
  tray.classList.toggle("failed", job.status === "failed");
  tray.classList.toggle("complete", job.status === "complete");
  $("#job-title").textContent = job.filename || "Video analysis";
  $("#job-percent").textContent = `${Math.round(job.percent || 0)}%`;
  $("#job-progress-bar").style.width = `${Math.max(0, Math.min(100, job.percent || 0))}%`;
  $("#job-message").textContent = job.error || job.message || job.phase || "Working…";
}

function showJobResult(job) {
  if (STATE.uploadPreviewUrl) {
    URL.revokeObjectURL(STATE.uploadPreviewUrl);
    STATE.uploadPreviewUrl = null;
  }
  const result = job.result || {};
  const types = Object.entries(result.events_by_type || {})
    .map(([type, count]) => `${count} ${type.replace(/_/g, " ")}`).join(", ") || "No incidents";
  $("#detail").innerHTML = `
    <div class="d-head">
      <h2>Video analysis complete</h2>
      <div class="meta">${job.filename} · ${result.wall_seconds || "—"}s processing time</div>
      <div class="chips"><span class="pill status">${result.events_total || 0} incidents</span></div>
    </div>
    <div class="section"><h3>Annotated output</h3>
      ${job.annotated_video ? `<video src="/api/jobs/${job.id}/video" poster="/api/jobs/${job.id}/poster"
        controls muted autoplay playsinline preload="auto" style="width:100%;max-height:62vh;background:#111"></video>
        <div class="note video-state">Loading annotated video…</div>` :
        `<div class="note">The analysis completed but the annotated renderer did not produce a video.</div>`}
      <p><strong>Detected:</strong> ${types}</p>
      <p class="note">The incident feed has been refreshed. Select an incident to inspect its physics, trajectories and evidence.</p>
    </div>`;
  activateVideo();
}

function showUploadPreview(file) {
  if (STATE.uploadPreviewUrl) URL.revokeObjectURL(STATE.uploadPreviewUrl);
  STATE.uploadPreviewUrl = URL.createObjectURL(file);
  $("#detail").innerHTML = `
    <div class="d-head">
      <h2>Analysing uploaded video</h2>
      <div class="meta"></div>
      <div class="chips"><span class="pill status">model running</span></div>
    </div>
    <div class="section"><h3>Uploaded video</h3>
      <video controls muted autoplay playsinline preload="metadata"
        style="width:100%;max-height:62vh;background:#111"></video>
      <div class="note video-state">Opening local preview…</div>
      <p class="note">The complete detector, tracker, trajectory and event pipeline is running. This preview will be replaced by the annotated result.</p>
    </div>`;
  $("#detail .meta").textContent = file.name;
  $("#detail video").src = STATE.uploadPreviewUrl;
  activateVideo();
}

function activateVideo() {
  const video = $("#detail video"), state = $("#detail .video-state");
  if (!video) return;
  const ready = () => { if (state) state.textContent = "Ready"; };
  video.addEventListener("loadeddata", ready, {once:true});
  video.addEventListener("canplay", ready, {once:true});
  video.addEventListener("error", () => {
    if (state) state.textContent = "Video could not be decoded. Reload once or download the generated WebM.";
  }, {once:true});
  video.play().catch(() => {});
}

async function pollJob(id) {
  clearTimeout(STATE.jobTimer);
  try {
    const job = await api(`/api/jobs/${id}`);
    STATE.activeJob = job;
    showJob(job);
    if (job.status === "complete") {
      await refresh();
      showJobResult(job);
      toast("Video analysis complete");
      return;
    }
    if (job.status === "failed") return;
    STATE.jobTimer = setTimeout(() => pollJob(id), 1000);
  } catch (e) {
    $("#job-message").textContent = `Status unavailable: ${e.message}`;
    STATE.jobTimer = setTimeout(() => pollJob(id), 2500);
  }
}

function uploadVideo(file) {
  if (!file) return;
  showUploadPreview(file);
  const job = { filename: file.name, status: "uploading", percent: 0,
                message: "Uploading video" };
  showJob(job);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
  xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) {
      job.percent = Math.round(100 * e.loaded / e.total);
      job.message = `Uploading ${Math.round(e.loaded / 1048576)} of ${Math.round(e.total / 1048576)} MB`;
      showJob(job);
    }
  };
  xhr.onerror = () => showJob({...job, status: "failed", error: "Upload connection failed"});
  xhr.onload = () => {
    let response;
    try { response = JSON.parse(xhr.responseText); } catch (_) { response = null; }
    if (xhr.status < 200 || xhr.status >= 300 || !response?.id) {
      showJob({...job, status: "failed", error: response?.detail || `Upload failed (${xhr.status})`});
      return;
    }
    STATE.activeJob = response;
    pollJob(response.id);
  };
  xhr.send(file);
}

/* ---------------- list ---------------- */

function renderList() {
  const cam = $("#f-camera").value, st = $("#f-status").value;
  const ty = STATE.typeFilter;
  if (ty === "problem_set") return renderProblemVideos();
  if (ty === "upload") return renderUploadJobs();
  const rows = STATE.incidents.filter(i =>
    i.source_kind === "traffic" &&
    (!cam || i.camera_id === cam) && (!st || i.status === st) &&
    (ty === "traffic" || i.event_type === ty ||
      (ty === "blockage" && i.event_type === "abnormal_stop")));

  $("#count").textContent = rows.length;

  if (!rows.length) {
    $("#list").innerHTML = `<div class="empty"><p>No incidents match this filter.</p></div>`;
    return;
  }

  $("#list").innerHTML = rows.map(i => `
    <div class="inc sev-${i.severity_label} ${STATE.selected === i.id ? "sel" : ""}" data-id="${i.id}">
      <div class="r1">
        <span class="ttl">${eventLabel(i)}</span>
        <span class="pill ${i.severity_label}">${i.severity_label}</span>
      </div>
      <div class="meta">
        INC-${String(i.id).padStart(5, "0")} · ${i.camera_id}${i.corridor_id ? " · " + i.corridor_id : ""}
        · onset ${fmt.t(i.started_t)} · conf ${fmt.num(i.confidence)}
      </div>
      <div class="why">${i.explanation || ""}</div>
      <div class="meta" style="margin-top:5px">
        <span class="pill status">${i.status}</span>
        ${i.needs_verification && i.status === "detected"
          ? '<span class="pill verify">needs verification</span>' : ""}
      </div>
    </div>`).join("");

  $$(".inc").forEach(el => el.onclick = () => select(Number(el.dataset.id)));
}

function renderProblemVideos() {
  $("#count").textContent = STATE.problemVideos.length;
  $("#list").innerHTML = STATE.problemVideos.map(row => {
    const event = (row.events || [])[0];
    const stem = row.file.replace(/\.[^.]+$/, "");
    const duration = Number(row.video?.duration || 0);
    return `<div class="inc ${event ? `sev-${event.severity_label}` : ""}
                 ${STATE.selected === `problem:${stem}` ? "sel" : ""}"
                 data-problem="${encodeURIComponent(stem)}">
      <div class="r1"><span class="ttl">${row.file}</span>
        <span class="pill ${event?.severity_label || "status"}">${event ? "candidate" : "no candidate"}</span></div>
      <div class="meta">ProblemSet accident video · ${duration ? duration.toFixed(1) : "—"}s</div>
      <div class="why">${event ? `${event.explanation} · onset ${fmt.t(event.started_t)}` :
        "No accident candidate generated; retained for honest 16/16 review."}</div>
    </div>`;
  }).join("");
  $$("[data-problem]").forEach(el => el.onclick = () =>
    selectProblem(decodeURIComponent(el.dataset.problem)));
}

function selectProblem(stem) {
  const row = STATE.problemVideos.find(v => v.file.replace(/\.[^.]+$/, "") === stem);
  if (!row) return;
  STATE.selected = `problem:${stem}`;
  renderProblemVideos();
  const event = (row.events || [])[0];
  $("#detail").innerHTML = `<div class="d-head">
      <h2>${row.file}</h2>
      <div class="meta">ProblemSet · fixed 16-video accident evaluation set</div>
      <div class="chips"><span class="pill ${event?.severity_label || "status"}">
        ${event ? "accident candidate" : "no candidate detected"}</span></div>
    </div>
    <div class="section"><h3>Annotated video</h3>
      <video src="/api/problem-videos/${encodeURIComponent(stem)}/video"
             poster="/api/problem-videos/${encodeURIComponent(stem)}/poster"
             controls muted autoplay playsinline preload="auto"
             style="width:100%;max-height:65vh;background:#111"></video>
      <div class="note video-state">Loading annotated video…</div>
      ${event ? `<p><strong>Onset:</strong> ${fmt.t(event.started_t)} · <strong>Confidence:</strong> ${fmt.num(event.confidence)}</p>
        <p>${event.explanation || ""}</p>` :
        `<div class="assurance"><strong>Known miss.</strong> The detector generated no accident candidate for this clip. The video remains visible instead of being silently omitted.</div>`}
    </div>`;
  activateVideo();
}

function renderUploadJobs() {
  $("#count").textContent = STATE.uploadJobs.length;
  if (!STATE.uploadJobs.length) {
    $("#list").innerHTML = `<div class="empty"><p>No uploaded videos yet.</p>
      <button class="btn primary" id="empty-upload">Analyse a video</button></div>`;
    $("#empty-upload").onclick = () => $("#video-input").click();
    return;
  }
  $("#list").innerHTML = STATE.uploadJobs.map(job => `<div class="inc" data-job="${job.id}">
    <div class="r1"><span class="ttl">${job.filename}</span><span class="pill status">${job.status}</span></div>
    <div class="meta">${job.phase} · ${Math.round(job.percent || 0)}%</div>
    <div class="why">${job.error || job.message || ""}</div>
  </div>`).join("");
  $$("[data-job]").forEach(el => el.onclick = () => {
    const job = STATE.uploadJobs.find(j => j.id === el.dataset.job);
    if (job?.status === "complete") showJobResult(job); else if (job) showJob(job);
  });
}

/* ---------------- detail ---------------- */

function bar(label, value) {
  const pct = Math.max(0, Math.min(100, (Number(value) || 0) * 100));
  return `<div class="bar">
    <span class="lbl">${label}</span>
    <span class="track"><span class="fill" style="width:${pct}%"></span></span>
    <span class="num">${fmt.num(value)}</span>
  </div>`;
}

function kvTable(obj) {
  const rows = Object.entries(obj || {})
    .filter(([k]) => !k.startsWith("_"))
    .map(([k, v]) => {
      let val = v;
      if (v && typeof v === "object") val = JSON.stringify(v);
      else if (typeof v === "number") val = Number.isInteger(v) ? v : v.toFixed(3);
      return `<tr><td>${k.replace(/_/g, " ")}</td><td>${val}</td></tr>`;
    }).join("");
  return `<table class="kv">${rows}</table>`;
}

async function select(id) {
  STATE.selected = id;
  renderList();
  const d = await api(`/api/incidents/${id}`);
  renderDetail(d);
}

function renderDetail(d) {
  const ev = d.evidence || {};
  const loc = d.location || {};
  const sev = d.severity_parts || {};
  const stepIdx = WORKFLOW.indexOf(d.status);
  const isCollision = String(d.event_type || "").includes("collision");
  const channels = Number((d.triggers || {}).channels_agreeing || 0);
  const participants = (d.track_ids || []).length;
  const camera = STATE.cameras.find(c => c.camera_id === d.camera_id);
  const fixedCamera = Boolean(camera && (camera.corridors || []).length);
  const directionUnreviewed = d.event_type === "wrong_way" &&
    d.triggers?.legal_direction_reviewed === false;
  const context = d.operational_context || {};
  const hasMappedRoad = Boolean(loc.road_edge_id);

  const media = [];
  if (ev.annotated_frame)
    media.push(`<figure><img src="/api/incidents/${d.id}/evidence/${ev.annotated_frame}" alt="annotated evidence frame">
      <figcaption>Annotated evidence — implicated tracks and motion</figcaption></figure>`);
  if (ev.clip)
    media.push(`<figure><video src="/api/incidents/${d.id}/evidence/${ev.clip}" controls muted loop></video>
      <figcaption>Evidence clip — spans the recovered onset${
        ev.clip_span_s ? ` (${ev.clip_span_s[0]}s → ${ev.clip_span_s[1]}s)` : ""}</figcaption></figure>`);

  $("#detail").innerHTML = `
    <div class="d-head">
      <h2>${eventLabel(d)}</h2>
      <div class="meta" style="color:var(--ink-3);font-family:var(--mono);font-size:12px">
        INC-${String(d.id).padStart(5, "0")} · ${d.camera_id}${d.corridor_id ? " · " + d.corridor_id : ""}
      </div>
      <div class="chips">
        <span class="pill ${d.severity_label}">severity ${d.severity_label} · ${fmt.num(d.severity)}</span>
        <span class="pill status">confidence ${fmt.num(d.confidence)}</span>
        <span class="pill status">${d.status}</span>
        ${d.needs_verification ? '<span class="pill verify">human verification required</span>' : ""}
      </div>
      ${isCollision ? `<div class="assurance">
        <strong>Suspected collision — not automatically confirmed.</strong>
        ${channels || "No"} independent motion channel${channels === 1 ? "" : "s"} agreed;
        ${participants ? `${participants} implicated track${participants === 1 ? "" : "s"}.` : "participant location is withheld."}
        ${fixedCamera
          ? "This fixed camera requires at least two independent channels before promotion."
          : "This clip has no reviewed fixed-camera geometry, so the alert remains a human-verification candidate."}
      </div>` : ""}
      ${directionUnreviewed ? `<div class="assurance">
        <strong>Wrong-side candidate — legal direction is not yet reviewed.</strong>
        This alert compares motion with observed majority flow. Confirm the legal
        carriageway direction in Camera &amp; lanes before enforcement action.
      </div>` : ""}
    </div>

    <div class="section">
      <h3>Why this fired</h3>
      <p style="margin:0 0 12px">${d.explanation || ""}</p>
      <div class="grid">
        <div class="f"><div class="k">Onset</div><div class="v">${fmt.t(d.started_t)}</div></div>
        <div class="f"><div class="k">Detected</div><div class="v">${fmt.t(d.detected_t)}</div></div>
        <div class="f"><div class="k">Detection delay</div><div class="v">${fmt.t(d.detection_delay)}</div></div>
        <div class="f"><div class="k">Duration</div><div class="v">${fmt.t(d.duration)}</div></div>
        <div class="f"><div class="k">Onset method</div><div class="v">${d.onset_method || "detection"}</div></div>
        <div class="f"><div class="k">Onset recovered</div><div class="v">${fmt.t(d.onset_recovered_s)}</div></div>
      </div>
    </div>

    ${media.length ? `<div class="section"><h3>Visual evidence</h3>
      <div class="media">${media.join("")}</div></div>` : ""}

    <details class="disclosure">
      <summary>Technical evidence</summary>
      <div class="section">
      <h3>Severity breakdown</h3>
      <div class="bars">
        ${bar("Flow loss", sev.flow_loss)}
        ${bar("Obstruction", sev.obstruction)}
        ${bar("Extent", sev.extent)}
        ${bar("Duration", sev.duration)}
        ${bar("Risk exposure", sev.risk)}
        ${bar("Impact subscore", sev.impact_subscore)}
      </div>
      <div class="note" style="margin-top:12px">
        <strong>Traffic-impact severity, not injury severity.</strong>
        Computed from observable variables only. Confidence is reported
        separately and is deliberately never folded into this score.
      </div>
    </div>

    <div class="section">
      <h3>Trigger values</h3>
      ${kvTable(d.triggers)}
    </div>
    </details>

    <div class="section">
      <h3>Location</h3>
      <div class="grid">
        <div class="f"><div class="k">Camera</div><div class="v">${loc.camera_id || d.camera_id}</div></div>
        <div class="f"><div class="k">Zone</div><div class="v">${loc.zone || "—"}</div></div>
        <div class="f"><div class="k">Road</div><div class="v">${loc.road_name || "—"}</div></div>
        <div class="f"><div class="k">Coordinates</div><div class="v">${
          loc.latitude != null ? `${Number(loc.latitude).toFixed(4)}, ${Number(loc.longitude).toFixed(4)}` : "—"}</div></div>
        <div class="f"><div class="k">Precision</div><div class="v">${loc.precision || "—"}</div></div>
        <div class="f"><div class="k">Road edge</div><div class="v">${loc.road_edge_id || "—"}</div></div>
      </div>
      ${loc.note ? `<div class="note" style="margin-top:12px">${loc.note}</div>` : ""}
    </div>

    <div class="section">
      <h3>Response</h3>
      <p style="margin:0 0 14px"><strong>Recommended:</strong> ${d.recommended_action || "Operator review"}</p>
      ${context.classification ? `<div class="note" style="margin:0 0 14px">
        <strong>Operational context:</strong> ${context.classification.replace(/_/g, " ")}.
        ${context.causality ? `Cause status: ${context.causality}.` : ""}
        ${context.operator_note || ""}
      </div>` : ""}
      <div class="flow">
        ${WORKFLOW.map((s, i) => `<span class="step ${
          i < stepIdx ? "done" : i === stepIdx ? "now" : ""}">${s}</span>`).join("")}
      </div>
      <div class="actions" style="margin-top:14px">
        <button class="btn primary" data-act="verify">Verify</button>
        <button class="btn" data-act="assign">Assign</button>
        <button class="btn" data-act="responding">Responding</button>
        <button class="btn" data-act="resolved">Resolved</button>
        <button class="btn" data-act="closed">Close</button>
        <button class="btn danger" data-act="reject">Reject…</button>
      </div>
      <div class="actions" style="margin-top:10px">
        <button class="btn" data-act="close_road" ${hasMappedRoad ? "" : "disabled"}>Confirm carriageway closed → divert</button>
        <button class="btn" data-act="route" ${hasMappedRoad ? "" : "disabled"}>Show diversion</button>
      </div>
      ${hasMappedRoad ? "" : `<div class="note" style="margin-top:10px">
        Diversion unavailable for this camera: no reviewed road-graph edge is configured.
        Response and escalation workflow remain available.
      </div>`}
      <div id="routeout" style="margin-top:12px"></div>
    </div>

    <details class="disclosure">
      <summary>Audit trail</summary>
      <table class="kv">
        ${(d.history || []).map(h => `<tr>
          <td>${fmt.clock(h.changed_at)}</td>
          <td>${h.old_status || "—"} → <strong>${h.new_status}</strong>
              ${h.actor ? ` · ${h.actor}` : ""}${h.reason ? ` · ${h.reason}` : ""}</td>
        </tr>`).join("") || "<tr><td>—</td><td>no changes yet</td></tr>"}
      </table>
    </details>`;

  $$("#detail .btn").forEach(b => b.onclick = () => action(d, b.dataset.act));
}

/* ---------------- operator actions ---------------- */

async function action(d, act) {
  try {
    if (act === "verify") {
      await api(`/api/incidents/${d.id}/verify`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ status: "verified", actor: "operator" }) });
      toast("Verified");
    } else if (act === "assign") {
      const owner = prompt("Assign to:", "traffic-response-1");
      if (!owner) return;
      await api(`/api/incidents/${d.id}/assign`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ owner, team: "field" }) });
      toast(`Assigned to ${owner}`);
    } else if (act === "reject") {
      const reasons = STATE.health?.rejection_reasons || ["other"];
      const reason = prompt(`Reject — reason?\n${reasons.join(", ")}`, reasons[0]);
      if (!reason) return;
      await api(`/api/incidents/${d.id}/reject`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ reason, actor: "operator" }) });
      toast("Rejected — recorded as labelled data");
    } else if (act === "close_road") {
      const r = await api(`/api/incidents/${d.id}/close_road`, { method: "POST" });
      toast(`Edge ${r.closed_edge} closed`);
    } else if (act === "route") {
      return showRoute(d);
    } else {
      await api(`/api/incidents/${d.id}/status`, {
        method: "PATCH", headers: { "content-type": "application/json" },
        body: JSON.stringify({ status: act, actor: "operator" }) });
      toast(`Status → ${act}`);
    }
    await refresh();
    await select(d.id);
  } catch (e) {
    toast(`Failed: ${e.message.slice(0, 90)}`);
  }
}

async function showRoute(d) {
  const out = $("#routeout");
  try {
    const graph = await api("/api/graph");
    if (!graph.features?.length) { out.innerHTML = `<div class="note">No road graph configured.</div>`; return; }
    const src = prompt("Route from node:", "N1");
    const dst = prompt("Route to node:", "N5");
    if (!src || !dst) return;
    const r = await api(`/api/route?source=${encodeURIComponent(src)}&target=${encodeURIComponent(dst)}`);
    if (!r.current) { out.innerHTML = `<div class="note">${r.note || r.error || "no route"}</div>`; return; }
    out.innerHTML = `
      <div class="grid">
        <div class="f"><div class="k">Baseline route</div><div class="v">${r.baseline.edge_ids.join(" → ")}</div></div>
        <div class="f"><div class="k">Incident-aware route</div><div class="v">${r.current.edge_ids.join(" → ")}</div></div>
        <div class="f"><div class="k">Diverted</div><div class="v">${r.diverted ? "yes" : "no"}</div></div>
        <div class="f"><div class="k">Extra time</div><div class="v">${fmt.num(r.extra_seconds, 1)}s</div></div>
        <div class="f"><div class="k">Extra distance</div><div class="v">${fmt.num(r.extra_metres, 0)}m</div></div>
      </div>
      <div class="note" style="margin-top:10px">
        Incident-aware routing: the affected edge's cost is raised in proportion to
        severity. This is <strong>not</strong> live traffic-aware routing — no live
        network speeds are available.
      </div>`;
  } catch (e) {
    out.innerHTML = `<div class="note">${e.message.slice(0, 120)}</div>`;
  }
}

/* ---------------- boot ---------------- */

async function refresh() {
  try {
    const snapshot = await api("/api/dashboard");
    const {summary, incidents, cameras} = snapshot;
    renderKpis(summary);
    updateCameraOptions(cameras);
    const signature = incidents.map(i => `${i.id}:${i.status}:${i.updated_at}`).join("|");
    if (signature !== STATE.incidentSignature) {
      const list = $("#list");
      const scrollTop = list.scrollTop;
      STATE.incidents = incidents;
      STATE.incidentSignature = signature;
      renderList();
      list.scrollTop = scrollTop;
    }
  } catch (e) {
    toast(`Refresh failed: ${e.message.slice(0, 70)}`);
  }
}

function updateCameraOptions(cameras) {
  const signature = cameras.map(c => `${c.camera_id}:${c.incidents}`).join("|");
  if (signature === STATE.cameraSignature) return;
  STATE.cameraSignature = signature;
  STATE.cameras = cameras;
  const select = $("#f-camera"), cur = select.value;
  select.innerHTML = `<option value="">All cameras</option>` +
    STATE.cameras.filter(c => Number(c.incidents || 0) > 0 &&
        !/^(PROBLEMSET_|ACCIDENTS_|UPLOAD_)/.test(c.camera_id))
      .map(c => `<option value="${c.camera_id}" ${c.camera_id === cur ? "selected" : ""}>${c.name || c.camera_id} · ${c.incidents} incidents</option>`).join("");
}

async function boot() {
  try {
    const [health, snapshot, problemVideos, uploadJobs] = await Promise.all([
      api("/api/health"), api("/api/dashboard"),
      api("/api/problem-videos"), api("/api/jobs"),
    ]);
    const {cameras, summary, incidents} = snapshot;
    STATE.health = health;
    STATE.problemVideos = problemVideos;
    STATE.uploadJobs = uploadJobs;
    STATE.cameras = cameras;
    STATE.incidents = incidents;
    STATE.incidentSignature = incidents.map(i => `${i.id}:${i.status}:${i.updated_at}`).join("|");
    updateCameraOptions(cameras);
    renderKpis(summary);
    renderList();
  } catch (e) {
    $("#detail").innerHTML = `<div class="empty"><p>Cannot reach the API.</p>
      <p style="font-family:var(--mono);font-size:12px">${e.message}</p></div>`;
  }
  ["f-camera", "f-status"].forEach(id => $("#" + id).onchange = renderList);
  $$(".event-tab").forEach(button => button.onclick = () => {
    STATE.typeFilter = button.dataset.type || "";
    $$(".event-tab").forEach(x => x.classList.toggle("on", x === button));
    $(".filters").hidden = ["problem_set", "upload"].includes(STATE.typeFilter);
    renderList();
  });
  $("#upload-trigger").onclick = () => $("#video-input").click();
  $("#video-input").onchange = e => {
    uploadVideo(e.target.files?.[0]);
    e.target.value = "";
  };
  $("#job-close").onclick = () => { $("#job-tray").hidden = true; };
  setInterval(refresh, 10000);
}

boot();
