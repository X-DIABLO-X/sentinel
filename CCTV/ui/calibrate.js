const $ = s => document.querySelector(s);
let cameras = [];

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function reviewed(camera) {
  return (camera.corridors || []).length > 0 &&
    !String(camera.notes || "").toUpperCase().includes("DRAFT");
}

function arrow(c, width, height) {
  const points = c.polygon || [];
  if (!points.length) return "";
  const x = points.reduce((s, p) => s + p[0], 0) / points.length;
  const y = points.reduce((s, p) => s + p[1], 0) / points.length;
  const d = c.direction || [0, 0];
  const scale = Math.max(35, Math.min(width, height) * .08);
  return `<line x1="${x - d[0] * scale}" y1="${y - d[1] * scale}"
    x2="${x + d[0] * scale}" y2="${y + d[1] * scale}"
    class="flow-arrow" marker-end="url(#arrow)"/>`;
}

function render(camera) {
  const [width, height] = camera.frame_size || [1280, 720];
  const corridors = camera.corridors || [];
  const isReviewed = reviewed(camera);
  const polygons = corridors.map((c, i) => `
    <polygon points="${(c.polygon || []).map(p => p.join(",")).join(" ")}" class="stream stream-${i % 4}"/>
    ${arrow(c, width, height)}
    <text x="${(c.polygon?.[0] || [8, 18])[0]}" y="${(c.polygon?.[0] || [8, 18])[1]}" class="stream-label">${c.id}</text>`).join("");
  $("#camera-canvas").innerHTML = `
    <img src="/api/cameras/${encodeURIComponent(camera.camera_id)}/frame" alt="First frame from ${camera.camera_id}">
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Observed traffic stream overlay">
      <defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z"/></marker></defs>
      ${polygons}
    </svg>`;
  $("#camera-state").innerHTML = `
    <span class="${corridors.length ? "ok" : "bad"}">${corridors.length} observed traffic streams</span>
    <span class="${isReviewed ? "ok" : "warn"}">${isReviewed ? "REVIEWED legal direction" : "DRAFT — legal direction not reviewed"}</span>
    <span>${camera.frame_size?.join("×") || "unknown size"}</span>
    <span>${camera.incidents || 0} recorded incidents</span>`;
  $("#camera-meta").innerHTML = `
    <dl class="kv">
      <dt>Camera ID</dt><dd>${camera.camera_id}</dd>
      <dt>Zone</dt><dd>${camera.zone || "—"}</dd>
      <dt>Road</dt><dd>${camera.road_name || "—"}</dd>
      <dt>Analysis rate</dt><dd>${camera.analysis_fps || "—"} fps</dd>
      <dt>Calibration</dt><dd>${isReviewed ? "reviewed" : "draft"}</dd>
    </dl>
    <p class="note">${camera.notes || "No calibration notes recorded."}</p>`;
  $("#corridors").innerHTML = corridors.map(c => {
    const deg = Math.atan2(c.direction?.[1] || 0, c.direction?.[0] || 0) * 180 / Math.PI;
    const support = String(c.name || "").match(/\((\d+) tracks\)/)?.[1] || "—";
    return `<tr><td>${c.id}</td><td>${deg.toFixed(0)}°</td><td>${support} tracks</td><td>${isReviewed ? "legal" : "observed"}</td></tr>`;
  }).join("") || `<tr><td colspan="4">No usable motion corridor learned.</td></tr>`;
}

async function boot() {
  try {
    cameras = await api("/api/cameras");
    $("#camera-select").innerHTML = cameras.map((c, i) =>
      `<option value="${i}">${c.name || c.camera_id} · ${(c.corridors || []).length} streams</option>`).join("");
    if (!cameras.length) throw new Error("No camera configurations found");
    render(cameras[0]);
    $("#camera-select").onchange = e => render(cameras[Number(e.target.value)]);
  } catch (e) {
    $("#camera-state").innerHTML = `<span class="bad">Cannot load cameras: ${e.message}</span>`;
  }
}

boot();
