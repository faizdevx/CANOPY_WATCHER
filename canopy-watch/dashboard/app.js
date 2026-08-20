/* =========================================================================
   Canopy Watch — Ops Console
   -------------------------------------------------------------------------
   Single-page dashboard. No build step, no framework — open index.html
   directly or serve the folder statically.

   Points at the Dashboard API (see backend/dashboard.py):
     GET {API_BASE}/summary?station=&risk=&since=
     GET {API_BASE}/sightings?station=&risk=&since=&q=&limit=

   If those endpoints aren't reachable (backend not running yet, wrong
   API_BASE, CORS, etc.) this falls back to generated sample data so the
   page is never just a blank error — you can see the UI/UX immediately,
   then point API_BASE at your real backend.
   ========================================================================= */

const API_BASE = "/api/dashboard";   // <-- change to e.g. "http://localhost:8000/api/dashboard"
const POLL_MS = 15000;

let MOCK_MODE = false;
let allSightings = [];   // last full fetch, pre-filter (mock mode filters client-side)
let stationList = [];

const el = (id) => document.getElementById(id);

/* ---------------------------------------------------------------------- */
/* Clock + connection indicator                                           */
/* ---------------------------------------------------------------------- */
function tickClock() {
  el("clock").textContent = new Date().toISOString().replace("T", " ").slice(0, 19) + "Z";
}
setInterval(tickClock, 1000);
tickClock();

function setConnState(ok) {
  el("connDot").className = "dot " + (ok ? "dot--live" : "dot--offline");
  el("connLabel").textContent = ok ? "LIVE" : (MOCK_MODE ? "SAMPLE DATA (API UNREACHABLE)" : "OFFLINE");
}

/* ---------------------------------------------------------------------- */
/* Fetch helpers                                                          */
/* ---------------------------------------------------------------------- */
function currentFilters() {
  return {
    station: el("fStation").value,
    risk: el("fRisk").value,
    since: el("fTime").value,
    q: el("fSearch").value.trim(),
  };
}

function qs(params) {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v) p.set(k, v); });
  const s = p.toString();
  return s ? `?${s}` : "";
}

async function apiGet(path, params) {
  const res = await fetch(`${API_BASE}${path}${qs(params)}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

/* ---------------------------------------------------------------------- */
/* Mock data (used only if the real API can't be reached)                 */
/* ---------------------------------------------------------------------- */
const MOCK_STATIONS = [
  { id: "CAM-N01-A", sector: "SECTOR_NORTH", is_connected: true },
  { id: "CAM-N01-B", sector: "SECTOR_NORTH", is_connected: true },
  { id: "CAM-S04",   sector: "SECTOR_SOUTH", is_connected: false },
  { id: "CAM-E02",   sector: "SECTOR_EAST",  is_connected: true },
];
const MOCK_CLASSES = [
  ["Panthera tigris", 0.91, 42], ["Human (Unarmed)", 0.88, 55],
  ["Human (Armed)", 0.96, 94], ["Elephas maximus", 0.85, 20],
  ["Vehicle (4x4)", 0.84, 88], ["Unknown / Small mammal", 0.61, 12],
  ["Axis axis", 0.93, 8], ["Chainsaw sound signature", 0.77, 76],
];
function buildMockSightings(n = 60) {
  const now = Date.now();
  const rows = [];
  for (let i = 0; i < n; i++) {
    const st = MOCK_STATIONS[Math.floor(Math.random() * MOCK_STATIONS.length)];
    const [cls, conf, riskBase] = MOCK_CLASSES[Math.floor(Math.random() * MOCK_CLASSES.length)];
    const risk = Math.max(0, Math.min(100, Math.round(riskBase + (Math.random() * 16 - 8))));
    const ageMinutes = Math.floor(Math.random() * 60 * 24 * 6); // up to ~6 days back
    rows.push({
      sighting_id: `REC-${900000 - i}`,
      timestamp: new Date(now - ageMinutes * 60000).toISOString(),
      station: { id: st.id, sector: st.sector, is_connected: st.is_connected, coordinates: { lat: 29.53, lng: 78.77 } },
      inference: {
        model_version: "yolov8x-wildlife:v2.4.1",
        class_name: cls,
        confidence: conf,
        risk_score: risk,
        bounding_box: [142, 88, 512, 640],
      },
      alert: {
        status: risk >= 70 ? (Math.random() > 0.4 ? "SMS_DISPATCHED" : "ACKNOWLEDGED") : "LOGGED",
      },
    });
  }
  return rows.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
}

function riskBand(score) {
  if (score >= 70) return "high";
  if (score >= 40) return "medium";
  return "low";
}
function sinceToMs(since) {
  return { "1h": 3600e3, "24h": 86400e3, "7d": 7 * 86400e3, all: Infinity }[since] ?? Infinity;
}

function filterMock(rows, f) {
  const cutoff = Date.now() - sinceToMs(f.since);
  return rows.filter((r) => {
    if (f.station && r.station.id !== f.station) return false;
    if (f.risk && riskBand(r.inference.risk_score) !== f.risk) return false;
    if (f.q && !r.inference.class_name.toLowerCase().includes(f.q.toLowerCase())) return false;
    if (new Date(r.timestamp).getTime() < cutoff) return false;
    return true;
  });
}

/* ---------------------------------------------------------------------- */
/* Rendering                                                              */
/* ---------------------------------------------------------------------- */
function fmtTime(iso) {
  const d = new Date(iso);
  return d.toISOString().replace("T", " ").slice(0, 19) + "Z";
}
function relTime(iso) {
  const diffMs = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diffMs / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
function alertClass(status) {
  if (status === "SMS_DISPATCHED" || status === "DISPATCH_PENDING") return "alert-pill--dispatched";
  if (status === "ACKNOWLEDGED") return "alert-pill--acknowledged";
  return "";
}

function renderStats(summary) {
  el("statTotal").textContent = summary.total_detections ?? "—";
  el("statHigh").textContent = summary.high_risk_detections ?? "—";
  el("statTotalSub").textContent = summary.window_label ? `in ${summary.window_label}` : "";
  el("statHighSub").textContent = summary.high_risk_pct != null ? `${summary.high_risk_pct}% of total` : "";

  const stations = summary.stations || [];
  const online = stations.filter((s) => s.is_connected).length;
  el("statStations").textContent = stations.length ? `${online}/${stations.length}` : "—";
  el("statStationsSub").textContent = stations.length ? "online / total" : "";

  const dispatched = summary.alerts_dispatched;
  el("statAlerts").textContent = dispatched != null ? dispatched : "—";
  el("statAlertsSub").textContent = dispatched != null ? "this window" : "";
}

function populateStationFilter(stations) {
  const sel = el("fStation");
  const current = sel.value;
  sel.innerHTML = '<option value="">All stations</option>' +
    stations.map((s) => `<option value="${s.id}">${s.id}${s.is_connected === false ? " (offline)" : ""}</option>`).join("");
  sel.value = current;
}

function renderFeed(rows) {
  const body = el("feedBody");
  el("resultCount").textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;

  if (!rows.length) {
    body.innerHTML = `<tr class="feed__empty"><td colspan="7">No sightings match these filters.</td></tr>`;
    return;
  }

  body.innerHTML = rows.map((r, i) => {
    const band = riskBand(r.inference.risk_score);
    return `
      <tr class="risk-${band}" data-idx="${i}">
        <td class="mono" title="${fmtTime(r.timestamp)}">${relTime(r.timestamp)}</td>
        <td class="mono">${r.station.id}</td>
        <td class="class-cell">${r.inference.class_name}</td>
        <td class="mono">${Math.round(r.inference.confidence * 100)}%</td>
        <td><span class="badge badge--${band}">${r.inference.risk_score}</span></td>
        <td><span class="alert-pill ${alertClass(r.alert.status)}">${(r.alert.status || "LOGGED").replace("_", " ")}</span></td>
        <td class="chev">›</td>
      </tr>`;
  }).join("");

  [...body.querySelectorAll("tr[data-idx]")].forEach((tr) => {
    tr.addEventListener("click", () => openDrawer(rows[Number(tr.dataset.idx)]));
  });
}

/* ---------------------------------------------------------------------- */
/* Detail drawer                                                          */
/* ---------------------------------------------------------------------- */
function openDrawer(r) {
  el("drawerId").textContent = `SIGHTING · ${r.sighting_id}`;
  const band = riskBand(r.inference.risk_score);
  el("drawerBody").innerHTML = `
    <div class="drawer__row"><span>Timestamp</span><span class="mono">${fmtTime(r.timestamp)}</span></div>
    <div class="drawer__row"><span>Station</span><span class="mono">${r.station.id} (${r.station.sector || "—"})</span></div>
    <div class="drawer__row"><span>Connectivity</span><span>${r.station.is_connected ? "Online" : "Offline"}</span></div>
    <div class="drawer__section-label">Inference</div>
    <div class="drawer__row"><span>Class</span><span>${r.inference.class_name}</span></div>
    <div class="drawer__row"><span>Confidence</span><span class="mono">${Math.round(r.inference.confidence * 100)}%</span></div>
    <div class="drawer__row"><span>Risk score</span><span class="badge badge--${band}">${r.inference.risk_score}</span></div>
    <div class="drawer__row"><span>Model</span><span class="mono">${r.inference.model_version || "—"}</span></div>
    ${r.inference.bounding_box ? `<div class="drawer__row"><span>Bounding box</span><span class="mono">[${r.inference.bounding_box.join(", ")}]</span></div>` : ""}
    <div class="drawer__section-label">Alert</div>
    <div class="drawer__row"><span>Status</span><span class="${alertClass(r.alert.status)}">${(r.alert.status || "LOGGED").replace("_", " ")}</span></div>
    ${r.alert.assigned_patrol ? `<div class="drawer__row"><span>Assigned patrol</span><span>${r.alert.assigned_patrol}</span></div>` : ""}
    ${r.inference.image_url ? `<img src="${r.inference.image_url}" alt="Detection crop">` : ""}
  `;
  el("drawer").dataset.open = "true";
  el("drawer").setAttribute("aria-hidden", "false");
}
function closeDrawer() {
  el("drawer").dataset.open = "false";
  el("drawer").setAttribute("aria-hidden", "true");
}
el("drawerClose").addEventListener("click", closeDrawer);
el("drawerScrim").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

/* ---------------------------------------------------------------------- */
/* Main load / refresh cycle                                              */
/* ---------------------------------------------------------------------- */
async function loadAll() {
  const f = currentFilters();
  try {
    const [summary, feed] = await Promise.all([
      apiGet("/summary", { station: f.station, risk: f.risk, since: f.since }),
      apiGet("/sightings", { station: f.station, risk: f.risk, since: f.since, q: f.q, limit: 200 }),
    ]);
    MOCK_MODE = false;
    stationList = summary.stations || stationList;
    allSightings = feed.sightings || [];
    populateStationFilter(stationList);
    renderStats(summary);
    renderFeed(allSightings);
    setConnState(true);
    el("dataModeNote").textContent = "";
  } catch (err) {
    // Fall back to sample data so the page is never just an error.
    MOCK_MODE = true;
    if (!allSightings.length) allSightings = buildMockSightings();
    if (!stationList.length) stationList = MOCK_STATIONS;
    populateStationFilter(stationList);

    const filtered = filterMock(allSightings, f);
    const high = filtered.filter((r) => r.inference.risk_score >= 70).length;
    renderStats({
      total_detections: filtered.length,
      high_risk_detections: high,
      high_risk_pct: filtered.length ? Math.round((high / filtered.length) * 100) : 0,
      stations: stationList,
      alerts_dispatched: filtered.filter((r) => r.alert.status === "SMS_DISPATCHED").length,
      window_label: { "1h": "last 1h", "24h": "last 24h", "7d": "last 7d", all: "all time" }[f.since],
    });
    renderFeed(filtered);
    setConnState(false);
    el("dataModeNote").textContent =
      `Showing sample data — could not reach ${API_BASE} (${err.message}). ` +
      `Point API_BASE in dashboard/app.js at your running Ingest API to see real sightings.`;
  }
}

/* ---------------------------------------------------------------------- */
/* Wire up filter controls                                                */
/* ---------------------------------------------------------------------- */
["fStation", "fRisk", "fTime"].forEach((id) => el(id).addEventListener("change", loadAll));
let searchDebounce;
el("fSearch").addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(loadAll, 250);
});
el("fReset").addEventListener("click", () => {
  el("fStation").value = "";
  el("fRisk").value = "";
  el("fTime").value = "24h";
  el("fSearch").value = "";
  loadAll();
});

loadAll();
setInterval(loadAll, POLL_MS);