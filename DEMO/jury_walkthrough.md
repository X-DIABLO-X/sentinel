# Jury walkthrough — drive the system yourself

Written so a jury member can run this **unaided**, on a laptop with no GPU, from
a cold start. Every command is copy-pasteable. Every step states what you should
see, so you can tell a working step from a broken one.

Paths below assume the repository root is `FINAL/`. Shell is **Windows
PowerShell**; Bash equivalents are noted where they differ.

---

## 0. Status of each component before you start

Read this first so nothing surprises you mid-demo.

| Component | Port | State today | Notes |
|---|---|---|---|
| CCTV backend (`CCTV/run.py serve`) | **8000** | **Working** — verified `import netra`, 82 camera configs, FastAPI app builds | This is the part to lean on |
| DRONE backend | **8011** | **NO SERVER YET** — `FINAL/DRONE/` has real config and standalone scripts (`gmc.py`, `hover_mode.py`, `telemetry_ingest.py`, `thermal_presence.py`), but no FastAPI app binds port 8011 as of this writing | Section 4 states the contract it must satisfy and the honest fallback |
| APP (Next.js console) | **3000** | **IN PROGRESS, UNVERIFIED** — `FINAL/APP/` has a real `package.json` and route scaffolding (`app/cctv`, `app/drone`, `app/incidents`, `app/map`, `app/upload`, `app/calibrate`) as of this writing, but this doc has not confirmed `npm run dev` actually serves a working page — verify live before relying on it | Use the built-in CCTV dashboard at `:8000` instead (section 3) if APP isn't confirmed working on the day |
| Built-in CCTV dashboard | 8000 | **Working** — served by the CCTV backend itself at `/` | Full incident feed, evidence, severity, workflow |
| Incident database | — | **Empty** — `CCTV/netra.db` has 0 rows in `incidents` | You must either process a video (section 2) or load the seed snapshot (`DEMO/seed_data/README.md`) |
| Road graph | — | **Absent** — `CCTV/config/road_graph.json` does not exist | Diversion/route buttons will be disabled. See section 6 |

Nothing in this table is a surprise on the day if you read it before the day.

---

## 1. Start the CCTV backend (port 8000)

### 1.1 One-time environment setup

```powershell
cd D:\HARSHIT\ELCIA\FINAL\CCTV
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bash: `source .venv/Scripts/activate`.

`requirements.txt` pins CUDA torch (`torch==2.7.1+cu126`). On a CPU-only judging
laptop install CPU torch instead, then the rest:

```powershell
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Verify the environment before anything else:**

```powershell
python -c "import netra; print('netra ok')"
python -c "import fastapi, uvicorn, cv2, ultralytics; print('deps ok')"
```

Expected: `netra ok` then `deps ok`. Anything else — stop, fix, do not continue.

### 1.2 Pin the detector weights (do this; it is the most common demo failure)

`config/config.yaml` has `detector.weights: null`. With `null`, the detector asks
Ultralytics to resolve `yolo26n.pt` **by name**, which looks in the current
working directory and then **downloads from the internet**. The checkpoints in
this repo live in `CCTV/models/`, not in `CCTV/`, so a null weight on an offline
machine fails.

Fix it once, before the demo:

```powershell
# from CCTV/
python - <<'PY'
from pathlib import Path
p = Path("config/config.yaml"); t = p.read_text(encoding="utf-8")
t = t.replace("  weights: null", "  weights: models/yolo26m.pt", 1)
p.write_text(t, encoding="utf-8")
print("detector.weights pinned to models/yolo26m.pt")
PY
```

The same trap applies to `run.py --low-resource`, which hardcodes
`<CCTV>/yolo26n.pt` — a path that does not exist. Either copy the checkpoints to
the repo root or avoid `--low-resource` in front of judges:

```powershell
Copy-Item models\yolo26n.pt .\yolo26n.pt
```

### 1.3 Confirm what is configured

```powershell
python run.py status
```

Expected: a list of 82 cameras with corridor and zone counts, followed by an
incident summary. On a fresh clone the incident summary is empty — that is
correct, not a fault.

### 1.4 Start the server

```powershell
python run.py serve --port 8000
```

Expected console line: `NETRA dashboard -> http://127.0.0.1:8000/`

Leave this window running. Open a **second** terminal for everything below.

### 1.5 Health check

Browser: **http://127.0.0.1:8000/api/health**

Expected JSON containing `"status": "ok"`, `"cameras": 82`, a
`severity_model` block with `impact_weights` `{flow 0.40, obstruction 0.35,
extent 0.25}`, `total_weights` `{impact 0.60, duration 0.25, risk 0.15}`, the
Low/Medium/High bands, the six workflow `statuses`, and a `severity_disclaimer`.

`"road_graph_edges": 0` is expected today. See section 6.

---

## 2. Put incidents in the database

Two routes. **Route A is the safe one for a live jury session.**

### Route A — load the pre-computed snapshot (no GPU, seconds)

Follow `DEMO/seed_data/README.md`. It restores a `netra.db` and an `evidence/`
tree produced by a real pipeline run, so the dashboard is fully populated
without running inference. The UI must show its **REPLAY** badge in this mode;
if you cannot see the badge, say out loud that this is replayed data.

### Route B — process a video live (slow, honest, GPU strongly preferred)

The camera JSONs carry `source` paths from the machine they were calibrated on
(`data\problems\Traffic\...`, `ProblemSet\...`), and **those video files are not
in this repository**. So you must override the source with `--video`:

```powershell
python run.py process --camera TRAFFIC_13105330_3840_2160_30fps `
                      --video <path-to-that-clip>.mp4 `
                      --seconds 60
```

Expected: a live progress line
`t=… frames_analysed=… tracks=… events=…`, then a summary block reporting
frames analysed, wall seconds, analysis FPS, **realtime factor**, detector p95
latency in ms, incident count, alerts per video-hour, and a per-event-type
breakdown. It writes a JSON report under `reports/`.

Quote the realtime factor from *that* printout — it is measured on the machine in
front of you. Do not quote a remembered number.

CPU fallback (much slower, still real):

```powershell
python run.py process --camera <ID> --video <clip>.mp4 --seconds 30 --cpu --fps 4 --imgsz 640
```

### Route C — drag a video into the UI

The dashboard has an `Analyse video` button (top nav) that POSTs to
`/api/jobs`. A progress tray appears bottom-right with a percentage and phase.
When it completes, the clip shows up under the **Uploads** tab with an annotated
video. Use this if a judge hands you a file. It runs real inference, so on a CPU
laptop expect minutes, not seconds — say so before you click.

---

## 3. Drive the operator console

Open **http://127.0.0.1:8000/**

The layout: KPI strip across the top; incident list on the left with type tabs;
detail panel on the right.

### 3.1 Queue

1. Click the **`Queues`** tab.
2. Click any incident row. The detail panel loads.
3. Read the header chips: `severity <band> · <value>`, `confidence <value>`,
   `<status>`, and `human verification required` where applicable.
4. Read **`Why this fired`** — plain-English explanation plus Onset, Detected,
   Detection delay, Duration, Onset method, Onset recovered.
5. Expand **`Technical evidence`** → **`Trigger values`**. Expected: the actual
   measured numbers that fired the rule (vehicle count, stopped fraction, speed
   loss, occupancy, CUSUM statistic).
6. Under **`Visual evidence`**: an annotated frame and a `clip.webm` you can
   scrub. The caption states the clip's time span.

**What to say:** queue length is in **vehicles**, not metres — this camera has no
homography, and image-plane pixels are not a distance.

### 3.2 Wrong-side movement

1. Click the **`Wrong-side`** tab, open an incident.
2. Expected: an amber panel reading *"Wrong-side candidate — legal direction is
   not yet reviewed."*
3. In `Trigger values`, expected keys `legal_direction_reviewed: false` and
   `direction_source: observed majority flow; legal direction unreviewed`.
4. Open **http://127.0.0.1:8000/ui/calibrate.html** — "Camera & lane geometry".
   This is where an operator confirms the legal direction per corridor; a
   confirmation POSTs to `/api/cameras/{camera_id}/calibration`.

**What to say:** the system learned where vehicles *do* go, not where they are
*allowed* to go. Until a human confirms, it stays a candidate and must not drive
enforcement.

### 3.3 Blockage

Click the **`Blockage`** tab. **Expect it to be empty.** The blockage head is
implemented and unit-tested but produced **0 findings across the 16 ELCIA
traffic clips**, because none of those clips contains a blockage. Zero
detections on an unlabelled set establishes nothing about recall. Say that
rather than hiding the tab.

### 3.4 Collision candidate

1. Click the **`Accidents`** tab (labelled from `PROBLEMSET_*` cameras) or find a
   `collision_candidate` under **`Traffic`**.
2. Expected banner: *"Suspected collision — not automatically confirmed"*, with
   the count of independent motion channels that agreed and the number of
   implicated tracks — or an explicit statement that **participant location is
   withheld**.
3. On a calibrated fixed camera the banner states that at least two independent
   channels are required before promotion (`collision.fixed_camera_min_channels: 2`).

**If a judge asks "is this an accident?"** — the answer is: it is a candidate for
human verification. See `CCTV/LIMITATIONS.md` and `DEMO/results_summary.md` for
the false-alarm rate that makes that the only defensible answer.

### 3.5 Severity

In any incident, expand `Technical evidence` → **`Severity breakdown`**. Five
bars: Flow loss, Obstruction, Extent, Duration, Risk exposure, plus the Impact
subscore. Below them: *"Traffic-impact severity, not injury severity."*

Weights are visible at `/api/health` and are editable in
`config/config.yaml`/`netra/severity.py` — a judge can change them and re-render.

### 3.6 Response and escalation workflow

In the incident detail, the action row:

| Button | Endpoint | Expected result |
|---|---|---|
| `Verify` | `POST /api/incidents/{id}/verify` | status → `verified`, workflow stepper advances |
| `Assign` | `POST /api/incidents/{id}/assign` | prompts for an assignee; status → `assigned` |
| `Responding` | `POST /api/incidents/{id}/status` | status → `responding` |
| `Resolved` | same | status → `resolved` |
| `Close` | same | status → `closed`; drops out of the active count |
| `Reject…` | `POST /api/incidents/{id}/reject` | pick a reason from the list at `/api/health` → `rejection_reasons`; incident is withdrawn with the reason recorded |

Every transition is written to `status_history` in `netra.db`, so the audit trail
is inspectable:

```powershell
python -c "import sqlite3;print(list(sqlite3.connect('netra.db').execute('select * from status_history order by id desc limit 10')))"
```

---

## 4. Start the DRONE backend (port 8011)

**Honest status: this backend does not exist yet.** `FINAL/DRONE/` has real
content now — `config/drone_config.yaml`, `models/detector/README.md` (the
VisDrone fine-tune plan, unexecuted), and standalone scripts (`gmc.py`,
`hover_mode.py`, `telemetry_ingest.py`, `thermal_presence.py`) — but none of
it is wired into a FastAPI app, and there is still no server to start on
8011. Re-check `find FINAL/DRONE -type f` on the day; this section may be
stale by then.

**When it exists**, it must satisfy this contract so the walkthrough and the APP
both keep working:

```powershell
cd D:\HARSHIT\ELCIA\FINAL\DRONE
python run.py serve --port 8011
```

- Health: `GET http://127.0.0.1:8011/api/health` → `{"status":"ok", ...}` and a
  field that names the pipeline as scaffolding, so the APP can render the
  placeholder banner automatically rather than relying on a human to remember.
- Incidents: `GET http://127.0.0.1:8011/api/dashboard` returning the same
  `{summary, incidents, cameras}` shape the CCTV API returns, so the APP has one
  renderer and not two.
- It is a **separate process** from CCTV on purpose: different detector, different
  viewing geometry, different failure modes.

**Fallback for the demo day:** play the placeholder clip from
`DEMO/clips/drone_placeholder/` directly, with the banner described in
`DEMO/script.md` §S8, and say plainly that the drone pipeline is scaffolding with
no measured accuracy. Do **not** start an empty server and gesture at it.

---

## 5. Start the APP (Next.js operator console, port 3000)

**Honest status: in progress, unverified from the DEMO side.** `FINAL/APP/`
now has a real `package.json` (Next.js 14, Leaflet for the map, Tailwind) and
route folders under `app/` (`cctv`, `drone`, `incidents`, `map`, `upload`,
`calibrate`) plus populated `components/` and `lib/`. This doc has not run
`npm install && npm run dev` against the current state of that tree — do
that yourself before recording day rather than trusting this paragraph's age.
If it fails, fall back to the built-in CCTV dashboard for everything except
the map view.

**Commands, once you've confirmed it runs:**

```powershell
cd D:\HARSHIT\ELCIA\FINAL\APP
npm install
npm run dev          # http://localhost:3000
```

Environment it must read (put in `APP/.env.local`):

```
NEXT_PUBLIC_CCTV_API=http://127.0.0.1:8000
NEXT_PUBLIC_DRONE_API=http://127.0.0.1:8011
NEXT_PUBLIC_REPLAY=1          # 1 = serve DEMO/seed_data snapshot, 0 = live backends
```

Both backends already send `Access-Control-Allow-Origin: *` (CORS middleware in
`netra/api.py`), so a browser app on `:3000` can call `:8000` directly with no
proxy.

**Until APP ships, use the built-in dashboard on port 8000.** It covers every
requirement in section 3. Nothing in this walkthrough depends on APP except the
map view in section 6.

---

## 6. Map, diversion and responder access route

**Today this is disabled, and the UI says so rather than drawing a fake route.**

Why: `CCTV/config/road_graph.json` does not exist, so `RoadGraph` loads empty and
`/api/health` reports `road_graph_edges: 0`. `/api/graph` returns no features and
the dashboard prints *"No road graph configured."* The two map buttons —
`Confirm carriageway closed → divert` and `Show diversion` — are `disabled`
unless the incident's location carries a `road_edge_id`.

Only two of the 82 cameras carry one: `CUTTACK_LINK_01` → `E_LINK_RD` and
`GANGTOK_6MILE_01` → `E_NH10`. Every ELCIA/ProblemSet camera is zone-only,
because no road metadata was supplied with those clips.

**To enable it**, `config/road_graph.json` must exist with this shape (the loader
is `netra/location.py`):

```json
{
  "nodes": [{"id": "N1", "lat": 20.4625, "lon": 85.8830}, {"id": "N2", "lat": 20.4661, "lon": 85.8902}],
  "edges": [{"id": "E_LINK_RD", "from": "N1", "to": "N2", "name": "Link Road",
             "length_m": 900, "speed_kph": 40, "oneway": false}]
}
```

Then, on an incident whose camera has a matching `road_edge_id`:

1. Click **`Confirm carriageway closed → divert`** → `POST /api/incidents/{id}/close_road`.
   Expected: the edge enters the closed set; the route recomputes around it.
2. Click **`Show diversion`** → `GET /api/route?source=<node>&target=<node>`.
   Expected: a `current` path (under incident costs) and a `baseline` path (clean
   network), so the cost of the diversion is visible rather than asserted.
3. **`Reopen`** → `POST /api/incidents/{id}/reopen_road` restores the edge.

Rendering convention for the APP map view: **diversion route = solid red**
(traffic routed *around* the incident); **responder access route = dashed blue**
(responders routed *to* the incident). The responder route is currently a
**simulated** overlay — see `DEMO/results_summary.md`.

Severity penalises rather than deletes an edge: cost multiplier `1 + 9·S`, so a
maximum-severity incident is ten times more expensive to traverse but the graph
stays connected. Infinite cost disconnects networks and produces "no route".

---

## 7. If a judge asks to change a parameter

This is expected and the system is built for it. All thresholds are in
`CCTV/config/config.yaml`, grouped by event type. Good ones to offer:

| Ask | Change | Effect to expect |
|---|---|---|
| "Make the queue detector more sensitive" | `queue.min_vehicles: 4 → 3`, or `queue.cusum_h: 3.0 → 2.0` | More queue findings, shorter time-to-alert, more marginal ones |
| "Make wrong-side stricter" | `wrong_way.min_persistence_s: 1.5 → 3.0` | Fewer candidates; brief turns stop qualifying |
| "Raise the collision bar" | `collision.stationary_gate: 0.42 → 0.55` | On our measured data this takes collision recall to **0/15** while only dropping clean alarms 21 → 10. Offer to show that table in `CCTV/LIMITATIONS.md` — it is the strongest evidence that the false-alarm problem is not a threshold problem |
| "Run it without a GPU" | add `--cpu --fps 4 --imgsz 640` to `run.py process` | Works; slower; the printed realtime factor tells the truth |

After editing `config.yaml`, **restart** `run.py serve` (config is read at
startup) and re-run `process` for the change to affect detection.

---

## 8. Fast troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach the API.` in the dashboard | Server not running, or wrong port | Restart section 1.4; check `/api/health` |
| Detector tries to download weights / fails offline | `detector.weights: null` | Section 1.2 |
| `--low-resource` fails on a missing checkpoint | `run.py` points at `<CCTV>/yolo26n.pt`, which does not exist | `Copy-Item models\yolo26n.pt .\yolo26n.pt` |
| `no camera config at config/cameras/<ID>.json` | Wrong camera ID | `python run.py status` lists all 82 |
| Video opens but never plays in the browser | Legacy `mp4v`-encoded file | The API transcodes to WebM on first request — reload once after a pause |
| Incident list empty | `netra.db` has no rows | Section 2 |
| Map buttons greyed out | No road graph, or no `road_edge_id` on that camera | Section 6 — this is intended behaviour, not a bug |
| Port 8000 in use | Another process | `python run.py serve --port 8010` and adjust URLs |

---

## 9. The 3-minute version, if time is short

1. `python run.py serve --port 8000` → open `http://127.0.0.1:8000/`
2. `Queues` tab → open one → expand `Technical evidence` → point at the trigger numbers
3. `Wrong-side` tab → open one → point at the "legal direction not reviewed" banner
4. Accident/collision → point at "Suspected collision — not automatically confirmed"
5. `Verify` → `Assign` → status advances
6. Open `DEMO/results_summary.md` and read the failure-cases section aloud
