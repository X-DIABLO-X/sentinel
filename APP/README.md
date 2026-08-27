# ELCIA Operator Console (APP)

Next.js (App Router, TypeScript, Tailwind) operator console for the ELCIA
Smart City Drone-AI Challenge 2026 submission. This directory owns only the
dashboard — it talks to the two backend FastAPI processes over HTTP and
fabricates nothing on its own.

## What this is

A rebuild in React of the concepts in two existing vanilla-JS UIs — not a
port of their code:

- `FINAL/CCTV/ui/index.html` + `app.js` — NETRA's incident console: event-type
  tabs, the `detected → verified → assigned → responding → resolved → closed`
  / `rejected` workflow, the upload job tray.
- `FINAL/CCTV/ui/calibrate.html` + `calibrate.js` — the ~90s camera onboarding
  review (observed motion streams, reviewed-vs-draft legal direction).
- `TEST/app/frontend/index.html` — the 3-column operator console (incident
  feed / map + run controls / detail panel), and its map-line convention:
  **solid red = diversion route for other traffic**, **dashed blue =
  simulated responder access route**.

Every route this app calls is real — read from `FINAL/CCTV/netra/api.py`,
not invented. Where the backend genuinely has no endpoint for something the
brief implies (per-incident diversion routing, a responder access-route
solver, a live drone backend, a live MJPEG stream, an HTTP auto-calibration
trigger), `lib/api.ts` documents it under `MISSING_ENDPOINTS` and the
relevant page renders an explicit, honest empty state instead of mock data.

## Two backends, two ports, on purpose

CCTV and DRONE are separate FastAPI processes (see the architecture note in
`FINAL/README.md`): the drone is a hover-based escalation/verification asset
dispatched to an already-confirmed CCTV incident, not a continuous patrol,
and it runs a differently-trained detector (top-down aerial appearance is
fundamentally different from eye-level CCTV). This console therefore carries
**two base URLs**, never one:

```
NEXT_PUBLIC_CCTV_API   default http://localhost:8000
NEXT_PUBLIC_DRONE_API  default http://localhost:8011
```

Copy `.env.local.example` to `.env.local` to override either. The header's
`BackendStatus` indicator polls both independently — a green CCTV light never
implies the drone backend is answering.

## Honesty rules this app follows

- `lib/types.ts` documents that the CCTV backend's severity bands
  (`severity.py` `BANDS`) only ever emit **Low / Medium / High** —
  `CRITICAL` exists in the `Severity` union for the drone escalation path and
  the brief's four-tier ladder, but nothing in the shipped CCTV pipeline
  produces it today. The UI never manufactures a tier the backend didn't send.
- `app/drone/page.tsx` renders a permanent, prominent
  **"detector not yet fine-tuned on VisDrone"** banner (`lib/types.ts`
  `DRONE_STATUS`), independent of whether the DRONE backend happens to be
  reachable — that banner is a model-readiness fact, not a connectivity fact.
- Nothing in this console fabricates plausible-looking data. Every empty
  state in `components/EmptyState.tsx` names the real reason (endpoint
  missing, backend unreachable, no rows yet) and, where relevant, the exact
  route that was tried.

## Run it

```bash
npm install
npm run dev        # http://localhost:3000, expects CCTV on :8000 (and optionally DRONE on :8011)
npm run typecheck  # tsc --noEmit
npm run build
```

Start the real backend from `FINAL/CCTV`:

```bash
python run.py serve
```

## Structure

```
app/
  page.tsx                overview — KPI cards from GET /api/dashboard
  incidents/page.tsx       filterable incident feed (event-type tabs, status, camera)
  incidents/[id]/page.tsx  detail: evidence, measurements, triggers, workflow actions
  map/page.tsx             3-column console: feed / Leaflet map + route controls / detail
  cctv/page.tsx            camera picker, calibration still, per-camera event feed
  drone/page.tsx           same shape, DRONE backend, permanent not-fine-tuned banner
  calibrate/page.tsx       camera & lane geometry review (ported concept from calibrate.html)
  upload/page.tsx          drag-and-drop upload + live job tray
components/                IncidentCard, SeverityBadge, MapView (+MapCanvas), EvidencePanel,
                            JobTray, StatusWorkflow, KpiCard, EmptyState, BackendStatus, NavLinks
lib/
  api.ts                   typed client for every real CCTV/DRONE route, + MISSING_ENDPOINTS
  types.ts                 Incident/Camera/Severity/Job/RouteGeometry types mirrored from the backend
  useApi.ts                fetch-on-mount + poll hook, surfaces ApiError (incl. "backend offline")
  format.ts                display helpers (durations, timestamps, event labels)
```

## react-leaflet under the App Router

`leaflet`/`react-leaflet` touch `window` at module-eval time, which breaks
server rendering. `components/MapCanvas.tsx` holds the actual Leaflet JSX;
`components/MapView.tsx` is a `"use client"` wrapper that loads it via
`next/dynamic(..., { ssr: false })`. Pages import `MapView`, never
`MapCanvas`, directly.
