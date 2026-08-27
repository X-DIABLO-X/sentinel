# Electronic City Phase 1 — CCTV-to-Hospital Demo Workflow

## Purpose

Demonstrate the operator journey from a **synthetic CCTV congestion signal** to an **explainable, simulated hospital access route**. This is a showcase workflow, not a live dispatch or navigation system.

## What is real, illustrative and deliberately unavailable

| Item | Status | Rule for presenting it |
|---|---|---|
| CCTV cameras | Synthetic Phase 1 showcase locations | Call them “demo cameras”; they are not installed devices. |
| Hospitals | Publicly mapped reference locations from the existing facility registry | Call them “nearby reference facilities”; do not claim bed, trauma or ambulance availability. |
| Route calculation | Local Dijkstra calculation on a small illustrative road graph | Say “shortest route in the demo road model,” not “live Google/OSM traffic route.” |
| Congestion | Operator-triggered simulated closure | It demonstrates the decision flow; it is not a live traffic feed. |
| Dispatch | Not implemented | The workflow recommends a route only; it never contacts a hospital or emergency service. |

## Presenter setup

1. Start the CCTV API (`FINAL/CCTV`, `python run.py serve`) and the Next.js console (`FINAL/APP`, `npm run dev`).
2. Open **Map**. The *Electronic City Phase 1 showcase* is enabled by default.
3. Keep the real incident feed visible on the left; this proves the showcase layer does not replace pipeline data.

## 90-second demo script

| Time | Presenter action | What the audience sees | Point to make |
|---:|---|---|---|
| 0:00 | Open Map | Four teal demo CCTV sites and three medical-reference pins around Phase 1 | Existing cameras are the persistent sensing layer. |
| 0:15 | Point to `EC-P1-CCTV-02` | Neeladri Junction is marked as the source of a congestion candidate | An incident is location-aware, but still needs operator verification. |
| 0:25 | Click **Simulate congestion** | The Neeladri approach changes to a dotted red closed segment | Operator confirmation changes the routing constraint, not the detector’s claim. |
| 0:35 | Read the decision rail | The system compares routes and chooses the shortest reachable hospital in the demo graph | The route is computed after excluding the congested edge, not merely drawn over it. |
| 0:50 | Follow the solid blue line | Hospital → alternate corridor → incident path, distance and ETA shown | This is a simulated responder-access route; it is visually separate from public diversion. |
| 1:05 | Click **Clear congestion** | The closure is removed and the direct route is restored | Decisions are reversible and auditable. |
| 1:15 | Select a real incident card | Evidence clip, severity and verify/reject controls remain available | The production workflow is evidence → human decision → action. |

## Operator logic behind the interaction

```text
CCTV candidate (synthetic map trigger for this showcase)
  → operator reviews evidence in the existing incident workflow
  → operator confirms congestion / closure
  → route engine removes the affected demo edge
  → Dijkstra evaluates every nearby reference hospital
  → shortest reachable facility-to-incident path is highlighted
  → operator may assign a team in the real incident workflow
  → no facility is contacted and no automatic dispatch occurs
```

## Data used by the map

- **Synthetic CCTV sites:** Phase 1 Gate, Neeladri Junction, Wipro Link and Velankani Link.
- **Reference hospitals:** Kauvery Hospital, Best E City Hospital and Ramakrishna Hospital, using the coordinates already registered in `CCTV/config/facilities.json`.
- **Demo speed assumption:** 22 km/h for the displayed ETA. It is a transparent assumption, not live traffic speed.
- **Road graph:** nine illustrative edges. The “Neeladri Junction → incident” edge is removed during simulated congestion, then restored on clear.

## Acceptance checklist

- The map is visibly centred on Electronic City Phase 1 rather than Bengaluru at large.
- Camera, hospital, incident, closure and route marks have distinct colours/labels.
- Triggering congestion visibly changes the selected route or route state.
- The interface labels the result as a synthetic demo and simulated route.
- Existing backend-driven incidents and evidence panels remain unaffected.
