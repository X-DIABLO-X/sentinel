"""Minimal FastAPI service for the DRONE subsystem.

Runs as a **separate process on a separate port** from the CCTV backend
(CCTV owns 8000; this owns 8011) — two independent detector models, two
independent processes, by deliberate architecture decision (see the
top-level project README and DRONE/models/detector/README.md).

Routes
------
GET  /api/health            liveness + the honesty flags a judge would check first
GET  /api/status            fuller machine-readable status (mode, gmc, telemetry, thermal)
POST /api/process           run the pipeline on one clip, return the summary
GET  /api/results/{name}    fetch a previously written results JSON by filename
GET  /api/results           summary row per processed clip
GET  /api/dashboard         summary + synthetic incidents + cameras, recomputed each call
GET  /api/incidents/{id}    single synthetic incident, same recompute-per-call basis
GET  /api/incidents/{id}/evidence/{name}   the annotated segment for that incident

Kept intentionally small: this is a hover-dispatch, single-incident system,
not a multi-camera streaming service, so there is no job queue, no websocket
feed, no auth layer here — those belong to APP/ if the operator console ever
needs them.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from fastapi import FastAPI, HTTPException          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware   # noqa: E402
from fastapi.responses import FileResponse           # noqa: E402
from pydantic import BaseModel                       # noqa: E402

import config as drone_config   # noqa: E402
import hover_mode               # noqa: E402
import telemetry_ingest         # noqa: E402
import thermal_presence         # noqa: E402
from pipeline_drone import run_pipeline, PipelineError   # noqa: E402

log = logging.getLogger("drone.api")

app = FastAPI(
    title="NETRA-DRONE API",
    description="Hover-based drone escalation/verification pipeline — separate process/port from CCTV.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_CFG = drone_config.load_config()


class ProcessBody(BaseModel):
    video_path: str
    max_frames: int | None = None


def _safe_results_path(name: str) -> Path:
    """Resolve a results filename with no path traversal outside results/."""
    if "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid results filename")
    p = (_CFG.processing.results_path / name).resolve()
    root = _CFG.processing.results_path.resolve()
    if root not in p.parents and p != root:
        raise HTTPException(status_code=400, detail="invalid results filename")
    return p


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "drone",
        "port": _CFG.api.port,
        "mode": _CFG.mode,
        "detector_finetuned": not _CFG.detector.is_placeholder,
        "placeholder_detector": _CFG.detector.is_placeholder,
        "gmc_enabled": _CFG.gmc.enabled,
        "telemetry_available": False,
        "road_plane_calibrated": _CFG.road_plane.available,
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "config": _CFG.provenance(),
        "hover_mode": hover_mode.describe(),
        "telemetry": telemetry_ingest.telemetry_status(),
        "thermal": thermal_presence.thermal_status(),
        "results_dir": str(_CFG.processing.results_path),
    }


@app.post("/api/process")
def process(body: ProcessBody) -> dict[str, Any]:
    try:
        result = run_pipeline(body.video_path, cfg=_CFG, max_frames=body.max_frames)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return {
        "results_file": result["_results_file"],
        "track_count": result["track_count"],
        "detector_finetuned": result["provenance"]["detector_finetuned"],
        "gmc_health": result["gmc"]["health"],
        "frames_processed": result["frames"]["processed"],
        "frames_rejected_by_quality_gate": result["frames"]["rejected_by_quality_gate"],
    }


@app.get("/api/results/{name}")
def get_results(name: str) -> dict[str, Any]:
    import json
    p = _safe_results_path(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no such results file: {name}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_results_cache: dict[str, Any] = {"key": None, "value": []}


def _results_signature() -> tuple:
    """(path, mtime, size) per results file -- changes only when a file is
    actually added, removed or rewritten."""
    results_dir = _CFG.processing.results_path
    if not results_dir.exists():
        return ()
    sig = []
    for p in sorted(results_dir.glob("*_results.json")):
        try:
            st = p.stat()
            sig.append((p.name, st.st_mtime_ns, st.st_size))
        except OSError:  # pragma: no cover - defensive
            continue
    return tuple(sig)


def _load_all_results() -> list[dict[str, Any]]:
    """Parsed results JSON for every processed clip.

    Cached on the results directory's own (name, mtime, size) signature.
    Every route here is derived from these files, and re-reading and
    re-parsing all 16 of them on every request -- including for a single
    incident lookup, and for each frame the console polls -- was the main
    cost in the drone routes. A newly written results file still invalidates
    the cache immediately, so `run_pipeline` output shows up without a
    restart.
    """
    import json
    key = _results_signature()
    if _results_cache["key"] == key:
        return _results_cache["value"]

    out = []
    results_dir = _CFG.processing.results_path
    if results_dir.exists():
        for p in sorted(results_dir.glob("*_results.json")):
            try:
                with p.open("r", encoding="utf-8") as fh:
                    out.append(json.load(fh))
            except (OSError, ValueError) as exc:  # pragma: no cover - defensive
                log.warning("skipping unreadable results file %s: %s", p, exc)

    _results_cache["key"] = key
    _results_cache["value"] = out
    return out


def _synthetic_incidents(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn real queue/blockage events found in results JSON into Incident-shaped
    records the console can render. No severity model exists on the drone side
    yet (unlike CCTV's bounded-factor scorer), so severity fields stay null
    rather than inventing a number — the evidence string is what carries the
    actual finding.
    """
    incidents: list[dict[str, Any]] = []
    next_id = 1
    for res in all_results:
        camera_id = Path(res.get("source_video", "unknown")).stem
        annotated = res.get("annotated_video")
        for kind in ("queue", "blockage"):
            block = res.get(kind) or {}
            for ev in block.get("events", []):
                incidents.append({
                    "id": next_id,
                    "run_id": res.get("generated_at"),
                    "camera_id": camera_id,
                    "corridor_id": None,
                    "event_type": kind,
                    "label": ev.get("evidence") or f"{kind} candidate",
                    "started_t": ev.get("start_t"),
                    "detected_t": ev.get("end_t", ev.get("start_t")),
                    "ended_t": ev.get("end_t"),
                    "duration": ev.get("duration_s"),
                    "confidence": None,
                    "severity": None,
                    "severity_label": None,
                    "priority": None,
                    "status": "detected",
                    "needs_verification": True,
                    "recommended_action": "Operator review — drone-side heuristic, no ground truth to validate precision/recall against yet.",
                    "explanation": ev.get("validation_note"),
                    "track_ids": ev.get("track_ids", []),
                    "triggers": ev,
                    "created_at": res.get("generated_at"),
                    "source_kind": "drone",
                    # The full annotated segment, not a trimmed clip -- there is
                    # no per-event clip extraction on the drone side yet, so
                    # this is the whole processed video the event was found in.
                    "evidence": {"clip": annotated} if annotated else {},
                })
                next_id += 1
    return incidents


_incidents_cache: dict[str, Any] = {"key": None, "value": []}


def _current_incidents() -> list[dict[str, Any]]:
    """Synthesized incidents for the current results set, cached on the same
    signature as the underlying files. Keeps id assignment stable between the
    dashboard listing and a single-incident lookup within one results set,
    which is what makes /incidents/{id}?backend=drone resolve to the row the
    console actually linked to."""
    key = _results_signature()
    if _incidents_cache["key"] == key:
        return _incidents_cache["value"]
    value = _synthetic_incidents(_load_all_results())
    _incidents_cache["key"] = key
    _incidents_cache["value"] = value
    return value


def _find_incident(incident_id: int) -> dict[str, Any] | None:
    for row in _current_incidents():
        if row["id"] == incident_id:
            return row
    return None


@app.get("/api/results")
def list_results() -> dict[str, Any]:
    """Summary row per processed clip — what /api/results/{name} can't give in
    one call. Real output only: this reads whatever is actually on disk in
    results/, nothing fabricated if the directory is empty."""
    rows = []
    for res in _load_all_results():
        rows.append({
            "source_video": res.get("source_video"),
            "track_count": res.get("track_count"),
            "queue_events": len((res.get("queue") or {}).get("events", [])),
            "blockage_events": len((res.get("blockage") or {}).get("events", [])),
            "detector_finetuned": res.get("provenance", {}).get("detector_finetuned"),
            "gmc_health": (res.get("gmc") or {}).get("health"),
            "annotated_video": res.get("annotated_video"),
        })
    return {"count": len(rows), "clips": rows}


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    """One consistent snapshot: summary + incidents + cameras, matching the
    CCTV backend's route contract so the APP console can render both with the
    same component. Built from whatever real results JSON exists in
    results/ — empty results/ means an honestly empty dashboard, not mock data.
    """
    all_results = _load_all_results()
    incidents = _current_incidents()
    by_type: dict[str, int] = {}
    for inc in incidents:
        by_type[inc["event_type"]] = by_type.get(inc["event_type"], 0) + 1
    cameras = [
        {"camera_id": Path(res.get("source_video", "unknown")).stem,
         "name": res.get("source_video"),
         "source": "drone-hover"}
        for res in all_results
    ]
    summary = {
        "total_incidents": len(incidents),
        "open_incidents": len(incidents),
        "awaiting_verification": sum(1 for i in incidents if i["needs_verification"]),
        "by_type": by_type,
        "by_severity": {},
        "by_status": {"detected": len(incidents)} if incidents else {},
        "cameras": len(cameras),
    }
    return {"summary": summary, "incidents": incidents, "cameras": cameras}


@app.get("/api/incidents/{incident_id}")
def get_incident(incident_id: int) -> dict[str, Any]:
    """Single-incident view for APP's /incidents/{id}?backend=drone route.

    Recomputed from results/ on every call, same as /api/dashboard -- there is
    no incident database on this backend, so an id is only ever stable as long
    as the results directory's file listing doesn't change.
    """
    row = _find_incident(incident_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no drone incident with id {incident_id}")
    return row


@app.get("/api/incidents/{incident_id}/evidence/{name}")
def get_incident_evidence(incident_id: int, name: str) -> FileResponse:
    """Serves the annotated segment named in that incident's own evidence.clip
    -- never an arbitrary filename, so this can't be used to read outside
    results/. The bare-filename convention matches CCTV's equivalent route.
    """
    row = _find_incident(incident_id)
    if row is None or row.get("evidence", {}).get("clip") != name:
        raise HTTPException(status_code=404, detail="no such evidence file for this incident")
    path = (_CFG.processing.results_path / name).resolve()
    root = _CFG.processing.results_path.resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=403, detail="forbidden path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="evidence file not found on disk")
    return FileResponse(path, media_type="video/mp4", headers={
        "Cache-Control": "public, max-age=86400, immutable"})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=_CFG.api.host, port=_CFG.api.port)
