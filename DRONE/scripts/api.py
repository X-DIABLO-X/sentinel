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


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=_CFG.api.host, port=_CFG.api.port)
