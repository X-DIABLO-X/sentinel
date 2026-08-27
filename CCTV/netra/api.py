"""FastAPI service: incidents, evidence, calibration, routing, metrics.

Python end to end. The model, the event engines and the API are one process,
which for a system of this size is the right call -- a message broker and three
services would add operational surface without removing any real coupling.

Endpoints exist to serve exactly what the brief asks a dashboard to show:
event type, severity, location/zone, timestamp, visual evidence, alert and
escalation logic, and a recommended response.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import load_config
from .db import REJECTION_REASONS, STATUSES, IncidentStore
from .location import RoadGraph, describe_location
from .jobs import VideoJobManager, safe_video_name
from .scene import SceneModel, load_all
from .severity import BANDS, DISCLAIMER, W_IMPACT, W_TOTAL

ROOT = Path(__file__).resolve().parents[1]


def _without_deprecated_aliases(models: dict[str, SceneModel]) -> dict[str, SceneModel]:
    """Hide legacy camera IDs when the same source has a canonical ELCIA ID.

    Earlier runs named the supplied traffic videos ``TRAFFIC_*``; the verified
    no-accident audit names them ``ELCIADATASET_*``. Showing both makes one
    physical camera look like two and resurfaces stale false-collision runs.
    History remains in SQLite; only the operator's current camera catalogue is
    deduplicated.
    """
    canonical = set(models)
    return {
        cid: scene for cid, scene in models.items()
        if not (cid.startswith("TRAFFIC_")
                and "ELCIADATASET_" + cid.removeprefix("TRAFFIC_") in canonical)
    }


# -- request bodies --------------------------------------------------------

class StatusBody(BaseModel):
    status: str
    actor: str = "operator"
    reason: str = ""
    comment: str = ""


class AssignBody(BaseModel):
    owner: str
    team: str = ""


class RejectBody(BaseModel):
    reason: str = "other"
    actor: str = "operator"
    comment: str = ""


class RouteQuery(BaseModel):
    source: str
    target: str


class CalibrationBody(BaseModel):
    camera_id: str
    payload: dict[str, Any]


def create_app(config: dict | None = None) -> FastAPI:
    cfg = config or load_config()
    paths = cfg.get("paths", {})
    store = IncidentStore(paths.get("database", "netra.db"))
    evidence_root = Path(paths.get("evidence", "evidence"))
    cameras_dir = Path(paths.get("cameras", "config/cameras"))
    graph_path = Path(paths.get("road_graph", "config/road_graph.json"))
    road_graph = RoadGraph(graph_path if graph_path.exists() else None)
    video_cache_lock = threading.Lock()

    app = FastAPI(title="NETRA -- Road Incident Intelligence", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def _no_cache_on_errors(request, call_next):
        """Never let a failure response be cached.

        Media routes hand back long-lived, immutable Cache-Control on success,
        which is right -- an evidence clip for a closed incident never changes.
        But a 404 served while an asset was still being deployed could be
        heuristically cached by the browser and then keep showing "could not be
        loaded" long after the file was actually in place. Errors are always
        transient here, so mark them explicitly uncacheable.
        """
        response = await call_next(request)
        if response.status_code >= 400:
            response.headers["Cache-Control"] = "no-store"
        return response

    scene_cache: dict[str, SceneModel] | None = None

    def scenes(refresh: bool = False) -> dict[str, SceneModel]:
        nonlocal scene_cache
        if refresh or scene_cache is None:
            scene_cache = _without_deprecated_aliases(load_all(cameras_dir))
        return scene_cache

    jobs = VideoJobManager(ROOT / "uploads", cfg,
                           on_complete=lambda: scenes(refresh=True))

    def enrich_incidents(rows: list[dict]) -> list[dict]:
        """Apply current safety semantics to historical stored incidents."""
        camera_models = scenes()
        rows = [row for row in rows if row.get("camera_id") in camera_models]
        for row in rows:
            cid = str(row.get("camera_id") or "")
            row["source_kind"] = (
                "problem_set" if cid.startswith("PROBLEMSET_") else
                "upload" if cid.startswith("UPLOAD_") else
                "legacy_accident" if cid.startswith("ACCIDENTS_") else
                "traffic")
            scene = camera_models.get(row.get("camera_id"))
            if scene is not None and not row.get("location"):
                proxy = type("EventLocation", (), {
                    "corridor_id": row.get("corridor_id")})()
                row["location"] = describe_location(scene, proxy)
            if row.get("event_type") == "wrong_way" and scene is not None:
                row.setdefault("triggers", {})["legal_direction_reviewed"] = scene.legal_direction_reviewed
                row["triggers"]["direction_source"] = (
                    "human-reviewed legal direction" if scene.legal_direction_reviewed
                    else "observed majority flow; legal direction unreviewed")
                if not scene.legal_direction_reviewed:
                    row["needs_verification"] = True
        return rows

    def browser_video(path: Path, cache_path: Path | None = None) -> Path:
        """Return a browser-native video, converting legacy mp4v once."""
        if path.suffix.lower() != ".mp4":
            return path
        browser_copy = cache_path or path.with_suffix(".browser.webm")
        browser_copy.parent.mkdir(parents=True, exist_ok=True)
        if browser_copy.exists() and browser_copy.stat().st_size > 0:
            return browser_copy
        with video_cache_lock:
            if browser_copy.exists() and browser_copy.stat().st_size > 0:
                return browser_copy
            import cv2
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return path
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
            target_fps = min(8.0, source_fps)
            stride = max(1, int(round(source_fps / target_fps)))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale = min(1.0, 960.0 / max(w, h, 1))
            out_size = (int(round(w * scale)), int(round(h * scale)))
            pending = browser_copy.with_name(browser_copy.stem + ".pending.webm")
            writer = cv2.VideoWriter(str(pending), cv2.VideoWriter_fourcc(*"VP80"),
                                     source_fps / stride, out_size)
            frame_index = 0
            while writer.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    if scale < 1.0:
                        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
                    writer.write(frame)
                frame_index += 1
            writer.release()
            cap.release()
            if pending.exists() and pending.stat().st_size > 0:
                pending.replace(browser_copy)
            else:
                pending.unlink(missing_ok=True)
        return browser_copy if browser_copy.exists() else path

    def current_incidents() -> list[dict]:
        return enrich_incidents(store.incidents(
            limit=2000, latest_only=True, consolidated_only=True))

    def summary_from(rows: list[dict]) -> dict:
        by_type = Counter(r["event_type"] for r in rows)
        by_severity = Counter(r["severity_label"] for r in rows)
        by_status = Counter(r["status"] for r in rows)
        active = sum(r["status"] not in ("closed", "rejected", "resolved")
                     for r in rows)
        return {
            "total_incidents": len(rows), "open_incidents": active,
            "awaiting_verification": sum(
                bool(r["needs_verification"]) and r["status"] == "detected"
                for r in rows),
            "by_type": dict(by_type), "by_severity": dict(by_severity),
            "by_status": dict(by_status), "cameras": len(scenes()),
            "severity_disclaimer": DISCLAIMER,
        }

    def cameras_from(rows: list[dict]) -> list[dict]:
        known = {c["camera_id"] for c in store.cameras()}
        counts = Counter(row["camera_id"] for row in rows)
        out = []
        for cid, scene in scenes().items():
            item = scene.to_dict()
            item["incidents"] = counts[cid]
            item["known_to_db"] = cid in known
            out.append(item)
        return out

    # -- health & meta ----------------------------------------------------
    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "cameras": len(scenes()),
            "road_graph_edges": road_graph.G.number_of_edges(),
            "severity_model": {
                "impact_weights": W_IMPACT,
                "total_weights": W_TOTAL,
                "bands": [{"below": b, "label": l} for b, l in BANDS],
                "disclaimer": DISCLAIMER,
            },
            "statuses": STATUSES,
            "rejection_reasons": REJECTION_REASONS,
        }

    @app.get("/api/summary")
    def summary():
        return summary_from(current_incidents())

    @app.get("/api/dashboard")
    def dashboard():
        """One consistent snapshot; avoids three repeated incident scans."""
        rows = current_incidents()
        return {"summary": summary_from(rows), "incidents": rows,
                "cameras": cameras_from(rows)}

    # -- cameras ----------------------------------------------------------
    @app.get("/api/cameras")
    def cameras():
        return cameras_from(current_incidents())

    @app.get("/api/cameras/{camera_id}")
    def camera(camera_id: str):
        sc = scenes().get(camera_id)
        if sc is None:
            raise HTTPException(404, f"unknown camera {camera_id}")
        return sc.to_dict()

    @app.get("/api/cameras/{camera_id}/frame")
    def camera_frame(camera_id: str):
        """First frame of the camera's source -- the canvas for calibration."""
        sc = scenes().get(camera_id)
        if sc is None:
            raise HTTPException(404, f"unknown camera {camera_id}")
        snap = ROOT / "reports" / f"frame_{camera_id}.jpg"
        if not snap.exists():
            import cv2
            cap = cv2.VideoCapture(sc.source)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(404, "could not read a frame from the source")
            snap.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snap), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # A camera's first frame never changes for a fixed source clip, and the
        # console re-requests it on every camera switch -- let the browser keep it.
        return FileResponse(snap, media_type="image/jpeg", headers={
            "Cache-Control": "public, max-age=86400"})

    @app.post("/api/cameras/{camera_id}/calibration")
    def save_calibration(camera_id: str, body: CalibrationBody):
        """Persist a scene model edited in the calibration screen."""
        try:
            sc = SceneModel.from_dict(body.payload)
        except Exception as exc:
            raise HTTPException(400, f"invalid scene model: {exc}") from exc
        sc.camera_id = camera_id
        path = cameras_dir / f"{camera_id}.json"
        sc.save(path)
        store.upsert_camera(sc)
        scenes(refresh=True)
        return {"saved": str(path), "corridors": len(sc.corridors), "zones": len(sc.zones)}

    # -- incidents --------------------------------------------------------
    @app.get("/api/incidents")
    def incidents(camera_id: str | None = None, status: str | None = None,
                  event_type: str | None = None, limit: int = Query(200, le=2000),
                  latest_only: bool = True):
        return enrich_incidents(store.incidents(
            camera_id=camera_id, status=status, event_type=event_type,
            limit=limit, latest_only=latest_only, consolidated_only=True))

    @app.get("/api/incidents/{incident_id}")
    def incident(incident_id: int):
        d = store.incident(incident_id)
        if d is None:
            raise HTTPException(404, f"incident {incident_id} not found")
        return enrich_incidents([d])[0]

    @app.post("/api/incidents/{incident_id}/verify")
    def verify(incident_id: int, body: StatusBody | None = None):
        actor = body.actor if body else "operator"
        return store.set_status(incident_id, "verified", actor=actor, reason="verified")

    @app.post("/api/incidents/{incident_id}/reject")
    def reject(incident_id: int, body: RejectBody):
        if body.reason not in REJECTION_REASONS:
            raise HTTPException(400, f"reason must be one of {REJECTION_REASONS}")
        # A rejection with a reason is not a deletion -- it is labelled data for
        # the next iteration of thresholds.
        return store.set_status(incident_id, "rejected", actor=body.actor,
                                reason=body.reason, comment=body.comment)

    @app.post("/api/incidents/{incident_id}/assign")
    def assign(incident_id: int, body: AssignBody):
        return store.assign(incident_id, body.owner, body.team)

    @app.patch("/api/incidents/{incident_id}/status")
    def set_status(incident_id: int, body: StatusBody):
        try:
            return store.set_status(incident_id, body.status, actor=body.actor,
                                    reason=body.reason, comment=body.comment)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    def _evidence_dir(folder: str) -> Path | None:
        """Locate an incident's evidence directory on THIS host.

        EvidenceWriter records an absolute path from whichever machine ran the
        analysis (e.g. a Windows ``D:\\...`` path). Deployed on Linux that
        string is not a path at all, just an odd single filename, so every
        evidence request 404'd even with the files present. The tail of the
        record -- ``<camera_id>/<run_hash>/<INC-xxxxx>`` -- is host-independent,
        so rebuild against the local evidence root and fall back to the stored
        path only when analysis and serving happen on the same machine.
        """
        parts = Path(str(folder).replace("\\", "/")).parts
        if len(parts) >= 3:
            cand = evidence_root.joinpath(*parts[-3:])
            if cand.is_dir():
                return cand
        direct = Path(folder)
        return direct if direct.is_dir() else None

    @app.get("/api/incidents/{incident_id}/evidence/{name}")
    def evidence_file(incident_id: int, name: str):
        d = store.incident(incident_id)
        if d is None:
            raise HTTPException(404, "incident not found")
        folder = (d.get("evidence") or {}).get("dir")
        if not folder:
            raise HTTPException(404, "no evidence recorded for this incident")
        base = _evidence_dir(folder)
        if base is None:
            raise HTTPException(
                404, "evidence directory for this incident is not present on this host")
        p = base / name
        if not p.exists() or not p.is_file():
            raise HTTPException(404, f"no evidence file {name}")
        # never let a path fragment escape the evidence root
        try:
            p.resolve().relative_to(evidence_root.resolve())
        except ValueError:
            raise HTTPException(403, "forbidden path")
        # Legacy evidence was encoded as FMP4/mp4v. It reads in OpenCV/VLC but
        # is not supported by Chromium, which looks like an infinite loader.
        # Convert once on demand and cache the browser-native VP8/WebM copy.
        served = browser_video(p)
        media = "video/webm" if served.suffix == ".webm" else (
            "video/mp4" if served.suffix == ".mp4" else
            "image/jpeg" if served.suffix in (".jpg", ".jpeg") else "application/json")
        return FileResponse(served, media_type=media, headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    # -- routing ----------------------------------------------------------
    @app.get("/api/graph")
    def graph():
        return road_graph.to_geojson()

    @app.get("/api/route")
    def route(source: str, target: str):
        if road_graph.G.number_of_nodes() == 0:
            raise HTTPException(404, "no road graph configured")
        return road_graph.route(source, target)

    @app.post("/api/incidents/{incident_id}/close_road")
    def close_road(incident_id: int):
        """Operator confirms the carriageway is blocked; the edge is removed."""
        d = store.incident(incident_id)
        if d is None:
            raise HTTPException(404, "incident not found")
        edge = (d.get("location") or {}).get("road_edge_id")
        if not edge:
            raise HTTPException(400, "incident has no mapped road edge")
        road_graph.apply_incident(edge, 1.0, confirmed_closed=True)
        return {"closed_edge": edge, "closed": sorted(road_graph.closed)}

    @app.post("/api/incidents/{incident_id}/reopen_road")
    def reopen_road(incident_id: int):
        d = store.incident(incident_id)
        if d is None:
            raise HTTPException(404, "incident not found")
        edge = (d.get("location") or {}).get("road_edge_id")
        if edge:
            road_graph.clear_incident(edge)
        return {"reopened_edge": edge}

    # -- metrics ----------------------------------------------------------
    @app.get("/api/metrics")
    def metrics(run_id: str | None = None):
        return {"runs": store.runs(), "metrics": store.metrics(run_id)}

    @app.get("/api/reports")
    def reports():
        out = []
        rdir = ROOT / "reports"
        if rdir.exists():
            for p in sorted(rdir.glob("*.json")):
                try:
                    out.append({"name": p.name, "data": json.loads(p.read_text("utf-8"))})
                except Exception:
                    continue
        return out

    # -- uploaded video analysis -----------------------------------------
    @app.post("/api/jobs")
    async def create_video_job(request: Request):
        """Stream one video to disk, then enqueue the complete NETRA pipeline."""
        filename = unquote(request.headers.get("x-filename", "video.mp4"))
        try:
            safe_video_name(filename)
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from exc
        job_id, target = jobs.allocate(filename)
        pending = target.with_suffix(target.suffix + ".part")
        size = 0
        max_bytes = 2 * 1024 * 1024 * 1024
        try:
            with pending.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(413, "video exceeds the 2 GB upload limit")
                    handle.write(chunk)
                    jobs.update(job_id, percent=min(99, round(size / max_bytes * 100)))
            if size == 0:
                raise HTTPException(400, "empty video")
            pending.replace(target)
            jobs.submit(job_id, size)
            return jobs.get(job_id)
        except Exception as exc:
            pending.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            jobs.fail_upload(job_id, str(getattr(exc, "detail", exc)))
            raise

    @app.get("/api/jobs")
    def video_jobs():
        return jobs.list()

    # -- fixed 16-video accident review set ------------------------------
    @app.get("/api/problem-videos")
    def problem_videos():
        summary_path = ROOT / "ProblemSet" / "Results_release_candidate" / "summary.json"
        if not summary_path.exists():
            return []
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = payload.get("rows") or []
        return sorted(rows, key=lambda row: str(row.get("file") or "").lower())

    @app.get("/api/problem-videos/{stem}/video")
    def problem_video(stem: str):
        rows = problem_videos()
        row = next((r for r in rows if Path(str(r.get("file", ""))).stem == stem), None)
        if row is None:
            raise HTTPException(404, "ProblemSet video not found")
        rel = row.get("annotated_video")
        if not rel:
            raise HTTPException(404, "annotated video not available")
        root = (ROOT / "ProblemSet" / "Results_release_candidate").resolve()
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HTTPException(403, "forbidden path") from exc
        if not path.is_file():
            raise HTTPException(404, "annotated video not found")
        cache = ROOT / "uploads" / "browser_cache" / f"{stem}.webm"
        served = browser_video(path, cache)
        media = "video/webm" if served.suffix.lower() == ".webm" else "video/mp4"
        return FileResponse(served, media_type=media, headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    @app.get("/api/problem-videos/{stem}/poster")
    def problem_poster(stem: str):
        rows = problem_videos()
        row = next((r for r in rows if Path(str(r.get("file", ""))).stem == stem), None)
        if row is None or not row.get("annotated_video"):
            raise HTTPException(404, "ProblemSet video not found")
        root = (ROOT / "ProblemSet" / "Results_release_candidate").resolve()
        source = (root / row["annotated_video"]).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise HTTPException(403, "forbidden path") from exc
        poster = ROOT / "uploads" / "browser_cache" / f"{stem}.jpg"
        if not poster.exists():
            import cv2
            cap = cv2.VideoCapture(str(source))
            event = (row.get("events") or [{}])[0]
            cap.set(cv2.CAP_PROP_POS_MSEC,
                    max(0.0, float(event.get("started_t", 1.0)) - 0.5) * 1000)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(404, "could not decode poster frame")
            h, w = frame.shape[:2]
            scale = min(1.0, 960.0 / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            poster.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        return FileResponse(poster, media_type="image/jpeg", headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    # -- physics-render demo clips (rotation_gate.py, CCTV/demo/) --------
    # Genuinely our own build output, not NETRA's ProblemSet -- lists
    # whatever *_physics_result.json files actually exist in demo/, so this
    # never drifts out of sync with what's really on disk.
    demo_index_cache: dict[str, Any] = {"key": None, "value": []}

    def _demo_signature() -> tuple:
        demo_dir = ROOT / "demo"
        if not demo_dir.is_dir():
            return ()
        sig = []
        for p in sorted(demo_dir.glob("*_physics_result.json")):
            try:
                st = p.stat()
                sig.append((p.name, st.st_mtime_ns, st.st_size))
            except OSError:  # pragma: no cover - defensive
                continue
        return tuple(sig)

    @app.get("/api/demo-videos")
    def demo_videos():
        """Cached on the demo directory's own (name, mtime, size) signature.

        These result files are multi-megabyte -- 13_physics_result.json alone
        carries every track's full per-frame history -- and the console polls
        this route, so re-parsing all nine on every request was pure waste. A
        re-rendered clip changes its signature and invalidates the cache, so
        new output still appears without a restart.
        """
        key = _demo_signature()
        if demo_index_cache["key"] == key:
            return demo_index_cache["value"]

        demo_dir = ROOT / "demo"
        rows = []
        for result_path in demo_dir.glob("*_physics_result.json"):
            stem = result_path.name[: -len("_result.json")]
            video_path = demo_dir / f"{stem}.mp4"
            if not video_path.is_file():
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            collision = payload.get("collision") or {}
            rows.append({
                "file": f"{stem}.mp4",
                "stem": stem,
                "duration_s": payload.get("duration_s"),
                "track_count": len(payload.get("tracks") or {}),
                "collision_score": collision.get("score"),
                "collision_confident": collision.get("confident"),
                "interaction": collision.get("interaction"),
                "relative_heading_deg": collision.get("relative_heading_deg"),
                "contact_t": collision.get("contact_t"),
                "track_ids": collision.get("track_ids"),
            })
        rows.sort(key=lambda r: r["file"].lower())
        demo_index_cache["key"] = key
        demo_index_cache["value"] = rows
        return rows

    @app.get("/api/demo-videos/{stem}/video")
    def demo_video(stem: str):
        # render_physics_demo.py already re-encodes its output to H.264/mp4
        # (see its --no-ffmpeg flag, off by default) -- unlike the legacy
        # mp4v ProblemSet clips, there is nothing here for browser_video() to
        # fix. Calling it anyway was actively harmful: on at least one real
        # deployment its OpenCV/VP8 encode path hung indefinitely mid-file,
        # permanently holding video_cache_lock and deadlocking every other
        # video route (ProblemSet included) behind it. Serve the real file.
        demo_dir = (ROOT / "demo").resolve()
        path = (demo_dir / f"{stem}.mp4").resolve()
        try:
            path.relative_to(demo_dir)
        except ValueError as exc:
            raise HTTPException(403, "forbidden path") from exc
        if not path.is_file():
            raise HTTPException(404, "demo video not found")
        return FileResponse(path, media_type="video/mp4", headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    @app.get("/api/demo-videos/{stem}/poster")
    def demo_poster(stem: str):
        demo_dir = (ROOT / "demo").resolve()
        source = (demo_dir / f"{stem}.mp4").resolve()
        try:
            source.relative_to(demo_dir)
        except ValueError as exc:
            raise HTTPException(403, "forbidden path") from exc
        if not source.is_file():
            raise HTTPException(404, "demo video not found")
        poster = ROOT / "uploads" / "browser_cache" / f"demo_{stem}.jpg"
        if not poster.exists():
            import cv2
            contact_t = 1.0
            result_path = demo_dir / f"{stem}_result.json"
            if result_path.is_file():
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    contact_t = float((payload.get("collision") or {}).get("contact_t") or 1.0)
                except (OSError, ValueError, TypeError):
                    pass
            cap = cv2.VideoCapture(str(source))
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, contact_t - 0.5) * 1000)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(404, "could not decode poster frame")
            h, w = frame.shape[:2]
            scale = min(1.0, 960.0 / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            poster.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        return FileResponse(poster, media_type="image/jpeg", headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    @app.get("/api/jobs/{job_id}")
    def video_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "analysis job not found")
        return job

    @app.get("/api/jobs/{job_id}/video")
    def analysed_video(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "analysis job not found")
        path = job.get("annotated_video")
        if not path or not Path(path).is_file():
            raise HTTPException(404, "annotated video is not ready")
        target = Path(path).resolve()
        try:
            target.relative_to((ROOT / "uploads" / "results").resolve())
        except ValueError as exc:
            raise HTTPException(403, "forbidden path") from exc
        media = "video/webm" if target.suffix.lower() == ".webm" else "video/mp4"
        # No Content-Disposition attachment header: this endpoint is consumed
        # by an inline <video>, not a download button.
        return FileResponse(target, media_type=media, headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    @app.get("/api/jobs/{job_id}/poster")
    def analysed_video_poster(job_id: str):
        job = jobs.get(job_id)
        path = Path(job.get("annotated_video")) if job and job.get("annotated_video") else None
        if path is None or not path.is_file():
            raise HTTPException(404, "annotated video is not ready")
        target = path.resolve()
        try:
            target.relative_to((ROOT / "uploads" / "results").resolve())
        except ValueError as exc:
            raise HTTPException(403, "forbidden path") from exc
        poster = target.with_suffix(".jpg")
        if not poster.exists():
            import cv2
            cap = cv2.VideoCapture(str(target))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise HTTPException(404, "could not decode poster frame")
            cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        return FileResponse(poster, media_type="image/jpeg", headers={
            "Cache-Control": "public, max-age=86400, immutable"})

    # -- UI ---------------------------------------------------------------
    ui_dir = ROOT / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = ui_dir / "index.html"
        if idx.exists():
            return HTMLResponse(idx.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>NETRA</h1><p>UI not found.</p>")

    return app


app = create_app()
