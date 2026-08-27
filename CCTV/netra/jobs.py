"""Bounded background video-analysis jobs for the local operator dashboard.

Uploads are streamed by the API and submitted here only after the file is
complete.  A single worker owns the detector and serialises GPU work: concurrent
model runs made both the dashboard and CUDA context unreliable, while a queue
keeps HTTP reads responsive and reuses the already-loaded model.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import Callable


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def safe_video_name(name: str) -> str:
    """Return a short filesystem-safe video name, preserving its extension."""
    raw = Path(name or "video.mp4").name
    suffix = Path(raw).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video type {suffix or '(none)'}")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(raw).stem).strip("_-")
    return f"{(stem or 'video')[:80]}{suffix}"


class VideoJobManager:
    """One model worker, O(1) job status lookup, and bounded queue metadata."""

    def __init__(self, root: Path, config: dict,
                 on_complete: Callable[[], None] | None = None) -> None:
        self.root = Path(root)
        self.input_root = self.root / "inputs"
        self.result_root = self.root / "results"
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.config = copy.deepcopy(config)
        self.on_complete = on_complete
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._queue: Queue[str | None] = Queue(maxsize=16)
        self._load_existing()
        self._worker = threading.Thread(target=self._work, daemon=True,
                                        name="netra-video-worker")
        self._worker.start()

    def _load_existing(self) -> None:
        """Recover completed uploads so a server restart does not orphan them."""
        for folder in sorted(self.result_root.iterdir(),
                             key=lambda p: p.stat().st_mtime):
            if not folder.is_dir():
                continue
            saved = folder / "job.json"
            job = None
            if saved.exists():
                try:
                    job = json.loads(saved.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    job = None
            if job is None:
                videos = sorted(folder.glob("*_annotated.webm"))
                reports = [p for p in folder.glob("*.json")
                           if not p.name.endswith("_candidates.json") and p.name != "job.json"]
                if not videos or not reports:
                    continue
                try:
                    report = json.loads(reports[0].read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                inputs = sorted(self.input_root.glob(f"{folder.name}_*"))
                filename = (inputs[0].name[len(folder.name) + 1:]
                            if inputs else reports[0].stem)
                now = folder.stat().st_mtime
                result = {
                    "events_total": report.get("events_total", 0),
                    "events_by_type": report.get("events_by_type", {}),
                    "events_by_severity": report.get("events_by_severity", {}),
                    "stats": report.get("stats", {}),
                    "events": report.get("events", []),
                    "wall_seconds": (report.get("stats") or {}).get("wall_seconds"),
                }
                job = {"id": folder.name, "filename": filename,
                       "input": str(inputs[0]) if inputs else None,
                       "status": "complete", "phase": "complete", "percent": 100,
                       "message": "Analysis complete", "created_at": now,
                       "updated_at": now, "result": result, "error": None,
                       "annotated_video": str(videos[0]),
                       "camera_id": report.get("camera_id")}
            job_id = str(job.get("id") or folder.name)
            job["id"] = job_id
            self._jobs[job_id] = job
            self._order.append(job_id)

    def _persist(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            return
        folder = self.result_root / job_id
        folder.mkdir(parents=True, exist_ok=True)
        pending = folder / "job.json.tmp"
        pending.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
        pending.replace(folder / "job.json")

    def allocate(self, filename: str) -> tuple[str, Path]:
        clean = safe_video_name(filename)
        job_id = uuid.uuid4().hex[:12]
        path = self.input_root / f"{job_id}_{clean}"
        now = time.time()
        job = {"id": job_id, "filename": clean, "input": str(path),
               "status": "uploading", "phase": "uploading", "percent": 0,
               "message": "Uploading video", "created_at": now,
               "updated_at": now, "result": None, "error": None}
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            # Status lookup stays O(1); retain only bounded old metadata.
            if len(self._order) > 100:
                old = self._order.pop(0)
                self._jobs.pop(old, None)
        return job_id, path

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)
                job["updated_at"] = time.time()

    def submit(self, job_id: str, size: int) -> None:
        self.update(job_id, status="queued", phase="queued", percent=1,
                    size_bytes=size, message="Queued for analysis")
        self._queue.put_nowait(job_id)

    def fail_upload(self, job_id: str, message: str) -> None:
        self.update(job_id, status="failed", phase="failed", error=message,
                    message=message)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def list(self) -> list[dict]:
        with self._lock:
            return [copy.deepcopy(self._jobs[j]) for j in reversed(self._order)
                    if j in self._jobs]

    def _work(self) -> None:
        detector = None
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self.get(job_id)
            if not job:
                continue
            store = None
            try:
                from .db import IncidentStore
                from .detect import Detector
                from scripts.run_problems import process_one

                self.update(job_id, status="running", phase="loading", percent=2,
                            message="Loading road-user detector")
                cfg = copy.deepcopy(self.config)
                # Eight analysed frames/second is enough for the temporal event
                # engines and keeps an ordinary laptop responsive. All engines
                # still run; this changes cadence, not the reasoning graph.
                cfg["pipeline"]["analysis_fps"] = 8.0
                cfg["pipeline"]["resize_long_side"] = 960
                cfg["detector"]["imgsz"] = 640
                cfg.setdefault("render", {})["timeline_stride"] = max(
                    1, int(cfg.get("render", {}).get("timeline_stride", 2)))
                if detector is None:
                    detector = Detector(
                        weights=cfg["detector"].get("weights") or "yolo26n.pt",
                        backend=cfg["detector"].get("backend", "auto"),
                        device=cfg["detector"].get("device", "auto"),
                        imgsz=640, aux_imgsz=cfg["detector"].get("aux_imgsz", 0),
                        conf=cfg["detector"].get("conf", 0.1),
                        iou=cfg["detector"].get("iou", 0.55))
                    detector.warmup()

                store = IncidentStore(cfg["paths"]["database"])
                out = self.result_root / job_id
                path = Path(job["input"])

                def progress(payload):
                    self.update(job_id, status="running", **payload)

                row = process_one(
                    path, f"UPLOAD_{job_id[:6]}", cfg, detector, store,
                    out, out, do_calibrate=True, max_seconds=None,
                    render=True, dump_candidates=True, progress=progress)
                annotated = row.get("annotated_video")
                self.update(job_id, status="complete", phase="complete",
                            percent=100, message="Analysis complete", result=row,
                            annotated_video=(str(out / annotated) if annotated else None),
                            camera_id=row.get("camera_id"))
                self._persist(job_id)
                if self.on_complete:
                    self.on_complete()
            except Exception as exc:
                self.update(job_id, status="failed", phase="failed",
                            error=f"{type(exc).__name__}: {exc}",
                            message="Analysis failed")
            finally:
                if store is not None:
                    store.close()
                self._queue.task_done()
