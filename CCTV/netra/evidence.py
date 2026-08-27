"""Evidence packets.

An alert without evidence is worse than no alert. An operator who cannot see
*why* the system fired learns within a week to ignore it, and then the system
has negative value -- it consumed budget and attention and returned noise.

So every NETRA incident carries a packet that lets a human reach their own
verdict in about five seconds:

* the annotated key frame, with the corridor, the legal direction arrow, the
  implicated tracks and their trajectories drawn on it
* a short clip spanning *before* and *after* the recovered onset -- before
  matters, because the useful question is what led up to this
* the trigger values that actually fired, in numbers
* the severity breakdown, component by component
* the model-run identity, so a result can be reproduced

The ring buffer is what makes the "before" possible. We hold a rolling window
of recent frames in memory at reduced scale; when an event fires we already
have its history rather than wishing we had started recording.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import class_name

# BGR. Severity uses a traffic-signal vocabulary because that is the visual
# language a control room already reads fluently.
COLOUR_BY_SEVERITY = {
    "Low": (120, 190, 90),
    "Medium": (60, 180, 240),
    "High": (60, 60, 235),
}
CORRIDOR_COLOUR = (190, 150, 60)
EXCLUSION_COLOUR = (120, 120, 120)
TRACK_COLOUR = (230, 200, 120)
IMPLICATED_COLOUR = (60, 60, 235)


@dataclass
class BufferedFrame:
    t: float
    frame: np.ndarray


class FrameBuffer:
    """Rolling window of recent frames, kept small enough for 16 GB of RAM.

    At 0.6 scale on 720p that is roughly 0.8 MB per frame; 30 s at 8 Hz is
    ~190 MB, which is affordable. The scale is configurable because a 1080p
    portrait phone video and a 410p CCTV feed have very different budgets.
    """

    def __init__(self, seconds: float = 30.0, fps: float = 8.0, scale: float = 0.6):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"FrameBuffer fps must be positive, got {fps!r}")
        self.seconds = seconds
        self.fps = float(fps)
        self.scale = scale
        self.buf: deque[BufferedFrame] = deque(maxlen=int(seconds * self.fps) + 8)

    def push(self, frame: np.ndarray, t: float) -> None:
        f = frame
        if self.scale != 1.0:
            f = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_AREA)
        self.buf.append(BufferedFrame(t=t, frame=f))

    def window(self, t0: float, t1: float) -> list[tuple[float, np.ndarray]]:
        return [(b.t, b.frame) for b in self.buf if t0 <= b.t <= t1]

    def all(self) -> list[tuple[float, np.ndarray]]:
        return [(b.t, b.frame) for b in self.buf]

    def nearest(self, t: float):
        if not self.buf:
            return None
        b = min(self.buf, key=lambda x: abs(x.t - t))
        return b.t, b.frame

    def scaled_box(self, box) -> tuple[float, float, float, float]:
        s = self.scale
        return (box[0] * s, box[1] * s, box[2] * s, box[3] * s)


class EvidenceWriter:
    """Renders and persists the packet for one incident."""

    def __init__(self, root: str | Path = "evidence", clip_fps: float = 8.0,
                 pre_roll_s: float = 6.0, post_roll_s: float = 6.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.set_clip_fps(clip_fps)
        self.pre_roll_s = pre_roll_s
        self.post_roll_s = post_roll_s

    def set_clip_fps(self, fps: float) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"evidence clip fps must be positive, got {fps!r}")
        self.clip_fps = float(fps)

    def incident_dir(self, camera_id: str, event_id: int,
                     run_id: str | None = None) -> Path:
        # Event ids are process-local and restart at one.  Without the run
        # namespace, repeating a benchmark silently overwrote evidence from
        # the earlier run while its report still pointed at that directory.
        d = self.root / camera_id
        if run_id:
            d = d / run_id
        d = d / f"INC-{event_id:05d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- drawing -----------------------------------------------------------
    def annotate(self, frame: np.ndarray, scene, tracks, event, scale: float = 1.0):
        """Draw the scene model, the tracks and the event's own reasoning."""
        img = frame.copy()
        h, w = img.shape[:2]

        overlay = img.copy()
        for corridor in scene.corridors:
            pts = (np.asarray(corridor.polygon, dtype=np.float32) * scale).astype(np.int32)
            active = (event is not None and event.corridor_id == corridor.id)
            colour = (70, 190, 255) if active else CORRIDOR_COLOUR
            cv2.fillPoly(overlay, [pts], colour)
            cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)

            # the legal direction arrow -- the thing a judge asks about first
            cx, cy = corridor.centroid
            cx, cy = cx * scale, cy * scale
            d = corridor.direction
            L = max(28.0, min(w, h) * 0.09)
            p0 = (int(cx - d[0] * L / 2), int(cy - d[1] * L / 2))
            p1 = (int(cx + d[0] * L / 2), int(cy + d[1] * L / 2))
            cv2.arrowedLine(img, p0, p1, (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.35)
            cv2.arrowedLine(img, p0, p1, colour, 2, cv2.LINE_AA, tipLength=0.35)
            cv2.putText(img, corridor.id, (int(cx) - 20, int(cy) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        for z in scene.zones:
            pts = (np.asarray(z.polygon, dtype=np.float32) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, EXCLUSION_COLOUR, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)

        implicated = set(event.track_ids) if event else set()
        for tr in tracks:
            box = [v * scale for v in tr.box]
            x1, y1, x2, y2 = [int(v) for v in box]
            hot = tr.track_id in implicated
            colour = IMPLICATED_COLOUR if hot else TRACK_COLOUR
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3 if hot else 1, cv2.LINE_AA)

            label = f"{class_name(tr.cls)} #{tr.track_id}"
            cv2.putText(img, label, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

            pts = tr.points(4.0)
            if len(pts) > 1:
                arr = (np.asarray(pts, dtype=np.float32) * scale).astype(np.int32)
                cv2.polylines(img, [arr], False, colour, 3 if hot else 1, cv2.LINE_AA)
                if hot:
                    d = tr.direction(1.5, min_span=3.0)
                    if d is not None:
                        p = arr[-1]
                        q = (int(p[0] + d[0] * 45), int(p[1] + d[1] * 45))
                        cv2.arrowedLine(img, tuple(p), q, IMPLICATED_COLOUR, 3,
                                        cv2.LINE_AA, tipLength=0.4)

        if event is not None:
            self._banner(img, event)
        return img

    def _banner(self, img, event) -> None:
        h, w = img.shape[:2]
        colour = COLOUR_BY_SEVERITY.get(event.severity_label, (60, 60, 235))
        pad = 10
        lines = [
            f"{event.label.upper()}  [{event.severity_label}]",
            event.explain()[:110],
            (f"confidence {event.confidence:.2f}   severity {event.severity:.2f}   "
             f"onset t={event.started_t:.1f}s   detected t={event.detected_t:.1f}s"),
        ]
        box_h = pad * 2 + 22 * len(lines)
        cv2.rectangle(img, (0, 0), (w, box_h), (18, 18, 18), -1)
        cv2.rectangle(img, (0, 0), (8, box_h), colour, -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (18, pad + 16 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 if i else 0.62,
                        (255, 255, 255) if i else colour, 1 if i else 2, cv2.LINE_AA)

    # -- persistence -------------------------------------------------------
    def write(self, event, scene, tracks, buffer: FrameBuffer,
              model_run: dict | None = None) -> dict:
        """Write frame, clip and JSON for one incident; return the manifest."""
        run_id = str((model_run or {}).get("run_id") or "") or None
        d = self.incident_dir(scene.camera_id, event.id, run_id=run_id)
        manifest: dict = {"dir": str(d)}

        key = buffer.nearest(event.detected_t)
        if key is not None:
            _, frame = key
            annotated = self.annotate(frame, scene, tracks, event, scale=buffer.scale)
            p = d / "annotated.jpg"
            cv2.imwrite(str(p), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            manifest["annotated_frame"] = p.name

            raw = d / "original.jpg"
            cv2.imwrite(str(raw), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            manifest["original_frame"] = raw.name

        t0 = event.started_t - self.pre_roll_s
        t1 = max(event.detected_t, event.started_t) + self.post_roll_s
        frames = buffer.window(t0, t1)
        if len(frames) >= 4:
            # VP8/WebM is directly decodable by Chromium/Firefox. OpenCV's
            # default mp4v output is valid on disk but leaves browser video
            # elements spinning indefinitely on many Windows installations.
            clip = d / "clip.webm"
            ok = self._write_clip(clip, frames, scene, tracks, event, buffer.scale)
            if ok:
                manifest["clip"] = clip.name
                manifest["clip_span_s"] = [round(frames[0][0], 2), round(frames[-1][0], 2)]

        record = event.to_dict()
        record["severity_disclaimer"] = (
            "Traffic-impact severity, not injury severity."
        )
        if model_run:
            record["model_run"] = model_run
        (d / "incident.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        manifest["record"] = "incident.json"
        return manifest

    def _write_clip(self, path: Path, frames, scene, tracks, event, scale) -> bool:
        h, w = frames[0][1].shape[:2]
        codecs = ("VP80",) if path.suffix.lower() == ".webm" else ("mp4v", "avc1", "MJPG")
        for fourcc in codecs:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                     self.clip_fps, (w, h))
            if writer.isOpened():
                for t, f in frames:
                    # only mark up frames at or after onset, so the operator can
                    # see the scene as it was before the system intervened
                    if t >= event.started_t:
                        f = self.annotate(f, scene, tracks, event, scale=scale)
                    else:
                        f = self.annotate(f, scene, [], None, scale=scale)
                    writer.write(f)
                writer.release()
                # ``isOpened`` is not enough: some OpenCV/codec combinations
                # accept the writer and still leave an empty or undecodable
                # file.  Only advertise evidence that can be read back.
                probe = cv2.VideoCapture(str(path))
                readable, _ = probe.read() if probe.isOpened() else (False, None)
                probe.release()
                if readable and path.exists() and path.stat().st_size > 0:
                    return True
            writer.release()
            # This path was created by this call and just failed read-back;
            # remove it before trying another codec so an invalid non-empty
            # file can never masquerade as evidence.
            if path.exists():
                path.unlink()
        return False
