"""Render a reviewable annotated video of what the system saw and decided.

This is the artefact a human actually judges the system by. Not a metric, not a
JSON dump -- a video where you can watch a vehicle move, watch the trail build,
and watch the alert appear with the numbers that caused it printed alongside.

Why a second pass
-----------------
The pipeline records a per-frame snapshot of tracks as it runs, so rendering
costs drawing time only -- the detector is never run twice. More importantly,
the second pass can draw an event banner from its **recovered onset**, which is
several seconds *before* the frame where the system actually raised it. A live
overlay cannot do that, because at onset time the event had not yet been
detected. Being able to show "the impact was here; we alerted here" in one
video is the clearest possible demonstration of why onset recovery matters.

Layout
------
    +--------------------------------------------------+
    |  banner: active event, severity, reason, numbers  |
    |                                                   |
    |            frame with corridors, tracks,          |
    |            trails, direction arrows               |
    |                                                   |
    |  HUD: time, tracks, change-point z                |
    |  timeline strip: every event, playhead            |
    +--------------------------------------------------+
"""

from __future__ import annotations

import bisect
from pathlib import Path

import cv2
import numpy as np

from .detect import class_name
from .footprint import Footprint

SEV_COLOUR = {"Low": (110, 180, 90), "Medium": (60, 175, 235), "High": (70, 70, 240)}
TYPE_COLOUR = {
    "collision_candidate": (70, 70, 240),
    "wrong_way": (80, 120, 255),
    "queue": (60, 175, 235),
    "blockage": (80, 200, 255),
    "lane_violation": (200, 160, 60),
    "pedestrian_on_carriageway": (240, 140, 200),
    "abnormal_stop": (170, 170, 170),
}
TRACK_COLOUR = (225, 195, 120)
HOT_COLOUR = (70, 70, 240)
# Predicted path: green while the vehicle is following it.
FORECAST_COLOUR = (120, 220, 120)
LANE_FILL = (185, 175, 150)
LANE_LINE = (205, 200, 180)
FORECAST_BREACH_COLOUR = (70, 70, 240)
CORRIDOR_COLOUR = (200, 155, 70)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fit(frame, long_side: int):
    h, w = frame.shape[:2]
    m = max(h, w)
    if long_side and m > long_side:
        s = long_side / m
        return cv2.resize(frame, (int(round(w * s)), int(round(h * s))),
                          interpolation=cv2.INTER_AREA), s
    return frame, 1.0


class VideoAnnotator:
    """Draws the pipeline's own reasoning back onto the source video."""

    def __init__(self,
                 scene,
                 timeline: list[dict],
                 events: list[dict],
                 proc_long_side: int,
                 out_long_side: int = 1280,
                 banner_hold_s: float = 4.0,
                 marker_lead_s: float = 0.0) -> None:
        self.scene = scene
        self.timeline = timeline
        self.events = sorted(events, key=lambda e: e["started_t"])
        self.proc_long_side = proc_long_side
        self.out_long_side = out_long_side
        self.banner_hold_s = banner_hold_s
        self.marker_lead_s = marker_lead_s
        self._ts = [f["t"] for f in timeline]
        # Once a vehicle has been implicated in a collision it stays marked for
        # every frame it appears in, before and after the impact. A reviewer
        # needs to follow that specific car through the whole clip -- see it
        # arrive, see it hit, see it come to rest -- not just glimpse it during
        # a four-second banner.
        self._last_hot: dict[int, list] = {}
        self.implicated: set[int] = set()
        # Boxes of the vehicles attribution actually chose. Kept separately
        # from track ids because the vehicles that matter most are often the
        # ones the tracker never locked onto.
        self.participant_boxes: list[dict] = []
        for e in self.events:
            if e.get("type") != "collision_candidate":
                continue
            if (e.get("triggers") or {}).get("attribution") == "unattributed":
                continue
            self.implicated.update(e.get("track_ids") or [])
            trig = e.get("triggers") or {}
            for i, box in enumerate(trig.get("participant_boxes") or []):
                probs = trig.get("participant_p_crashed") or []
                tids = e.get("track_ids") or []
                self.participant_boxes.append({
                    "box": box,
                    "track_id": tids[i] if i < len(tids) else None,
                    # The red box appears when the collision is CONFIRMED, not
                    # at the recovered onset.
                    #
                    # started_t is walked backwards to the moment the incident
                    # began -- which for a path conflict is when the vehicles
                    # were still approaching each other. Drawing from there put
                    # a red "COLLISION" box on a car seconds before anything had
                    # happened to it, which is both wrong and misleading: it
                    # reads as a prediction the system is not entitled to make.
                    # detected_t is the frame at which the evidence actually
                    # closed.
                    # Never before the collision. detected_t is when the
                    # evidence closed, and the recovered onset is deliberately
                    # NOT used here -- it walks backwards to when the incident
                    # began, which for a conflict is while the vehicles were
                    # still approaching each other.
                    "from_t": max(float(e.get("detected_t", e["started_t"])),
                                  float(e.get("started_t", 0.0))),
                    "until_t": float(e.get("detected_t", e["started_t"])) + 0.35,
                    "p": probs[i] if i < len(probs) else None,
                })

    # -- lookups -----------------------------------------------------------
    def snapshot_at(self, t: float) -> dict | None:
        if not self._ts:
            return None
        i = bisect.bisect_left(self._ts, t)
        if i >= len(self._ts):
            i = len(self._ts) - 1
        if i > 0 and abs(self._ts[i - 1] - t) <= abs(self._ts[i] - t):
            i -= 1
        if abs(self._ts[i] - t) > 1.5:
            return None
        return self.timeline[i]

    def active_events(self, t: float) -> list[dict]:
        """Events whose banner should be on screen at time ``t``.

        An event is shown from its recovered onset until either it ends or the
        hold expires -- so the viewer sees the marking appear at the moment the
        system believes the incident began, not the moment it noticed.
        """
        out = []
        for e in self.events:
            start = e["started_t"] - self.marker_lead_s
            end = e.get("ended_t") or e["detected_t"]
            end = max(end, e["detected_t"]) + self.banner_hold_s
            if start <= t <= end:
                out.append(e)
        return out

    # -- drawing -----------------------------------------------------------
    def draw_scene(self, img, scale):
        overlay = img.copy()
        for c in self.scene.corridors:
            pts = (np.asarray(c.polygon, np.float32) * scale).astype(np.int32)
            cv2.fillPoly(overlay, [pts], CORRIDOR_COLOUR)
            cv2.polylines(img, [pts], True, CORRIDOR_COLOUR, 2, cv2.LINE_AA)
            cx, cy = c.centroid
            cx, cy = cx * scale, cy * scale
            d = c.direction
            L = max(34.0, min(img.shape[:2]) * 0.10)
            p0 = (int(cx - d[0] * L / 2), int(cy - d[1] * L / 2))
            p1 = (int(cx + d[0] * L / 2), int(cy + d[1] * L / 2))
            cv2.arrowedLine(img, p0, p1, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.35)
            cv2.arrowedLine(img, p0, p1, CORRIDOR_COLOUR, 2, cv2.LINE_AA, tipLength=0.35)
            cv2.putText(img, f"{c.id} legal {c.heading:.0f}deg",
                        (int(cx) - 46, int(cy) - 14), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        for z in self.scene.zones:
            pts = (np.asarray(z.polygon, np.float32) * scale).astype(np.int32)
            cv2.polylines(img, [pts], True, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.13, img, 0.87, 0, img)
        return img

    @staticmethod
    def _corner_box(img, x1, y1, x2, y2, colour, thickness=3, frac=0.28):
        """Corner-bracket box. Reads as a targeting lock rather than a label,
        which is what we want on the vehicles judged to have collided -- and it
        leaves the vehicle itself unobscured."""
        w, h = x2 - x1, y2 - y1
        L = int(max(10, min(w, h) * frac))
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * L, py), colour, thickness, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * L), colour, thickness, cv2.LINE_AA)

    def _remember_hot(self, snap, hot_ids):
        """Cache the last known box of each implicated vehicle.

        A collision frequently ends the track: the vehicle is occluded by the
        one that hit it, or the detector loses a crumpled shape it no longer
        recognises. When that happens the red box vanishes mid-clip, exactly
        when a reviewer most wants to keep watching it. Holding the last known
        position keeps the mark on screen for the rest of the video, drawn
        dimmer and labelled as a last-known position so it is not mistaken for
        a live detection.
        """
        for tr in snap.get("tracks", []):
            if tr["id"] in hot_ids:
                self._last_hot[tr["id"]] = [v for v in tr["box"]]

    def draw_lost_hot(self, img, snap, scale, hot_ids):
        live = {tr["id"] for tr in snap.get("tracks", [])} if snap else set()
        for tid in hot_ids:
            if tid in live or tid not in self._last_hot:
                continue
            x1, y1, x2, y2 = [int(v * scale) for v in self._last_hot[tid]]
            self._corner_box(img, x1, y1, x2, y2, (60, 60, 170), 2)
            label = f"COLLISION #{tid} (last seen)"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 7)), (x1 + tw + 8, max(0, y1)),
                          (30, 30, 30), -1)
            cv2.putText(img, label, (x1 + 4, max(9, y1 - 5)), FONT, 0.42,
                        (90, 90, 220), 1, cv2.LINE_AA)
        return img

    def draw_lanes(self, img, snap, scale):
        """Lanes as this camera's own traffic revealed them.

        Drawn faintly and underneath everything else: they are context for the
        reasoning, not a claim about road law. What is shown is where vehicles
        have actually been driving, which on an unmarked or worn road is the
        only lane structure there is.
        """
        lanes = snap.get("lanes") or []
        if not lanes:
            return img
        overlay = img.copy()
        for ln in lanes:
            pts = np.asarray(ln.get("pts") or [], dtype=np.float32)
            if len(pts) < 2:
                continue
            arr = (pts * scale).astype(np.int32)
            half = max(2.0, 0.5 * float(ln.get("w", 40.0)) * scale)
            d = np.gradient(pts, axis=0)
            n = np.stack([-d[:, 1], d[:, 0]], axis=1)
            nrm = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(nrm, 1e-6)
            edge_a = ((pts + n * half) * scale).astype(np.int32)
            edge_b = ((pts - n * half) * scale).astype(np.int32)
            cv2.fillPoly(overlay, [np.concatenate([edge_a, edge_b[::-1]])],
                         LANE_FILL)
            # a dashed centreline, as a lane marking would be
            for i in range(0, len(arr) - 1, 2):
                cv2.line(img, tuple(arr[i]), tuple(arr[i + 1]),
                         LANE_LINE, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)
        return img

    def draw_forecasts(self, img, snap, scale, hot_ids: set[int]):
        """Draw where each vehicle is predicted to go, as a widening cone.

        The blue trail behind a vehicle is what happened; this is what the
        motion model expects next. A collision is precisely a vehicle failing to
        arrive inside its own cone, so showing the cone shows the reasoning
        rather than only the verdict.
        """
        fc = snap.get("forecasts") or {}
        if not fc:
            return img
        overlay = img.copy()
        for tid, f in fc.items():
            pts = np.asarray(f.get("points") or [], dtype=np.float32)
            sig = np.asarray(f.get("sigma") or [], dtype=np.float32)
            if len(pts) < 2 or len(sig) != len(pts):
                continue
            breached = int(tid) in hot_ids
            colour = FORECAST_BREACH_COLOUR if breached else FORECAST_COLOUR

            d = np.gradient(pts, axis=0)
            n = np.stack([-d[:, 1], d[:, 0]], axis=1)
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            n = n / np.maximum(norm, 1e-6)
            off = n * (2.0 * sig[:, None])

            upper = ((pts + off) * scale).astype(np.int32)
            lower = ((pts - off) * scale).astype(np.int32)
            poly = np.concatenate([upper, lower[::-1]], axis=0)
            cv2.fillPoly(overlay, [poly], colour)

            # Keep the drawn line inside the frame. A forecast that leaves
            # the image is telling the viewer about road that is not in shot.
            h_img, w_img = img.shape[:2]
            centre = (pts * scale).astype(np.int32)
            centre[:, 0] = np.clip(centre[:, 0], 0, w_img - 1)
            centre[:, 1] = np.clip(centre[:, 1], 0, h_img - 1)
            cv2.polylines(img, [centre], False, colour, 2, cv2.LINE_AA)
            # a tick at the horizon, so the prediction distance is legible
            cv2.circle(img, tuple(centre[-1]), 3, colour, -1, cv2.LINE_AA)

        # the cone is context, not the subject: keep it faint
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        return img

    def draw_tracks(self, img, snap, scale, hot_ids: set[int]):
        """Every road user gets a trajectory. Implicated vehicles get a lock.

        The two jobs are deliberately different. *All* vehicles are drawn with
        their path so a reviewer can see the traffic as the system saw it --
        that is the context the judgement was made in. Only the vehicles the
        collision engine attributed the impact to get the red corner-bracket
        box, so the claim being made is unambiguous and checkable.
        """
        tracks = snap.get("tracks", [])

        # pass 1: trajectories for everything, underneath the boxes
        for tr in tracks:
            trail = tr.get("trail") or []
            if len(trail) < 2:
                continue
            hot = tr["id"] in hot_ids
            arr = (np.asarray(trail, np.float32) * scale).astype(np.int32)
            if hot:
                cv2.polylines(img, [arr], False, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.polylines(img, [arr], False, HOT_COLOUR, 3, cv2.LINE_AA)
            else:
                # fade the tail so direction of travel reads at a glance
                n = len(arr)
                for i in range(1, n):
                    a = i / n
                    c = tuple(int(v * (0.35 + 0.65 * a)) for v in TRACK_COLOUR)
                    cv2.line(img, tuple(arr[i - 1]), tuple(arr[i]), c, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(arr[-1]), 5 if hot else 3,
                       HOT_COLOUR if hot else TRACK_COLOUR, -1, cv2.LINE_AA)

        # pass 1b: ground-plane footprints.
        #
        # These are the patches of road the collision logic actually compares.
        # Drawing them makes the geometry auditable: a reviewer can see that
        # two vehicles whose boxes appear to touch are in fact standing well
        # apart on the road, which is the confusion that produced most of the
        # earlier false contacts.
        for tr in tracks:
            f = Footprint.from_box([v * scale for v in tr['box']])
            hot = tr['id'] in hot_ids
            cv2.ellipse(img, (int(f.cx), int(f.cy)),
                        (max(1, int(f.a)), max(1, int(f.b))), 0, 0, 360,
                        HOT_COLOUR if hot else (90, 140, 90),
                        2 if hot else 1, cv2.LINE_AA)

        # pass 2: boxes and labels on top
        for tr in tracks:
            box = [v * scale for v in tr["box"]]
            x1, y1, x2, y2 = [int(v) for v in box]
            hot = tr["id"] in hot_ids
            name = class_name(tr["cls"])

            if hot:
                self._corner_box(img, x1, y1, x2, y2, (255, 255, 255), 5)
                self._corner_box(img, x1, y1, x2, y2, HOT_COLOUR, 3)
                cv2.rectangle(img, (x1, y1), (x2, y2), HOT_COLOUR, 1, cv2.LINE_AA)
                label = f"COLLISION  {name} #{tr['id']}"
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 2)
                ly = max(th + 6, y1)
                cv2.rectangle(img, (x1, ly - th - 8), (x1 + tw + 10, ly), HOT_COLOUR, -1)
                cv2.putText(img, label, (x1 + 5, ly - 5), FONT, 0.50,
                            (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), TRACK_COLOUR, 1, cv2.LINE_AA)
                label = f"{name} #{tr['id']}"
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.38, 1)
                cv2.rectangle(img, (x1, max(0, y1 - th - 5)), (x1 + tw + 6, max(0, y1)),
                              (18, 18, 18), -1)
                cv2.putText(img, label, (x1 + 3, max(9, y1 - 4)), FONT, 0.38,
                            TRACK_COLOUR, 1, cv2.LINE_AA)
        return img

    def draw_stationary(self, img, snap, scale):
        """Vehicles found sitting in the background image -- stopped or crashed."""
        for o in (snap.get("stationary") or []):
            x1, y1, x2, y2 = [int(v * scale) for v in o["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 255), 2, cv2.LINE_AA)
            bits = [f"stopped {o['dwell']:.0f}s"]
            if o.get("persons"):
                bits.append(f"{o['persons']} person(s) out")
            if o.get("debris"):
                bits.append(f"{o['debris']} debris")
            label = " | ".join(bits)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.40, 1)
            cv2.rectangle(img, (x1, y2), (x1 + tw + 8, y2 + th + 8), (18, 18, 18), -1)
            cv2.putText(img, label, (x1 + 4, y2 + th + 3), FONT, 0.40,
                        (80, 200, 255), 1, cv2.LINE_AA)
        return img

    def draw_participants(self, img, t, scale):
        """Draw a brief frozen-box fallback when no live track can be followed.

        A stored event box describes one instant. Keeping that rectangle fixed
        while the vehicle drives away makes a correct attribution look metres
        wrong. Tracked participants are already drawn from the time-indexed
        timeline by ``draw_tracks``/``draw_lost_hot``; only an untracked box is
        shown here, and only around the alert frame.
        """
        for p in self.participant_boxes:
            if p.get("track_id") is not None:
                continue
            if t < p["from_t"] or t > p["until_t"]:
                continue
            x1, y1, x2, y2 = [int(v * scale) for v in p["box"]]
            self._corner_box(img, x1, y1, x2, y2, (255, 255, 255), 5)
            self._corner_box(img, x1, y1, x2, y2, HOT_COLOUR, 3)
            cv2.rectangle(img, (x1, y1), (x2, y2), HOT_COLOUR, 1, cv2.LINE_AA)
            f = Footprint.from_box([x1, y1, x2, y2])
            cv2.ellipse(img, (int(f.cx), int(f.cy)),
                        (max(1, int(f.a)), max(1, int(f.b))), 0, 0, 360,
                        HOT_COLOUR, 2, cv2.LINE_AA)
            label = "COLLISION"
            if p.get("p") is not None:
                label += f"  p={p['p']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.50, 2)
            ly = max(th + 6, y1)
            cv2.rectangle(img, (x1, ly - th - 8), (x1 + tw + 10, ly), HOT_COLOUR, -1)
            cv2.putText(img, label, (x1 + 5, ly - 5), FONT, 0.50,
                        (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def draw_impact(self, img, events, scale):
        """Mark the estimated impact point, when one was localised."""
        for e in events:
            pt = (e.get("triggers") or {}).get("impact_point")
            if not pt:
                continue
            x, y = int(pt[0] * scale), int(pt[1] * scale)
            r = int(max(14, min(img.shape[:2]) * 0.035))
            cv2.circle(img, (x, y), r, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), r + 4, HOT_COLOUR, 2, cv2.LINE_AA)
            cv2.line(img, (x - r - 8, y), (x - r + 2, y), HOT_COLOUR, 2, cv2.LINE_AA)
            cv2.line(img, (x + r - 2, y), (x + r + 8, y), HOT_COLOUR, 2, cv2.LINE_AA)
            cv2.line(img, (x, y - r - 8), (x, y - r + 2), HOT_COLOUR, 2, cv2.LINE_AA)
            cv2.line(img, (x, y + r - 2), (x, y + r + 8), HOT_COLOUR, 2, cv2.LINE_AA)
            cv2.putText(img, "impact", (x + r + 10, y + 4), FONT, 0.42,
                        HOT_COLOUR, 1, cv2.LINE_AA)
        return img

    def draw_banner(self, img, events, t):
        if not events:
            return img
        h, w = img.shape[:2]
        ev = max(events, key=lambda e: e.get("priority", 0))
        colour = SEV_COLOUR.get(ev["severity_label"], (70, 70, 240))

        lines = [
            f"{ev['label'].upper()}   severity {ev['severity_label']} "
            f"({ev['severity']:.2f})   confidence {ev['confidence']:.2f}",
            ev.get("explanation", "")[:118],
        ]
        extra = []
        if ev.get("onset_recovered_s", 0) > 0.25:
            extra.append(f"onset recovered {ev['onset_recovered_s']:.1f}s earlier "
                         f"({ev.get('onset_method', '')})")
        extra.append(f"onset {ev['started_t']:.1f}s / alerted {ev['detected_t']:.1f}s")
        attr = (ev.get("triggers") or {}).get("attribution")
        if attr == "unattributed":
            extra.append("VEHICLES NOT IDENTIFIED - no box drawn")
        if ev.get("needs_verification"):
            extra.append("HUMAN VERIFICATION REQUIRED")
        lines.append("   |   ".join(extra)[:118])

        pad, lh = 9, 21
        bh = pad * 2 + lh * len(lines)
        cv2.rectangle(img, (0, 0), (w, bh), (16, 16, 16), -1)
        cv2.rectangle(img, (0, 0), (7, bh), colour, -1)
        # pulse the border while the incident is fresh, so the eye is drawn to it
        if t - ev["started_t"] < 2.0 and int(t * 4) % 2 == 0:
            cv2.rectangle(img, (0, 0), (w - 1, bh), colour, 2)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (16, pad + 15 + i * lh), FONT,
                        0.56 if i == 0 else 0.44,
                        colour if i == 0 else (232, 232, 232),
                        2 if i == 0 else 1, cv2.LINE_AA)

        if len(events) > 1:
            tag = f"+{len(events) - 1} more active"
            cv2.putText(img, tag, (w - 150, 18), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        return img

    def draw_hud(self, img, snap, t, duration, n_events):
        h, w = img.shape[:2]
        strip = 34
        cv2.rectangle(img, (0, h - strip), (w, h), (16, 16, 16), -1)

        n_tracks = len(snap.get("tracks", [])) if snap else 0
        cp = snap.get("changepoint", 0.0) if snap else 0.0
        txt = (f"{self.scene.camera_id}   t={t:6.2f}s / {duration:.1f}s   "
               f"tracks={n_tracks:2d}   motion-changepoint z={cp:5.2f}   "
               f"incidents={n_events}")
        cv2.putText(img, txt, (10, h - strip + 22), FONT, 0.45,
                    (205, 205, 205), 1, cv2.LINE_AA)

        if not snap or not snap.get("geometry_valid", True):
            cv2.putText(img, "GEOMETRY INVALID - camera moved", (w - 330, h - strip + 22),
                        FONT, 0.45, (60, 170, 240), 1, cv2.LINE_AA)

        # timeline of every incident in the clip, with a playhead
        bar_y, bar_h = h - strip - 12, 8
        cv2.rectangle(img, (10, bar_y), (w - 10, bar_y + bar_h), (34, 34, 34), -1)
        span = max(duration, 1e-6)
        for e in self.events:
            x = int(10 + (w - 20) * min(1.0, max(0.0, e["started_t"] / span)))
            c = TYPE_COLOUR.get(e["type"], (200, 200, 200))
            cv2.rectangle(img, (x - 2, bar_y - 3), (x + 2, bar_y + bar_h + 3), c, -1)
        px = int(10 + (w - 20) * min(1.0, max(0.0, t / span)))
        cv2.line(img, (px, bar_y - 6), (px, bar_y + bar_h + 6), (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # -- main --------------------------------------------------------------
    def render(self, src: str, dst: str | Path, max_seconds: float | None = None,
               progress=None) -> dict:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise FileNotFoundError(src)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (n / fps) if n else (max_seconds or 0)

        ok, probe = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"no frames in {src}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sized, out_scale = _fit(probe, self.out_long_side)
        oh, ow = sized.shape[:2]

        # the timeline was recorded in processing coordinates; map to output
        proc_scale = 1.0
        ph, pw = probe.shape[:2]
        m = max(ph, pw)
        if self.proc_long_side and m > self.proc_long_side:
            proc_scale = self.proc_long_side / m
        draw_scale = out_scale / max(proc_scale, 1e-9)

        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        codecs = ("VP80",) if dst.suffix.lower() == ".webm" else ("mp4v", "avc1", "MJPG")
        for fourcc in codecs:
            writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*fourcc), fps, (ow, oh))
            if writer.isOpened():
                break
            writer.release()
            writer = None
        if writer is None:
            cap.release()
            raise RuntimeError("no usable video writer codec")

        i = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if max_seconds and t > max_seconds:
                break

            img, _ = _fit(frame, self.out_long_side)
            img = self.draw_scene(img, draw_scale)

            snap = self.snapshot_at(t)
            active = self.active_events(t)
            hot: set[int] = set(self.implicated)
            for e in active:
                hot.update(e.get("track_ids") or [])
            if snap:
                self._remember_hot(snap, hot)
                img = self.draw_lanes(img, snap, draw_scale)
                img = self.draw_forecasts(img, snap, draw_scale, hot)
            img = self.draw_tracks(img, snap, draw_scale, hot)
            img = self.draw_lost_hot(img, snap, draw_scale, hot)
            img = self.draw_participants(img, t, draw_scale)
            if snap:
                img = self.draw_stationary(img, snap, draw_scale)
            img = self.draw_banner(img, active, t)
            img = self.draw_hud(img, snap, t, duration or t, len(self.events))

            writer.write(img)
            written += 1
            if progress and written % 200 == 0:
                progress(written, n)

        writer.release()
        cap.release()
        return {"output": str(dst), "frames": written, "fps": fps,
                "size": [ow, oh], "events_marked": len(self.events)}
