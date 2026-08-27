"""SQLite incident store and operator workflow.

SQLite rather than PostgreSQL, deliberately. The brief asks for reproducibility;
a reviewer should be able to clone the repo, run one command and have the whole
system stand up. A database that lives in a single file and needs no server is
worth more here than PostGIS features we would not use -- the road graph is a
few dozen edges and networkx handles it in memory.

The schema is built around three ideas the brief explicitly asks for:

* every incident carries its **evidence** and the numbers that fired it
* every incident has an **owner** and a **status**, so it can be worked
* every incident records the **model run** that produced it, so a number in the
  report can be traced back to a commit, a checkpoint and a threshold set
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id     TEXT PRIMARY KEY,
    name          TEXT,
    zone          TEXT,
    road_name     TEXT,
    road_edge_id  TEXT,
    latitude      REAL,
    longitude     REAL,
    source        TEXT,
    analysis_fps  REAL,
    frame_w       INTEGER,
    frame_h       INTEGER,
    has_homography INTEGER DEFAULT 0,
    config_json   TEXT,
    created_at    REAL
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id           TEXT PRIMARY KEY,
    detector         TEXT,
    detector_backend TEXT,
    device           TEXT,
    imgsz            INTEGER,
    tracker          TEXT,
    engine_version   TEXT,
    threshold_hash   TEXT,
    config_json      TEXT,
    started_at       REAL,
    finished_at      REAL,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id               INTEGER PRIMARY KEY,
    run_id           TEXT,
    camera_id        TEXT,
    corridor_id      TEXT,
    event_type       TEXT,
    label            TEXT,
    started_t        REAL,
    detected_t       REAL,
    ended_t          REAL,
    duration         REAL,
    detection_delay  REAL,
    onset_method     TEXT,
    onset_recovered_s REAL,
    confidence       REAL,
    severity         REAL,
    severity_label   TEXT,
    priority         REAL,
    status           TEXT,
    needs_verification INTEGER,
    recommended_action TEXT,
    explanation      TEXT,
    track_ids        TEXT,
    triggers_json    TEXT,
    severity_json    TEXT,
    location_json    TEXT,
    evidence_json    TEXT,
    wall_clock       REAL,
    created_at       REAL,
    updated_at       REAL,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);

CREATE INDEX IF NOT EXISTS idx_inc_cam    ON incidents(camera_id);
CREATE INDEX IF NOT EXISTS idx_inc_type   ON incidents(event_type);
CREATE INDEX IF NOT EXISTS idx_inc_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_inc_prio   ON incidents(priority DESC);
CREATE INDEX IF NOT EXISTS idx_inc_cam_run ON incidents(camera_id, run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS status_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  INTEGER,
    old_status   TEXT,
    new_status   TEXT,
    actor        TEXT,
    reason       TEXT,
    comment      TEXT,
    changed_at   REAL,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);

CREATE TABLE IF NOT EXISTS assignments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  INTEGER,
    owner        TEXT,
    team         TEXT,
    assigned_at  REAL,
    acknowledged_at REAL,
    completed_at REAL,
    FOREIGN KEY (incident_id) REFERENCES incidents(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT,
    camera_id  TEXT,
    key        TEXT,
    value      REAL,
    detail     TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_metric_cam_key_time
    ON metrics(camera_id, key, created_at DESC);
"""

# The operator workflow the brief asks for. Rejection is a first-class outcome:
# a rejected incident with a reason is training data for the next iteration.
STATUSES = ["detected", "verified", "assigned", "responding", "resolved", "closed", "rejected"]

REJECTION_REASONS = [
    "tracking error", "legal turn", "camera shake", "false detection",
    "normal signal queue", "legitimate stop", "duplicate", "other",
]


class IncidentStore:
    def __init__(self, path: str | Path = "netra.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI executes sync endpoints in a worker pool.  A SQLite
        # connection may be shared with check_same_thread=False, but operations
        # on that connection still must not overlap: simultaneous cursors caused
        # intermittent IndexError/InterfaceError and an empty dashboard.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # -- cameras -----------------------------------------------------------
    def upsert_camera(self, scene) -> None:
        fw, fh = (scene.frame_size or (None, None))
        self.conn.execute(
            """INSERT INTO cameras (camera_id, name, zone, road_name, road_edge_id,
                   latitude, longitude, source, analysis_fps, frame_w, frame_h,
                   has_homography, config_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(camera_id) DO UPDATE SET
                   name=excluded.name, zone=excluded.zone, road_name=excluded.road_name,
                   road_edge_id=excluded.road_edge_id, latitude=excluded.latitude,
                   longitude=excluded.longitude, source=excluded.source,
                   analysis_fps=excluded.analysis_fps, frame_w=excluded.frame_w,
                   frame_h=excluded.frame_h, has_homography=excluded.has_homography,
                   config_json=excluded.config_json""",
            (scene.camera_id, scene.name, scene.zone, scene.road_name, scene.road_edge_id,
             scene.latitude, scene.longitude, scene.source, scene.analysis_fps, fw, fh,
             int(scene.has_metric_scale), json.dumps(scene.to_dict()), time.time()),
        )
        self.conn.commit()

    def cameras(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
        return [dict(r) for r in rows]

    # -- model runs --------------------------------------------------------
    def start_run(self, run_id: str, info: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO model_runs
               (run_id, detector, detector_backend, device, imgsz, tracker,
                engine_version, threshold_hash, config_json, started_at, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, info.get("detector"), info.get("backend"), info.get("device"),
             info.get("imgsz"), info.get("tracker"), info.get("engine_version"),
             info.get("threshold_hash"), json.dumps(info.get("config", {})),
             time.time(), info.get("notes", "")),
        )
        self.conn.commit()

    def finish_run(self, run_id: str) -> None:
        self.conn.execute("UPDATE model_runs SET finished_at=? WHERE run_id=?",
                          (time.time(), run_id))
        self.conn.commit()

    def runs(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM model_runs ORDER BY started_at DESC").fetchall()
        return [dict(r) for r in rows]

    # -- incidents ---------------------------------------------------------
    def insert_incident(self, event, scene, run_id: str, location: dict,
                        recommendation: str, evidence: dict | None = None) -> int:
        now = time.time()
        d = event.to_dict()
        cur = self.conn.execute(
            """INSERT INTO incidents
               (run_id, camera_id, corridor_id, event_type, label, started_t, detected_t,
                ended_t, duration, detection_delay, onset_method, onset_recovered_s,
                confidence, severity, severity_label, priority, status,
                needs_verification, recommended_action, explanation, track_ids,
                triggers_json, severity_json, location_json, evidence_json,
                wall_clock, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, scene.camera_id, event.corridor_id, event.type, event.label,
             d["started_t"], d["detected_t"], d["ended_t"], d["duration"],
             d["detection_delay"], event.onset_method, event.onset_recovered_s,
             event.confidence, event.severity, event.severity_label, event.priority,
             event.status, int(event.needs_verification), recommendation,
             event.explain(), json.dumps(event.track_ids),
             json.dumps(event.triggers), json.dumps(event.severity_parts),
             json.dumps(location), json.dumps(evidence or {}), now, now, now),
        )
        self.conn.commit()
        incident_id = int(cur.lastrowid)
        self._history(incident_id, None, event.status, "system", "created")
        return incident_id

    def update_incident(self, incident_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [time.time(), incident_id]
        with self._lock:
            self.conn.execute(f"UPDATE incidents SET {cols}, updated_at=? WHERE id=?", vals)
            self.conn.commit()

    def set_status(self, incident_id: int, new_status: str, actor: str = "operator",
                   reason: str = "", comment: str = "") -> dict:
        if new_status not in STATUSES:
            raise ValueError(f"unknown status {new_status!r}; expected one of {STATUSES}")
        with self._lock:
            row = self.conn.execute("SELECT status FROM incidents WHERE id=?",
                                    (incident_id,)).fetchone()
            if row is None:
                raise KeyError(f"incident {incident_id} not found")
            old = row["status"]
            self.conn.execute("UPDATE incidents SET status=?, updated_at=? WHERE id=?",
                              (new_status, time.time(), incident_id))
            self.conn.commit()
            self._history(incident_id, old, new_status, actor, reason, comment)
        return {"id": incident_id, "old_status": old, "new_status": new_status}

    def _history(self, incident_id, old, new, actor, reason="", comment="") -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO status_history
                   (incident_id, old_status, new_status, actor, reason, comment, changed_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (incident_id, old, new, actor, reason, comment, time.time()),
            )
            self.conn.commit()

    def assign(self, incident_id: int, owner: str, team: str = "") -> dict:
        with self._lock:
            self.conn.execute(
                "INSERT INTO assignments (incident_id, owner, team, assigned_at) VALUES (?,?,?,?)",
                (incident_id, owner, team, time.time()),
            )
            self.conn.commit()
        return self.set_status(incident_id, "assigned", actor=owner,
                               reason="assigned", comment=team)

    # -- queries -----------------------------------------------------------
    def incidents(self, camera_id: str | None = None, status: str | None = None,
                  event_type: str | None = None, limit: int = 500,
                  latest_only: bool = False,
                  consolidated_only: bool = False) -> list[dict]:
        q = "SELECT * FROM incidents i WHERE 1=1"
        args: list[Any] = []
        if latest_only:
            # Metrics are written for every completed analysis, including a
            # clean run with zero incidents. Choosing the latest run only from
            # the incidents table made such a clean run invisible and kept an
            # older false alarm alive forever on the dashboard.
            q += (" AND i.run_id=COALESCE((SELECT m.run_id FROM metrics m "
                  "WHERE m.camera_id=i.camera_id AND m.key='system.video_seconds' "
                  "ORDER BY m.created_at DESC LIMIT 1), "
                  "(SELECT i2.run_id FROM incidents i2 WHERE i2.camera_id=i.camera_id "
                  "ORDER BY i2.created_at DESC LIMIT 1))")
        if consolidated_only:
            q += (" AND (i.event_type!='collision_candidate' OR i.id=("
                  "SELECT i3.id FROM incidents i3 WHERE i3.camera_id=i.camera_id "
                  "AND i3.run_id=i.run_id AND i3.event_type='collision_candidate' "
                  "ORDER BY i3.priority DESC, i3.detected_t DESC LIMIT 1))")
        if camera_id:
            q += " AND camera_id=?"; args.append(camera_id)
        if status:
            q += " AND status=?"; args.append(status)
        if event_type:
            q += " AND event_type=?"; args.append(event_type)
        q += " ORDER BY priority DESC, detected_t DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        return [self._hydrate(dict(r)) for r in rows]

    def incident(self, incident_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM incidents WHERE id=?",
                                    (incident_id,)).fetchone()
            if row is None:
                return None
            history = self.conn.execute(
                "SELECT * FROM status_history WHERE incident_id=? ORDER BY changed_at",
                (incident_id,)).fetchall()
            assignments = self.conn.execute(
                "SELECT * FROM assignments WHERE incident_id=? ORDER BY assigned_at",
                (incident_id,)).fetchall()
        d = self._hydrate(dict(row))
        d["history"] = [dict(r) for r in history]
        d["assignments"] = [dict(r) for r in assignments]
        return d

    @staticmethod
    def _hydrate(d: dict) -> dict:
        for key, target in (("triggers_json", "triggers"), ("severity_json", "severity_parts"),
                            ("location_json", "location"), ("evidence_json", "evidence"),
                            ("track_ids", "track_ids")):
            raw = d.pop(key, None)
            try:
                d[target] = json.loads(raw) if raw else ({} if target != "track_ids" else [])
            except (json.JSONDecodeError, TypeError):
                d[target] = {} if target != "track_ids" else []
        d["needs_verification"] = bool(d.get("needs_verification"))
        return d

    # -- metrics -----------------------------------------------------------
    def record_metric(self, run_id: str, camera_id: str, key: str,
                      value: float, detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO metrics (run_id, camera_id, key, value, detail, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, camera_id, key, float(value), json.dumps(detail or {}), time.time()),
        )
        self.conn.commit()

    def metrics(self, run_id: str | None = None) -> list[dict]:
        with self._lock:
            if run_id:
                rows = self.conn.execute(
                    "SELECT * FROM metrics WHERE run_id=? ORDER BY created_at",
                    (run_id,)).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM metrics ORDER BY created_at DESC LIMIT 500").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d.get("detail") or "{}")
            except json.JSONDecodeError:
                d["detail"] = {}
            out.append(d)
        return out

    # -- summary -----------------------------------------------------------
    def summary(self, latest_only: bool = False) -> dict:
        scope = (" WHERE i.run_id=(SELECT i2.run_id FROM incidents i2 "
                 "WHERE i2.camera_id=i.camera_id "
                 "ORDER BY i2.created_at DESC LIMIT 1)" if latest_only else "")
        and_scope = scope.replace(" WHERE ", " AND ", 1) if scope else ""
        total = self.conn.execute(
            f"SELECT COUNT(*) c FROM incidents i{scope}").fetchone()["c"]
        by_type = {r["event_type"]: r["c"] for r in self.conn.execute(
            f"SELECT event_type, COUNT(*) c FROM incidents i{scope} GROUP BY event_type")}
        by_sev = {r["severity_label"]: r["c"] for r in self.conn.execute(
            f"SELECT severity_label, COUNT(*) c FROM incidents i{scope} GROUP BY severity_label")}
        by_status = {r["status"]: r["c"] for r in self.conn.execute(
            f"SELECT status, COUNT(*) c FROM incidents i{scope} GROUP BY status")}
        open_count = self.conn.execute(
            "SELECT COUNT(*) c FROM incidents i WHERE status NOT IN "
            f"('closed','rejected','resolved'){and_scope}"
        ).fetchone()["c"]
        needs = self.conn.execute(
            "SELECT COUNT(*) c FROM incidents i WHERE needs_verification=1 "
            f"AND status='detected'{and_scope}"
        ).fetchone()["c"]
        return {
            "total_incidents": total,
            "open_incidents": open_count,
            "awaiting_verification": needs,
            "by_type": by_type,
            "by_severity": by_sev,
            "by_status": by_status,
            "cameras": len(self.cameras()),
        }
