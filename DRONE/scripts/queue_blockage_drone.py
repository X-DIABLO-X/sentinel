"""Corridor-free queue and blockage detection for the DRONE subsystem.

CCTV's queue engine (``CCTV/netra/events/queueing.py``) and blockage engine
(``CCTV/netra/events/blockage.py``) both lean on a **calibrated corridor
polygon** — a hand-drawn or auto-fit (``scripts/autocalibrate.py``) region of
the fixed camera's own field of view that says "this is a lane, this is its
direction, this is its occupancy denominator". A hovering or repositioned
drone has no such thing: there is no fixed frame to calibrate a corridor
against, and building one per clip would be exactly the kind of invented
precision this project's honesty rules exist to prevent.

This module is a **simpler, honestly-scoped, corridor-free equivalent** of
the same two ideas, operating entirely on GMC-compensated reference-frame
track positions:

Queue
-----
CCTV's definition — density + slowness + coverage, sustained — survives
without a corridor by replacing "coverage of the corridor polygon" with
**spatial clustering**: several vehicles simultaneously slow, close together
in reference-frame pixels. Proximity is measured in units of the scene's own
median vehicle-box diagonal (not an absolute pixel count), so the same
config works whether the drone is at 60 m or 120 m AGL. Free-flow speed is
learned from the clip's own traffic, the same "never an absolute threshold"
principle CCTV's queue engine states in its own module docstring.

**There is no ground truth to validate this against** — same documented
limitation as CCTV's own queue engine (see that module's docstring: speed is
"compared against a learned per-corridor baseline, never an absolute
threshold", and its own precision/recall has never been measured either,
only used as one signal among several). A queue candidate here is reported
with its full evidence trail, not as a validated detection.

Blockage
--------
A single track stationary for a sustained duration. Where CCTV distinguishes
"impeding traffic" from "legally stopped" using flow-drop against a
corridor's own reference and occupancy of the corridor polygon, this module
can only ask a strictly weaker, corridor-free question: **did any other
vehicle pass close by while this one was stationary, and if so, did it slow
down doing it?** Three honest outcomes fall out of that:

* no other vehicle ever came near it -> ``stationary_undetermined`` — there
  is no geometric basis to call this blockage vs. parked, and this module
  says so rather than guessing.
* vehicles passed near it and at least one visibly slowed doing so ->
  ``blockage_candidate``, with the slowing neighbour(s) as evidence.
* vehicles passed near it and none slowed -> ``stationary_likely_parked``.

This is a materially weaker discriminator than CCTV's corridor+flow-drop
approach and is not claimed to be as good — it is the honest ceiling of what
geometry alone can support without a corridor. The project's own stated
principle for the CCTV blockage engine applies identically here: "zero
validated recall is a different, honest claim from broken."
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from physics_drone import TrackSample, _Arr, windowed_speed, position_at  # noqa: E402

__all__ = ["detect_queues", "detect_blockages", "median_vehicle_diagonal"]


# --------------------------------------------------------------------------
# shared geometry helper
# --------------------------------------------------------------------------

def median_vehicle_diagonal(track_samples: dict[int, Sequence[TrackSample]],
                            vehicle_classes: set[int]) -> float:
    """Median box diagonal (px) over every vehicle-class sample in the clip.

    The scale unit for "close together" in both engines below — using a
    fraction of the vehicles' own apparent size rather than a raw pixel
    count is what makes the thresholds meaningful regardless of altitude,
    zoom, or resolution.
    """
    diags = []
    for samples in track_samples.values():
        for s in samples:
            if int(s.cls) not in vehicle_classes:
                continue
            x1, y1, x2, y2 = s.ref_box
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if w > 0 and h > 0:
                diags.append(float(np.hypot(w, h)))
    if not diags:
        return 0.0
    return float(np.median(diags))


def _union_find_clusters(ids: list[int], positions: np.ndarray, radius: float) -> list[list[int]]:
    """Connected components of ``ids`` under the "within radius" relation."""
    n = len(ids)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(positions[i] - positions[j])) <= radius:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(ids[i])
    return list(groups.values())


# --------------------------------------------------------------------------
# Queue detection
# --------------------------------------------------------------------------

def detect_queues(track_samples: dict[int, list[TrackSample]],
                  class_names: dict[int, str], vehicle_classes: set[int],
                  kcfg, qcfg) -> dict[str, Any]:
    """Corridor-free spatial-clustering queue detection. See module docstring.

    Returns ``{"events": [...], "diagnostics": {...}}``. ``events`` is a list
    of queue-candidate dicts, each with a full evidence trail. An empty list
    is a real, reportable result ("no queue candidate found in this clip"),
    not an error.
    """
    if not qcfg.enabled or not track_samples:
        return {"events": [], "diagnostics": {"enabled": bool(qcfg.enabled), "n_tracks": len(track_samples)}}

    arrs: dict[int, _Arr] = {
        tid: _Arr.from_samples(s) for tid, s in track_samples.items()
        if s and int(s[-1].cls) in vehicle_classes
    }
    if not arrs:
        return {"events": [], "diagnostics": {"reason": "no_vehicle_class_tracks"}}

    diag = median_vehicle_diagonal(track_samples, vehicle_classes)
    proximity_px = qcfg.proximity_diagonals * diag if diag > 0 else 80.0

    # -- free-flow baseline: causal windowed-speed samples across the clip --
    baseline_samples: list[float] = []
    for tid, arr in arrs.items():
        for t in arr.t:
            sp = windowed_speed(arr, float(t), kcfg.speed_window_s)
            if sp is not None and sp > 1.0:
                baseline_samples.append(sp)
    baseline = None
    if len(baseline_samples) >= qcfg.min_baseline_samples:
        baseline = float(np.percentile(baseline_samples, qcfg.baseline_percentile))
    slow_thresh = (baseline * qcfg.slow_ratio) if baseline else qcfg.slow_abs_px_s

    t_max = max(float(arr.t[-1]) for arr in arrs.values())
    grid = np.arange(0.0, t_max + qcfg.sample_dt_s, qcfg.sample_dt_s)

    # active_events: list of dicts (member_ids:set, start_t, last_t, sizes:list[(t,size,speed)])
    active_events: list[dict[str, Any]] = []
    finished_events: list[dict[str, Any]] = []

    for t in grid:
        t = float(t)
        slow_ids: list[int] = []
        slow_pos: list[tuple[float, float]] = []
        slow_speed: dict[int, float] = {}
        for tid, arr in arrs.items():
            pos = position_at(arr, t, tol=max(qcfg.sample_dt_s, 0.5))
            if pos is None:
                continue
            sp = windowed_speed(arr, t, kcfg.speed_window_s)
            if sp is None or sp >= slow_thresh:
                continue
            slow_ids.append(tid)
            slow_pos.append(pos)
            slow_speed[tid] = sp

        clusters = _union_find_clusters(slow_ids, np.array(slow_pos) if slow_pos else np.empty((0, 2)),
                                        proximity_px)
        clusters_big = [set(c) for c in clusters if len(c) >= qcfg.min_vehicles]

        # close events that have gone silent past the gap tolerance
        still_active = []
        for ev in active_events:
            if t - ev["last_t"] > qcfg.gap_tolerance_s:
                finished_events.append(ev)
            else:
                still_active.append(ev)
        active_events = still_active

        for cluster in clusters_big:
            best_ev, best_overlap = None, 0.0
            for ev in active_events:
                recent = ev["member_snapshots"][-1] if ev["member_snapshots"] else set()
                union_n = len(cluster | recent)
                overlap = (len(cluster & recent) / union_n) if union_n else 0.0
                if overlap > best_overlap:
                    best_ev, best_overlap = ev, overlap
            avg_speed = float(np.mean([slow_speed[i] for i in cluster]))
            if best_ev is not None and best_overlap >= qcfg.merge_overlap:
                best_ev["last_t"] = t
                best_ev["all_members"] |= cluster
                best_ev["member_snapshots"].append(cluster)
                best_ev["size_history"].append((t, len(cluster)))
                best_ev["speed_history"].append(avg_speed)
            else:
                active_events.append({
                    "start_t": t, "last_t": t,
                    "all_members": set(cluster),
                    "member_snapshots": [cluster],
                    "size_history": [(t, len(cluster))],
                    "speed_history": [avg_speed],
                })

    finished_events.extend(active_events)

    events_out = []
    qualifying_events = [ev for ev in finished_events if ev["last_t"] - ev["start_t"] >= qcfg.min_duration_s]
    for ev in qualifying_events:
        duration = ev["last_t"] - ev["start_t"]
        sizes = [n for _, n in ev["size_history"]]
        names = set()
        for tid in ev["all_members"]:
            s = track_samples.get(tid)
            if s:
                names.add(class_names.get(int(s[-1].cls), str(s[-1].cls)))
        events_out.append({
            "start_t": round(ev["start_t"], 2),
            "end_t": round(ev["last_t"], 2),
            "duration_s": round(duration, 2),
            "track_ids": sorted(ev["all_members"]),
            "peak_vehicle_count": int(max(sizes)) if sizes else 0,
            "first_vehicle_count": int(sizes[0]) if sizes else 0,
            "mean_cluster_speed_px_s": round(float(np.mean(ev["speed_history"])), 2) if ev["speed_history"] else None,
            "class_names": sorted(names),
            "evidence": (
                f"{len(ev['all_members'])} distinct tracks clustered within "
                f"{qcfg.proximity_diagonals:.1f} vehicle-diagonals "
                f"({proximity_px:.0f}px) of each other, each below the slow "
                f"threshold ({slow_thresh:.1f} px/s"
                + (f", {qcfg.slow_ratio:.0%} of the clip's own {baseline:.1f} px/s "
                   "free-flow baseline" if baseline else ", fallback absolute threshold "
                   "-- too few free-flow samples in this clip to learn a baseline")
                + f"), sustained (with gaps <= {qcfg.gap_tolerance_s}s tolerated) for "
                f"{duration:.1f}s."
            ),
            "validation_note": (
                "corridor-free spatial-clustering heuristic; no ground truth exists "
                "to measure this engine's precision or recall, the same documented "
                "limitation as CCTV's own calibrated queue engine."
            ),
        })

    events_out.sort(key=lambda e: e["start_t"])
    return {
        "events": events_out,
        "diagnostics": {
            "enabled": True,
            "n_vehicle_tracks_considered": len(arrs),
            "free_flow_baseline_px_s": round(baseline, 2) if baseline else None,
            "baseline_sample_count": len(baseline_samples),
            "slow_threshold_px_s": round(slow_thresh, 2),
            "median_vehicle_diagonal_px": round(diag, 1),
            "proximity_px": round(proximity_px, 1),
        },
    }


# --------------------------------------------------------------------------
# Blockage detection
# --------------------------------------------------------------------------

def detect_blockages(track_samples: dict[int, list[TrackSample]],
                     class_names: dict[int, str], vehicle_classes: set[int],
                     kcfg, bcfg) -> dict[str, Any]:
    """Corridor-free single-track blockage detection. See module docstring.

    Returns ``{"events": [...], "diagnostics": {...}}``. Every event carries
    an explicit ``classification`` of ``blockage_candidate`` /
    ``stationary_likely_parked`` / ``stationary_undetermined`` — never a bare
    "blockage" label without that qualification.
    """
    if not bcfg.enabled or not track_samples:
        return {"events": [], "diagnostics": {"enabled": bool(bcfg.enabled), "n_tracks": len(track_samples)}}

    arrs: dict[int, _Arr] = {
        tid: _Arr.from_samples(s) for tid, s in track_samples.items()
        if s and int(s[-1].cls) in vehicle_classes
    }
    if not arrs:
        return {"events": [], "diagnostics": {"reason": "no_vehicle_class_tracks"}}

    diag = median_vehicle_diagonal(track_samples, vehicle_classes)
    proximity_px = bcfg.proximity_diagonals * diag if diag > 0 else 100.0

    events_out = []
    for tid, arr in arrs.items():
        cls = int(track_samples[tid][-1].cls)
        # -- stationary flag per sample (causal windowed speed) --
        flags = []
        for t in arr.t:
            sp = windowed_speed(arr, float(t), kcfg.speed_window_s)
            flags.append(sp is not None and sp <= kcfg.stationary_speed_px_s)

        # -- maximal True runs, tolerating short gaps --
        runs: list[tuple[int, int]] = []
        run_start = None
        gap_count = 0
        for i, f in enumerate(flags):
            if f:
                if run_start is None:
                    run_start = i
                gap_count = 0
            else:
                if run_start is not None:
                    gap_t = arr.t[i] - arr.t[i - 1] if i > 0 else 0.0
                    gap_count += gap_t
                    if gap_count > bcfg.gap_tolerance_s:
                        runs.append((run_start, i - 1))
                        run_start = None
                        gap_count = 0
        if run_start is not None:
            runs.append((run_start, len(flags) - 1))

        for (i0, i1) in runs:
            t0, t1 = float(arr.t[i0]), float(arr.t[i1])
            duration = t1 - t0
            if duration < bcfg.min_stationary_s:
                continue

            pos = (float(np.median(arr.gx[i0:i1 + 1])), float(np.median(arr.gy[i0:i1 + 1])))

            # -- neighbours: other vehicle tracks that came within proximity --
            neighbours_observed = []
            neighbours_judgeable = []
            evidence = []
            for other_id, other_arr in arrs.items():
                if other_id == tid:
                    continue
                # sample the neighbour's own timestamps that overlap the run
                lo = np.searchsorted(other_arr.t, t0 - bcfg.gap_tolerance_s)
                hi = np.searchsorted(other_arr.t, t1 + bcfg.gap_tolerance_s)
                if hi <= lo:
                    continue
                sub_t = other_arr.t[lo:hi]
                sub_x = other_arr.gx[lo:hi]
                sub_y = other_arr.gy[lo:hi]
                dists = np.hypot(sub_x - pos[0], sub_y - pos[1])
                if dists.size == 0 or float(dists.min()) > proximity_px:
                    continue
                closest_idx = int(np.argmin(dists))
                t_close = float(sub_t[closest_idx])
                neighbours_observed.append(other_id)

                speed_at_approach = windowed_speed(other_arr, t_close, kcfg.speed_window_s)
                # own recent free speed: speeds while this neighbour was NOT
                # near the stationary vehicle, in a window around the pass
                far_mask = dists > proximity_px * 1.5
                far_speeds = []
                for k in np.where(far_mask)[0]:
                    sp = windowed_speed(other_arr, float(sub_t[k]), kcfg.speed_window_s)
                    if sp is not None:
                        far_speeds.append(sp)
                if not far_speeds:
                    # fall back to the neighbour's whole-track median speed
                    for tt in other_arr.t:
                        sp = windowed_speed(other_arr, float(tt), kcfg.speed_window_s)
                        if sp is not None:
                            far_speeds.append(sp)
                own_recent_free_speed = float(np.median(far_speeds)) if far_speeds else None

                # A neighbour only lets us judge obstruction if it was
                # actually moving freely at some point -- another mutually
                # stationary vehicle (e.g. a fellow queue member) has no
                # "free speed" to slow down from, so it cannot provide
                # evidence either way and must not be counted as if it could.
                if (own_recent_free_speed is None
                        or own_recent_free_speed < bcfg.min_neighbor_free_speed_px_s):
                    continue
                neighbours_judgeable.append(other_id)

                if (speed_at_approach is not None
                        and speed_at_approach <= bcfg.slowdown_fraction * own_recent_free_speed):
                    evidence.append({
                        "neighbor_track_id": other_id,
                        "own_recent_free_speed_px_s": round(own_recent_free_speed, 2),
                        "speed_at_closest_approach_px_s": round(speed_at_approach, 2),
                        "drop_fraction": round(1.0 - speed_at_approach / own_recent_free_speed, 3),
                        "closest_approach_t": round(t_close, 2),
                    })

            if not neighbours_judgeable:
                classification = "stationary_undetermined"
                if neighbours_observed:
                    note = (
                        f"{len(neighbours_observed)} nearby track(s) were observed but none had "
                        "a usable free-flow speed to judge against (e.g. a fellow stationary "
                        "vehicle) -- there is no geometric basis to distinguish blockage from a "
                        "vehicle legally parked off to the side, so this is reported as "
                        "undetermined rather than guessed."
                    )
                else:
                    note = (
                        "no other vehicle track came within "
                        f"{bcfg.proximity_diagonals:.1f} vehicle-diagonals ({proximity_px:.0f}px) "
                        "of this stationary vehicle during the interval -- there is no "
                        "geometric basis to distinguish blockage from a vehicle legally "
                        "parked off to the side, so this is reported as undetermined "
                        "rather than guessed."
                    )
            elif evidence:
                classification = "blockage_candidate"
                note = (
                    f"{len(evidence)} of {len(neighbours_judgeable)} judgeable passing "
                    "vehicle(s) slowed measurably near this stationary vehicle's position -- "
                    "consistent with (not proof of) an obstruction."
                )
            else:
                classification = "stationary_likely_parked"
                note = (
                    f"{len(neighbours_judgeable)} vehicle(s) passed within "
                    f"{bcfg.proximity_diagonals:.1f} vehicle-diagonals without a "
                    "measurable slowdown -- consistent with a vehicle parked clear "
                    "of moving traffic, not proof of it."
                )

            events_out.append({
                "track_id": tid,
                "cls_name": class_names.get(cls, str(cls)),
                "start_t": round(t0, 2),
                "end_t": round(t1, 2),
                "stationary_s": round(duration, 2),
                "position_ref": [round(pos[0], 1), round(pos[1], 1)],
                "classification": classification,
                "neighbours_observed": len(neighbours_observed),
                "neighbours_judgeable": len(neighbours_judgeable),
                "evidence": evidence,
                "note": note,
                "validation_note": (
                    "corridor-free geometric heuristic; no ground truth exists to "
                    "measure this engine's precision or recall against. See module "
                    "docstring -- this is deliberately a weaker discriminator than "
                    "CCTV's calibrated-corridor blockage engine, not a claim of "
                    "equal capability."
                ),
            })

    events_out.sort(key=lambda e: e["start_t"])
    return {
        "events": events_out,
        "diagnostics": {
            "enabled": True,
            "n_vehicle_tracks_considered": len(arrs),
            "median_vehicle_diagonal_px": round(diag, 1),
            "proximity_px": round(proximity_px, 1),
        },
    }


if __name__ == "__main__":   # pragma: no cover - manual smoke check
    # Synthetic scene: 4 cars stationary and clustered (queue), plus one
    # isolated stationary car with a passing neighbour that slows (blockage)
    # and one that doesn't (parked). No footage needed.
    import config as drone_config
    cfg = drone_config.load_config()

    samples: dict[int, list[TrackSample]] = {}
    # queue cluster: 4 cars near (100,100)..(160,100), near-zero speed, 10s
    for i in range(4):
        tid = i + 1
        samples[tid] = [
            TrackSample(t=t, ref_box=(100 + i * 20, 100, 140 + i * 20, 140),
                       px_box=(100 + i * 20, 100, 140 + i * 20, 140), score=0.8, cls=3)
            for t in np.arange(0.0, 10.0, 0.2)
        ]

    # isolated stationary car at (500,500), stationary 10s
    samples[10] = [
        TrackSample(t=t, ref_box=(500, 500, 540, 540), px_box=(500, 500, 540, 540),
                   score=0.8, cls=3)
        for t in np.arange(0.0, 10.0, 0.2)
    ]
    # a fast neighbour that passes close to (500,500) and slows down there
    pts = []
    x = 300.0
    for t in np.arange(0.0, 10.0, 0.2):
        if 4.0 <= t <= 6.0:
            x += 3.0     # slows near the stationary car
        else:
            x += 40.0
        pts.append((t, x))
    samples[11] = [
        TrackSample(t=t, ref_box=(x - 20, 495, x + 20, 535), px_box=(x - 20, 495, x + 20, 535),
                   score=0.8, cls=3) for t, x in pts
    ]

    names = {3: "car"}
    veh = {2, 3, 4, 5, 6, 7, 8, 9}
    q = detect_queues(samples, names, veh, cfg.kinematics, cfg.queue)
    b = detect_blockages(samples, names, veh, cfg.kinematics, cfg.blockage)
    import json
    print("QUEUES:", json.dumps(q, indent=2)[:2000])
    print("BLOCKAGES:", json.dumps(b, indent=2)[:3000])
