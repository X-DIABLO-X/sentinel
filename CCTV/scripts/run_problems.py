"""Run every clip in data/problems through NETRA and render reviewable output.

Two folders, two very different regimes, handled honestly rather than uniformly:

**Accidents/** -- short clips, mostly moving or arbitrary viewpoints, each
containing a collision. There is no fixed camera geometry here, so corridor-based
reasoning (wrong-way, queue, blockage) is meaningless and is *switched off*
rather than left to produce noise. What runs is the part that needs no map: the
pairwise trajectory-conflict test and the global motion change-point detector.

**Traffic/** -- longer fixed-camera and aerial footage. Here the scene model is
worth building, so each clip is auto-calibrated first and the full event set is
active.

For every clip the script writes:
    results/<group>/<name>_annotated.mp4   what the system saw and decided
    results/<group>/<name>.json            incidents, triggers, timings
    results/index.html                     one page to review all of it
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.config import load_config                      # noqa: E402
from netra.db import IncidentStore                        # noqa: E402
from netra.detect import Detector                         # noqa: E402
from netra.pipeline import Pipeline                       # noqa: E402
from netra.render import VideoAnnotator                   # noqa: E402
from netra.scene import Corridor, SceneModel              # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from autocalibrate import (build_scene, cluster_directions,   # noqa: E402
                           collect_trajectories)


def write_json(path: Path, payload: dict) -> None:
    """Write a report completely, then publish it with one replace.

    A killed batch must leave either the previous valid report or the new valid
    report, never a half-written JSON file that the scorer silently skips.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pending.replace(path)


class _CalArgs:
    """Minimal stand-in for autocalibrate's argparse namespace."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "width": w, "height": h,
            "duration": (n / fps) if fps > 0 else 0.0}


def make_uncalibrated_scene(cam_id: str, src: str, info: dict, group: str) -> SceneModel:
    """A scene with no road map.

    Every corridor-dependent engine degrades to silence, which is the correct
    behaviour: without knowing which way a lane runs, a wrong-way claim would be
    invented rather than measured.
    """
    return SceneModel(
        camera_id=cam_id, name=cam_id, source=src, zone=group,
        road_name="(uncalibrated clip)",
        frame_size=(info["width"], info["height"]),
        corridors=[], zones=[],
        notes=("Uncalibrated clip: no corridor geometry available, so wrong-way, "
               "queue, lane-crossing and blockage reasoning are disabled. Active "
               "detectors are pairwise trajectory conflict and global motion "
               "change-point, neither of which requires a road map."),
    )


def calibrate(video: str, cam_id: str, detector: Detector, group: str,
              seconds: float, imgsz: int, long_side: int) -> SceneModel | None:
    try:
        traj, frame_size, _ = collect_trajectories(video, detector, seconds, 8.0, long_side)
        clusters = cluster_directions(traj, min_points=6, min_span_px=40.0)
        clusters = [c for c in clusters if c["n_tracks"] >= 3]
        if not clusters:
            return None
        args = _CalArgs(camera_id=cam_id, name=cam_id, zone=group, road_name="",
                        road_edge_id=None, lat=None, lon=None, analysis_fps=8.0,
                        video=video, dilate_px=26.0, max_corridors=4, min_tracks=3)
        return build_scene(args, clusters, frame_size)
    except Exception:
        return None


def saved_calibration_for(cam_cfg: Path, video_stem: str) -> Path | None:
    """Find geometry for the same video even when its group was renamed."""
    if cam_cfg.exists():
        return cam_cfg
    matches = sorted(cam_cfg.parent.glob(f"*_{video_stem}.json"))
    return matches[0] if matches else None


def process_one(path: Path, group: str, cfg: dict, detector: Detector,
                store: IncidentStore, out_dir: Path, results_root: Path,
                do_calibrate: bool, max_seconds: float | None,
                render: bool = True, dump_candidates: bool = False,
                reuse_calibration: bool = False, progress=None) -> dict:
    info = probe(path)
    cam_id = f"{group.upper()}_{path.stem}"
    src = str(path)
    long_side = int(cfg["pipeline"]["resize_long_side"])

    row: dict = {"group": group, "file": path.name, "camera_id": cam_id,
                 "video": info, "calibrated": False}
    t0 = time.perf_counter()

    cam_cfg = ROOT / "config" / "cameras" / f"{cam_id}.json"
    scene = None
    saved_cfg = saved_calibration_for(cam_cfg, path.stem) if reuse_calibration else None
    if do_calibrate and saved_cfg is not None:
        try:
            scene = SceneModel.load(saved_cfg)
            scale = min(1.0, long_side / max(info["width"], info["height"]))
            target_size = (int(round(info["width"] * scale)),
                           int(round(info["height"] * scale)))
            scene = scene.scaled_to(target_size)
            # Geometry is reusable across a renamed set; operational identity
            # is not. Keep incidents and dashboard rows namespaced to this run.
            scene.camera_id = cam_id
            scene.name = cam_id
            scene.zone = group
            row["calibration_source"] = str(saved_cfg.relative_to(ROOT))
        except Exception:
            scene = None
    if do_calibrate and scene is None:
        if progress:
            progress({"phase": "calibrating", "percent": 5,
                      "message": "Learning traffic motion corridors"})
        scene = calibrate(src, cam_id, detector, group,
                          min(30.0, info["duration"] or 30.0),
                          int(cfg["detector"]["imgsz"]), long_side)
        if scene is not None:
            row["calibration_source"] = "auto"
    if scene is None:
        scene = make_uncalibrated_scene(cam_id, src, info, group)
    else:
        scene.source = src
        row["calibrated"] = True
    row["corridors"] = len(scene.corridors)

    scene.save(cam_cfg)

    run_cfg = json.loads(json.dumps(cfg))
    if not render:
        run_cfg.setdefault("render", {})["record_timeline"] = False
    if max_seconds:
        run_cfg["pipeline"]["max_seconds"] = max_seconds

    pipe = Pipeline(scene, run_cfg, detector=detector, store=store,
                    evidence_root=str(ROOT / "evidence"))
    if progress:
        progress({"phase": "analysing", "percent": 15,
                  "message": "Detecting, tracking and reasoning over events"})

    duration = max(float(info.get("duration") or 0.0), 1e-6)

    def analysis_progress(item):
        if progress:
            fraction = min(1.0, float(item.get("t", 0.0)) / duration)
            progress({"phase": "analysing", "percent": 15 + round(60 * fraction),
                      "message": (f"Analysed {item.get('t', 0):.1f}s; "
                                  f"{item.get('tracks', 0)} active tracks; "
                                  f"{item.get('events', 0)} events")})

    pipe.run(progress=analysis_progress)
    # One accident per clip: report the strongest candidate, not the
    # first one to pass a gate.
    dropped = pipe.consolidate_collisions()
    rep = pipe.report()
    row["collision_candidates_dropped"] = dropped

    out_dir.mkdir(parents=True, exist_ok=True)

    # Measuring the false-alarm rate needs the incident report, not a re-encoded
    # 4K video, and rendering costs several times the analysis itself. Only the
    # video is optional here -- the report is always written, because it is what
    # the results index and the evaluation script read.
    if render:
        if progress:
            progress({"phase": "rendering", "percent": 76,
                      "message": "Rendering annotated review video"})
        annotator = VideoAnnotator(
            scene=scene,
            timeline=pipe.timeline,
            events=[e.to_dict() for e in pipe.events],
            proc_long_side=long_side,
            out_long_side=1280,
            banner_hold_s=4.0,
        )
        # Browser-native output: mp4v files are valid but Chromium often cannot
        # decode them, presenting an endless loading spinner to the operator.
        video_out = out_dir / f"{path.stem}_annotated.webm"
        try:
            def render_progress(written, total):
                if progress:
                    fraction = min(1.0, written / max(total, 1))
                    progress({"phase": "rendering",
                              "percent": 76 + round(23 * fraction),
                              "message": f"Rendered {written} of {total} frames"})

            rinfo = annotator.render(src, video_out, max_seconds=max_seconds,
                                     progress=render_progress)
            row["annotated_video"] = str(video_out.relative_to(results_root))
            row["render"] = rinfo
        except Exception as exc:
            row["render_error"] = f"{type(exc).__name__}: {exc}"
    else:
        row["render"] = {"skipped": True}

    if dump_candidates:
        cands = pipe.conflict_candidates()
        write_json(out_dir / f"{path.stem}_candidates.json", {
            "clip": path.stem,
            "crash_free": group.lower().startswith("traffic"),
            "video": info,
            "analysis_long_side": long_side,
            "candidates": cands,
        })
        row["candidates_dumped"] = len(cands)

    report_path = out_dir / f"{path.stem}.json"
    write_json(report_path, rep)
    row["report"] = str(report_path.relative_to(results_root))

    row["events_total"] = rep["events_total"]
    row["events_by_type"] = rep["events_by_type"]
    row["events_by_severity"] = rep["events_by_severity"]
    row["alerts_per_video_hour"] = rep["alerts_per_video_hour"]
    row["stats"] = rep["stats"]
    row["events"] = [{
        "type": e["type"], "label": e["label"],
        "started_t": e["started_t"], "detected_t": e["detected_t"],
        "detection_delay": e["detection_delay"],
        "onset_recovered_s": e["onset_recovered_s"],
        "confidence": e["confidence"], "severity": e["severity"],
        "severity_label": e["severity_label"],
        "explanation": e["explanation"],
        "needs_verification": e["needs_verification"],
    } for e in rep["events"]]
    row["wall_seconds"] = round(time.perf_counter() - t0, 1)
    if progress:
        progress({"phase": "complete", "percent": 100,
                  "message": (f"Complete: {row['events_total']} incident"
                              f"{'s' if row['events_total'] != 1 else ''}")})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="data/problems")
    ap.add_argument("--results", default="results")
    ap.add_argument("--flat-results", action="store_true",
                    help="write clip reports directly under --results; use this "
                         "for a single selected group such as ProblemSet")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="analysis fps; 0 = every frame at the source rate")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conf", type=float, default=None,
                    help="detector confidence floor (default: config value)")
    ap.add_argument("--track-low", type=float, default=None,
                    help="ByteTrack low-confidence association floor; lower "
                         "with --conf during a measured recall experiment")
    ap.add_argument("--weights", default=None,
                    help="detector checkpoint; defaults to yolo12m for accident recall")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--only", default=None, help="substring filter on filename")
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--dump-candidates", action="store_true",
                    help="write every conflict candidate with its physical "
                         "measurements, for fitting weights on labelled clips")
    ap.add_argument("--no-render", action="store_true",
                    help="analyse only; skip annotated-video output")
    ap.add_argument("--calibrate", dest="calibrate", action="store_true",
                    default=None,
                    help="force auto-calibration on (all engines run)")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="force auto-calibration off: collision and "
                         "change-point only, as used for the accident sets")
    ap.add_argument("--reuse-calibration", action="store_true",
                    help="reuse an existing camera JSON when available; avoids "
                         "silently replacing reviewed lane/corridor geometry")
    args = ap.parse_args()

    cfg = load_config()
    cfg["pipeline"]["analysis_fps"] = args.fps
    cfg["pipeline"]["resize_long_side"] = args.long_side
    cfg["detector"]["imgsz"] = args.imgsz
    cfg["detector"]["device"] = args.device
    if args.conf is not None:
        cfg["detector"]["conf"] = args.conf
    if args.track_low is not None:
        cfg["tracker"]["low_thresh"] = args.track_low
    cfg["pipeline"]["warmup_seconds"] = 1.5      # clips are short

    problems = Path(args.problems)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    groups = args.groups or [d.name for d in sorted(problems.iterdir()) if d.is_dir()]
    detector = Detector(weights=args.weights or "yolo26m.pt",
                        imgsz=args.imgsz, conf=cfg["detector"]["conf"],
                        device=args.device)
    detector.warmup()
    store = IncidentStore(cfg["paths"]["database"])

    print(f"detector: {detector.weights} imgsz={detector.imgsz} device={detector.device}")

    rows = []
    expected_reports: list[Path] = []
    for group in groups:
        gdir = problems / group
        out_dir = results if args.flat_results else results / group
        files = sorted([p for p in gdir.glob("*.mp4")
                        if not args.only or args.only in p.name])
        # Accident clips have no usable fixed geometry; traffic clips do.
        # Folder-name matching is a poor way to choose an analysis mode --
        # a new clip set called anything else silently gets the wrong one.
        # The flag wins when given; the name is only the fallback.
        do_cal = (args.calibrate if args.calibrate is not None
                  else "accident" not in group.lower())
        mode = "auto-calibrated" if do_cal else "uncalibrated: collision + change-point only"

        # Short accident clips need a denser sample rate and a shorter warm-up:
        # the change-point detector needs ~12 samples before its z-score means
        # anything, and on a 4-second clip an 8 Hz cadence with a 1.5 s warm-up
        # leaves almost no usable window. Traffic clips are long enough to prefer
        # the cheaper cadence and a longer warm-up for baseline learning.
        gcfg = json.loads(json.dumps(cfg))
        gcfg["pipeline"]["analysis_fps"] = args.fps
        if do_cal:
            gcfg["pipeline"]["warmup_seconds"] = 2.0
        else:
            gcfg["pipeline"]["warmup_seconds"] = 0.5
            gcfg["signals"]["cp_window"] = 6

        print(f"\n=== {group}: {len(files)} clips ({mode}, "
              f"{gcfg['pipeline']['analysis_fps']:.0f} analysis fps) ===")
        for k, p in enumerate(files, 1):
            expected_reports.append(out_dir / f"{p.stem}.json")
            print(f"[{k}/{len(files)}] {p.name} ... ", end="", flush=True)
            try:
                # Release cached GPU blocks between clips. A 6 GB card
                # fragments over a long batch and fell over with an
                # out-of-memory error nine clips in, losing the whole run.
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                row = process_one(p, group, gcfg, detector, store,
                                  out_dir, results,
                                  do_cal, args.max_seconds,
                                  render=not args.no_render,
                                  dump_candidates=args.dump_candidates,
                                  reuse_calibration=args.reuse_calibration)
                rows.append(row)
                types = ", ".join(f"{v}x{k2.replace('_', ' ')}"
                                  for k2, v in sorted(row["events_by_type"].items())) or "none"
                print(f"{row['events_total']} incidents ({types})  "
                      f"[{row['wall_seconds']}s]")
            except Exception as exc:
                print(f"FAILED {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
                rows.append({"group": group, "file": p.name,
                             "error": f"{type(exc).__name__}: {exc}"})

                # A CUDA or cuDNN failure poisons the context: every later clip
                # in the batch then fails too, silently, and the run finishes
                # "successfully" having processed a third of the data. That is
                # exactly how several measurements in this project were taken on
                # partial results without anyone noticing. Rebuild the detector
                # so the batch survives, and say so loudly.
                msg = f"{type(exc).__name__}: {exc}".lower()
                if "cuda" in msg or "cudnn" in msg or "out of memory" in msg:
                    print("  GPU context lost -- rebuilding the detector before "
                          "continuing, so the rest of the batch is not silently "
                          "skipped")
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        detector = Detector(
                            weights=detector.weights, imgsz=detector.imgsz,
                            conf=detector.conf, device=detector.device)
                        detector.warmup()
                        print("  detector rebuilt")
                    except Exception as rebuild_exc:
                        print(f"  could not rebuild the detector: {rebuild_exc}")
                        raise

    missing_reports = [str(p) for p in expected_reports if not p.is_file()]
    failed_rows = [r for r in rows if r.get("error")]
    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detector": detector.describe(),
        "analysis_fps": args.fps,
        "clips": len(rows),
        "total_incidents": sum(r.get("events_total", 0) for r in rows),
        "output_validation": {
            "expected_reports": len(expected_reports),
            "written_reports": len(expected_reports) - len(missing_reports),
            "missing_reports": missing_reports,
            "failed_clips": [r.get("file") for r in failed_rows],
            "complete": not missing_reports and not failed_rows,
        },
        "rows": rows,
    }
    write_json(results / "summary.json", summary)
    print(f"\nsummary -> {results / 'summary.json'}")
    print(f"clips={len(rows)}  incidents={summary['total_incidents']}")
    if missing_reports or failed_rows:
        print("BATCH INCOMPLETE: see output_validation in summary.json")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
