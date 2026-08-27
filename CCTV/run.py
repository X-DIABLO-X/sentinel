#!/usr/bin/env python
"""NETRA entry point.

    python run.py calibrate --video data/raw/clip.webm --camera-id CAM_01
    python run.py process   --camera CUTTACK_LINK_01 [--seconds 120] [--cpu]
    python run.py serve     [--port 8000]
    python run.py status

One command per thing a person actually wants to do. `process` runs the full
pipeline and writes incidents, evidence and a JSON report; `serve` puts the
dashboard up over whatever is already in the database.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_process(args) -> int:
    from netra.config import load_config
    from netra.db import IncidentStore
    from netra.location import RoadGraph
    from netra.pipeline import Pipeline
    from netra.scene import SceneModel

    cfg = load_config()
    if args.seconds:
        cfg["pipeline"]["max_seconds"] = args.seconds
    if args.fps:
        cfg["pipeline"]["analysis_fps"] = args.fps
    if args.imgsz:
        cfg["detector"]["imgsz"] = args.imgsz
        cfg["pipeline"]["resize_long_side"] = max(args.imgsz, 960)
    if args.cpu:
        cfg["detector"]["device"] = "cpu"
        if args.openvino:
            cfg["detector"]["backend"] = "openvino"
    if args.low_resource:
        # Judge/demo profile for ordinary CPU hardware. It deliberately
        # spends compute on temporal persistence rather than dense inference.
        cfg["detector"].update({
            "weights": str(ROOT / "yolo26n.pt"),
            "device": "cpu",
            "imgsz": 640,
            "aux_imgsz": 0,
        })
        cfg["pipeline"].update({"analysis_fps": 4.0, "resize_long_side": 960})
        if args.openvino:
            cfg["detector"]["backend"] = "openvino"

    path = Path(cfg["paths"]["cameras"]) / f"{args.camera}.json"
    if not path.exists():
        print(f"no camera config at {path}. Run:  python run.py calibrate --video ... "
              f"--camera-id {args.camera}")
        return 1
    scene = SceneModel.load(path)
    if args.video:
        scene.source = args.video

    graph_path = Path(cfg["paths"]["road_graph"])
    graph = RoadGraph(graph_path) if graph_path.exists() else None
    store = IncidentStore(cfg["paths"]["database"])

    pipe = Pipeline(scene, cfg, store=store, road_graph=graph,
                    evidence_root=cfg["paths"]["evidence"])
    print(f"processing {scene.source}")
    print(f"  camera={scene.camera_id} corridors={len(scene.corridors)} "
          f"device={pipe.detector.device} backend={pipe.detector.backend} "
          f"imgsz={pipe.detector.imgsz}")

    def show(p):
        print(f"    t={p['t']:>7.1f}s  frames={p['frames_analysed']:>5d}  "
              f"tracks={p['tracks']:>3d}  events={p['events']:>3d}", end="\r")

    pipe.run(progress=show)
    print()

    report = pipe.report()
    out = Path(cfg["paths"]["reports"]) / f"{scene.camera_id}_{pipe.run_id}.json"
    pipe.save_report(out)

    s = report["stats"]
    print(f"\n  analysed {s['frames_analysed']} frames of {s['video_seconds']}s video "
          f"in {s['wall_seconds']}s")
    print(f"  analysis {s['analysis_fps']} FPS   realtime factor {s['realtime_factor']}x   "
          f"detector p95 {s['detector_latency'].get('p95_ms', '?')} ms")
    print(f"  incidents {report['events_total']}  "
          f"({report['alerts_per_video_hour']} alerts / video-hour)")
    for k, v in sorted(report["events_by_type"].items()):
        print(f"    {k:32s} {v}")
    print(f"\n  report -> {out}")
    print(f"  dashboard: python run.py serve")
    return 0


def cmd_calibrate(args) -> int:
    script = ROOT / "scripts" / "autocalibrate.py"
    cmd = [sys.executable, str(script), "--video", args.video,
           "--camera-id", args.camera_id]
    for flag, val in (("--name", args.name), ("--zone", args.zone),
                      ("--road-name", args.road_name), ("--road-edge-id", args.road_edge_id)):
        if val:
            cmd += [flag, str(val)]
    if args.lat is not None:
        cmd += ["--lat", str(args.lat)]
    if args.lon is not None:
        cmd += ["--lon", str(args.lon)]
    cmd += ["--seconds", str(args.seconds), "--imgsz", str(args.imgsz)]
    return subprocess.call(cmd)


def cmd_serve(args) -> int:
    import uvicorn
    from netra.config import load_config
    cfg = load_config()
    host = args.host or cfg["api"]["host"]
    port = args.port or cfg["api"]["port"]
    print(f"NETRA dashboard -> http://{host}:{port}/")
    uvicorn.run("netra.api:app", host=host, port=port, reload=False, log_level="warning")
    return 0


def cmd_status(args) -> int:
    from netra.config import load_config
    from netra.db import IncidentStore
    from netra.scene import load_all
    cfg = load_config()
    cams = load_all(cfg["paths"]["cameras"])
    store = IncidentStore(cfg["paths"]["database"])
    print("cameras:")
    for cid, sc in cams.items():
        print(f"  {cid:22s} corridors={len(sc.corridors)} zones={len(sc.zones)} "
              f"metric={sc.has_metric_scale} src={Path(sc.source).name}")
    print("\nincidents:", json.dumps(store.summary(), indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="netra", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="run the pipeline over a camera's video")
    p.add_argument("--camera", required=True)
    p.add_argument("--video", default=None, help="override the camera's source")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--fps", type=float, default=None, help="analysis fps")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("--openvino", action="store_true", help="with --cpu, use OpenVINO IR")
    p.add_argument("--low-resource", action="store_true",
                   help="YOLO26n, 4 Hz and one detector pass on CPU")
    p.set_defaults(func=cmd_process)

    c = sub.add_parser("calibrate", help="bootstrap a camera scene model")
    c.add_argument("--video", required=True)
    c.add_argument("--camera-id", required=True)
    c.add_argument("--name", default="")
    c.add_argument("--zone", default="")
    c.add_argument("--road-name", default="")
    c.add_argument("--road-edge-id", default=None)
    c.add_argument("--lat", type=float, default=None)
    c.add_argument("--lon", type=float, default=None)
    c.add_argument("--seconds", type=float, default=60)
    c.add_argument("--imgsz", type=int, default=1280)
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("serve", help="run the API and dashboard")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    st = sub.add_parser("status", help="what is configured and what has been found")
    st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
