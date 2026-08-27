"""Bootstrap a camera's scene model from its own traffic.

The calibration screen lets a human draw corridors in about ninety seconds. This
script does the first draft for them, so the human is *correcting* a proposal
rather than starting from an empty frame.

How it works
------------
Watch a stretch of nominal traffic, track everything, and keep the trajectories
that actually went somewhere. Each trajectory has a direction; cluster those
directions on the unit circle and every dense cluster is a stream of traffic
moving the same way -- which is exactly what a corridor is. The polygon is then
fitted as a ribbon along that cluster's flow axis (see ``ribbon_polygon``), and
the direction is the cluster's mean heading.

What this can and cannot do
---------------------------
It can find where vehicles drive and which way they normally go. It **cannot**
know which way they are legally *allowed* to go -- if every vehicle in the
sample is violating, the system will happily learn the violation as normal. It
also cannot see lane markings, so solid-versus-dashed boundaries and junction
exclusion zones must be set by a human.

That is why the output is explicitly a draft, flagged as such in the JSON, and
why the calibration UI exists. Stating this limit is the difference between an
honest tool and one that quietly launders an assumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netra.config import load_config          # noqa: E402
from netra.detect import Detector, MOTORISED_CLASSES  # noqa: E402
from netra.geometry import unit               # noqa: E402
from netra.scene import Corridor, SceneModel, Zone    # noqa: E402
from netra.track import ByteTracker           # noqa: E402


def collect_trajectories(video: str, detector: Detector, seconds: float,
                         analysis_fps: float, resize_long_side: int):
    """Run detection+tracking over a warm-up stretch and return trajectories."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    stride = max(1, int(round(fps / analysis_fps)))
    tracker = ByteTracker()

    trajectories: dict[int, list[tuple[float, float]]] = {}
    classes: dict[int, int] = {}
    frame_size = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        t = (i - 1) / fps
        if t > seconds:
            break
        if (i - 1) % stride:
            continue
        if resize_long_side:
            h, w = frame.shape[:2]
            m = max(h, w)
            if m > resize_long_side:
                s = resize_long_side / m
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
        frame_size = (frame.shape[1], frame.shape[0])
        dets = detector.detect_array(frame)
        for tr in tracker.update(dets, t):
            if tr.cls not in MOTORISED_CLASSES:
                continue
            trajectories.setdefault(tr.track_id, []).append(tr.ground_point)
            classes[tr.track_id] = tr.cls
    cap.release()
    return trajectories, frame_size, i


def cluster_directions(trajectories: dict[int, list], min_points: int,
                       min_span_px: float, n_bins: int = 36):
    """Bin trajectory headings on the unit circle; dense bins become corridors."""
    entries = []
    for tid, pts in trajectories.items():
        if len(pts) < min_points:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        span = float(np.hypot(*(arr[-1] - arr[0])))
        if span < min_span_px:
            continue
        d = unit(arr[-1] - arr[0])
        if not d.any():
            continue
        angle = float(np.arctan2(d[1], d[0]))
        entries.append({"id": tid, "pts": arr, "dir": d, "angle": angle, "span": span})

    if not entries:
        return []

    bins: dict[int, list] = {}
    for e in entries:
        b = int(((e["angle"] + np.pi) / (2 * np.pi)) * n_bins) % n_bins
        bins.setdefault(b, []).append(e)

    # merge each bin with its immediate neighbours so a stream that straddles a
    # bin edge is not split into two half-strength corridors
    merged: dict[int, list] = {}
    for b, items in bins.items():
        group = list(items)
        for nb in ((b - 1) % n_bins, (b + 1) % n_bins):
            group.extend(bins.get(nb, []))
        merged[b] = group

    used: set[int] = set()
    clusters = []
    for b in sorted(merged, key=lambda k: -len(merged[k])):
        group = [e for e in merged[b] if e["id"] not in used]
        if len(group) < 2:
            continue
        for e in group:
            used.add(e["id"])
        mean_dir = unit(np.mean([e["dir"] for e in group], axis=0))
        pts = np.vstack([e["pts"] for e in group])
        clusters.append({"dir": mean_dir, "points": pts, "n_tracks": len(group)})
    return clusters


def ribbon_polygon(points: np.ndarray, direction: np.ndarray, dilate_px: float,
                   frame_size, n_bins: int = 10) -> list:
    """Fit a road-shaped ribbon along the cluster's flow axis.

    A convex hull is the obvious thing to reach for and it is wrong here: hulls
    of a receding road swallow the sky, the footpath and the opposing
    carriageway, and overlapping corridors make corridor assignment meaningless.

    Instead, work in the flow frame. Project the ground points onto the
    direction of travel (``u``) and its perpendicular (``v``), slice along
    ``u``, and take a robust lateral spread within each slice. Stitching the
    slice edges gives a band that follows the road and narrows with perspective,
    which is what a lane actually looks like from a traffic camera.
    """
    if len(points) < 8:
        return []
    u = unit(direction)
    v = np.array([-u[1], u[0]], dtype=np.float64)

    s = points @ u                     # along-flow coordinate
    w = points @ v                     # lateral coordinate

    lo, hi = np.percentile(s, [2, 98])
    if hi - lo < 20:
        return []
    edges = np.linspace(lo, hi, n_bins + 1)

    left, right = [], []
    for i in range(n_bins):
        m = (s >= edges[i]) & (s <= edges[i + 1])
        if m.sum() < 3:
            continue
        wl, wr = np.percentile(w[m], [12, 88])
        half = max((wr - wl) / 2.0, 6.0) + dilate_px
        mid = (wl + wr) / 2.0
        centre_s = (edges[i] + edges[i + 1]) / 2.0
        left.append(centre_s * u + (mid - half) * v)
        right.append(centre_s * u + (mid + half) * v)

    if len(left) < 2:
        return []

    poly = np.vstack([np.asarray(left), np.asarray(right)[::-1]])
    out = []
    for p in poly:
        x, y = float(p[0]), float(p[1])
        if frame_size:
            x = float(np.clip(x, 0, frame_size[0] - 1))
            y = float(np.clip(y, 0, frame_size[1] - 1))
        out.append([round(x, 1), round(y, 1)])
    return out


def disjoin(corridors, frame_size):
    """Force corridors to be mutually exclusive.

    Overlapping corridors are the single biggest source of false wrong-way
    alerts: a vehicle travelling legally in one stream sits inside the polygon
    of the opposing one, and its heading is then compared against the wrong
    reference vector.

    Contested pixels are awarded to whichever corridor they sit *deepest*
    inside, measured with a distance transform, and each corridor's polygon is
    re-extracted from the resulting exclusive mask. The result is a partition of
    the road surface, which is what a set of lanes actually is.
    """
    if len(corridors) < 2 or not frame_size:
        return corridors
    w, h = frame_size
    masks, dists = [], []
    for c in corridors:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.asarray(c.polygon, np.int32)], 255)
        masks.append(m)
        dists.append(cv2.distanceTransform(m, cv2.DIST_L2, 5))

    stack = np.stack(dists, axis=0)
    owner = np.argmax(stack, axis=0)
    covered = stack.max(axis=0) > 0

    out = []
    for i, c in enumerate(corridors):
        excl = ((owner == i) & covered).astype(np.uint8) * 255
        excl = cv2.morphologyEx(excl, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(excl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) < 400:
            continue
        approx = cv2.approxPolyDP(big, epsilon=0.008 * cv2.arcLength(big, True), closed=True)
        poly = [(round(float(p[0][0]), 1), round(float(p[0][1]), 1)) for p in approx]
        if len(poly) < 3:
            continue
        c.polygon = poly
        out.append(c)
    return out


def build_scene(args, clusters, frame_size) -> SceneModel:
    corridors = []
    kept = [c for c in clusters if c["n_tracks"] >= args.min_tracks]
    for i, c in enumerate(kept[: args.max_corridors]):
        poly = ribbon_polygon(c["points"], c["dir"], args.dilate_px, frame_size)
        if len(poly) < 3:
            continue
        corridors.append(Corridor(
            id=f"c{i + 1}",
            name=f"stream {i + 1} ({c['n_tracks']} tracks)",
            polygon=[tuple(p) for p in poly],
            direction=c["dir"],
            lanes=1,
        ))

    corridors = disjoin(corridors, frame_size)

    # Deliberately NOT marking any boundary as solid.
    #
    # The tempting default is "opposing corridors must be separated by a solid
    # line". On a real junction that is wrong, and it was measurably wrong here:
    # it turned every legitimate turn across the centre of a Cuttack junction
    # into a lane-violation alert. Whether a marking is solid is a fact about
    # paint on the road that this script cannot see, so it stays unset until a
    # human sets it in the calibration screen. An unset boundary raises no
    # alert; a wrongly-set one raises a stream of them.

    return SceneModel(
        camera_id=args.camera_id,
        name=args.name or args.camera_id,
        source=args.video,
        zone=args.zone,
        road_name=args.road_name,
        road_edge_id=args.road_edge_id,
        latitude=args.lat,
        longitude=args.lon,
        frame_size=frame_size,
        analysis_fps=args.analysis_fps,
        corridors=corridors,
        zones=[],
        notes=("DRAFT auto-calibration from observed traffic. Directions are the "
               "OBSERVED majority flow, not verified legal directions. Junction "
               "exclusion zones and solid/dashed lane boundaries still require "
               "human confirmation in the calibration screen."),
    )


def preview(scene: SceneModel, video: str, out_path: Path, resize_long_side: int):
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if resize_long_side:
        h, w = frame.shape[:2]
        m = max(h, w)
        if m > resize_long_side:
            s = resize_long_side / m
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    from netra.evidence import EvidenceWriter
    img = EvidenceWriter().annotate(frame, scene, [], None, scale=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--zone", default="")
    ap.add_argument("--road-name", default="")
    ap.add_argument("--road-edge-id", default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--analysis-fps", type=float, default=8.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--resize-long-side", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--min-span-px", type=float, default=45.0)
    ap.add_argument("--dilate-px", type=float, default=28.0)
    ap.add_argument("--max-corridors", type=int, default=4)
    ap.add_argument("--min-tracks", type=int, default=4,
                    help="drop clusters supported by fewer tracks than this")
    ap.add_argument("--out-dir", default="config/cameras")
    args = ap.parse_args()

    cfg = load_config()
    det = Detector(imgsz=args.imgsz, conf=0.10, device=args.device)
    det.warmup()
    print(f"[1/4] watching {args.seconds:.0f}s of {Path(args.video).name} "
          f"at {args.analysis_fps} fps, imgsz={args.imgsz}")

    trajectories, frame_size, frames = collect_trajectories(
        args.video, det, args.seconds, args.analysis_fps, args.resize_long_side)
    print(f"      {len(trajectories)} tracks over {frames} frames, frame_size={frame_size}")

    print("[2/4] clustering trajectory directions")
    clusters = cluster_directions(trajectories, args.min_points, args.min_span_px)
    clusters = [c for c in clusters if c["n_tracks"] >= args.min_tracks]
    for i, c in enumerate(clusters[: args.max_corridors]):
        ang = np.degrees(np.arctan2(c["dir"][0], -c["dir"][1])) % 360
        print(f"      stream {i + 1}: {c['n_tracks']:3d} tracks, heading {ang:5.1f} deg, "
              f"{len(c['points'])} points")
    if not clusters:
        print("      no usable trajectory clusters -- try a longer --seconds or "
              "a lower --min-span-px")
        return 1

    print("[3/4] building scene model")
    scene = build_scene(args, clusters, frame_size)

    out = Path(args.out_dir) / f"{args.camera_id}.json"
    scene.save(out)
    print(f"      saved -> {out}")

    print("[4/4] rendering preview")
    p = preview(scene, args.video, Path("reports") / f"calib_{args.camera_id}.jpg",
                args.resize_long_side)
    if p:
        print(f"      preview -> {p}")

    print("\nDRAFT. Open the calibration screen to confirm legal directions, add "
          "junction exclusion zones, and mark solid lane boundaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
