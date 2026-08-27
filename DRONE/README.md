# DRONE — hover-based escalation/verification pipeline

## What this is

The moving-camera half of the submission. CCTV (`../CCTV/`) is the persistent
backbone: 82 fixed cameras, always on. This subsystem is dispatched to an
**already-confirmed** incident, hovers over it, and gives the overhead view a
pole-mounted camera cannot — queue tail length beyond the CCTV field of view,
blockage extent, lane occupancy. It is not a patrol platform; see
`scripts/hover_mode.py` for the three independent reasons (DGCA regulation,
airframe endurance, ego-motion drift) `PATROL` raises `NotImplementedError`
instead of pretending to support it.

## The one honest fact that governs everything else here

**No drone footage exists yet.** Everything below was built and tested against
synthetic frames, a synthetic pan sequence, and code-level unit checks — not
against a real aerial clip, because there is no real aerial clip in this repo.
The pipeline shape is final; the two things that are still placeholders
(the detector weights, the road-plane calibration) are placeholders because
the inputs they need don't exist yet, not because the code is unfinished.

## What works today, mechanically, on any aerial clip

Run this right now against any `.mp4`/`.avi` you have — dashcam, YouTube
drone footage, anything with a moving camera — and it will run end to end:

```
cd DRONE
pip install -r requirements.txt
python scripts/pipeline_drone.py path/to/any_clip.mp4 --max-frames 300 -v
```

What actually executes:

- **Frame quality gate** — real. Laplacian-variance blur check, brightness
  bounds. Rejected frames are counted and reported, not silently skipped.
- **Detection** — real inference runs, using a generic COCO-pretrained model
  because `models/detector/visdrone_config.yaml` has `weights: null`. It
  prints a loud banner every run and stamps `detector_finetuned: false` /
  `placeholder_detector: true` on every result. **The class labels and box
  quality it produces on an aerial frame are not meaningful** — COCO has
  never seen a nadir/oblique viewing angle. See
  `models/detector/README.md` for why VisDrone is the correct fine-tune
  target and why the CCTV side's eye-level models (BMD-45/UVH-26) would be
  the wrong choice even if reused.
- **Global Motion Compensation (`scripts/gmc.py`)** — real, and the one piece
  of this subsystem that is genuinely novel work rather than scaffolding.
  Masks detected-vehicle boxes out, ORB-matches the remaining static
  background frame-to-frame, RANSAC-estimates a homography, chains it back to
  a reference frame. Rejects and reports (never fabricates) a homography when
  matches/inliers are too few or the transform is physically implausible for
  a hovering drone. Has its own synthetic self-check:
  `python scripts/gmc.py`.
- **Tracking (`scripts/track_drone.py`)** — real, running on GMC-compensated
  (reference-frame) coordinates, so a parked car reads as stationary
  regardless of drone drift. ByteTrack-shaped two-stage association; greedy
  IoU matching rather than the CCTV side's Hungarian solver — a real,
  documented accuracy trade to avoid a `scipy` dependency for a
  single-incident, few-track workload (see the module docstring for the
  honest trade-off statement, not a silent downgrade).
- **Kinematics** — pixel-space speed (post-GMC) is always computed. Metric
  speed (km/h) is only ever produced when `road_plane.homography` is
  calibrated, which it is not yet — every metric field is `null` with an
  explicit `metric_reason: "no_road_plane_homography"` rather than a guessed
  scale.
- **Results JSON** — written to `DRONE/results/`, carrying a `provenance`
  block that answers, for every run: is the detector fine-tuned, is
  telemetry available, is the road plane calibrated, what mode is active.
- **API (`scripts/api.py`)** — a real FastAPI process on port **8011**
  (CCTV owns 8000; they never share a port or a process).
  `GET /api/health` reports `detector_finetuned: false` today, honestly.

Verify the core modules import cleanly:

```
cd DRONE
python -c "import sys; sys.path.insert(0,'scripts'); import gmc, hover_mode, config"
```

## What needs real footage before it means anything

| Piece | Status today | What unblocks it |
|---|---|---|
| Detector accuracy | Generic COCO model, not fine-tuned | VisDrone fine-tune (`models/detector/README.md` — planned, `finetune_plan.executed: false`), then a second site-adaptation stage on real footage |
| Road-plane calibration | `road_plane.homography: null` — all metric speeds report `null` | 4+ ground correspondences with known metric separation, taken from real footage of a real site |
| Telemetry-assisted georeferencing | `scripts/telemetry_ingest.py` — signature frozen, `load_telemetry()` returns `None` unconditionally | A real flight log (GPS+IMU+gimbal) from a real flight; no format is parsed yet |
| Thermal presence trigger | `scripts/thermal_presence.py` — stub, returns `[]` unconditionally | Thermal sensor + footage; scope is permanently limited to a total-darkness presence trigger, never classification/motion/severity — see that file's docstring |
| GMC accuracy | Unit-tested on synthetic pans only | Real hover footage with any ground-truth (survey markers, known-speed pass) to measure RMSE against |

**No accuracy number for any of the above appears anywhere in this repo**,
because none has been measured. The literature band this approach targets —
homography-based UAV speed RMSE roughly 0.53–16 km/h, 9.7–15% MAE for
calibration-free monocular — is recorded in `scripts/gmc.py`'s docstring as
the expectation to be checked against, not as a result already achieved.
When real numbers exist they go in
`models/detector/visdrone_config.yaml -> finetune_plan.measured_metrics`
and this table, dated.

## Layout

```
config/drone_config.yaml       all tunables; every path relative to PROJECT_ROOT (DRONE/)
models/detector/                VisDrone class map, fine-tune plan, weights (none yet)
scripts/config.py               YAML -> dataclass config loader, no absolute paths
scripts/gmc.py                  Global Motion Compensation (the core module)
scripts/hover_mode.py           HOVER implemented / PATROL raises NotImplementedError
scripts/telemetry_ingest.py     direct-georeferencing ingest — signature frozen, returns None
scripts/thermal_presence.py     total-darkness presence-only stub
scripts/detect_drone.py         Ultralytics YOLO wrapper, placeholder-aware
scripts/track_drone.py          tracking on GMC-compensated coordinates
scripts/pipeline_drone.py       orchestrates the full run, writes results JSON
scripts/api.py                  FastAPI service, port 8011 (CCTV owns 8000)
results/                        pipeline output JSON lands here
```

## Why two separate detector models (CCTV vs DRONE)

Settled architecture decision, not an open question. A nadir/oblique aerial
view removes a vehicle's side profile entirely — no wheels, no windscreen
rake, no grille, a roof rectangle 25-50 px on a side from 80-120 m AGL. The
features an eye-level detector keys on are physically absent from the image;
reusing the CCTV model would be a domain-mismatch design error, not a small
accuracy loss. Full argument in `models/detector/README.md`.
