# Drone demo samples

Real output from this session's first genuine (non-synthetic) test of the drone
pipeline against real DJI footage — genuinely top-down/nadir, genuinely
near-static platform (hovering, not patrolling).

| File | What it is |
|---|---|
| `sample_raw_footage.mp4` | Unprocessed source segment (`50m_90d_morning_congkhuA_22_3_part006`), GPU-compressed from the original DJI recording, no analysis applied |
| `sample_physics_annotated.mp4` | The same footage after the full pipeline: detection, tracking, physics HUD |
| `sample_results.json` | The exact structured output behind that annotated video — every number on screen is traceable here |

## Pipeline behind this sample

- **Detector**: `dronefreak/visdrone-yolov8x` — a real VisDrone-fine-tuned YOLO
  checkpoint (HuggingFace), not the generic-COCO placeholder this project
  started with. `detector_finetuned: true` in the results JSON, not a claim
  made only in prose.
- **Tracker**: Ultralytics' native BoT-SORT (`botsort.yaml`, with ReID) —
  chosen over this project's earlier hand-rolled tracker specifically because
  its built-in motion compensation and appearance re-identification are
  well-suited to a near-static aerial platform with frequent occlusion
  (vehicles passing close together look far more overlapped from directly
  overhead than from an oblique angle).
- **Physics**: speed (px/s, honestly labelled, plus a class-width km/h
  *estimate* — never presented as calibrated), acceleration over a real time
  window (not single-frame differencing, which earlier work on this project's
  CCTV side already measured to be noise, not signal), momentum from
  per-class mass priors, full trajectory.
- **Queue / blockage**: corridor-free, since a moving/hovering drone has no
  calibrated corridors the way a fixed CCTV camera does — spatial clustering
  of simultaneously slow tracks for queue, sustained-stationary tracks for
  blockage. `queue_events` and `blockage_candidates` are visible on screen
  every frame, including when they're zero.

## What this sample does and doesn't show

This specific clip shows real, correctly-tracked traffic (14 concurrent
vehicles through a roundabout in the full result set) with **no queue or
blockage event** — genuinely free-flowing traffic, not a failure to detect
one. That is reported plainly rather than picking a more dramatic-looking
clip to feature instead. See the root `README.md` §7 for the fuller first-look
result set and any findings across all 16 segments processed.
