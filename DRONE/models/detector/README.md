# Drone detector — model choice, status, and fine-tune plan

## Current status: NOT YET FINE-TUNED

There is **no trained drone detector in this repository.** `visdrone_config.yaml`
has `weights: null`, and that is accurate, not a placeholder we forgot to fill in.

What happens today when you run the pipeline:

- `scripts/detect_drone.py` sees `weights: null`
- it loads a **generic COCO-pretrained YOLO** so the pipeline is mechanically runnable
- it prints a multi-line banner on **every single run** saying the results are not meaningful
- every results JSON carries `"detector_finetuned": false` and
  `"placeholder_detector": true`
- `GET /api/health` reports `detector_finetuned: false`

**No accuracy number for the drone detector appears anywhere in this repo, because
none has been measured.** When one is measured it goes in
`visdrone_config.yaml -> finetune_plan.measured_metrics`, with the date and the
split it was measured on.

---

## Why VisDrone, and not BMD-45 / UVH-26

The CCTV subsystem is trained against eye-level road-traffic data. Reusing that
model on drone footage would be a domain mismatch severe enough to be a design
error, not just a small accuracy loss.

**The aerial view removes the side profile of a vehicle.** That is the whole
argument, and it is a statement about the image, not about the network.

From 80–120 m AGL at nadir or steep oblique, a car presents as:

- a roof rectangle, plus maybe a windscreen sliver
- no wheels
- no bumper, no grille, no number plate
- no headlights or tail-lights
- an aspect ratio governed by the drone's yaw, not by the vehicle's heading
- a length of roughly 25–50 px on a 4K frame

An eye-level detector's learned evidence for "car" is wheels, windscreen rake,
grille, and a side-profile silhouette. **None of those pixels exist in the
aerial image.** The mismatch is not a distribution shift you can close with
augmentation; the discriminative features are physically absent.

Three further mismatches compound it:

| | Eye-level CCTV (BMD-45 / UVH-26) | Aerial (VisDrone) |
|---|---|---|
| Object scale | large, varies 10× across one frame with range | uniformly tiny, near-constant across the frame |
| Canonical orientation | there is an "up"; vehicles are upright | no canonical up; drone yaw is arbitrary |
| Occlusion mode | vehicles occlude each other front-to-back | almost none between vehicles; buildings/trees occlude instead |
| Dominant failure | scale extremes | small-object recall |

The last row is why `imgsz: 1280` is in the config rather than the usual 640.

**VisDrone2019-DET** is the right fine-tuning target because it is the same
sensor geometry as our task — consumer drone, 60–120 m, urban traffic, oblique
and nadir — and because its class list already contains `tricycle` and
`awning-tricycle`, which matters for Indian urban traffic in a way that no
Western aerial dataset covers.

### Consequence: two separate detector models

CCTV and DRONE run **separate detector weights**. There is no shared checkpoint
and no plan for one. That is a deliberate architectural decision, already
settled — see the top-level project README. The pipeline code is shared in
shape only.

---

## Fine-tune plan (planned — NOT YET EXECUTED)

Recorded so it is auditable. `finetune_plan.executed: false` in the config.

**Stage 1 — VisDrone base.**
`yolo11m.pt` → VisDrone2019-DET-train, 100 epochs, `imgsz 1280`, batch 8.
Aerial-appropriate augmentation: `degrees: 180` and `flipud: 0.5`, both of
which are nonsense at eye level and correct overhead, because a nadir view has
no canonical orientation. Validate on VisDrone2019-DET-val, and report
**AP-small specifically**, not just mAP50 — mAP50 over all sizes will flatter
the model, since the objects we actually care about are the small ones.

**Stage 2 — site adaptation.**
Once real footage from the target site arrives, hand-label a slice of it and
run a second, shorter fine-tune. VisDrone is Chinese urban traffic. The
expected gaps for Indian traffic: autorickshaw density, two-wheeler density,
lane-sharing / non-lane-following behaviour, and heavier mixed-traffic
occlusion. The `tricycle`, `awning-tricycle` and `motor` classes will need the
most correction.

**Stage 3 — measure and publish.**
Fill in `measured_metrics` with the date and the exact split. Nothing gets
quoted in the submission until it is in that block.

---

## Where a checkpoint goes when it exists

```
DRONE/models/detector/visdrone_yolo.pt
```

then set in `models/detector/visdrone_config.yaml`:

```yaml
weights: models/detector/visdrone_yolo.pt
```

and in `config/drone_config.yaml`:

```yaml
detector:
  weights: models/detector/visdrone_yolo.pt
```

Paths are resolved relative to `DRONE/` (the `PROJECT_ROOT` in
`scripts/config.py`). No code changes are required — the placeholder banner
disappears on its own and `detector_finetuned` flips to `true`.

---

## Thermal

Thermal has exactly one sanctioned role in this system: a **total-darkness
vehicle-presence trigger**. It never classifies, never contributes motion or
speed, and never feeds severity. See `scripts/thermal_presence.py` for the
reasoning. It is a stub today.
