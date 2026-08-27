# NETRA — Road Incident Intelligence

**Network for Event Tracking, Response & Analysis**
ELCIA Smart City Drone-AI Challenge 2026 · Track 1: Smart Mobility & Road Incident Intelligence

> The model recognises objects. The system recognises incidents.

NETRA turns fixed traffic-camera video into explainable, evidence-backed incidents:
wrong-side movement, wrong lane crossing, queues, blockages and suspected
collision-related disruption — each with the numbers that produced it, a severity
derived from observable impact, a location, a recommended response and an
operator workflow.

One neural network runs continuously. Everything after the tracker is geometry,
temporal statistics and deterministic state machines, which is what makes every
alert auditable and cheap enough to run on commodity CPUs.

Full design rationale, measured results and honest limits: **[`docs/ARCHITECTURE.html`](docs/ARCHITECTURE.html)**

Proposal-by-proposal evidence and the exact remaining deployment gaps:
**[`docs/SUBMISSION_READINESS.md`](docs/SUBMISSION_READINESS.md)**

---

## Quick start

```bash
pip install -r requirements.txt

# model weights are not in the repository (see .gitignore); fetch them once
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo26m.pt")          # detector used for every reported figure
YOLO("yolo26n.pt")          # CPU / edge profile
PY

# optional: the India-specific detector (IISc AIM, Bengaluru CCTV)
#   https://huggingface.co/iisc-aim/UVH-26  ->  models/UVH-26-MV-YOLOv11-S.pt

# 1. teach a camera its own road (~90 seconds, then confirm in the UI)
python run.py calibrate --video data/raw/clip.webm --camera-id CAM_01 \
    --zone "Electronics City" --road-name "Hosur Road" --lat 12.845 --lon 77.660

# 2. run the pipeline
python run.py process --camera CAM_01

# 3. open the dashboard
python run.py serve            # http://127.0.0.1:8000/
```

CPU deployment path (the civic argument — no GPU per camera):

```bash
python run.py process --camera CAM_01 --cpu --openvino
```

---

## What it detects

Fourteen anomaly types from **one detector and one tracker**. Every "None" in the
third column is a row where geometry was spent instead of GPU.

| Anomaly | Mechanism | Extra model? |
|---|---|---|
| Wrong-side / wrong-way | Track direction vs corridor vector, cosine < −τ, CUSUM-persisted | None |
| Wrong lane crossing | Corridor transition across a boundary marked solid | None |
| Queue / congestion | Count + median speed + stopped ratio + rasterised occupancy | None |
| Road blockage | Stationary in corridor + flow impairment + long-background presence | None |
| Abnormal / illegal stop | Blockage logic inside a no-stop zone | None |
| Pedestrian on carriageway | Person track dwelling inside a travel corridor | None |
| Sudden braking | Smoothed speed derivative over track history | None |
| Illegal U-turn | Heading reversal inside a prohibited corridor | None |
| Near-miss / conflict | Proximity ∧ approach angle ∧ mutual deceleration | None |
| **Suspected collision** | Momentum exchange between two vehicles, or a vehicle at rest with crash cues | None |
| Tailgating | Time headway = gap / follower speed | None (needs homography) |
| Speeding (km/h) | Homography + Haversine on ground coordinates | None (needs homography) |
| Restricted-zone entry | Track crosses ROI polygon + dwell | None |
| Unknown anomaly | Deviation from the camera's learned normal | None |

Deliberately **not** claimed: collision type, injury, fault, per-vehicle GPS.
See [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Where this stands, on the benchmark's own metrics

The ACCIDENT benchmark (2026) annotates the exact second and location of every
accident, and all sixteen held-out clips here are in it. That makes a real
comparison possible instead of "we detected 14 of 16", which is not comparable
to anything — a clip firing on a stopped bus seventeen seconds after the
collision counts as a detection under that phrasing.

| | temporal **T** | spatial **S** | type **C** | unified |
|---|---|---|---|---|
| Human annotators | 0.979 | 0.995 | 0.923 | — |
| Molmo-7B (best published automatic) | 0.343 | 0.596 | 0.270 | 0.412 |
| DINOv2 linear probe | — | — | 0.440 | — |
| Optical-flow heuristic | — | 0.273 | — | — |
| Naive content-agnostic prior | — | — | — | 0.245 |
| **NETRA (this system)** | **0.454** | 0.173 | 0.250 | 0.250 |

**Temporal localisation is ahead of the best published automatic result** and
roughly half of human. **Spatial is well behind** — when the wrong vehicles are
named the predicted location is a vehicle-width out, and the metric is
unforgiving about that. Collision type is comparable to a 7B vision-language
model and comes free from the detection geometry: the shape of the conflict that
fired *is* a claim about what kind of accident it was, so no classifier is
involved.

Two caveats, stated rather than buried. The published figures are over the full
benchmark test split and ours over sixteen clips, which is a small sample with a
wide interval. And the spatial sigma is our reconstruction — the paper specifies
an anisotropic Gaussian but not its constants, so the annotated accident box
sets the scale. Treat the comparison as indicative, not as an official
reproduction.

```bash
python scripts/benchmark_compare.py --results ProblemSet/Results
python scripts/score_against_truth.py    # signed temporal error, per clip
```

Reproduce the supplied 16-clip evaluation without the historical nested-output
path ambiguity:

```bash
python scripts/run_problems.py --problems . --groups ProblemSet --results ProblemSet/Results_candidate --flat-results --no-calibrate --no-render --dump-candidates --device cuda:0
python scripts/bundle_evidence.py --results ProblemSet/Results_candidate
python scripts/make_results_index.py --results ProblemSet/Results_candidate
python scripts/score_against_truth.py --results ProblemSet/Results_candidate
```

`--fps 0` analyses every source frame. Evidence is intentionally sampled at
8 FPS, independently of analysis, so native-rate inference cannot create a
multi-gigabyte ring buffer. Every batch summary includes an `output_validation`
record, and exits non-zero if a clip failed or a report is missing.
`ProblemSet/Results` is the preserved best validated snapshot; fresh experiments
go to `Results_candidate` until their metrics are compared and accepted.

---

---

## Commands

| Command | Purpose |
|---|---|
| `python run.py calibrate` | Bootstrap a camera's corridors from its own traffic |
| `python run.py process` | Run the pipeline; writes incidents, evidence, JSON report |
| `python run.py serve` | FastAPI + dashboard + calibration screen |
| `python run.py status` | What's configured and what's been found |
| `python scripts/run_problems.py` | Batch-process a labelled clip set, render annotated video |
| `python scripts/evaluate.py` | Confusion matrix against folder labels |
| `python scripts/make_results_index.py` | Build `results/index.html` review page |
| `python scripts/sweep_detector.py` | Measure accuracy/latency across models and resolutions |

---

## Architecture in one screen

```
video ─┬─ quality gate ────────── corrupt frames dropped; camera-shift check
       │                          suspends geometry rather than lying
       ├─ DETECTOR ────────────── the only always-on network
       ├─ ByteTrack ───────────── two-stage association keeps weak boxes
       ├─ scene model ─────────── corridors, legal directions, exclusion zones,
       │                          road mask learned from moving traffic
       ├─ cheap signals ───────── dual-window background · motion change-point
       │                          · CUSUM accumulation
       ├─ background detection ── stopped/crashed vehicles found on the
       │                          background image  (the AI City winners' method)
       ├─ event engines ───────── wrong-way · queue · blockage · collision
       ├─ onset recovery ─────── backward Lucas–Kanade, once per incident
       └─ severity → evidence → location → routing → operator workflow
```

Research lineage for each stage is documented inline in the source and in
`docs/ARCHITECTURE.html`.

---

## Key engineering decisions

**Resolution beats model size.** Measured, not assumed: on dense Indian traffic,
`yolo26n` at 1280px yields ~5× more confident detections than at 640px for
similar latency. Raise input resolution before reaching for a bigger network.
Reproduce with `scripts/sweep_detector.py`.

**Crashed vehicles are found on the background image, not in live frames.** A
crashed vehicle is the *hardest* object in a live frame — motion-blurred,
occluded, oddly oriented. Measured on our own footage: 0.08 confidence live.
In the background image, where moving traffic has melted away, it is crisp and
isolated. Every winning AI City anomaly entry does this.

**A road mask is not optional.** Without one, a car park in frame produced 22
"stationary vehicles" that all corroborated each other as crashes. The mask is
learned online from the ground points of vehicles that were *moving* — no
segmentation model needed.

**Global motion change-point may never fire alone.** It trips on camera pans and
close passes. Measured: firing alone flagged 13 of 16 clean traffic clips. It is
now a corroborator only.

**A queue and a rear-end collision are the same shape, so geometry cannot
separate them — but physics can.** Every state-based cue (stopped, stopped
nearby, stopped and damaged-looking) fires on queues, which is fatal on Indian
urban roads where stopping is the most common thing a vehicle does. The AI City
winners' pipeline assumes *stopped = anomalous*, which holds on highways and does
not hold here.

So NETRA adds a channel that measures the impact rather than its aftermath. Two
vehicles that brake in a queue change velocity in the *same* direction, and their
momentum changes add. Two vehicles that collide exchange momentum, so the changes
*oppose and cancel*. That difference is dimensionless, needs no calibration, and
has no benign generator:

| scenario | momentum exchange |
|---|---|
| rear-end into a slower car | 1.00 |
| T-bone, unequal masses | 1.00 |
| two-wheeler vs bus (80:1 mass) | 1.00 |
| both brake for a signal | 0.00 |
| one brakes, other steady | 0.00 |

On crash-free footage the measured median is **0.049**. When this channel fires,
attribution is free and stronger than any classifier: a bystander cannot take
part in a momentum exchange.

**The appearance classifier is disabled by default, on evidence.** It scores
0.993 AUC held-out and is close to anti-correlated inside the accident clips,
because its negatives came from cameras its positives never appeared on. The full
account is in [`LIMITATIONS.md`](LIMITATIONS.md#negative-results-and-how-they-were-caught).

**Thresholds are set on crash-free footage before positives are examined.** A
threshold taken from clean data cannot be flattered by the positives, because
there are none to fit to. Two separate 0.95+ AUC results in this project were
produced the other way round and both were wrong.

**Never box a vehicle you cannot justify.** Naming the wrong car is worse than
naming none. If attribution fails, the overlay says
`VEHICLES NOT IDENTIFIED — no box drawn`.

---

## Detector choice

| Deployment | Checkpoint | Why |
|---|---|---|
| **Bengaluru / ELCIA** | `models/UVH-26-MV-YOLOv11-S.pt` | IISc AIM, fine-tuned on ~2,800 Bengaluru Safe City CCTV cameras, 14 India-specific classes. Measured better *and* 5× faster than COCO models on Indian traffic. No person class — pair with a COCO model for pedestrian events. |
| General / crash clips | `yolo26m.pt` @1920 | Best measured recall on collision regions; has `person` for the occupant-exit cue |
| CPU deployment | `yolo26n.pt` + OpenVINO INT8 | NMS-free head, ~43% faster CPU ONNX than YOLO11n at higher mAP |

---

## Reproducibility

Every threshold lives in `config/config.yaml`, which is hashed into every
`model_runs` row alongside the checkpoint, tracker, engine version and commit.
Any number in any report traces back to the exact settings that produced it.

Deadline builds are guarded by two empirical checks, not only unit tests:

```powershell
python scripts/check_release_gates.py `
  --problem-results ProblemSet/Results_release_candidate `
  --traffic-results results/ElciaDataSet_audit
```

The gate protects the four team-reviewed time-and-vehicle anchors, permits at
most one collision alert per accident clip, requires complete coverage of the
16 confirmed crash-free ElciaDataSet videos, and requires zero collision alerts
across that negative set. The reviewed baseline remains in
`ProblemSet/Results_localization_baseline`; presentation-safe videos are written
separately to `ProblemSet/Results_release_candidate/annotated_videos`.

Data provenance and licensing: [`DATA.md`](DATA.md).
Known limits and what the system may not claim: [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Requirements

Python 3.11+, PyTorch, Ultralytics ≥ 8.4 (for YOLO26), OpenCV, FastAPI, SciPy,
NetworkX. GPU optional — used for offline benchmarking; the deployment target is
CPU via OpenVINO.

Developed and measured on: Ryzen 7 7840HS · RTX 4050 Laptop (6 GB) · 16 GB RAM.
