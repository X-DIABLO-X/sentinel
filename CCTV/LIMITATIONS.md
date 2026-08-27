# Limitations, test conditions and future work

The brief asks for this section explicitly. It is written to be *useful* rather
than defensive: every limit below carries the evidence behind it, which turns the
weakest part of a submission into the part a technical reviewer can most easily
trust.

## What NETRA may not claim

**Not "we detect accidents."**
We raise *suspected collision-related disruption* and route it to a human. On the
2026 ACCIDENT benchmark — 2,027 real CCTV accident clips — the best published
system scores 0.571 on the unified metric against a human inter-annotator ceiling
of 0.923–0.995, and needs roughly 13 GPU-hours on an RTX PRO 6000. Claiming
reliable automatic accident detection on a laptop would be claiming something the
field has not achieved.

**Not collision type** (head-on / rear-end / T-bone / sideswipe / single-vehicle).
A frozen 7B vision-language model scores 0.115 on that task — *below* the
majority-class floor of 0.335. We do not attempt it.

**Not injury, casualty or fault.** Nothing in RGB video supports it.

**Not km/h or metres without calibration.** Image-plane speed is
perspective-dependent: 30 px/s near the camera and 30 px/s at the vanishing point
are wildly different real speeds. Without a homography we report vehicle counts
and relative motion, never physical units. Queue length is reported in vehicles.

**Not per-vehicle GPS.** The camera is a fixed geospatial sensor, so we report
*camera-associated* location and label its precision as such. Projecting an image
point to world coordinates needs camera pose, intrinsics and a ground model.

**Not "real-time" as an unqualified claim.** At 1920 px and native frame rate this
laptop runs below real time. The CPU deployment path trades resolution and frame
rate for throughput; both are measured and reported per run rather than assumed.

**Not congestion causality.** We can observe that a queue formed after a
disruption. We cannot show from video alone that a crash caused it rather than a
signal cycle or roadworks.

## Measured collision performance, stated plainly

On the 31 supplied clips, at full frame rate, YOLO26m @1920, RTX 4050:

| | value |
|---|---|
| collision clips detected | **11 / 15** |
| collision clips with a vehicle named | **8 / 15** |
| false collision alarms on crash-free footage | **155.8 per hour**, on 16 / 16 clips |

That false-alarm rate is not deployable, and no threshold fixes it. The trigger
score on crash-free clips is *higher* than on collision clips -- median 0.529
against 0.488, maximum 0.992 against 0.548 -- so raising the gate destroys
recall before it touches precision:

| gate | clean alarms kept | collision clips kept |
|---|---|---|
| 0.42 (current) | 21 | 11 / 15 |
| 0.50 | 12 | 5 / 15 |
| 0.55 | 10 | **0 / 15** |

The momentum-exchange channel does not rescue it either. Its physics is sound in
isolation -- every momentum-conserving impact returns 1.00 and every co-braking
pair 0.00, verified in `tests/test_collision_physics.py` -- but as an estimator
over tracked boxes it does not separate on this footage. Best pair score reached
0.577-0.673 on collision clips and **0.695-0.736 on crash-free clips**. The
cause is multiple comparisons: crash-free clips carry 60-70 simultaneous tracks
against 2-27 in the collision clips, so roughly 100,000 pairs are evaluated per
clip and the maximum of a noisy estimator reaches 0.99 by chance alone. A test
that is valid for one pair is not valid for the maximum over 100,000 of them.

Six of the fifteen collision clips are additionally below the channel's
frame-rate floor (8.0 to 15.0 fps), where an impact occupies a single frame and
cannot be distinguished from a dropped detection.

**What this means for the demo.** Collision findings are surfaced as
`collision_candidate` requiring operator verification, never as confirmed
accidents, and the dashboard's verify/reject workflow exists precisely because
this channel cannot be trusted unattended. The incident types that rest on
sustained geometry rather than a single instantaneous event -- queue, wrong-side
movement, blockage, pedestrian on carriageway -- do not share this weakness,
because they are defined by conditions that persist for seconds and can be
confirmed over time.

**Context.** The ACCIDENT benchmark (2026) evaluated heuristics, DINOv2/SigLIP2
probes and 7B vision-language models on this exact task: best automatic temporal
localisation 0.343 against 0.979 for humans, best unified score 0.412 against a
content-agnostic baseline of 0.245. The task is unsolved in the literature. That
is context for our numbers, not an excuse for them.

## Negative results, and how they were caught

These are recorded because they cost the most time, because each one *looked
like success first*, and because the method that caught them is the only reason
the rest of this document can be trusted.

**A crash classifier that had learned "stopped equals crashed".**
A fine-tuned yolo11n-cls scored **0.954 AUC** on held-out crops, then confidently
labelled parked cars as collisions at p = 0.99. The dataset was at fault, not the
threshold: every positive crop was taken *after* an accident, when the vehicle had
stopped, and every negative *before* it, while traffic moved. The cheapest way to
fit that data is to detect stillness. It scores beautifully on that validation
split and is worthless in deployment, where the only crops it is ever shown are of
stationary vehicles.

**The same model, rebuilt, had learned the camera.**
Adding 1,041 stationary-but-undamaged hard negatives raised held-out performance
to **0.993 AUC**, with zero still-negatives above threshold. In deployment it
barely moved. Re-scored on vehicles *inside* the accident clips, a parked truck
went from 0.993 to **0.995** — up. Every hard negative had come from crash-free
clips, i.e. cameras no positive ever appeared on, so "which scene is this"
separates the classes as well as "is this vehicle wrecked". The validation split
shared the confound and certified the shortcut instead of exposing it.

On clip 1 the model rated six of eight vehicles above 0.8, gave an untracked
bystander 0.995, and gave the vehicle that had visibly rolled — 265 px/s, a
41.7 px/s² stop, aspect swing 0.74 — only **0.695**. In-domain it is close to
anti-correlated with the truth. **The classifier is therefore disabled by
default.** The code and training scripts remain so the finding is reproducible.

**A hard rule written as a zero in a weighted sum is not a hard rule.**
The parked-vehicle veto zeroed the *prior*, but candidate scoring was
`0.75·p_crashed + 0.25·prior`, leaving `0.75 × 0.995 = 0.746` — still over the
bar. The gate was real and was being outvoted three to one. Vetoes now apply
after fusion, not inside it.

**Two config defaults silently shadowed the code.**
`ParticipantSelector` was constructed with literals from the event engine, so
`max_gap_lengths` stayed at 1.6 vehicle-lengths after the geometry switched to
footprint units, and the solo bar stayed at 0.62 after it was lowered. The
footprint geometry never took effect at all until this was found.

**Absence of evidence was being scored as evidence of absence.**
A stationary object with debris on the carriageway reached the trigger at 0.032
instead of 0.47, because two penalties multiplied: `stop_decel = 0` was read as
"stopped gently" when it meant *never measured*, and `arrived_moving = False` was
read as "watched it sit still" when it meant *no track ever matched it*. Both now
require the measurement to exist.

**The momentum test never executed once.**
`velocity_change(t)` needs samples on both sides of `t` and was being asked about
the present instant, where nothing later exists yet. It returned `None` on every
call. Across **299,499 residuals from sixteen clips, zero pairs were evaluated** —
a dead channel that looked like a working one with no false positives. Everything
is now evaluated at a deliberate lag.

**Per-frame impulse measured the filter, not the vehicle.**
Scored frame to frame on crash-free traffic, "exceeded driver control authority"
had a median of 0.271 and a 99.9th percentile of **0.986** — ordinary driving
saturating the measure. Differentiating a Kalman estimate across a 33 ms step
cannot separate the noise correction from real acceleration. Velocity change is
now taken across a 0.30 s window.

**NIS is not calibrated, so it gates nothing.**
Normalised innovation squared should be χ²(2) distributed. Measured on clean
footage: median 0.52 against an expected 1.39, p99 **95.7** against an expected
9.2, max 3185. The tracker's covariance is far too confident in the tail because
identity switches and box jitter are not in its noise model. It is retained as a
diagnostic only.

### The method that caught them

Every one of these was found by **measuring on crash-free footage first and
setting thresholds from that tail before looking at a single accident clip**. A
threshold taken from clean data cannot be flattered by the positives, because
there are none to fit to. Both AUC numbers above were produced the other way
round, and both were wrong.

## Measured failure modes

**Detection recall is the binding constraint, not reasoning.**
Five of fifteen crash clips are missed. They separate cleanly on one variable —
tracks created: median 17 on missed clips versus 106 on detected ones, with two
producing only 5–6 tracks across the entire video. Neither the
trajectory-conflict channel (needs two tracked parties) nor the
background-stationary channel (needs a detected stopped vehicle) can fire without
tracks. The remedy is fine-tuning the detector on crash imagery — a training job,
not a threshold change.

**Small, dark, motion-blurred participants.** Measured on one night clip: the two
colliding vehicles scored 0.08 and 0.05 with a nano detector — below any usable
threshold — while undamaged parked cars scored confidently. Mitigated by
detecting on the background image and by a larger model at higher resolution.
Mitigated, not solved.

**Auto-calibration cannot see paint.** It learns where vehicles *do* drive and
which way they *usually* go. It cannot know which way they are legally allowed to
go, and it cannot see whether a lane boundary is solid or dashed. If every vehicle
in the sample is violating, it will learn the violation as normal. This is why its
output is flagged DRAFT and why solid boundaries stay unset until a human confirms
them: an unset boundary raises no alert, a wrongly-set one raises a stream.

**The road mask needs traffic to learn from.** On clips shorter than ~10 s it will
not have enough moving trajectories to form. It then falls *open* — permitting
everything — rather than falsely rejecting. That is the safe direction, but it
means very short clips get weaker car-park filtering.

**Camera movement invalidates all geometry.** Corridors are drawn in image
coordinates; if the camera pans or is knocked, those polygons describe the wrong
piece of road. Detected by phase correlation, after which the system *suspends*
geometric events rather than emitting confident nonsense.

**Onset recovery succeeds on a minority of incidents.** Where participants are
poorly tracked there is nothing to trace backwards, so reported onset falls back
to detection time. Detection delay is always reported alongside onset, so the gap
is visible rather than hidden.

## Test conditions

- **Hardware:** Ryzen 7 7840HS · RTX 4050 Laptop (6 GB) · 16 GB RAM
- **Data:** 31 supplied clips (15 with collisions, 16 confirmed crash-free) plus
  3 CC/PD-licensed Wikimedia clips including Indian urban traffic
- **Frame rate:** native — no frame dropping — for the reported figures
- **Detector:** YOLO26m @ 1920 px, CUDA
- **No ELCIA footage exists**, so every threshold is fitted on public corpora.
  Transfer to the deployment domain is unvalidated, and the ACCIDENT benchmark's
  geo-OOD split is evidence that the risk is real. The per-camera calibration
  screen is the mitigation, not a fix.

## Future work, in order of expected value

1. **Fine-tune the detector on crash imagery.** This is the binding constraint.
   The ACCIDENT training split and its CARLA-generated synthetic clips exist for
   exactly this purpose.
2. **Lower the detector confidence floor from 0.10 to 0.05.** We currently discard
   weak boxes *before* ByteTrack sees them, which defeats the one property we
   chose ByteTrack for. Should be A/B'd against the current recall baseline.
3. **Second-stage temporal verifier** (X3D-S) on candidate clips only — raises
   precision without touching the always-on cost.
4. **Per-camera homography**, unlocking metric speed, queue length in metres,
   time headway and time-to-collision.
5. **Operator feedback as training data.** Rejections already record a reason;
   those labels are the cheapest available route to better thresholds.
