# NETRA localization failure analysis

This audit is based on dense before/during/after frame sequences from all 16
ProblemSet videos, the 15 hand-marked screenshots, the ACCIDENT metadata, the
stored NETRA events, and a fresh YOLO26m detector audit at the marked time.

## Root cause

There is no global coordinate conversion error. Red boxes coincide with real
vehicles in source coordinates. The dominant defect is **wrong participant
selection**: a path-crossing candidate can occur near the accident time while
referring to a bystander elsewhere in the image. Event timing and participant
identity were optimized separately, so a temporally plausible event could still
name the wrong vehicle.

The detector overlaps the annotated accident region in 14/16 clips and exceeds
0.1 IoU in 13/16. Consequently, detector fine-tuning alone cannot repair most
failures. It is justified for the small/distant/night exceptions, but the main
repair belongs after detection: track continuity, candidate generation,
physics-aware ranking, and participant verification.

A second, presentation-level defect amplified the problem: alert-time boxes
were frozen and drawn while vehicles continued moving. The renderer now uses
time-indexed track boxes when a track exists and limits a frozen fallback box
to the alert frame.

## Clip-level diagnosis

| Clip | Observed failure | Required repair |
|---|---|---|
| `0eoMti_njew_00` | Correct pair is available; alert is slightly late. | Preserve pair; improve event-time selection. |
| `-AztVDZ6cEE_00` | Best current spatial/temporal case. | Regression anchor; do not trade it away. |
| `-PpBteU0p3Q_00` | Single-vehicle/post-impact case; localization depends on an aspect-change candidate. | Single-vehicle state verifier and temporal track continuity. |
| `-PpjzmhI_PE_00` | Very early single-vehicle impact with little pre-impact history; stored benchmark timing and user visual judgment need separate reporting. | First-seconds mode, post-impact state evidence, no long warm-up. |
| `-RrDtLjWsT4_00` | Correct region and usable time. | Regression anchor; consolidate later false candidates. |
| `-6SQSDj8cYU_00` | First alert is useful, followed by three false collision alerts. | One-event consolidation and suppress post-event candidates. |
| `-7-vQ4obVwQ_00` | Early unrelated pair plus another false alert; detector overlap with the true region is only 0.019 IoU. | Same-camera detector adaptation plus candidate reranking and consolidation. |
| `0ThWw_efieo_01` | The overturned vehicle is detected (0.808 IoU), but a foreground moving car is selected as a struck-object candidate. | Reject transient aspect/turn changes; verify persistent post-impact state. |
| `2W7S_-7F6S8_00` | Accident is in the upper junction; a foreground solo `rollover` candidate wins. | Penalize uncorroborated rollover/aspect-change candidates. |
| `29O6I-sITyw_00` | Distant barrier strike is primarily a dust burst; later passing vehicles are selected. | Small-object tiled detection plus localized appearance/change evidence; do not treat dust as a vehicle box. |
| `-dmYsQc-odI_00` | True lorry collision is near the camera; a distant solo rollover wins. | Participant ranking by post-event speed loss/persistence and candidate location continuity. |
| `-FQxK6HdxNU_00` | Detector sees the annotated region, but an unrelated crossing pair is selected. | Spatially supervised candidate ranking. |
| `-i9bRJWMtTo_00` | True side impact is detected visually (0.749 IoU), but no candidate is selected until an unrelated event around 24 s. | Candidate generation across track-ID changes and pair/vehicle outcome evidence. |
| `-NgnSm_oEB4_00` | Correct rear-end area is detected (0.707 IoU); a distant solo rollover candidate wins at nearly the same time. | Disable weak standalone rollover and rank the interacting pair. |
| `-Qt5bDJNT84_00` | Foreground through-traffic produces an early deflection candidate; actual roadworks collision is farther away. | Require outcome evidence and handle stationary/infrastructure partners. |
| `-RE3XseZINA_00` | No event; night lorry interaction has too little stable tracking, although a marked-frame vehicle detection exists. | Night/small-object detector adaptation, truck/trailer association, track-loss recovery. |

## Training target

The trained model is a candidate reranker, not a frame classifier. Its inputs
are interpretable physical measurements: rule score, PET, footprint separation,
deceleration, heading/turn change, speed loss, lateral acceleration, participant
count, identity change, and geometry type. A positive candidate must satisfy
both conditions:

1. occur within the accident-time tolerance; and
2. overlap the annotated accident region.

Ground-truth position creates the training label only. It is never an inference
feature. Evaluation holds out an entire accident clip at a time, preventing the
model from reading its own answer during validation. A final all-clip fit may be
used for the deadline demonstration but must be labelled as in-sample adaptation.

## Measured training and localisation results

The deadline experiments were evaluated and **not promoted when they failed**:

| Experiment | Held-out / deployable result | Decision |
|---|---:|---|
| Current consolidated pipeline | 6/16 within ±1 s; 4/16 right time and annotated vehicle | Baseline only |
| Raw physics-candidate reranker | correct candidate exists in 13/16; leave-one-video-out top-1 5/13 | Reject for deployment |
| Reranker after the live 0.55 hard gate | usable positive in 6/16; leave-one-video-out top-1 3/6 | Reject; the gate discards ten answers |
| Existing crash-appearance verifier | 4/16 correct vehicle even at oracle/true time | Reject as localizer |
| Optical-flow acceleration | 2/16 correct vehicle at oracle/true time | Reject as standalone localizer |
| Newly-stopped/persistent-place heuristic | 0/16 at oracle/true time | Reject as standalone localizer |
| Closest detected vehicle pair | 3/16 at oracle/true time | Reject as standalone localizer |
| Learned combination of appearance, flow, persistence, contact and scale | leave-one-video-out top-1 6/14 where a detectable box exists; only 1/16 right time and vehicle at NETRA timestamps | Experimental only; do not wire into live output |

This establishes the current bottleneck precisely: the detector usually sees
the crash, but the correct participant/outcome hypothesis is frequently never
admitted by the event gate. Training only the final ranker cannot recover a
hypothesis that was discarded upstream.

The refreshed annotated videos draw stored alert boxes only at their source
frame and contain one consolidated collision alert per clip. All 16 outputs
were read back at both first and last frame successfully.

## No-compromise traffic-anomaly architecture

The original objective remains the deployment contract:

> Detect wrong-way movement, queue buildup, abnormal stops and road blockages
> from existing camera feeds, emphasizing lightweight tracking/temporal
> reasoning without high-end GPUs.

Collision work does not replace those heads. For a fixed camera, NETRA must
retain:

- learned corridor/lane direction for wrong-way and lane-crossing events;
- per-corridor density, slow ratio and stopped ratio for queue buildup;
- per-track motion-to-rest transitions for abnormal stops;
- long-background occupancy and persistence for road blockage;
- the collision candidate generator/reranker as an additional event head;
- event-level output with time range, participants, location, confidence and
  physical explanation.

Uncalibrated moving-camera clips cannot support a defensible wrong-way claim;
that head must fail closed until lane direction is known. This is a truthfulness
constraint, not a feature reduction.

## Acceptance gates

1. One consolidated collision event per ProblemSet clip.
2. Report time-only and time-plus-location scores separately.
3. No frozen box may persist after its source frame; tracked boxes must move.
4. Candidate-ranker leave-one-clip-out results must be reported before the
   in-sample final fit.
5. Crash-free Traffic clips must not gain collision false alarms.
6. Wrong-way, queue, abnormal-stop and blockage tests and smoke runs must remain
   operational.

## Release blockers found by the audit

- The stored crash-free Traffic run contains 16 path-crossing collision alerts
  over 16 clips (118.7 false collision alerts/hour). Collision promotion on
  continuous feeds is therefore blocked until this is brought down on a
  camera-held-out validation set.
- The same stored run still demonstrates 11 wrong-way and 7 queue events. Those
  heads must remain independent while collision precision is repaired.
- `-RE3XseZINA_00` needs detector/tracker recovery; `-7-vQ4obVwQ_00` and
  `29O6I-sITyw_00` need small/distant evidence. They cannot be fixed by ranking.
- Model attribution must fail closed: when participant evidence is not stable,
  report the event region as unverified instead of drawing an unrelated car.

## Implementation order

1. **Ship the presentation-safe renderer now.** One clip-level alert, alert-frame
   boxes only, honest `VEHICLE NOT MATCHED`/`MODEL MISS` labels, and retained raw
   candidates for audit.
2. **Repair event hypothesis admission.** Preserve a 2–3 s per-track state buffer
   through ID changes and form candidates from pre-contact approach plus
   post-contact outcome (speed loss, heading impulse, persistence, partner
   response). Do not use aspect-ratio change as a standalone rollover proof.
3. **Add a triggered participant verifier.** At an event timestamp, rank all
   nearby tracked/detected vehicles using pre/post state and contact outcome;
   return no box below a grouped-validated confidence threshold.
4. **Adapt detection only for the three detector-bound clips.** Fine-tune a small
   detector with clip-disjoint validation for night trucks, distant vehicles and
   post-impact deformation; use tiling only on distant ROIs. This is not a
   replacement for the participant verifier.
5. **Eliminate Traffic collision false alarms before promotion.** Calibrate per
   fixed camera, require outcome corroboration, measure false alarms/hour, and
   keep collision candidates operator-verified until the acceptance target is
   met.
6. **Protect the original solution contract.** Run independent regression suites
   for wrong-way, queue, abnormal stop and blockage after every collision change;
   retain CPU/light-GPU detector-tracker inference and trigger expensive work
   only around a candidate event.

The code test suite currently passes 58/58 tests. This verifies mechanics and
regressions covered by tests; it does not override the empirical Traffic and
ProblemSet failures above.
