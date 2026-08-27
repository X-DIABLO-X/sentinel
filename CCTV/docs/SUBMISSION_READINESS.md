# ELCIA submission readiness — 26 August 2026

## Proposal contract

> Build video-AI systems using ELCIA-provided sample traffic videos to detect
> queues, blockage, wrong-side movement and accident-related congestion.
> Classify incident type and severity, display location-aware alerts and
> recommend diversion, response or escalation workflows.

## What is implemented and verified

| Requirement | Implementation | Evidence today | Honest status |
|---|---|---|---|
| Queue buildup | Per-corridor vehicle count, union occupancy, stopped fraction, free-flow-relative speed loss and CUSUM persistence | 6 findings across the 16 ELCIA traffic clips | Implemented; precision is not measurable without queue labels |
| Wrong-side movement | Track heading versus corridor direction, displacement/speed floors, exclusion zones and CUSUM persistence | 8 findings across the 16 ELCIA traffic clips | Candidate detector works; all current ELCIA directions are draft observed flow, not reviewed legal direction |
| Blockage | Stationary track plus impaired upstream flow or corridor obstruction plus dual-window background persistence | 0 findings across the 16 ELCIA clips | Implemented but not demonstrated; no blockage ground truth exists in the supplied set |
| Accident-related congestion | Temporal/corridor fusion of queue with a collision or blockage hypothesis | Implemented as `suspected_accident_related_congestion` or `obstruction_related_congestion` | Never presented as proven causality; no composite example occurs in the current crash-free traffic set |
| Incident type | Structured event taxonomy, not frame labels | JSON reports, database and dashboard | Implemented |
| Severity | Flow loss, obstruction, affected extent, duration and risk exposure | Every incident has component breakdown and Low/Medium/High band | Implemented as traffic-impact severity only; not injury severity |
| Location-aware alert | Camera, zone, corridor, optional road name/coordinate/road edge and precision statement | Every new report stores a location block; historical ELCIA audit reports were backfilled | Current ELCIA clips are zone-only because road metadata was not supplied |
| Evidence | Original frame, annotated frame, short clip, tracks and trigger values | Evidence packets are linked from the dashboard | Implemented |
| Response/escalation | Recommended action plus detected → verified → assigned → responding → resolved → closed | API and dashboard controls | Implemented |
| Diversion | Incident-penalised directed road graph and closure workflow | Engine and API exist | Unavailable for ELCIA clips until a real road edge is mapped; UI now says this instead of offering a fake route |
| No high-end GPU | One YOLO detector, ByteTrack and deterministic temporal reasoning; `--low-resource` uses YOLO26n at 4 Hz on CPU | CPU execution path exists | Architecture requirement met; target-device latency is still unbenchmarked |

## Verified release gates

- 16/16 ProblemSet annotated videos exist in `ProblemSet/Results_release_candidate/annotated_videos`.
- The four user-approved videos are byte-identical to the accepted baseline.
- Each ProblemSet report has at most one collision candidate.
- 16/16 confirmed crash-free ELCIA traffic clips were analysed.
- Those 16 clips produced **zero collision candidates**.
- Test suite: **74 passed**.

## Exactly what remains

### Required before claiming measured accuracy

1. Label event intervals and affected corridors for queue, wrong-side and blockage examples. At present only “no accident” is ground truth for the ELCIA traffic set.
2. Review every camera's legal direction and junction exclusion zones. Until then, wrong-side outputs must remain “candidates” and must not trigger enforcement.
3. Supply at least several true blockage videos. Zero detections on an unlabelled set cannot establish recall.
4. Add camera metadata: actual camera name, road name, zone, coordinates and road-graph edge. Without it, location is honestly zone-only and diversion is disabled.
5. Benchmark `python run.py process --camera <ID> --low-resource` on the intended CPU. Choose the detector cadence from measured latency, not from a claim.

### Required to improve accident localization

1. Fine-tune or replace the vehicle detector for small, distant and deformed vehicles. `-RE3XseZINA_00` still has no usable participant track.
2. Collect vehicle-level temporal boxes rather than screenshot-only regions for missed accident clips.
3. Keep detection confidence separate from participant attribution; the release candidate already withholds unsupported boxes instead of accusing a bystander.

### Not required for today's defensible demo

- Injury severity estimation.
- Per-vehicle GPS without camera calibration.
- A VLM or LLM in the detection loop.
- A larger end-to-end video model.
- Pretending observed majority flow is legal direction.
- Pretending a queue was caused by an accident merely because they overlap in time.

## Demo order

1. Open the release candidate accident review page and show the four locked participant-localization anchors.
2. Open the ELCIA proposal coverage page and show the 16/16, zero-false-collision gate.
3. Open the dashboard: select a queue, inspect measured triggers, severity and evidence, then move it through verify and assign.
4. Select a wrong-side candidate and show the explicit legal-direction warning.
5. Show why diversion is unavailable on an unmapped clip, then explain that mapping a deployed camera to a road edge enables deterministic routing.
6. Close with the low-resource architecture: one detector at reduced cadence; tracking and temporal/physics reasoning do the rest.
