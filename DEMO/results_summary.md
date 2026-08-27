# Results summary — one honest page

Written for the submission form's "measured vs simulated vs future scope"
declaration and its "3 successful cases / 2 failure cases" requirement. Every
number below is quoted verbatim from a source file, with the file named next
to it. **If a number is not in this file, it has not been measured on this
project's code — do not speak it in the demo video or write it anywhere else
in `DEMO/`.**

## The one thing to understand before reading the table

**Update, this session: the physics engine is now wired into NETRA, and has
been re-measured — partially.** `netra/events/rotation_gate.py` (grep for
`rotation_floor`, `oncoming_min_deg`, `prior_crossing` in `FINAL/CCTV/netra`
now finds it) is called from `netra/events/collision.py` on every analysed
frame, gated on independent-channel agreement (`_rotation_gate_confirmed`) —
see the root `README.md` §6 for the full mechanism and the honestly-reported
numbers below. The two systems described below are history — what NETRA's
pre-merge collision channel measured, and what the physics engine measured
standalone in `IDEAS/COMBINED` before the port — kept for provenance, not as
claims about the shipped system.

1. **NETRA's shipped collision channel, pre-wiring** — lives in
   `FINAL/CCTV/netra/` (`pathconflict.py`, `attribution.py`, `predict.py`).
   Measured on 31 supplied clips (`CCTV/LIMITATIONS.md`): **11/15 collision
   clips detected, 8/15 with a vehicle named, 155.8 false collision
   alarms/hour on crash-free footage.** Not deployable stand-alone; gated to
   "candidate, human-verified" for exactly this reason.
2. **The physics participant-localization engine, pre-port** — measured
   standalone in `IDEAS/COMBINED/` (`run_inference2.py`) on the 4
   ground-truth-labelled videos of the 15-video accident set
   (`IDEAS/COMBINED/OUTPUT/2nd_inference/README.md`): **4/4 correct top-1
   participant-pair identification**, in-sample thresholds.
3. **The merged, wired system, re-measured this session** — `scripts/run_problems.py`
   against 15 Accidents + **6 of 16** Traffic clips (batch deliberately capped
   for GPU/time budget; the Traffic figure is a small, noisy partial sample,
   not a final number). **Accidents recall 12/15 (80%)**, but not attributable
   to rotation-gate — `rotation_gate_confirmed` was `False` on every one of
   those 12 detections (Accidents clips are uncalibrated, so the corroboration
   gate never applies). **3 of the 6 sampled Traffic clips produced a false
   collision candidate** (≈84/hour extrapolated, wide uncertainty at this
   sample size), including the exact clip already known to false-positive
   standalone (`13009518_1920_1080_30fps.mp4`, `CCTV/demo/README.md`) — the
   agreement gate did not suppress it. More pointedly: on all 3 Traffic false
   alarms, momentum-exchange (the pre-existing, stricter corroborator) never
   fired, meaning **none of the 3 would have promoted under the pre-wiring
   gate at all** — the new rotation-gate-agreement path gave a pre-existing
   false-alarm source (path-crossing on busy intersections) a second way
   through. Reported as a real, measured finding, not smoothed over; see
   root `README.md` §6 for the full breakdown.

They answer different questions (NETRA: "is there a collision on this
camera, right now, at any confidence" / physics engine: "given a labelled
collision, which two tracked vehicles were the participants") on different
data. Anywhere this document writes "TBD", that is the honest state — not an
oversight.

---

## Working today / simulated / not implemented

| Capability | Working today (measured) | Simulated or mocked | Not implemented (future scope) |
|---|---|---|---|
| Queue detection | NETRA corridor-based rule fires: **6 findings across the 16 ELCIA traffic clips** (`SUBMISSION_READINESS.md`); real evidence image in root `README.md` §2, from this session's own re-run of `12937197_3840_2160_30fps.mp4`. Persistence-gated (CUSUM), reports count in vehicles, not metres. | — | Precision/recall unmeasured — no queue ground-truth labels exist for the ELCIA clips. |
| Wrong-side movement | Heading-vs-corridor rule fires: **8 findings across the 16 ELCIA traffic clips**; real evidence image in root `README.md` §2, same re-run. | — | Legal direction unreviewed for every current camera — outputs stay "candidate," never enforcement, until a human confirms direction on the calibration screen. |
| Blockage | Detector code exists, unit-tested. | — | **0 findings across the 16 ELCIA clips** — no blockage examples exist in the supplied set, so recall is literally unmeasurable, not merely unmeasured. |
| Collision candidate (NETRA + rotation-gate, wired and re-measured) | Fires: **12/15 Accidents detected (80%)**, but that recall change is **not** attributable to rotation-gate (`rotation_gate_confirmed` was `False` on all 12 — see above). On a **partial, capped Traffic sample (6/16 clips)**: **3/6 clips false-positived**, and the agreement gate did not suppress the one known standalone false positive. | The two-channel agreement requirement (`fixed_camera_min_channels: 2`, now `(impulse_confirmed or rotation_confirmed) and channels>=2`) is real and verified in the triggers of every fired event, not just in the code — but on this sample it gave an *additional* promotion path to a pre-existing false-alarm source rather than closing it. | An unattended "this is confirmed" claim. It is never made — every collision, `CONFIRMED` or `POSSIBLE`, still requires human verification (`ALWAYS_VERIFY`). |
| Collision participant localization (physics engine) | **Ported and wired** (`netra/events/rotation_gate.py`, called from `netra/events/collision.py`). Pre-port: **4/4 top-1 on the 4 ground-truth-labelled videos**, in-sample thresholds, sample-size caveat stated by its own authors (`2nd_inference/README.md`); 0 false positives / 19 negatives pre-port (`RESULTS.md`). Post-port, in-process, on fresh NETRA footage: scores in `CCTV/demo/README.md` (0.289–0.738, including one confirmed false positive). | — | The full 16-clip Traffic false-alarm number (only 6/16 measured this session) and closing the false-positive-promotion gap found above. |
| Severity classification | Implemented: 5 measured components (flow loss, obstruction, extent, duration, risk exposure) → Low/Medium/High band. Every incident carries the component breakdown. | — | Not validated against any independent severity ground truth (no labelled severity set exists). Explicitly *traffic-impact* severity only — never injury severity, by design. |
| Location-aware alert | Camera + zone reported on every incident. | — | Road name/coordinate/road-edge precision — ELCIA clips carry no road metadata, so location is honestly zone-only for all but 2 of 82 configured cameras. |
| Diversion route (solid red) | Engine and API exist and are exercised by tests. | Only reachable for **2 of 82 cameras** (`CUTTACK_LINK_01`→`E_LINK_RD`, `GANGTOK_6MILE_01`→`E_NH10`) — every other camera has no `road_edge_id`, so the buttons stay disabled and the UI says so rather than faking a route. | A populated road graph for the ELCIA clips — `CCTV/config/road_graph.json` does not exist. |
| Responder access route (dashed blue) | Same routing engine, opposite direction of travel. | Same 2-camera-only reachability as diversion. Currently a **simulated overlay concept** — no live responder telemetry feeds it; it is a routed path on the same graph, not a tracked vehicle. | Real responder GPS/ETA integration. |
| Response/escalation workflow | Full state machine — detected → verified → assigned → responding → resolved → closed, plus reject-with-reason — implemented and exercised by the test suite; every transition is written to `status_history`. | — | — |
| Low-resource / CPU path | `--cpu --fps 4 --imgsz 640` flags exist and run; realtime factor is printed per run, not asserted. | — | Target-device latency is **unbenchmarked** — no number for "runs at N FPS on a typical control-room PC" exists yet; only "it runs, and the honest FPS prints." |
| Drone detector | Pipeline is mechanically complete and runs end-to-end. | Runs on a **generic COCO-pretrained YOLO placeholder** — `detector_finetuned: false` on every run, with a banner saying so (`DRONE/models/detector/README.md`). | **No drone detector has been fine-tuned or measured.** VisDrone fine-tune is a planned, unexecuted 3-stage plan. No accuracy number exists or is claimed. |
| Drone hover / GMC | Global motion compensation config and sanity envelope exist (`DRONE/config/drone_config.yaml`). | — | No real drone footage exists to test drift correction against; behaviour is unverified outside synthetic/placeholder input. |
| Thermal presence trigger | Scoped to exactly one role by design: total-darkness vehicle-presence trigger, never classification/motion/severity. | — | `scripts/thermal_presence.py` is a stub today — no thermal sensor input exists. |
| Seed-data replay (jury demo) | — | This is **by design** a mocked/replayed data path — `DEMO/seed_data/README.md` — real inference run once, frozen, served statically for GPU-free judging. It must never be presented as live inference. | — |
| APP (Next.js operator console) | Route scaffolding and `package.json` exist under `FINAL/APP/` (pages for `cctv`, `drone`, `incidents`, `map`, `upload`, `calibrate`). | — | **Functional status unverified as of this writing** — presence of files is not evidence of a working `npm run dev`. Confirm live on recording day per `DEMO/jury_walkthrough.md` before relying on it for any segment; the built-in CCTV dashboard at `:8000` is the fallback for everything except the map view. |
| DRONE backend server (port 8011) | Config and standalone scripts exist (`gmc.py`, `hover_mode.py`, `telemetry_ingest.py`, `thermal_presence.py`). | — | **No FastAPI server exists to bind port 8011** as of this writing. `DEMO/jury_walkthrough.md` §4 states the contract it must satisfy when it does. |
| End-to-end merged-system accuracy | — | — | **TBD.** No number in this file describes CCTV+DRONE+APP running together as one measured pipeline, because that pipeline has not been assembled yet. Re-measure and re-quote after integration; do not carry forward either subsystem's pre-merge number as if it described the merged submission. |

---

## Reconciling the collision false-alarm numbers (three documents, three numbers, one explanation)

Three CCTV docs report three different collision-false-alarm figures. They
are not a contradiction once you know what each measures:

| Source | Figure | What it measured |
|---|---|---|
| `CCTV/LIMITATIONS.md` | **155.8 alarms/hour**, 16/16 clips | Raw single-channel path-conflict gate, threshold 0.42, on 16 crash-free clips |
| `CCTV/docs/LOCALIZATION_FAILURE_ANALYSIS.md` | **118.7 alarms/hour** (16 alerts / 16 clips) | A separately audited stored Traffic run, also single-channel path-crossing, same 16 clips — the two audits used slightly different runs/timepoints of the same immature gate |
| `CCTV/docs/SUBMISSION_READINESS.md` | **0 collision candidates** | The pre-rotation-gate release-candidate gate, which required **two independent motion channels to agree** (`collision.fixed_camera_min_channels: 2`, satisfiable only via `impulse_confirmed` at the time) before a candidate was ever surfaced, on the same 16 clips |
| This session, post-wiring (root `README.md` §6) | **3 of 6 sampled clips flagged** (partial, capped Traffic sample) | The same two-channel gate, now `(impulse_confirmed or rotation_confirmed) and channels_agreeing >= 2` — rotation-gate agreement gives a *second* way through, and on this sample that second way is what let all 3 false alarms promote (`impulse_confirmed` was `False` on all 3) |

Read together: the raw single-channel signal is unusably noisy on its own
(both audits agree on that, within audit-to-audit variance), and the
two-channel agreement requirement was, at the "0 collision candidates" audit
point, the measured fix. **That is no longer the current state.** Wiring
rotation-gate in added a second path through the same gate, and on this
session's partial re-measurement that path let false alarms back in — so "0
collision candidates" must not be quoted as the current number; it describes
a specific, now-superseded gate configuration. Do not quote 155.8/hour as
"our current false-alarm rate" either — the current, partial, honestly-capped
number is the 3-of-6-clips figure above, and it needs the remaining 10
Traffic clips before it is a real figure.

---

## 3 successful cases

1. **NETRA release-candidate gate, two-channel requirement (historical).** On
   the 16 confirmed crash-free ELCIA traffic clips, the pre-rotation-gate
   release-candidate collision gate produced **zero collision candidates**
   (`SUBMISSION_READINESS.md`, "Verified release gates"). **This gate
   configuration has since changed** — wiring rotation-gate in added a second
   way through the same two-channel requirement, and re-measuring this
   session found 3 false alarms on a 6-clip partial Traffic sample (see
   above). Kept here for provenance, not as a current success claim.
2. **NETRA spatial/temporal localization regression anchor — `-AztVDZ6cEE_00`.**
   Flagged in `LOCALIZATION_FAILURE_ANALYSIS.md` as "best current spatial/temporal
   case" — correct region, usable time, explicitly called out as a case to
   protect ("do not trade it away") in later development.
3. **Physics participant-localization engine, video 4.** Top-1 pair
   `#2239↔#2254`, score **0.680**, classified `crossing` (T-bone), matching the
   ground-truth pair exactly (`2nd_inference/README.md`), pre-port. The
   cleanest margin of the four correct videos, and a genuine demonstration of
   the rotation-gate/interaction-geometry reasoning — the engine **is now
   merged and wired** into the shipped CCTV pipeline (root `README.md` §6);
   this specific in-sample number describes the pre-port `IDEAS/COMBINED`
   standalone run, not a claim re-verified inside NETRA.

## 2 failure cases

1. **NETRA's single collision channel is not usable unattended.** Best pair
   score on the raw single-channel gate reaches **0.695–0.736 on crash-free
   clips**, higher than the **0.577–0.673** it reaches on actual collision
   clips (`LIMITATIONS.md`). Root cause is measured, not guessed: crash-free
   clips carry 60–70 simultaneous tracks against 2–27 in collision clips, so
   roughly 100,000 pairs are scored per clean clip and the maximum of a noisy
   estimator over that many pairs reaches 0.99 by chance. This is why the
   channel is gated behind a two-channel agreement requirement and human
   verification rather than shipped as a standalone detector.
2. **`-RE3XseZINA_00` — a genuine miss, not a threshold problem.** Night
   lorry interaction with too little stable tracking; no collision event
   fires at all despite a marked-frame vehicle detection existing
   (`LOCALIZATION_FAILURE_ANALYSIS.md`). This clip sits in the broader pattern
   the same document measures: clips where NETRA misses a crash separate
   cleanly from detected ones by **track count** — median 17 tracks created
   on missed clips versus 106 on detected ones, two clips producing only 5–6
   tracks across the whole video. The fix is a training job (fine-tune the
   detector on crash imagery), explicitly not a threshold change — no amount
   of re-tuning the collision logic recovers an event with no tracks to
   reason over.
3. **The rotation-gate module's standalone failure mode survived being wired
   in.** First live in-process test on 5 of NETRA's own clips (`CCTV/demo/`,
   full detail in `CCTV/demo/README.md`) scored a Traffic-category
   (crash-free) clip at **0.738 — the highest score of the whole set**, traced
   to 227 tracker IDs over 221 frames of dense queued traffic and a spurious
   1915 px/s reading on a visually-stationary vehicle. The plan at the time
   was: require rotation-gate to agree with an independent NETRA channel
   before a candidate is `CONFIRMED`, not `POSSIBLE`. **That gate is now wired
   and verified working as designed** (`netra/events/collision.py`,
   root `README.md` §6) — but re-measuring it this session on the same clip
   found the agreement gate did **not** suppress this exact false positive: a
   pre-existing NETRA channel (path-crossing) already scored it 0.836–0.937 on
   its own, and rotation-gate's more moderate score (0.49 in this run) agreed
   with it rather than vetoing it. Worse, on this session's partial Traffic
   sample, momentum-exchange (the stricter pre-existing corroborator) never
   fired on any of the 3 false alarms found — meaning the new rotation-gate
   agreement path is what let all 3 promote at all. This is reported as a
   real, measured, unresolved finding, not a caught-before-merge success
   story; see root `README.md` §6 and §11 for what closing it would require.

---

## External benchmark context (not our numbers — quoted for scale)

- **2026 ACCIDENT benchmark** (2,027 real CCTV accident clips): best
  published automatic system scores **0.571** on its unified metric against a
  human inter-annotator ceiling of **0.923–0.995**, needing ~13 GPU-hours on
  an RTX PRO 6000. Best automatic temporal localization: **0.343** vs **0.979**
  human. (`CCTV/LIMITATIONS.md`)
- Collision-**type** classification (head-on/rear-end/T-bone/sideswipe/single-vehicle)
  by a frozen 7B vision-language model: **0.115**, below the majority-class
  floor of **0.335**. We do not attempt this task. (`CCTV/LIMITATIONS.md`)

This context is why NETRA claims "suspected collision, human-verified" and
nothing stronger — the unqualified task is unsolved in the published
literature, on far larger compute budgets than this project used.

## Test suite

**This session (27 Aug 2026), after wiring rotation-gate into `collision.py`:
`python -m pytest tests/` in `CCTV/` — 126 passed**, matching the 77
pre-existing NETRA tests plus 49 rotation-gate tests recorded before this
session (no regressions). Earlier snapshots for provenance:
`CCTV/docs/SUBMISSION_READINESS.md` (26 Aug 2026): **74 passed**;
`CCTV/docs/LOCALIZATION_FAILURE_ANALYSIS.md` (earlier still): **58/58
passed.** Re-verify with `python -m pytest` before quoting any of these on
demo day — this file is a snapshot, not a live number.
