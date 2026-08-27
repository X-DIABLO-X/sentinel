# Results summary — one honest page

Written for the submission form's "measured vs simulated vs future scope"
declaration and its "3 successful cases / 2 failure cases" requirement. Every
number below is quoted verbatim from a source file, with the file named next
to it. **If a number is not in this file, it has not been measured on this
project's code — do not speak it in the demo video or write it anywhere else
in `DEMO/`.**

## The one thing to understand before reading the table

This project currently ships **two separate, unmerged collision-related
systems**, evaluated on two different corpora, with two very different
results:

1. **NETRA's shipped collision channel** — lives in `FINAL/CCTV/netra/`
   (`pathconflict.py`, `attribution.py`, `predict.py`). This is what runs when
   you execute `FINAL/CCTV/run.py`. Measured on 31 supplied clips
   (`CCTV/LIMITATIONS.md`): **11/15 collision clips detected, 8/15 with a
   vehicle named, 155.8 false collision alarms/hour on crash-free footage.**
   Not deployable stand-alone; gated to "candidate, human-verified" for
   exactly this reason.
2. **The physics participant-localization engine** — lives in
   `IDEAS/COMBINED/` (`run_inference2.py`). It is **not present anywhere in
   `FINAL/CCTV/netra`** — grep for `rotation_floor`, `oncoming_min_deg`, or
   `prior_crossing` in that package and you get nothing. Measured on the 4
   ground-truth-labelled videos of the 15-video accident set
   (`IDEAS/COMBINED/OUTPUT/2nd_inference/README.md`): **4/4 correct top-1
   participant-pair identification**, in-sample thresholds.

They answer different questions (NETRA: "is there a collision on this
camera, right now, at any confidence" / physics engine: "given a labelled
collision, which two tracked vehicles were the participants") on different
data, and neither result describes the other system. **Nothing in this repo
has yet run the physics engine's logic inside NETRA's pipeline and
re-measured the result.** Until that merge and re-measurement happens, any
number describing "the collision system" must specify which of the two it
means. Anywhere this document writes "TBD", that is the honest state — not an
oversight.

---

## Working today / simulated / not implemented

| Capability | Working today (measured) | Simulated or mocked | Not implemented (future scope) |
|---|---|---|---|
| Queue detection | NETRA corridor-based rule fires: **6 findings across the 16 ELCIA traffic clips** (`SUBMISSION_READINESS.md`). Persistence-gated (CUSUM), reports count in vehicles, not metres. | — | Precision/recall unmeasured — no queue ground-truth labels exist for the ELCIA clips. |
| Wrong-side movement | Heading-vs-corridor rule fires: **8 findings across the 16 ELCIA traffic clips.** | — | Legal direction unreviewed for every current camera — outputs stay "candidate," never enforcement, until a human confirms direction on the calibration screen. |
| Blockage | Detector code exists, unit-tested. | — | **0 findings across the 16 ELCIA clips** — no blockage examples exist in the supplied set, so recall is literally unmeasurable, not merely unmeasured. |
| Collision candidate (NETRA, shipped) | Fires: **11/15 detected, 8/15 vehicle-named**, but **155.8 false alarms/hour, 16/16 crash-free clips** at the live single-channel gate (0.42). Raising the gate to 0.55 drives real-collision recall to **0/15** before it touches false alarms (`LIMITATIONS.md`). | The release-candidate UI requires **two independent motion channels to agree** before promoting a candidate (`SUBMISSION_READINESS.md`); on that gate, the same 16 crash-free clips produced **zero collision candidates** — a real, measured mitigation, not a demo trick, but it trades away most single-channel recall to get there (see reconciliation note below). | An unattended "this is confirmed" claim. It is never made. |
| Collision participant localization (physics engine) | **4/4 top-1 on the 4 ground-truth-labelled videos**, in-sample thresholds, sample-size caveat stated by its own authors (`2nd_inference/README.md`). Precision on the broader 12-pos/19-neg detection task: **0 false positives / 19 negatives** (`RESULTS.md`). | — | **Not merged into `FINAL/CCTV`.** Does not run when you start `run.py serve`. Numbers above describe the `IDEAS/COMBINED` prototype only, and must be re-measured inside NETRA before being claimed as this submission's collision-localization accuracy. |
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
| `CCTV/docs/SUBMISSION_READINESS.md` | **0 collision candidates** | The **release-candidate gate**, which requires **two independent motion channels to agree** (`collision.fixed_camera_min_channels: 2`) before a candidate is ever surfaced, on the same 16 clips |

Read together: the raw single-channel signal is unusably noisy on its own
(both audits agree on that, within audit-to-audit variance), and the
two-channel agreement requirement is the actual, measured fix that ships in
the release candidate. State it exactly that way if a judge asks — do not
quote 155.8/hour as "our current false-alarm rate" without the two-channel
gate context, and do not quote "zero false alarms" without naming what that
gate cost (raising the single-channel bar to 0.55 alone would have taken real
recall to 0/15; the two-channel requirement is a different, cheaper trade).

---

## 3 successful cases

1. **NETRA release-candidate gate, two-channel requirement.** On the 16
   confirmed crash-free ELCIA traffic clips, the shipped release-candidate
   collision gate produced **zero collision candidates** (`SUBMISSION_READINESS.md`,
   "Verified release gates"). This is the honest headline success: the
   mitigation for the false-alarm problem is measured, not asserted.
2. **NETRA spatial/temporal localization regression anchor — `-AztVDZ6cEE_00`.**
   Flagged in `LOCALIZATION_FAILURE_ANALYSIS.md` as "best current spatial/temporal
   case" — correct region, usable time, explicitly called out as a case to
   protect ("do not trade it away") in later development.
3. **Physics participant-localization engine, video 4.** Top-1 pair
   `#2239↔#2254`, score **0.680**, classified `crossing` (T-bone), matching the
   ground-truth pair exactly (`2nd_inference/README.md`). The cleanest margin
   of the four correct videos, and a genuine demonstration of the
   rotation-gate/interaction-geometry reasoning — with the caveat stated
   throughout this document that this engine is not yet merged into the
   shipped CCTV pipeline.

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

`CCTV/docs/SUBMISSION_READINESS.md` (26 Aug 2026, most recent): **74 passed.**
`CCTV/docs/LOCALIZATION_FAILURE_ANALYSIS.md` (earlier snapshot): **58/58
passed.** Use the 74-test figure as current; the difference reflects tests
added between the two audits, not a regression — verify with
`python -m pytest` in `CCTV/` before quoting either number on demo day, since
both are now stale relative to whatever the tree looks like at recording
time.
