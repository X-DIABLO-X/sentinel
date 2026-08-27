# 5-minute demonstration video — shot-by-shot script

**Hard limit: 5:00.** The budget below totals **4:58**, leaving 2 s of slack.
Overrunning is a disqualification risk, so the two compressible segments are
marked `[CUT FIRST]`.

**Submission rule this script is built around:** team members must *speak* and
must *demonstrate the actual working project*. A slides-only video is
disqualified. Therefore **every segment except S1 and S8 is screen capture of
software actually running on the machine** — no mockups, no After Effects
recreations of a UI. Where something is not real yet (the drone), the video says
so on screen, in words, while it is on screen.

## Speakers

| Tag | Role in the video |
|---|---|
| **A** | Narrates problem framing, CCTV detection, map/diversion, workflow |
| **B** | Narrates collision physics, severity, drone concept, limitations |

Replace A/B with the actual team member names in the final recording. Both
speakers must be audible; at least one on-camera intro is expected.

## Time budget

| # | Segment | Speaker | In | Out | Len |
|---|---|---|---|---|---|
| S1 | Problem + what we built | A | 0:00 | 0:28 | 0:28 |
| S2 | CCTV live: queue detection | A | 0:28 | 1:03 | 0:35 |
| S3 | CCTV live: wrong-side movement | A | 1:03 | 1:33 | 0:30 |
| S4 | CCTV: collision candidate + physics evidence | B | 1:33 | 2:23 | 0:50 |
| S5 | Severity classification | B | 2:23 | 2:48 | 0:25 |
| S6 | Map: diversion (solid red) + responder access (dashed blue) | A | 2:48 | 3:28 | 0:40 |
| S7 | Response / escalation workflow `[CUT FIRST]` | A | 3:28 | 3:50 | 0:22 |
| S8 | Drone hover-escalation concept, on placeholder footage | B | 3:50 | 4:22 | 0:32 |
| S9 | Measured results + known limitations | B then A | 4:22 | 4:58 | 0:36 |
| | **Total** | | | | **4:58** |

If you overrun in the field: cut S7 to 0:00 (the workflow is visible in S4's
detail panel anyway) and trim S1 to 0:20. That buys 30 s.

---

## S1 — Problem + what we built · 0:00–0:28 · Speaker A

**Visual:** Speaker A on camera for the first 8 s, then cut to a single static
title card holding the system name while A keeps talking. Do not spend more than
one card here.

**Narration (A):**

> "An urban traffic control room has hundreds of camera feeds and a handful of
> operators. The incident that matters — a queue forming, a vehicle going the
> wrong way, a crash blocking a lane — is on a screen nobody is looking at.
> We built two things. A fixed-camera pipeline that watches every feed
> continuously and raises location-aware incidents, and a drone that gets
> dispatched to hover over an incident the cameras have already confirmed. The
> cameras are the backbone; the drone is escalation, not patrol. Everything you
> are about to see is running live on this laptop."

**Direction:** say "running live on this laptop" only if you are genuinely
running it live. If you are running the pre-computed replay snapshot
(`DEMO/seed_data/`), change the line to *"running from a pre-computed snapshot
so it works without a GPU — the same code produced it"* and make sure the REPLAY
badge is visible in the UI. Do not blur this.

---

## S2 — CCTV live: queue detection · 0:28–1:03 · Speaker A

**Visual:** Full-screen browser at the operator console. Incident list on the
left, `Queues` tab selected. Click one queue incident. Detail panel opens with
the annotated evidence frame and the trigger table.

**Narration (A):**

> "This is the operator console. Left is the live incident feed across all
> configured cameras. I'll open a queue. The system did not classify a picture —
> it measured a corridor: how many vehicles are inside it, what fraction are
> stopped, how far speed has fallen below this camera's own free-flow baseline,
> and how long that has persisted. Those five numbers are on screen, and the
> persistence test is what stops a red light from being reported as a queue.
> Length is reported in vehicles, not metres, because this camera has no
> homography and we will not invent physical units we did not measure."

**On-screen must be visible:** the `Trigger values` table (open the
`Technical evidence` disclosure before recording so it is already expanded), and
the corridor polygon drawn on the annotated frame.

**Judge-bait moment (optional, +0 s):** leave `config/config.yaml` open in a
second window at the `queue:` block so a judge can see `min_vehicles: 4`,
`stopped_ratio: 0.45`, `cusum_h: 3.0` are real editable parameters. Judges are
told they may ask to change parameters — having the file already on screen
invites the question you can answer.

---

## S3 — CCTV live: wrong-side movement · 1:03–1:33 · Speaker A

**Visual:** Click the `Wrong-side` tab, open a wrong-side candidate. The amber
assurance banner reading *"Wrong-side candidate — legal direction is not yet
reviewed"* must be legible. Then a 3-second cut to `/ui/calibrate.html` showing
the corridor direction arrows.

**Narration (A):**

> "Wrong-side movement. The track's heading is compared against the corridor's
> configured direction, with displacement and speed floors so a parked car
> jittering does not trigger it. Note what the system refuses to do. It has
> learned this corridor's direction from observed traffic — it has not been told
> the *legal* direction, and it cannot see whether that lane marking is solid or
> dashed. So it says so, on the alert, and it stays a candidate. An operator
> confirms the legal direction on the calibration screen before this is ever
> enforcement evidence. An unset boundary raises no alert; a wrongly-set one
> raises a stream of them."

---

## S4 — Collision candidate + physics evidence · 1:33–2:23 · Speaker B

**Visual:** Two parts.
1. (0:22) The operator console: open a `collision_candidate`. The banner
   *"Suspected collision — not automatically confirmed"* and the channel count
   must be legible.
2. (0:28) Cut to the rendered physics-engine video — the true pair boxed red
   with score, interaction type and relative heading, the `<< CONTACT` marker at
   the refined contact instant, and the brake-to-avoid vehicles thin-boxed with
   their much lower scores. Play it at 1x through the impact, then freeze.

**Narration (B):**

> "Collisions. We do not claim to detect accidents — the published state of the
> art on this exact task does not either. We raise a *suspected collision* and
> route it to a human, and on a calibrated fixed camera we require two
> independent motion channels to agree before it is promoted at all.
>
> Here is the reasoning channel. The separating physics is one sentence:
> *braking decelerates you along your own axis, being struck rotates you.* So
> rotation is a multiplicative gate on impact evidence, not one term in a sum.
> Watch the vehicle that stands on its brakes to avoid the crash ahead — it
> sheds nearly all its momentum and rotates not at all, and it scores 0.03 to
> 0.05 against the true pair's 0.39. Before the gate, that braking bystander
> outranked the real collision. We also classify contact geometry: this pair is
> crossing at 53 degrees, a T-bone. Two vehicles at 178 degrees are passing
> safely on opposite sides of the road and their boxes only overlap because a
> 2-D box cannot encode depth."

**Direction:** B must say "suspected", not "detected". Do not say "our accident
detector".

---

## S5 — Severity classification · 2:23–2:48 · Speaker B

**Visual:** Same incident, scroll to the `Severity breakdown` bars: flow loss,
obstruction, extent, duration, risk exposure, impact subscore. The Low/Medium/
High band pill and the disclaimer line must be legible.

**Narration (B):**

> "Severity. Five observable components — flow loss, obstruction, affected
> extent, duration and risk exposure — combined into one score and a
> Low/Medium/High band. Two deliberate constraints. This is *traffic-impact*
> severity, not injury severity: nothing in RGB video supports an injury claim,
> so we do not make one. And confidence is reported beside severity, never
> folded into it — a high-confidence minor blockage and a low-confidence major
> one are different operational problems and must not average into the same
> number."

---

## S6 — Map: diversion + responder access · 2:48–3:28 · Speaker A

**Visual:** The map view. Incident pinned at the camera location. Press
`Confirm carriageway closed → divert`. The affected edge goes red/closed, the
**diversion route draws as a solid red polyline**, and the **responder access
route draws as a dashed blue polyline** arriving from the opposite direction.

**Narration (A):**

> "Location and response. The camera is a fixed geospatial sensor, so we report
> camera-associated location with its precision stated — we do not claim
> per-vehicle GPS. When an operator confirms the carriageway is closed, the
> incident penalises that edge in a directed road graph and we recompute two
> different routes for two different users. Solid red is the diversion for
> ordinary traffic: route *around* the incident. Dashed blue is the responder
> access route: get *to* the incident, from the least congested approach. The
> penalty is a multiplier, not infinity, so a severe incident makes an edge ten
> times more expensive rather than disconnecting the network."

**Blocker to clear before this segment can be recorded:** `CCTV/config/road_graph.json`
does not exist in the repo today, and only two cameras carry a `road_edge_id`
(`CUTTACK_LINK_01` → `E_LINK_RD`, `GANGTOK_6MILE_01` → `E_NH10`). With no graph,
`/api/graph` returns no features and both map buttons are disabled. Either the
APP track's OSMnx graph must be wired in, or a small hand-built
`config/road_graph.json` covering those two edges must exist. **If neither is
ready on recording day, do not fake this.** Replace S6 with the honest 25-second
version: show the disabled buttons and the UI's own "diversion unavailable on an
unmapped clip" state, say *"routing is implemented and tested; it is disabled
here because this clip has no road-edge mapping, and mapping a deployed camera
enables it deterministically"*, and give the 15 s saved to S9.

---

## S7 — Response / escalation workflow · 3:28–3:50 · Speaker A `[CUT FIRST]`

**Visual:** In the incident detail, click `Verify`, then `Assign`, then
`Responding`. The status pill changes at each step. Then click `Reject…` on a
*different* incident and show the reason list.

**Narration (A):**

> "Every incident is a piece of work with an owner and a state: detected,
> verified, assigned, responding, resolved, closed. And rejection is
> first-class — the operator picks a reason, and those rejections are the
> cheapest training labels we will ever get for fixing our own thresholds."

---

## S8 — Drone hover-escalation, on placeholder footage · 3:50–4:22 · Speaker B

**Visual:** The drone view. **A persistent on-screen banner must be burned into
this segment for its entire duration, not a title card before it:**

> `PLACEHOLDER FOOTAGE — NOT CHALLENGE DATA · DRONE PIPELINE IS SCAFFOLDING, NOT MEASURED`

**Narration (B):**

> "The drone. Be clear about what this is: this is stock aerial footage, not
> challenge data, and this pipeline is scaffolding — it has no measured accuracy
> and we are not reporting one. What is real is the design decision behind it.
> The drone is dispatched to an incident the cameras have already confirmed, and
> it *hovers*. Not patrol — endurance is twenty to forty minutes, and Indian
> DGCA rules do not approve beyond-visual-line-of-sight flight for traffic
> monitoring, so standard operations mean visual line of sight under a
> 120-metre ceiling. Hovering also turns ego-motion compensation from a large
> continuous correction into a small drift correction against a background
> homography. And it uses a different detector from the CCTV model on purpose:
> from above you see a roof, not a side profile, so we target VisDrone rather
> than the eye-level CCTV datasets."

**Direction:** the words "placeholder" and "not measured" must be spoken, not
just captioned. A judge who watches at 1.5x with captions off must still hear
them.

---

## S9 — Measured results + limitations · 4:22–4:58 · B then A

**Visual:** One results table on screen (generate it from
`DEMO/results_summary.md`, do not retype numbers by hand). Then both speakers on
camera for the last 8 s.

**Narration (B), 4:22–4:44:**

> "Numbers, measured, with their sample sizes attached. The collision physics
> engine ranks the true participant pair first on all four ground-truth-labelled
> videos — that is four videos, each worth twenty-five percentage points, and
> the thresholds were selected on those same four, so it is in-sample and we do
> not quote it as a hundred percent. On the sixteen confirmed crash-free traffic
> clips the shipped two-channel rule produced zero collision candidates."

**Narration (A), 4:44–4:58:**

> "And the open problem, because it is the honest headline. With the
> single-channel path-conflict gate, on crash-free footage, we measured one
> hundred and fifty-five point eight false collision alarms per hour, across
> sixteen out of sixteen clips. No threshold fixes it — raising the gate to
> 0.55 takes recall to zero before it touches precision. That is why collisions
> are candidates requiring a human, why the verify-and-reject workflow exists,
> and it is the first thing we would fix next. Thank you."

---

## Recording constraints for this script

- **Never speak a number that is not in `DEMO/results_summary.md`.** If a number
  is not in that file it has not been measured on the shipped code.
- **Never say "detects accidents", "real-time", "100% accurate", or "GPS
  location".** The correct phrases are: *suspected collision candidate*, *runs at
  N FPS on this hardware — measured*, *4/4 on four labelled videos, in-sample*,
  *camera-associated location with stated precision*.
- The REPLAY badge must be visible in any segment driven by
  `DEMO/seed_data/`. Do not crop it out of frame.
- S8's placeholder banner is non-negotiable and must survive any re-edit.
