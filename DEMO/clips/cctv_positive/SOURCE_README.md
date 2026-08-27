# 2nd inference — results

**4/4 correct on every labelled video** (was 3/4, and the one it got wrong was
video 13).

## What was wrong, and the physics that fixed it

### 1. A braking bystander outranked the real collision

Measured on video 13 at the contact instant:

```
#1320   decel 0.93   momentum 0.99   yaw 0.00   aspect 0.00
```

That vehicle sheds nearly all its speed and momentum and rotates **not at
all**. It is a driver standing on the brakes to avoid running into the crash
ahead — one of the vehicles queued behind the accident. Under a purely
additive score, "decelerated hard" looked identical to "was hit".

The separating physics is absolute:

> **Braking decelerates you along your own axis. Being struck rotates you.**

So rotation is now a multiplicative **gate** on impact evidence, not one
additive term among many. A vehicle with no rotation keeps only
`rotation_floor` (0.20) of its kinematic score. That single change is what
demotes the brake-to-avoid vehicles — in the rendered video 13 they now score
0.03–0.05 against the true pair's 0.39.

### 2. Oncoming traffic passing was counted as contact

The old top pick, #1320 ↔ #1322, had a relative heading of **178.4°** — two
vehicles travelling in *opposite directions*, on opposite sides of the road.
Their boxes overlap in the image as they pass because a 2-D box cannot encode
depth. They never touch.

The true pair sits at **53°** — a genuine crossing / T-bone, exactly as
described. Interaction geometry now classifies every contact:

| type | rel. heading | what it usually is |
|---|---|---|
| `crossing` | 45–160° | T-bone — the classic intersection collision |
| `following` | 0–45° | queuing / brake-to-avoid (occasionally a real rear-end) |
| `oncoming` | 160–180° | passing safely on opposite sides (occasionally a real head-on) |

This is a prior on *how much evidence to demand*, never a veto — real rear-end
and head-on collisions stay reachable when the rotation evidence is there.

**One correction found while testing this:** the first threshold put `oncoming`
at ≥135°, which classified video 11's *real* 138° collision as "passing
traffic" and demoted it. Two vehicles genuinely passing in opposing lanes are
near-exactly antiparallel (the video 13 false positive: 178°); 138° is a
converging angled impact. Threshold moved to 160°.

### 3. Full frame rate, with gap interpolation and sub-frame refinement

Tracking already ran every frame, but detectors drop frames — measured: 14
missing across #1320's 705-frame span. A collision lasts a handful of frames,
so the true closest approach can fall inside a dropout and be missed entirely.

Now: positions and boxes are linearly interpolated across gaps, the gap curve
is evaluated at **every** frame both tracks could exist on, and the minimum is
refined **sub-frame** by parabolic fit through the three samples around it. At
30 fps and 15 m/s a vehicle covers ~0.5 m per frame — the same order as the gap
being resolved, so this is not academic.

### 4. Appearance anchoring

The rule from the brief — *the vehicle the crash CNN is looking at is one
participant; the vehicle closest to it at that instant is the other* — is
implemented as an explicit candidate generator. Anchored pairs get a modest
bonus (0.20) and then compete on the same score as the geometric candidates
rather than short-circuiting them. It was set to 0.35 initially, which let the
anchor pick the wrong pair in video 14; reduced to a hint rather than a
determinant.

### 5. A track that dies at impact

Video 13's true participant #1287 has its last frame at t=5.919 — the contact
instant to three decimals. It has no "after", so its rotation is unmeasurable
and it scored near zero. That absence is itself evidence: the tracker lost it
*because* the vehicle deformed and rotated. `break_implies_rotation` now
substitutes for the rotation that could not be observed.

## Results — all 15 videos

| vid | fps | top pair | score | type | ground truth | result |
|---|---|---|---|---|---|---|
| 1 | 14.65 | #1↔#2 | 0.026 | crossing | — | |
| 2 | 30.00 | #266↔#378 | 0.233 | crossing | — | |
| 3 | 21.13 | #2↔#8 | 0.381 | crossing | — | |
| **4** | 24.08 | **#2239↔#2254** | 0.680 | crossing | (2239, 2254) | **CORRECT** |
| 5 | 15.06 | #52↔#58 | 0.105 | unknown | — | |
| 6 | 24.37 | #15↔#284 | 0.155 | unknown | — | |
| 7 | 22.94 | #312↔#395 | 0.064 | unknown | — | |
| 8 | 30.00 | #16↔#57 | 0.273 | crossing | — | |
| 9 | 14.94 | #66↔#159 | 0.265 | unknown | — | |
| 10 | 8.03 | — | — | — | — | no vehicle passed the parked/speed gates |
| **11** | 14.97 | **#22↔#94** | 0.401 | crossing | (22, 94) | **CORRECT** |
| 12 | 14.85 | #1↔#106 | 0.212 | crossing | — | |
| **13** | 29.73 | **#1287↔#1303** | 0.392 | crossing | (1287, 1303) | **CORRECT** |
| **14** | 29.87 | **#1896↔#1902** | 0.174 | crossing | (1896, 1902) | **CORRECT** |
| 15 | 12.82 | #2↔#6 | 0.164 | crossing | — | |

**top-1: 4/4 · in top-3: 4/4**

## What 4/4 is and is not worth

**It is four videos.** Each is worth 25 percentage points. 4/4 is not
meaningfully distinguishable from 3/4 at this sample size and must not be
quoted as 100% accuracy.

**The thresholds were selected using these same four videos.** `oncoming_min_deg`
135→160 and `prior_unknown` 0.75→0.60 each moved a labelled video from wrong to
correct. Both are physically defensible — 138° genuinely is not "passing
traffic", and unmeasurable geometry genuinely is weaker evidence than measured
crossing — but that is an argument, not held-out evidence. **There is no
held-out set.** These are in-sample numbers.

**All four confirmed collisions are `crossing`.** The `prior_crossing = 1.00`
against `following = 0.55` is therefore doing real work and has never been
tested against a true rear-end. If one appears in the unlabelled set, this
system will under-rank it. That is a known, deliberate bias, not an oversight.

**The 11 unlabelled videos are unverified predictions.** Video 1 (score 0.026,
only one candidate pair) and video 7 (0.064) are weak enough that they should
be read as "nothing found" rather than as answers.

## Files

- `<n>_inference2.json` — every candidate pair with full evidence breakdown:
  rotation, yaw °, aspect, decel, speed drop, momentum, track-break,
  appearance, contact gap, sub-frame contact time, relative heading,
  interaction type, anchor flag
- `<n>_inference2.mp4` — top pair boxed red with score + interaction type +
  relative heading, all other vehicles thin-boxed with their best pair score,
  `[GT]` marked where known, `<< CONTACT` at the refined contact instant
- `summary.json` — the table above

## Reproduce

```bash
python run_inference2.py --videos 1-15
python render_inference2.py --videos 1-15
```
