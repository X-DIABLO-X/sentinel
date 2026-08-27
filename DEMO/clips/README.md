# DEMO/clips — what belongs here and where to get it

Three folders, three different honesty statuses. Read `DEMO/results_summary.md`
first — this file tells you *where to source* footage; that one tells you what
you're allowed to claim about it.

```
clips/
  cctv_positive/                    real collision footage, CCTV-angle
  cctv_queue_wrongway_blockage/     real ELCIA-camera footage, queue/wrong-side
  drone_placeholder/                NOT real challenge footage — labelled as such
```

---

## `cctv_positive/`

**Purpose:** real, CCTV-angle vehicle-collision footage for script.md's S4
("Collision candidate + physics evidence").

**Source it from `IDEAS/COMBINED/OUTPUT/2nd_inference/`.** That directory
already contains rendered videos with the physics engine's reasoning burned
in: top pair boxed red with score + interaction type + relative heading, the
`<< CONTACT` marker at the refined sub-frame contact instant, and the
brake-to-avoid bystanders thin-boxed with their much lower score. Use the
four videos where this engine's top-1 pick is **confirmed correct against
ground truth** (`2nd_inference/README.md`):

| video | file | score | interaction type |
|---|---|---|---|
| 4 | `4_inference2.mp4` | 0.680 | crossing (cleanest margin — lead with this one) |
| 11 | `11_inference2.mp4` | 0.401 | crossing |
| 13 | `13_inference2.mp4` | 0.392 | crossing (the braking-bystander fix — good narrative if you want to explain the rotation gate) |
| 14 | `14_inference2.mp4` | 0.174 | crossing |

Copy command (PowerShell, from repo root):

```powershell
Copy-Item "IDEAS\COMBINED\OUTPUT\2nd_inference\4_inference2.mp4"  "FINAL\DEMO\clips\cctv_positive\"
Copy-Item "IDEAS\COMBINED\OUTPUT\2nd_inference\11_inference2.mp4" "FINAL\DEMO\clips\cctv_positive\"
Copy-Item "IDEAS\COMBINED\OUTPUT\2nd_inference\13_inference2.mp4" "FINAL\DEMO\clips\cctv_positive\"
Copy-Item "IDEAS\COMBINED\OUTPUT\2nd_inference\14_inference2.mp4" "FINAL\DEMO\clips\cctv_positive\"
```

**If you want a raw "before the overlay" establishing shot**, the same four
clips exist unannotated at `TESTING/VIDEOS/{4,11,13,14}.mp4` and (identical
footage) at `DATASET/ACCIDENT/POSITIVE/{4,11,13,14}.mp4`.

**The critical caveat, restated from `results_summary.md`:** this physics
engine is a research prototype in `IDEAS/COMBINED/` and is **not merged into
`FINAL/CCTV`**. It is not what `python run.py serve` produces. Script.md's S4
already keeps this straight in its narration ("here is the reasoning
channel" for the render, "suspected collision — not automatically confirmed"
for the live dashboard banner) — when you cut this footage into the video,
preserve that spoken transition. Do not let the render play immediately after
the dashboard screen without the line that separates them.

**For the shipped-system half of S4** (the dashboard's own
`collision_candidate` incident, banner and all), pull that from a live or
`seed_data/`-replayed screen capture of the actual CCTV dashboard — see
`DEMO/seed_data/README.md`. That capture belongs in this same folder once you
have it (name it clearly, e.g. `dashboard_collision_candidate_screencap.mp4`,
so it's obviously distinct from the physics-engine renders).

---

## `cctv_queue_wrongway_blockage/`

**Purpose:** real ELCIA-camera footage showing NETRA's queue and wrong-side
heads actually firing, for script.md's S2 and S3.

**The source footage is not in this workspace.** Every camera config under
`CCTV/config/cameras/TRAFFIC_*.json` points at
`data\problems\Traffic\<id>.mp4`, and every `ELCIADATASET_*.json` points at
`results\ElciaDataSet\<id>.mp4` — both relative paths from whatever machine
NETRA was originally developed and calibrated on. A full search of
`D:\HARSHIT\ELCIA` for `ProblemSet`, `data\problems`, or `results\ElciaDataSet`
turns up nothing outside those camera-config path strings. **These are the 16
ELCIA-supplied traffic clips referenced throughout `CCTV/docs/` — they must be
retrieved from wherever the team originally downloaded the challenge's sample
videos, not generated from anything already in this repo.**

**Once you have them, here's how to find out which ones actually fire:**

```powershell
cd FINAL\CCTV
python run.py status        # lists all 82 configured camera IDs — confirm the TRAFFIC_* ones you have
python run.py process --camera TRAFFIC_<id> --video <path-to-that-clip>.mp4 --seconds 60
```

`SUBMISSION_READINESS.md` reports **6 queue findings and 8 wrong-side
findings across the 16 ELCIA traffic clips in total** — but does not name
which specific clips produced them. Don't guess: process each clip you've
retrieved and read the dashboard's incident list (or the printed
`events=…` count in the console) to see which ones actually fired, then use
those. Once a clip is confirmed to produce a finding, save its annotated
output/evidence clip here.

**If the original videos cannot be located before recording day:**
script.md's S2/S3 already assume the camera is screen-captured live from the
dashboard, not played back as a standalone file — so this folder is only
needed for a raw "establishing shot" before the cut to the dashboard, and is
skippable without weakening the demo. Do not substitute footage from a
different, unrelated camera and call it an ELCIA queue/wrong-side example —
if nothing here fires by recording day, drop the establishing shot and go
straight to the dashboard capture.

---

## `drone_placeholder/`

**Purpose:** honest stand-in footage for script.md's S8. **This must never be
presented as real drone footage of this challenge's traffic, and the on-screen
banner from script.md S8 —**
`PLACEHOLDER FOOTAGE — NOT CHALLENGE DATA · DRONE PIPELINE IS SCAFFOLDING, NOT MEASURED`
**— must be burned in or overlaid for the clip's entire duration in this
folder, not added only at edit time.**

**Candidate source already in this workspace:**
`EVAL/SmartMobility-ELCIA-Techtronics/test_videos/` contains
`drone_accident_footage.mp4`, `drone_traffic.mp4`, `drone_stoppage.mp4`,
`wrong_way_drone1.mp4`, and `sample_traffic_mock.mp4`. That directory belongs
to a **different team's separate submission idea** bundled for reference
under `EVAL/` — before using anything from it:

1. **Open and watch it.** Confirm the viewpoint is genuinely elevated/oblique
   — the one thing this placeholder needs to sell is the hover-escalation
   concept, so an eye-level clip with an aerial-sounding filename defeats the
   purpose even before the license question.
2. **Confirm you have the right to use it.** It is not our footage, its
   license and origin are unknown from the filename alone, and it may itself
   be a placeholder the other team never cleared for reuse. Do not assume.

**If it doesn't check out**, use verifiably public-domain or Creative-Commons
aerial traffic footage (Wikimedia Commons, Pexels, Pixabay — CC0 or CC-BY
only, and note the source/license in this folder alongside the file).
`CCTV/LIMITATIONS.md` already establishes the precedent for this project: its
collision-channel thresholds were fitted in part on "3 CC/PD-licensed
Wikimedia clips including Indian urban traffic" — using cleared external
footage to fill a real data gap is already this project's normal practice,
not a shortcut invented for the demo.

**Whatever ends up here:** keep it out of `cctv_positive/` and vice versa —
the folder boundary is the honesty boundary, and it needs to survive into
however you organize the final edited video's source files, not just this
raw-clips staging area.
