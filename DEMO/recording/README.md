# Recording checklist

Practical companion to `DEMO/script.md` (what to say, timed) and
`DEMO/jury_walkthrough.md` (how to drive the system). This file is about the
mechanics of the recording session itself: format, what must already be
running, and the specific ways this exact demo has been documented to break.

## Resolution / format

- Record at **1920×1080 minimum**, native resolution of the presenting
  laptop — do not downscale a smaller capture and upscale it in post.
- 30 fps capture is enough. Nothing in the dashboard has motion fast enough
  to need more; the annotated collision renders (`DEMO/clips/cctv_positive/`)
  are themselves recorded at their source clip's native fps.
- Export at a bitrate that keeps small text legible under a paused frame —
  the `Trigger values` table, the JSON config, and the severity breakdown are
  exactly the things a judge will pause on. A muddy low-bitrate export
  defeats the "technical evidence" segments (S2, S4, S5 in `script.md`) more
  than any narration mistake would.
- Check the submission portal's upload size/duration limit **before** the
  final export, not after — re-encoding a finished 5:00 take because it's
  10 MB over the limit is an avoidable last-hour problem.
- Audio: a real external or headset microphone, not the laptop's built-in
  one over fan noise. Do a 10-second level check and listen back before the
  first real take.

## Before you press record — have running

1. **CCTV backend on port 8000.** `python run.py serve --port 8000` from
   `CCTV/`, with `detector.weights` pinned first
   (`DEMO/jury_walkthrough.md` §1.2 — this is documented as the single most
   common demo-day failure: an unpinned `null` weight tries to download from
   the internet). Confirm `http://127.0.0.1:8000/api/health` returns 200
   before doing anything else.
2. **The seed_data snapshot loaded**, per `DEMO/seed_data/README.md`, so the
   dashboard is populated without waiting on live inference — unless a
   segment is deliberately planned as a live Route B run, in which case the
   wait time must already be inside that segment's time budget in
   `script.md`, not improvised.
3. **DRONE backend on 8011, if and when it exists** (`jury_walkthrough.md`
   §4). As of this writing there is no server bound to that port — if that
   is still true on recording day, play `DEMO/clips/drone_placeholder/`
   directly for S8, with its banner already burned in, rather than starting
   an empty server and gesturing at a blank page.
4. **APP on port 3000, if and when it's ready** (`jury_walkthrough.md` §5).
   Verify `npm run dev` actually serves a working page — don't assume from
   file presence in `FINAL/APP/`. If it isn't working, use the built-in CCTV
   dashboard at `:8000` for every segment except S6 (the map), which is
   APP-only; if APP's map isn't ready either, use S6's disabled-state
   fallback described in `script.md`.
5. **`CCTV/config/config.yaml` open in a second window**, scrolled to the
   `queue:` block — script.md's S2 "judge-bait moment." Costs nothing to
   have ready even if the judges never ask.
6. **`DEMO/results_summary.md` open and ready to screen-capture** for S9.
   `script.md` is explicit that the results table must come from this file
   on screen, not be retyped by hand onto a slide.
7. Close every other browser tab, chat client, and notification source. One
   popup mid-take is one redo.
8. Windows Focus Assist / Do Not Disturb **on** before recording starts.
9. Laptop plugged in or a full charge confirmed — losing a take to a
   sleep/low-battery interrupt is entirely avoidable.
10. **Rehearse the full click path once with the recorder off**, immediately
    before the real take. `DEMO/jury_walkthrough.md` is written so this
    takes minutes; do it anyway, every session — a stale seed snapshot or a
    changed API response shape can silently break a step that worked
    yesterday.

## Common failure modes to avoid on the day

These are pulled directly from what `CCTV/docs/` and `DEMO/jury_walkthrough.md`
already document as known traps in this exact codebase — not generic advice.

- **Detector tries to download weights on a venue with no internet.**
  `detector.weights: null` in `config.yaml` resolves by name and reaches out
  to the internet. Pin it per `jury_walkthrough.md` §1.2 days before the
  recording session, not while a judge watches a spinning cursor.
- **`--low-resource` fails on a missing checkpoint.** It hardcodes
  `<CCTV>/yolo26n.pt`, which is not where the checkpoint actually lives
  (`CCTV/models/`). Either avoid `--low-resource` in front of judges or copy
  the checkpoint to the repo root beforehand (`jury_walkthrough.md` §1.2).
- **The replay state isn't visually obvious.** If APP's `NEXT_PUBLIC_REPLAY`
  badge isn't confirmed working (`seed_data/README.md`), the speaker must
  say "replay" out loud per `script.md` S1's direction. Silence here is the
  single fastest way to accidentally overclaim "running live."
- **Map buttons are disabled and nobody explains why.** If
  `CCTV/config/road_graph.json` still doesn't exist on recording day
  (it doesn't as of this writing — only 2 of 82 cameras carry a
  `road_edge_id`), use `script.md` S6's honest fallback paragraph. Do not
  skip the segment awkwardly or click a disabled button and hope it's not
  noticed.
- **Speaking a banned phrase.** "Detects accidents," "real-time," "100%
  accurate," "GPS location" are explicitly listed as forbidden in
  `script.md`'s closing recording constraints, with the honest replacement
  phrase given for each. Keep that list visible to whoever is speaking
  during the take, not filed away in the doc.
- **Overrunning 5:00.** Time every segment against `script.md`'s budget
  table on the rehearsal take. If over, cut in the order the script
  specifies — S7 first, then trim S1 — not by improvising a cut live.
- **The drone placeholder banner not surviving the edit.** It must be
  visible for S8's entire duration, not a title card before it. Some overlay
  methods (a separate subtitle track, a browser-window watermark) don't
  survive an export/re-encode — check the **final exported file**, not just
  the live capture in the recording tool.
- **Presenting the physics-engine render as if it's the live dashboard.**
  `DEMO/clips/cctv_positive/` footage comes from a research prototype that
  is not merged into `FINAL/CCTV` (`results_summary.md`). `script.md`'s S4
  narration already keeps this distinction spoken aloud — don't cut between
  the render and the dashboard screen without that line.
- **A stale seed_data snapshot.** If anyone touched `netra/db.py`,
  `netra/severity.py`, or the API response shape after the snapshot was
  frozen, the dashboard can throw errors reading old rows mid-recording.
  Regenerate per `seed_data/README.md` after any such change, and re-run the
  full click-through before trusting it in a real take.
- **Wrong camera ID typed live.** `python run.py status` lists every valid
  ID. Have that output already on screen or copy-pasted — don't type a
  camera ID from memory in front of the camera.
- **Quoting a number not in `results_summary.md`.** If it isn't in that
  file, it hasn't been measured on this project's shipped code. This applies
  to ad-libbed answers to judge questions as much as to the scripted
  narration.

## After recording

- Watch the full export once, full screen, sound on, before submitting.
  Catch dropped audio, a banner that didn't survive the export, and timing
  overruns here — not after the file has already been submitted.
- Re-check total runtime is **≤ 5:00**, including any intro/outro card.
- Re-check every spoken number against `DEMO/results_summary.md` on the
  **final edited audio**, not against the written script — a take can drift
  off-script in the moment, and that drift is what a judge actually hears.
