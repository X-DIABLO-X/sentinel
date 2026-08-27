# DEMO/seed_data — GPU-free jury replay snapshot

## What this is

A frozen copy of the three things `CCTV/run.py serve` reads to populate the
dashboard: the SQLite incident database, the evidence tree, and the reports
tree. Confirmed from `CCTV/config/config.yaml`'s `paths:` block:

```yaml
paths:
  database: netra.db      # CCTV/netra.db  — SQLite, opened by netra/db.py IncidentStore
  evidence: evidence       # CCTV/evidence/ — annotated frames + evidence clips, netra/evidence.py
  reports: reports         # CCTV/reports/  — per-run JSON reports and calibration previews
```

Produce these three once, with a real pipeline run on real footage, on
whatever GPU is available beforehand. On demo day, copy them into place and
start the server — the dashboard reads a database and serves static files, so
nothing at request time needs a GPU or re-runs inference. This is route A in
`DEMO/jury_walkthrough.md` §2.

**This is not a synthetic or fabricated incident set.** Every incident in the
snapshot must come from the real pipeline running on real footage — never
hand-write a row into `netra.db` to make the dashboard look populated. That
would violate this project's measured-not-estimated culture more directly
than not having a snapshot at all.

---

## How to generate it (once, before the demo, GPU available)

1. Fresh, verified `CCTV` environment. Follow
   `DEMO/jury_walkthrough.md` §1.1–1.3 first (venv, weights pinned, `run.py
   status` returns 82 cameras) — do not build a snapshot on a broken
   environment.

2. Pick the clips you want in the snapshot:
   - Collision: the four ground-truth videos from
     `DEMO/clips/cctv_positive/` (`POSITIVE_4`, `POSITIVE_11`, `POSITIVE_13`,
     `POSITIVE_14` camera IDs, or the equivalent `PROBLEMSET_*` IDs).
   - Queue / wrong-side: whichever `TRAFFIC_*` clips you confirmed fire, per
     `DEMO/clips/cctv_queue_wrongway_blockage/README.md` guidance in
     `DEMO/clips/README.md`.

3. Run each one:

   ```powershell
   cd FINAL\CCTV
   python run.py process --camera <CAMERA_ID> --video <path-to-clip>.mp4 --seconds 60
   ```

   Each run appends rows to `netra.db`, writes annotated frames/clips under
   `evidence/`, and drops a per-run JSON summary under `reports/`. Watch the
   printed summary block (frames analysed, wall seconds, realtime factor,
   incident count, alerts/video-hour) — if a clip you expected to fire
   produces zero events, that's real information, not a bug to hide; either
   drop it from the snapshot or pick a different clip.

4. **Inspect before freezing.** Start the server and click through every
   incident you just produced:

   ```powershell
   python run.py serve --port 8000
   ```

   Confirm the evidence frame, the clip, the trigger-value table and the
   severity breakdown all look right in the browser. A bug baked into a
   frozen snapshot gets re-demoed every time you load it — catch it now, not
   in front of a judge.

5. Stop the server, then copy the three artifacts out of `CCTV/` into this
   folder:

   ```powershell
   cd FINAL\CCTV
   Copy-Item netra.db "..\DEMO\seed_data\netra.db" -Force
   Copy-Item evidence "..\DEMO\seed_data\evidence" -Recurse -Force
   Copy-Item reports  "..\DEMO\seed_data\reports"  -Recurse -Force
   ```

6. Optionally run `python scripts\bundle_evidence.py` first (check its
   `--help`) to produce portable review clips for any older/historical
   report you want in the snapshot alongside a fresh run — it exists
   precisely to make a report's evidence clip reviewable without re-running
   inference.

7. Write a `manifest.json` here (template below) — this is your record of
   what's inside and your spoken "this is replayed data" evidence, since (as
   of this writing) there is no in-app REPLAY badge to point at. See the
   note in "How APP loads it" below.

### `manifest.json` template

```json
{
  "generated_at": "2026-08-27T00:00",
  "generated_on": "Ryzen 7 7840HS / RTX 4050 Laptop (6 GB) / 16 GB RAM",
  "cctv_checkout_state": "describe how you can identify this exact CCTV checkout — commit hash if under version control, otherwise file hashes of netra/pipeline.py and config/config.yaml",
  "cameras_included": ["POSITIVE_4", "POSITIVE_11", "POSITIVE_13", "POSITIVE_14"],
  "incident_types_present": ["collision_candidate", "queue", "wrong_way"],
  "notes": "Every incident here was produced by a real python run.py process run on real footage, then frozen. No incident was hand-written. Served statically at demo time; no GPU inference happens during jury replay."
}
```

Keep this file accurate — regenerate both the snapshot and the manifest
together, never one without the other.

---

## How to load it on demo day (no GPU needed)

```powershell
cd FINAL\CCTV

# back up whatever's currently there, in case you need to get back to it
Copy-Item netra.db netra.db.bak -ErrorAction SilentlyContinue

# load the frozen snapshot
Copy-Item ..\DEMO\seed_data\netra.db  .\netra.db  -Force
Copy-Item ..\DEMO\seed_data\evidence  .\evidence  -Recurse -Force
Copy-Item ..\DEMO\seed_data\reports   .\reports   -Recurse -Force

python run.py serve --port 8000
```

Open `http://127.0.0.1:8000/`. Verify the same incidents you inspected in
generation step 4 are present. `/api/health` will still report
`"cameras": 82` (the camera *configs* are unchanged and unrelated to the
incident snapshot) — what changed is the incident rows and evidence files.

---

## How APP loads it

APP has no database of its own — it is a thin client over the CCTV API
(`NEXT_PUBLIC_CCTV_API`, see `DEMO/jury_walkthrough.md` §5). "Loading the seed
snapshot for APP" is therefore nothing more than: **put the snapshot into
`CCTV/` and start the CCTV backend before starting APP**, per the steps
above. APP then sees the same populated dashboard through its normal API
calls — there is no separate APP-side loading step.

`NEXT_PUBLIC_REPLAY=1` is documented in `DEMO/jury_walkthrough.md` §5 as the
env var intended to make the APP UI visibly say "replay" instead of silently
looking live. **Verify with whoever owns the APP track that this flag
actually renders a visible badge before relying on it in a recording** — as
of this writing that has not been confirmed from the DEMO side. If no visible
badge exists yet, say the word "replay" out loud instead — `DEMO/script.md`'s
S1 direction already tells the speaker to do exactly that when the badge
isn't on screen. Do not silently let a replayed dashboard look like a live
GPU run.

---

## What this is not

- **Not a substitute for Route B or C** (`DEMO/jury_walkthrough.md` §2) if a
  judge explicitly asks to see inference actually run. Keep a GPU-capable
  machine or extra time budget available for that request; the seed snapshot
  answers "show me the system," not "prove it isn't canned."
- **Not automatically kept in sync with the code.** If `netra/db.py`'s
  schema, `netra/severity.py`'s scoring, or the API's response shape changes
  after you freeze this snapshot, the dashboard can throw errors reading old
  rows against new code. Regenerate the snapshot after any such change, and
  re-run the inspection in generation step 4 before trusting it again.
- **Not a place to store the raw source clips.** Those belong in
  `DEMO/clips/` (see `DEMO/clips/README.md`). This folder holds only the
  pipeline's *output* — the database, evidence, and reports — not the input
  videos.
