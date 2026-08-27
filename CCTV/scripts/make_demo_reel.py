"""Stitch annotated clips into one demo reel with explanatory title cards.

The brief asks for a demo video. A montage of annotated clips is not a
substitute for a live walkthrough, but it is the part that can be built
reproducibly from the pipeline's own outputs, so it is built here rather than
recorded by hand: rerunning the pipeline regenerates the reel exactly.

Each segment is introduced by a card stating what the viewer is about to see and
what the system claims about it, including when the claim is "detected, but no
vehicle could be identified". Showing the abstentions alongside the successes is
deliberate -- a reel of only the good cases tells a reviewer nothing about how
the system behaves when it is unsure, which is the thing an operator most needs
to know.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (18, 16, 15)
FG = (238, 238, 238)
DIM = (150, 150, 150)
ACCENT = (120, 220, 120)


def wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def title_card(size, eyebrow: str, title: str, body: str, seconds: float,
               fps: float, writer) -> None:
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    x = int(w * 0.09)
    y = int(h * 0.34)

    cv2.line(img, (x, y - int(h * 0.10)), (x + int(w * 0.06), y - int(h * 0.10)),
             ACCENT, 3, cv2.LINE_AA)
    cv2.putText(img, eyebrow.upper(), (x, y - int(h * 0.06)), FONT,
                w / 2200.0, ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, title, (x, y), FONT, w / 900.0, FG, 3, cv2.LINE_AA)

    yy = y + int(h * 0.09)
    for line in wrap(body, 74):
        cv2.putText(img, line, (x, yy), FONT, w / 2100.0, DIM, 2, cv2.LINE_AA)
        yy += int(h * 0.055)

    for _ in range(int(seconds * fps)):
        writer.write(img)


def banner(frame, text: str, colour=ACCENT):
    """A persistent strip naming what the system is claiming on this clip."""
    h, w = frame.shape[:2]
    bar = int(h * 0.075)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar), (w, h), BG, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (0, h - bar), (w, h - bar), colour, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (int(w * 0.02), h - int(bar * 0.32)), FONT,
                w / 2100.0, FG, 2, cv2.LINE_AA)
    return frame


def describe(result_json: Path) -> tuple[str, str]:
    """One honest line about what the system concluded for this clip."""
    if not result_json.exists():
        return "no result recorded", "unknown"
    d = json.loads(result_json.read_text(encoding="utf-8"))
    evs = d.get("events", [])
    if not evs:
        return "no incident raised", "none"
    parts, mode = [], "none"
    for e in evs:
        trig = e.get("triggers") or {}
        boxes = trig.get("participant_boxes") or []
        det = trig.get("detector", e.get("type", "incident"))
        sev = e.get("severity_band") or ""
        if not sev and isinstance(e.get("severity"), dict):
            sev = e["severity"].get("band", "")
        mode = trig.get("attribution", mode)
        if boxes:
            parts.append(f"{e.get('type', 'incident')} [{det}] "
                         f"{len(boxes)} vehicle(s) named{' - ' + sev if sev else ''}")
        else:
            parts.append(f"{e.get('type', 'incident')} [{det}] "
                         "detected, vehicles NOT identified")
    return "; ".join(parts[:2]), mode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/demo_reel.mp4")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="explicit list like Accidents/3 Traffic/13052943_...")
    ap.add_argument("--per-clip", type=float, default=12.0,
                    help="seconds taken from each clip")
    ap.add_argument("--card", type=float, default=3.5)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    res = Path(args.results)
    picks: list[tuple[str, Path, Path]] = []
    if args.clips:
        for c in args.clips:
            grp, stem = c.split("/", 1)
            picks.append((grp, res / grp / f"{stem}_annotated.mp4",
                          res / grp / f"{stem}.json"))
    else:
        # a spread: incidents that were attributed, incidents that were not,
        # and clean clips that correctly stayed silent
        for grp in ("Accidents", "Traffic"):
            vids = sorted((res / grp).glob("*_annotated.mp4"))
            picks += [(grp, v, v.parent / f"{v.stem.replace('_annotated','')}.json")
                      for v in vids[: 6 if grp == "Accidents" else 3]]

    picks = [(g, v, j) for (g, v, j) in picks if v.exists()]
    if not picks:
        print("no annotated clips found; run scripts/run_problems.py first")
        return 1

    w = args.width
    probe = cv2.VideoCapture(str(picks[0][1]))
    ok, fr = probe.read()
    probe.release()
    if not ok:
        print("could not read the first clip")
        return 1
    h = int(round(w * fr.shape[0] / fr.shape[1] / 2) * 2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
    if not writer.isOpened():
        print("could not open the writer")
        return 1

    title_card((w, h), "ELCIA Track 1", "NETRA",
               "One detector runs always-on. Every incident decision after the "
               "tracker is geometry, temporal statistics and deterministic state "
               "machines, so each alert carries the numbers that produced it.",
               args.card + 1.0, args.fps, writer)
    title_card((w, h), "how to read the overlay", "Blue is past, green is predicted",
               "Blue trails are where vehicles have been. Green cones are where the "
               "motion model expects them next, widened to two sigma so the cone "
               "shows confidence rather than a false-precision line. A collision is "
               "a vehicle failing to arrive inside its own cone.",
               args.card + 1.0, args.fps, writer)

    for grp, vid, js in picks:
        claim, mode = describe(js)
        stem = vid.stem.replace("_annotated", "")
        expected = ("this clip contains a collision" if grp == "Accidents"
                    else "this clip is confirmed crash-free")
        colour = ACCENT if (grp == "Traffic") == (mode == "none") else (70, 70, 240)
        title_card((w, h), f"{grp} / {stem}", expected.capitalize(),
                   f"System says: {claim}.", args.card, args.fps, writer)

        cap = cv2.VideoCapture(str(vid))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        limit = int(args.per_clip * src_fps)
        n = 0
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            banner(frame, f"{grp}/{stem}   {claim}"[:110], colour)
            writer.write(frame)
        cap.release()
        print(f"  + {grp}/{stem}: {n} frames  ({claim})")

    title_card((w, h), "honest by construction", "What it will not claim",
               "No injury, no fault, no collision type, no per-vehicle identity. "
               "Where attribution fails the overlay says so rather than boxing a "
               "vehicle it cannot justify. Seven negative results, each caught by "
               "measuring on crash-free footage first, are in LIMITATIONS.md.",
               args.card + 1.5, args.fps, writer)
    writer.release()
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
