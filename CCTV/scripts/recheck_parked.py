"""Re-score the exact vehicles the review called wrong, with the new classifier.

The hard negatives came entirely from crash-free traffic clips, i.e. from
cameras the positives never appear on. That leaves a confound: a model can post
a beautiful AUC by learning "this camera" rather than "this vehicle is wrecked",
and the aggregate number cannot tell the two apart.

The in-domain test is the one below. It takes the vehicles the previous run
boxed inside the *accident* clips -- same camera, same lighting, same footage as
the real collisions -- and re-scores them. Parked and queued vehicles there are
same-camera still negatives, which is precisely what the training set lacks.
If their scores fall, the model learned damage. If they stay near 1.0, it
learned the camera and the harvest has to be repeated on accident footage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.crashcls import CrashClassifier   # noqa: E402

PREV = Path(sys.argv[1] if len(sys.argv) > 1 else "prev")
CLIPS = Path("data/problems/Accidents")
LONG_SIDE = 1920


def frame_at(video: Path, t: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = fr.shape[:2]
    sc = LONG_SIDE / max(h, w)
    if sc < 1:
        fr = cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    return fr


def main() -> int:
    cc = CrashClassifier()
    if cc.model is None:
        print("classifier unavailable")
        return 1
    print(f"weights: {cc.weights}   threshold: {cc.threshold}\n")
    print(f"{'clip':6s} {'t':>7s} {'old p':>7s} {'new p':>7s}  verdict")
    print("-" * 52)

    olds, news = [], []
    for jf in sorted(PREV.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
        data = json.loads(jf.read_text(encoding="utf-8"))
        clip = CLIPS / f"{jf.stem}.mp4"
        if not clip.exists():
            continue
        for e in data.get("events", []):
            trig = e.get("triggers") or {}
            boxes = trig.get("participant_boxes") or []
            probs = trig.get("participant_p_crashed") or []
            if not boxes:
                continue
            fr = frame_at(clip, float(e.get("started_t", 0.0)))
            if fr is None:
                continue
            new = cc.score_boxes(fr, boxes)
            for i, b in enumerate(boxes):
                op = probs[i] if i < len(probs) and probs[i] is not None else float("nan")
                np_ = new[i] if i < len(new) and new[i] is not None else float("nan")
                olds.append(op)
                news.append(np_)
                delta = np_ - op
                if delta != delta:
                    mark = "n/a"
                else:
                    mark = "dropped" if delta < -0.25 else ("held" if abs(delta) <= 0.25 else "ROSE")
                print(f"{jf.stem:6s} {e.get('started_t', 0):>7.1f} {op:>7.3f} {np_:>7.3f}  {mark}")

    if olds:
        olds, news = np.array(olds, float), np.array(news, float)
        print("-" * 52)
        print(f"{'mean':6s} {'':>7s} {np.nanmean(olds):>7.3f} {np.nanmean(news):>7.3f}")
        print(f"\nabove threshold: was {(olds >= 0.72).sum()}/{len(olds)}, "
              f"now {(news >= 0.72).sum()}/{len(news)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
