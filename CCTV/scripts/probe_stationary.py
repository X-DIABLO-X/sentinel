"""Print every stationary object and its crash score, for one clip.

Clip 10 stopped firing entirely, and a result file with zero events says nothing
about why. The trigger is a threshold on ``StationaryDetector.crash_score``, so
the useful question is what that score actually was and which term suppressed
it. This wraps the scorer, records every call, and reports the highest-scoring
object together with the gates that fired on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from netra.config import load_config          # noqa: E402
from netra.detect import Detector             # noqa: E402
from netra.pipeline import Pipeline           # noqa: E402
from netra.stationary import StationaryDetector  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_problems import make_uncalibrated_scene, probe  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--long-side", type=int, default=1920)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    seen: list[tuple[float, dict]] = []
    original = StationaryDetector.crash_score

    def traced(o):
        score, detail = original(o)
        seen.append((float(score), dict(detail)))
        return score, detail

    StationaryDetector.crash_score = staticmethod(traced)

    cfg = load_config()
    cfg["pipeline"]["resize_long_side"] = args.long_side
    cfg["detector"]["imgsz"] = args.imgsz
    det = Detector(weights=cfg["detector"].get("weights", "yolo26m.pt"),
                   imgsz=args.imgsz, conf=cfg["detector"]["conf"], device=args.device)
    info = probe(Path(args.video))
    scene = make_uncalibrated_scene("PROBE", args.video, info, "Accidents")
    pipe = Pipeline(scene, cfg, detector=det, write_evidence=False)
    pipe.run()

    if not seen:
        print("crash_score was never called: no stationary object was ever formed.")
        print("The failure is upstream of scoring -- background modelling or detection.")
        return 0

    seen.sort(key=lambda x: -x[0])
    gate = float(cfg.get("events", {}).get("collision", {}).get("stationary_gate", 0.42))
    print(f"\n{len(seen)} stationary-object scorings; trigger gate is {gate}\n")
    print(f"{'score':>6s}  terms")
    print("-" * 78)
    for score, d in seen[: args.top]:
        if "gate" in d:
            print(f"{score:>6.3f}  VETOED: {d['gate']}")
            continue
        bits = []
        for k in ("road_term", "rollover_term", "person_term", "debris_term",
                  "companion_term", "dwell_term"):
            if k in d:
                bits.append(f"{k.split('_')[0]}={d[k]:.2f}")
        for k in ("arrived_moving", "ever_moved", "parked", "queue_member",
                  "present_from_start"):
            if k in d and d[k]:
                bits.append(k)
        if "stop_decel_px_s2" in d:
            sd = d["stop_decel_px_s2"]
            if sd <= 0.0:
                bits.append("decel=unmeasured")
            else:
                bits.append(f"decel={sd:.0f}" + ("  <gentle,x0.45>" if sd < 25 else ""))
        print(f"{score:>6.3f}  " + "  ".join(bits))

    best = seen[0][0]
    print("-" * 78)
    print(f"best {best:.3f} vs gate {gate:.2f} -> "
          + ("FIRES" if best >= gate else "SUPPRESSED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
