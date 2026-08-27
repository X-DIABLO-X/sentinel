"""Per-channel precision and recall, and the counterfactual for each gate.

The clip sets are labelled at the folder level -- every clip in ``Accidents/``
contains a collision, no clip in ``Traffic/`` does -- which makes it possible to
attribute every finding and every false alarm to the channel that raised it.
That attribution matters more than an overall accuracy figure, because the
channels fail in different places and the remedy differs.

It also reports a counterfactual that would otherwise cost a full re-run: what
recall and false-alarm rate *would* have been if a given channel were required
to be corroborated rather than allowed to fire alone. Each event records which
channels agreed, so the answer is already in the reports.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

COLLISION = "collision_candidate"


def load(group_dir: Path) -> list[dict]:
    out = []
    for f in sorted(group_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        d["_clip"] = f.stem
        out.append(d)
    return out


def channel_of(ev: dict) -> str:
    return (ev.get("triggers") or {}).get("detector", "unknown")


def momentum_fired(ev: dict) -> bool:
    ic = (ev.get("triggers") or {}).get("impulse_channel") or {}
    return str(ic.get("impulse", "")) == "momentum exchange"


def hours(reports: list[dict]) -> float:
    return sum((r.get("stats") or {}).get("video_seconds", 0.0)
               for r in reports) / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    res = Path(args.results)
    clean = load(res / "Traffic")
    crash = load(res / "Accidents")
    if not clean and not crash:
        print("no reports found; run scripts/run_problems.py first")
        return 1

    clean_h, crash_h = hours(clean), hours(crash)
    print(f"crash-free footage : {len(clean)} clips, {clean_h * 60:.1f} min")
    print(f"collision footage  : {len(crash)} clips, {crash_h * 60:.1f} min\n")

    # ---------------- false alarms, by channel ---------------------------
    fa = collections.Counter()
    fa_clips = collections.defaultdict(set)
    for r in clean:
        for e in r.get("events", []):
            if e.get("type") != COLLISION:
                continue
            fa[channel_of(e)] += 1
            fa_clips[channel_of(e)].add(r["_clip"])

    print("FALSE COLLISION ALARMS on confirmed crash-free clips")
    print(f"{'channel':28s} {'events':>7s} {'clips':>6s} {'per hour':>9s}")
    print("-" * 54)
    for ch, n in fa.most_common():
        print(f"{ch:28s} {n:>7d} {len(fa_clips[ch]):>6d} "
              f"{n / max(clean_h, 1e-9):>9.1f}")
    if not fa:
        print("  none")
    print(f"{'TOTAL':28s} {sum(fa.values()):>7d} "
          f"{len(set().union(*fa_clips.values()) if fa_clips else set()):>6d} "
          f"{sum(fa.values()) / max(clean_h, 1e-9):>9.1f}")

    # ---------------- recall, by channel ---------------------------------
    detected, by_ch = set(), collections.Counter()
    boxed = set()
    for r in crash:
        for e in r.get("events", []):
            if e.get("type") != COLLISION:
                continue
            detected.add(r["_clip"])
            by_ch[channel_of(e)] += 1
            if ((e.get("triggers") or {}).get("participant_boxes") or []):
                boxed.add(r["_clip"])

    print(f"\nRECALL on collision clips: {len(detected)}/{len(crash)} detected, "
          f"{len(boxed)}/{len(crash)} with a vehicle named")
    print(f"{'channel':28s} {'events':>7s}")
    print("-" * 38)
    for ch, n in by_ch.most_common():
        print(f"{ch:28s} {n:>7d}")

    # ---------------- counterfactual -------------------------------------
    # What if background-stationary had to be corroborated by the momentum test?
    kept_clean = collections.Counter()
    for r in clean:
        for e in r.get("events", []):
            if e.get("type") != COLLISION:
                continue
            if channel_of(e) == "momentum-exchange" or momentum_fired(e):
                kept_clean["kept"] += 1
            else:
                kept_clean["suppressed"] += 1

    kept_crash = set()
    for r in crash:
        for e in r.get("events", []):
            if e.get("type") != COLLISION:
                continue
            if channel_of(e) == "momentum-exchange" or momentum_fired(e):
                kept_crash.add(r["_clip"])

    print("\nCOUNTERFACTUAL: require momentum corroboration to raise a collision")
    print(f"  false alarms : {sum(fa.values())} -> {kept_clean['kept']} "
          f"({kept_clean['kept'] / max(clean_h, 1e-9):.1f}/h, "
          f"was {sum(fa.values()) / max(clean_h, 1e-9):.1f}/h)")
    print(f"  recall       : {len(detected)}/{len(crash)} -> "
          f"{len(kept_crash)}/{len(crash)}")
    print("\nThis is the whole trade, stated in one place. A channel that cannot")
    print("fire alone costs recall; one that can costs precision. Which is")
    print("right depends on whether an operator is drowning in alerts or")
    print("missing incidents, and that is a deployment decision, not ours.")

    # ---------------- other incident types --------------------------------
    other = collections.Counter()
    for r in clean + crash:
        for e in r.get("events", []):
            if e.get("type") != COLLISION:
                other[e.get("type", "?")] += 1
    if other:
        print("\nOTHER INCIDENT TYPES RAISED (both sets)")
        for k, v in other.most_common():
            print(f"  {v:4d}  {k.replace('_', ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
