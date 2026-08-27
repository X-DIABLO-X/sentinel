"""Build a presentation-safe result set from measured pipeline reports.

This does not rerun inference and never changes event time or event selection.
It applies the same participant-attribution gate now used by Pipeline, then
writes a separate release-candidate directory so the reviewed baseline remains
recoverable and byte-for-byte untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SKIP = {"annotation_comparison.json", "appearance_localizer_audit.json",
        "cv_candidate_ceiling.json", "cv_deployable.json"}


def safe_attribution(event: dict) -> bool:
    triggers = event.get("triggers") or {}
    selection = triggers.get("participant_selection") or {}
    if (triggers.get("attribution") == "stationary-object-track" and
            str(selection.get("mode", "")).startswith("single-vehicle") and
            not bool(selection.get("off_road"))):
        return False
    if triggers.get("attribution") != "path-crossing":
        return True
    ids = list(dict.fromkeys(event.get("track_ids") or []))
    if len(ids) < 2 or float(event.get("detected_t", 0.0)) < 1.5:
        return False
    pc = triggers.get("path_conflict_channel") or {}
    score = float(pc.get("score", 0.0) or 0.0)
    gates = pc.get("gates") or {}
    deviations = pc.get("heading_change_deg") or []
    if (str(gates.get("geometry", "")) == "rear-end" and score < 0.80 and
            max([float(x) for x in deviations] or [0.0]) < 12.0):
        return False
    corr = triggers.get("corroboration") or {}
    agreeing = int(corr.get("independent_geometries_at_this_moment", 0) or 0)
    return agreeing >= 2 or score >= 0.80


def withhold(event: dict) -> None:
    triggers = dict(event.get("triggers") or {})
    prior_attribution = triggers.get("attribution")
    pc = triggers.get("path_conflict_channel") or {}
    corr = triggers.get("corroboration") or {}
    triggers["participant_boxes"] = []
    triggers["participant_p_crashed"] = []
    triggers["participant_selection"] = {
        "mode": "unattributed after release gate",
        "reason": ("event time retained; participant identity lacked "
                   "independent outcome evidence required for a red box"),
        "prior_attribution": prior_attribution,
        "path_score": float(pc.get("score", 0.0) or 0.0),
        "independent_geometries": int(corr.get(
            "independent_geometries_at_this_moment", 0) or 0),
    }
    triggers["attribution"] = "unattributed"
    triggers["attribution_note"] = (
        "participant identity was not corroborated; no red box drawn rather "
        "than mark an uninvolved vehicle")
    event["track_ids"] = []
    event["triggers"] = triggers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ProblemSet/Results_localization_baseline")
    ap.add_argument("--target", default="ProblemSet/Results_release_candidate")
    args = ap.parse_args()
    source, target = Path(args.source), Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    copied = withheld_count = 0
    for path in sorted(source.glob("*.json")):
        if path.name in SKIP or path.name.endswith("_candidates.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name != "summary.json":
            for event in payload.get("events", []):
                if event.get("type") == "collision_candidate" and not safe_attribution(event):
                    withhold(event)
                    withheld_count += 1
        pending = target / f"{path.name}.tmp"
        pending.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        pending.replace(target / path.name)
        copied += 1

    # Preserve experiment audit files separately; they are not live reports.
    audit = target / "experimental_audit"
    audit.mkdir(exist_ok=True)
    for name in SKIP:
        src = source / name
        if src.is_file():
            shutil.copy2(src, audit / name)
    print(f"release candidate: {copied} reports, {withheld_count} unsafe participant attributions withheld")
    print(target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
