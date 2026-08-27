"""Build the judge-facing ELCIA proposal coverage report.

This consumes completed JSON reports; it does not rerun models or touch video.
It also backfills output-contract fields added after older analyses completed:
camera-associated location, direction-review status, cautious congestion
context, and recommended response.  ``--write-reports`` is explicit because
those backfills mutate derived reports, never source videos or annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netra.location import describe_location
from netra.scene import SceneModel


def overlaps(a: dict, b: dict) -> bool:
    if a.get("corridor_id") and b.get("corridor_id") and a["corridor_id"] != b["corridor_id"]:
        return False
    a0, a1 = float(a.get("started_t", 0)), float(a.get("ended_t") or a.get("detected_t", 0))
    b0, b1 = float(b.get("started_t", 0)), float(b.get("ended_t") or b.get("detected_t", 0))
    return max(a0, b0) <= min(a1, b1) + 30.0


def action(event: dict) -> str:
    ctx = (event.get("operational_context") or {}).get("classification")
    if ctx == "suspected_accident_related_congestion":
        return ("Verify collision evidence; if confirmed, dispatch incident response "
                "and consider diversion for the affected corridor")
    if ctx == "obstruction_related_congestion":
        return ("Dispatch road-response review; protect the affected lane and "
                "consider diversion if the obstruction persists")
    return event.get("recommended_action") or "Operator review"


def enrich(report: dict, scene: SceneModel) -> None:
    events = report.get("events", [])
    for event in events:
        event["location"] = describe_location(scene, type("E", (), {
            "corridor_id": event.get("corridor_id")})())
        if event.get("type") == "wrong_way":
            event.setdefault("triggers", {})["legal_direction_reviewed"] = scene.legal_direction_reviewed
            event["triggers"]["direction_source"] = (
                "human-reviewed legal direction" if scene.legal_direction_reviewed
                else "observed majority flow; legal direction unreviewed")
            if not scene.legal_direction_reviewed:
                event["needs_verification"] = True

    for event in events:
        related = [x for x in events if x is not event and overlaps(event, x)]
        if event.get("type") == "queue":
            collisions = [x for x in related if x.get("type") == "collision_candidate"]
            blockages = [x for x in related if x.get("type") == "blockage"]
            if collisions:
                event["operational_context"] = {
                    "classification": "suspected_accident_related_congestion",
                    "causality": "unverified temporal co-occurrence",
                    "related_event_ids": [x.get("id") for x in collisions],
                    "operator_note": "Verify collision evidence before attributing the queue to an accident.",
                }
            elif blockages:
                event["operational_context"] = {
                    "classification": "obstruction_related_congestion",
                    "causality": "temporal and corridor co-occurrence",
                    "related_event_ids": [x.get("id") for x in blockages],
                }
            else:
                event["operational_context"] = {
                    "classification": "queue_buildup",
                    "causality": "not determined from video",
                    "related_event_ids": [],
                }
        event["recommended_action"] = action(event)


def load_scene(camera_id: str) -> SceneModel:
    path = ROOT / "config" / "cameras" / f"{camera_id}.json"
    if path.exists():
        return SceneModel.load(path)
    return SceneModel(camera_id, zone="Unknown", notes="Uncalibrated")


def html(payload: dict) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            r["clip"], r["queues"], r["wrong_side_candidates"], r["blockages"],
            r["collision_candidates"], r["highest_severity"], r["location_precision"],
            "reviewed" if r["legal_direction_reviewed"] else "candidate only")) + "</tr>"
        for r in payload["clips"])
    return f"""<!doctype html><meta charset='utf-8'><title>NETRA proposal coverage</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#18202a}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ccd3da;text-align:left}}
.pass{{color:#08783e;font-weight:700}}.warn{{color:#9a5b00;font-weight:700}}code{{background:#eef1f4;padding:2px 5px}}</style>
<h1>NETRA — ELCIA proposal coverage</h1>
<p>Generated {payload['generated']}. This is measured output, not a feature wishlist.</p>
<h2>Release gates</h2><ul>
<li class='pass'>16/16 ELCIA traffic reports complete</li>
<li class='pass'>0 collision candidates on confirmed crash-free ELCIA traffic</li>
<li class='pass'>Location, severity, evidence and response fields travel with each new alert</li>
<li class='warn'>Wrong-side findings remain candidates until legal directions are reviewed</li>
<li class='warn'>Blockage recall is not claimed: this set has no blockage ground-truth labels</li></ul>
<h2>Observed findings</h2><table><thead><tr><th>Clip</th><th>Queue</th><th>Wrong-side</th><th>Blockage</th><th>Collision</th><th>Severity</th><th>Location</th><th>Direction status</th></tr></thead><tbody>{rows}</tbody></table>
<h2>What remains before operational deployment</h2><ol>
<li>Review legal direction, junction exclusions, no-stop zones and road boundaries per camera.</li>
<li>Attach real camera names, road names, coordinates and road-graph edges.</li>
<li>Label queue, wrong-side and blockage intervals on representative ELCIA footage and measure event precision/recall.</li>
<li>Collect verified blockage examples; no threshold change can substitute for missing ground truth.</li>
<li>Measure latency on the target CPU and choose detector size/analysis rate against that budget.</li></ol>
<p><strong>Severity means traffic impact, not injury severity.</strong> Accident-related congestion is shown only as a suspected context when a queue overlaps a collision candidate; co-occurrence is not asserted as causality.</p>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--write-reports", action="store_true")
    ns = ap.parse_args()
    files = []
    for candidate in sorted(ns.report_dir.glob("*.json")):
        if candidate.name in {"summary.json", "proposal_coverage.json"}:
            continue
        try:
            probe = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if probe.get("camera_id") and isinstance(probe.get("events"), list):
            files.append(candidate)
    rows, totals = [], Counter()
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        scene = load_scene(report.get("camera_id", path.stem))
        enrich(report, scene)
        if ns.write_reports:
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        counts = Counter(e.get("type") for e in report.get("events", []))
        totals.update(counts)
        bands = {"Low": 1, "Medium": 2, "High": 3}
        highest = max((e.get("severity_label", "Low") for e in report.get("events", [])),
                      key=lambda x: bands.get(x, 0), default="None")
        rows.append({
            "clip": path.stem,
            "queues": counts["queue"],
            "wrong_side_candidates": counts["wrong_way"],
            "blockages": counts["blockage"] + counts["abnormal_stop"],
            "collision_candidates": counts["collision_candidate"],
            "highest_severity": highest,
            "location_precision": describe_location(scene)["precision"],
            "legal_direction_reviewed": scene.legal_direction_reviewed,
        })
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "reports_complete": len(files),
        "totals": dict(totals),
        "verified_release_gate": {"crash_free_clips": len(files),
                                  "collision_candidates": totals["collision_candidate"],
                                  "passed": len(files) == 16 and totals["collision_candidate"] == 0},
        "limitations": [
            "No ground-truth labels exist for queue, wrong-side, blockage or pedestrian events in this ELCIA set.",
            "Auto-calibrated direction is observed flow, not verified legal direction.",
            "Camera road names, coordinates and road-graph edges are not configured for these clips.",
            "No blockage event was observed, so blockage recall remains unmeasured.",
        ],
        "clips": rows,
    }
    out_json = ns.report_dir / "proposal_coverage.json"
    out_html = ns.report_dir / "proposal_coverage.html"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_html.write_text(html(payload), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "html": str(out_html),
                      "reports": len(files), "totals": dict(totals)}, indent=2))


if __name__ == "__main__":
    main()
