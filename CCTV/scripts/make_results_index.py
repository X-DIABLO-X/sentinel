"""Build results/index.html -- one page to review every processed clip.

The point of this page is falsifiability. For each clip it shows the annotated
video next to the exact numbers that produced each incident, so a reviewer can
watch the footage and disagree with the system on specific, stated grounds
rather than in the abstract.

It also reports the two things that matter more than a headline accuracy figure:
how often the system fires when nothing is happening, and how late it is when
something is.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

TYPE_LABEL = {
    "collision_candidate": "Suspected collision",
    "wrong_way": "Wrong-side movement",
    "queue": "Queue / congestion",
    "blockage": "Road blockage",
    "lane_violation": "Wrong lane crossing",
    "pedestrian_on_carriageway": "Pedestrian on carriageway",
    "abnormal_stop": "Abnormal stop",
}


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def clip_card(row: dict, results_dir: Path) -> str:
    if "error" in row:
        return (f'<article class="clip err"><h3>{esc(row["file"])}</h3>'
                f'<p class="bad">Failed: {esc(row["error"])}</p></article>')

    v = row.get("video", {})
    st = row.get("stats", {})
    events = row.get("events", [])
    vid = row.get("annotated_video") or row.get("evidence_video")

    sev_counts = row.get("events_by_severity", {})
    chips = "".join(
        f'<span class="pill {k}">{v2} {k}</span>' for k, v2 in sorted(sev_counts.items())
    ) or '<span class="pill none">no incidents</span>'

    rows = ""
    for e in events:
        late = e["detection_delay"]
        rec = e["onset_recovered_s"]
        rows += f"""<tr>
          <td><span class="dot {esc(e['type'])}"></span>{esc(TYPE_LABEL.get(e['type'], e['type']))}</td>
          <td class="n">{e['started_t']:.2f}s</td>
          <td class="n">{e['detected_t']:.2f}s</td>
          <td class="n">{late:.2f}s</td>
          <td class="n">{rec:.2f}s</td>
          <td class="n">{e['confidence']:.2f}</td>
          <td class="n"><span class="pill {esc(e['severity_label'])}">{e['severity']:.2f}</span></td>
          <td class="why">{esc(e['explanation'])}{
              ' <b class="verify">verify</b>' if e.get('needs_verification') else ''}</td>
        </tr>"""
    table = f"""<table class="ev">
        <thead><tr><th>Event</th><th>Onset</th><th>Alerted</th><th>Delay</th>
        <th>Recovered</th><th>Conf</th><th>Sev</th><th>Why it fired</th></tr></thead>
        <tbody>{rows}</tbody></table>""" if rows else \
        '<p class="quiet">No incidents raised on this clip.</p>'

    video_html = (f'<video src="{esc(vid)}" controls preload="metadata" playsinline></video>'
                  if vid else '<p class="bad">No review video was bundled.</p>')

    return f"""<article class="clip" id="{esc(row['camera_id'])}">
      <header>
        <h3>{esc(row['file'])}</h3>
        <div class="chips">{chips}</div>
      </header>
      <div class="body">
        <div class="vid">{video_html}</div>
        <div class="meta">
          <table class="kv">
            <tr><td>Source</td><td>{v.get('width')}x{v.get('height')} ·
                {v.get('fps', 0):.0f} fps · {v.get('duration', 0):.1f}s</td></tr>
            <tr><td>Mode</td><td>{'auto-calibrated, ' + str(row.get('corridors', 0)) +
                ' corridors' if row.get('calibrated') else
                'uncalibrated — collision + change-point only'}</td></tr>
            <tr><td>Analysed</td><td>{st.get('frames_analysed', 0)} frames ·
                {st.get('analysis_fps', 0):.1f} fps · {st.get('realtime_factor', 0):.2f}x realtime</td></tr>
            <tr><td>Detector p95</td><td>{st.get('detector_latency', {}).get('p95_ms', '—')} ms</td></tr>
            <tr><td>Alert rate</td><td>{row.get('alerts_per_video_hour', 0):.0f} / video-hour</td></tr>
          </table>
        </div>
      </div>
      {table}
    </article>"""


def build(results_dir: Path) -> Path:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.get("group", "?"), []).append(r)

    # headline numbers, computed rather than asserted
    stats_html = ""
    for g, rs in groups.items():
        ok = [r for r in rs if "error" not in r]
        n = len(ok)
        with_ev = sum(1 for r in ok if r.get("events_total", 0) > 0)
        with_coll = sum(1 for r in ok
                        if r.get("events_by_type", {}).get("collision_candidate", 0) > 0)
        total_ev = sum(r.get("events_total", 0) for r in ok)
        secs = sum(r.get("video", {}).get("duration", 0) for r in ok)
        delays = [e["detection_delay"] for r in ok for e in r.get("events", [])]
        recs = [e["onset_recovered_s"] for r in ok for e in r.get("events", [])
                if e["onset_recovered_s"] > 0.25]
        med_delay = sorted(delays)[len(delays) // 2] if delays else 0.0
        stats_html += f"""<div class="gstat">
          <h3>{esc(g)}</h3>
          <div class="kpis">
            <div class="kpi"><b>{n}</b><span>clips</span></div>
            <div class="kpi"><b>{secs / 60:.1f}</b><span>minutes</span></div>
            <div class="kpi"><b>{total_ev}</b><span>incidents</span></div>
            <div class="kpi"><b>{with_ev}/{n}</b><span>clips with ≥1</span></div>
            <div class="kpi"><b>{with_coll}/{n}</b><span>with collision flag</span></div>
            <div class="kpi"><b>{med_delay:.1f}s</b><span>median alert delay</span></div>
            <div class="kpi"><b>{len(recs)}</b><span>onsets recovered</span></div>
          </div></div>"""

    body = ""
    for g, rs in groups.items():
        body += f'<section><h2>{esc(g)}</h2>'
        body += "".join(clip_card(r, results_dir) for r in rs)
        body += "</section>"

    det = summary.get("detector", {})
    out = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NETRA — Results Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{--bg:#0c1013;--panel:#141a1e;--p2:#1a2227;--line:#243037;--ink:#e6edf0;
--ink2:#9fb0b8;--ink3:#6b7f89;--acc:#4fb6cf;--low:#5aa86e;--med:#d99a2b;--high:#dc5450;
--mono:"IBM Plex Mono",monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:14px;line-height:1.55}}
.wrap{{max-width:1240px;margin:0 auto;padding:28px 20px 80px}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:.05em}}
h2{{font-size:15px;text-transform:uppercase;letter-spacing:.14em;color:var(--ink2);
margin:44px 0 16px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
h3{{font-size:15px;margin:0}}
.lede{{color:var(--ink2);max-width:70ch}}
.note{{background:rgba(217,154,43,.08);border:1px solid rgba(217,154,43,.3);
border-radius:3px;padding:12px 14px;margin:20px 0;color:var(--ink2);font-size:13px}}
.note b{{color:var(--med)}}
.gstat{{background:var(--panel);border:1px solid var(--line);border-radius:4px;
padding:14px 16px;margin-bottom:12px}}
.gstat h3{{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--ink3);margin-bottom:10px}}
.kpis{{display:flex;gap:10px;flex-wrap:wrap}}
.kpi{{background:var(--p2);border:1px solid var(--line);border-radius:3px;
padding:8px 14px;min-width:92px}}
.kpi b{{display:block;font-family:var(--mono);font-size:19px;font-variant-numeric:tabular-nums}}
.kpi span{{font-size:10px;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em}}
.clip{{background:var(--panel);border:1px solid var(--line);border-radius:4px;
margin-bottom:18px;overflow:hidden}}
.clip>header{{display:flex;justify-content:space-between;align-items:center;gap:12px;
padding:12px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}}
.body{{display:grid;grid-template-columns:1.6fr 1fr;gap:16px;padding:16px}}
@media(max-width:860px){{.body{{grid-template-columns:1fr}}}}
video{{width:100%;border-radius:3px;background:#000;display:block}}
table.kv{{width:100%;border-collapse:collapse;font-size:12px}}
table.kv td{{padding:5px 6px;border-bottom:1px solid #1c252a;vertical-align:top}}
table.kv td:first-child{{color:var(--ink3);width:40%}}
table.kv td:last-child{{font-family:var(--mono)}}
table.ev{{width:100%;border-collapse:collapse;font-size:12px;margin:0}}
table.ev th{{text-align:left;padding:8px 10px;background:var(--p2);color:var(--ink3);
font-size:10px;text-transform:uppercase;letter-spacing:.1em;font-weight:500;
border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}
table.ev td{{padding:8px 10px;border-bottom:1px solid #1c252a;vertical-align:top}}
td.n{{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}}
td.why{{color:var(--ink2);font-size:11.5px}}
.pill{{display:inline-block;font-family:var(--mono);font-size:10px;font-weight:600;
padding:2px 7px;border-radius:2px;letter-spacing:.06em}}
.pill.Low{{background:rgba(90,168,110,.15);color:var(--low);border:1px solid var(--low)}}
.pill.Medium{{background:rgba(217,154,43,.15);color:var(--med);border:1px solid var(--med)}}
.pill.High{{background:rgba(220,84,80,.15);color:var(--high);border:1px solid var(--high)}}
.pill.none{{background:var(--p2);color:var(--ink3);border:1px solid var(--line)}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;background:var(--ink3)}}
.dot.collision_candidate{{background:var(--high)}}
.dot.wrong_way{{background:#ff8c50}}
.dot.queue{{background:var(--med)}}
.dot.blockage{{background:#ffc850}}
.dot.lane_violation{{background:#3ca0c8}}
.dot.pedestrian_on_carriageway{{background:#c88cd0}}
.quiet{{color:var(--ink3);padding:14px 16px;margin:0;font-size:13px}}
.bad{{color:var(--high)}}
.verify{{color:var(--acc);font-family:var(--mono);font-size:10px;text-transform:uppercase}}
.chips{{display:flex;gap:6px;flex-wrap:wrap}}
code{{font-family:var(--mono);background:var(--p2);padding:1px 5px;border-radius:2px;font-size:12px}}
</style></head><body><div class="wrap">

<h1>NETRA — results review</h1>
<p class="lede">Every clip in <code>data/problems</code> put through the pipeline.
Each annotated video shows what the system tracked, which corridor it believed it was
in, and the moment an incident fired — with the trigger values printed on the frame.</p>

<div class="note">
<b>Annotation comparison:</b> yellow marks the human accident region at the hand-marked
time; red marks every stored NETRA alert-time participant box. The verdict at the top
uses the model alert closest to the human time, but all model alerts remain visible in
the video. These overlays evaluate preserved predictions; the human annotation was not
fed back into inference. One clip, <code>-FQxK6HdxNU_00</code>, has no hand screenshot and
is explicitly labelled as dataset ground truth.
</div>

<div class="note">
<b>Read the "Delay" and "Recovered" columns.</b> Onset is when the system believes the
incident began; Alerted is when it actually raised it. The gap is the honest cost of
detection. "Recovered" is how many seconds of that gap were clawed back by tracking
optical flow backwards from the stopped vehicle. On the AI City leaderboard this single
quantity separated first place from fifth.
</div>

<div class="note">
<b>The Accidents clips run without a road map.</b> They have no fixed camera geometry, so
wrong-way, queue, lane-crossing and blockage reasoning are switched off rather than left
to invent findings. Only the two detectors that need no map are active: pairwise
trajectory conflict, and the global motion change-point. Every collision finding is
labelled <i>suspected</i> and routed to a human — the current best published system on the
2026 ACCIDENT benchmark reaches 0.571 against a ~0.96 human ceiling, so certainty here
would be a false claim.
</div>

{stats_html}
{body}

<p class="quiet" style="margin-top:40px">
Detector {esc(det.get('weights'))} · imgsz {esc(det.get('imgsz'))} ·
backend {esc(det.get('backend'))} · device {esc(det.get('device'))} ·
generated {esc(summary.get('generated'))}
</p>
</div></body></html>"""

    p = results_dir / "index.html"
    p.write_text(out, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    p = build(Path(args.results))
    print(f"written -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
