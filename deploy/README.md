# Deployment

The console runs as three long-lived processes behind nginx:

| Service | Port | Unit |
|---|---|---|
| CCTV backend (FastAPI) | 8000 | `sentinel-cctv.service` |
| DRONE backend (FastAPI) | 8011 | `sentinel-drone.service` |
| Operator console (Next.js) | 3000 | `sentinel-app.service` |

nginx proxies `/` to :3000, `/cctv-api/` to :8000 and `/drone-api/` to :8011.

## Why these are systemd units

They were originally started with `nohup`. When the host rebooted, nothing
brought them back and the site served 502 until someone noticed — the two
unrelated services on the same box came back on their own precisely because
they were units. `Restart=always` also covers a process dying on its own.

Install (as root, from a checkout on the host):

```bash
cp deploy/systemd/sentinel-*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now sentinel-cctv sentinel-drone sentinel-app
```

## Operating

```bash
systemctl status  sentinel-cctv sentinel-drone sentinel-app
systemctl restart sentinel-cctv          # after changing CCTV/ python
journalctl -u sentinel-app -n 50         # or tail /var/log/sentinel-app.log
```

`TimeoutStartSec` is deliberately generous on the two backends: the CCTV
service loads torch, ultralytics and ~69 scene models before it binds, which
takes roughly 30 seconds on this CPU-only host.

## Deploying a frontend change

`sentinel-app` serves a prebuilt `.next`, so a change to `APP/` needs a build
before the restart:

```bash
cd /opt/sentinel/APP && npm run build
systemctl restart sentinel-app
```

**Run that build in the foreground.** A `setsid`-detached `npm run build` on
this host has been observed to stall indefinitely — parked in `ep_poll` at
~1% CPU with `.next/` untouched for half an hour — while the identical
command run in an attached shell completes normally. A healthy build pegs a
core (200%+ across workers); if it is sitting at ~1%, it is hung, not slow.
The build takes ~20 minutes here.

## Assets that are not in git

Two directories are deployed out of band because they are large and
regenerable, so a fresh host needs them copied across:

- `CCTV/evidence/` (~70MB) — per-incident annotated frames and clips.
  Without it every incident's evidence 404s.
- `CCTV/reports/frame_*.jpg` — cached camera first-frames. These *are*
  committed (see the `.gitignore` exception) because they cannot be
  regenerated from a clone: the source clips are not redistributed.
