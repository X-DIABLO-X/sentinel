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

## nginx and page-load latency

`nginx-sentinel.conf` (the `sentinel.cypherion.tech` server block) and
`nginx-sentinel-cache.conf` (a cache zone, kept in its own `conf.d` file so
the shared `nginx.conf` and the other sites on the box stay untouched).

Two things in there exist purely because **this host's CPU is very slow**.
Measured: a 5-million-iteration pure-Python loop takes **3.4 seconds**, where
a normal core does it in roughly 300ms — about 10x slower. Consequences that
show up everywhere:

- A TLS handshake costs **120ms-1000ms** of CPU. Once a connection is
  established and reused, responses come back in well under a millisecond, so
  the cost is per-connection, not per-request.
- Node needed **160-345ms** just to hand back a static `.js` file from disk.
- `npm run build` takes ~20 minutes.

So the config keeps the slow origin off the critical path rather than trying
to optimise the app around it:

- `/_next/static/` is served straight from disk by nginx. Content-hashed and
  immutable; there is no reason to wake Node for it.
- HTML gets a 5-minute nginx micro-cache with `proxy_cache_background_update`
  and `proxy_cache_lock`, so one request refreshes an entry while everyone
  else gets an instant hit instead of queueing behind a slow render. The pages
  are fully client-rendered shells with no per-user content, so this is safe.
- The `/cctv-api/` and `/drone-api/` routes are deliberately **not** cached —
  that data has to be live.

Measured end to end through the domain after these changes: `/` 3.51s -> 0.74s,
`/cctv` 1.41s -> 0.74s, `/incidents` 1.17s -> 0.70s.

### The two remaining levers

Neither is a code change:

1. **Cache HTML at the Cloudflare edge.** Static assets already come back
   `cf-cache-status: HIT`, but HTML is `DYNAMIC` — Cloudflare does not cache
   HTML by default regardless of origin headers, so every page navigation
   still crosses the network to this box. A Cache Rule in the Cloudflare
   dashboard matching `sentinel.cypherion.tech/*` with "Eligible for cache"
   would serve the shells from the edge and remove the origin from page loads
   almost entirely. This is the single biggest remaining win.
2. **A faster host.** The ~10x CPU deficit sets a floor that no amount of
   caching removes — it is why builds take 20 minutes and why the first
   connection from any new client is slow.

## Assets that are not in git

Two directories are deployed out of band because they are large and
regenerable, so a fresh host needs them copied across:

- `CCTV/evidence/` (~70MB) — per-incident annotated frames and clips.
  Without it every incident's evidence 404s.
- `CCTV/reports/frame_*.jpg` — cached camera first-frames. These *are*
  committed (see the `.gitignore` exception) because they cannot be
  regenerated from a clone: the source clips are not redistributed.
