"""Free path ahead: how far a vehicle can go before something is in the way.

The idea
--------
A vehicle's projected path is not free to run through the world. If another
vehicle is standing on the road ahead, the projection should *stop there* --
that is where this vehicle can get to, and no further. Review put it exactly:
the green line should not cross an object, and when the green line shrinks to
nothing, it means something is immediately ahead.

That single quantity, the length of the unobstructed path, turns out to express
several collision types at once and to be far more honest than the pairwise
tests it replaces:

* **driving into a stopped vehicle** -- the free path shortens to zero while the
  vehicle is still moving. No prediction about the other vehicle is needed,
  because it is not doing anything;
* **rear-end** -- the same, with the obstruction itself moving, so the path
  shortens only as fast as the gap actually closes;
* **anything ahead at all** -- the blocker does not have to be the vehicle we
  happened to pair with, which is what the pairwise tests kept getting wrong.

It is also directly checkable by eye, which matters more than it sounds. The
renderer clips the drawn line at the same point this function reports, so a
reviewer watching the video sees precisely what the detector believes, and can
disagree with it on the spot.

Why shadows cannot block the path
---------------------------------
Review raised this immediately and it is the right worry: a shadow, a wet patch,
a pothole or a lane marking must never stop the line. Nothing here looks at the
image at all. The only things that can obstruct a path are **other detected
vehicles' ground footprints** -- objects the detector has already committed to
as road users. A shadow is not a detection, so it cannot block anything, and a
dark patch of tarmac has no footprint to intersect.

The trade is that the blocker set is exactly as good as the detector: a vehicle
that was never detected cannot block anything either, and the path will run
straight through it. That failure is visible in the overlay rather than hidden,
which is the best that can be done without depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .footprint import Footprint


@dataclass
class Blockage:
    """Where a vehicle's forward path first meets another vehicle."""

    index: int                 # step along the projection at which it is blocked
    point: np.ndarray          # where the path stops
    time_s: float              # how long until it gets there
    blocker_id: int
    gap_px: float              # remaining clear distance right now

    @property
    def is_immediate(self) -> bool:
        """The path has essentially no room left: something is right there."""
        return self.time_s <= 0.20


def inside(f: Footprint, x: float, y: float, inflate: float = 1.0) -> bool:
    """Is a ground point standing on this vehicle's patch of road?"""
    dx = (x - f.cx) / max(f.a * inflate, 1e-6)
    dy = (y - f.cy) / max(f.b * inflate, 1e-6)
    return (dx * dx + dy * dy) <= 1.0


def blockers_from(tracks, exclude_id: int | None = None,
                  depth_ratio: float = 0.35) -> list[tuple[int, Footprint]]:
    """Footprints that can obstruct a path: other detected vehicles, nothing else."""
    out = []
    for tr in tracks:
        tid = int(tr.track_id)
        if exclude_id is not None and tid == exclude_id:
            continue
        out.append((tid, Footprint.from_box(tr.box, depth_ratio)))
    return out


def first_blockage(points, blockers, dt: float,
                   start_index: int = 1, inflate: float = 1.0) -> Blockage | None:
    """Walk a projected path forward and stop at the first vehicle in the way.

    ``points`` are successive ground positions of the projection, ``dt`` the time
    between them. Returns ``None`` when the path is clear for its whole length.

    The first step is skipped by default: a vehicle's own footprint overlaps its
    immediate surroundings, and neighbouring vehicles in dense traffic often
    overlap slightly at the start of the projection without anything being in
    the way.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2 or not blockers:
        return None
    lo = max(1, start_index)
    if lo >= len(pts):
        return None

    # Vectorised over every step and every blocker at once.
    #
    # The nested Python loop this replaces was the single slowest thing in the
    # system: a 4K clip carries seventy simultaneous tracks, each casting three
    # rays of up to ninety steps against seventy footprints, which is over a
    # million point-in-ellipse tests per frame and pushed one 59-second clip to
    # nearly twenty minutes. The arithmetic is identical; only the loop is gone.
    cx = np.array([f.cx for _, f in blockers], dtype=float)
    cy = np.array([f.cy for _, f in blockers], dtype=float)
    ax = np.maximum(np.array([f.a for _, f in blockers], dtype=float) * inflate, 1e-6)
    by = np.maximum(np.array([f.b for _, f in blockers], dtype=float) * inflate, 1e-6)

    seg = pts[lo:]
    dx = (seg[:, 0][:, None] - cx[None, :]) / ax[None, :]
    dy = (seg[:, 1][:, None] - cy[None, :]) / by[None, :]
    hits = (dx * dx + dy * dy) <= 1.0
    rows = np.flatnonzero(hits.any(axis=1))
    if rows.size == 0:
        return None
    i_rel = int(rows[0])
    i = lo + i_rel
    j = int(np.flatnonzero(hits[i_rel])[0])
    travelled = float(np.linalg.norm(pts[i] - pts[0]))
    return Blockage(index=i, point=pts[i], time_s=i * dt,
                    blocker_id=blockers[j][0], gap_px=travelled)


def swept_blockage(pos, vel, half_width_px: float, blockers, horizon_s: float,
                   step_px: float = 8.0, rays: int = 3,
                   inflate: float = 1.15) -> Blockage | None:
    """Block a vehicle's swept CORRIDOR, not a hairline down its centre.

    A vehicle is not a point. Review saw the consequence directly: the centre
    line passed cleanly down the side of another vehicle while the car itself
    ran straight into it, because the line has no width and the car is two
    metres across.

    So the projection is swept at the vehicle's own width -- a plane on the road
    rather than a ray -- by casting parallel rays at the centre and at each
    edge and taking the first thing any of them meets. The corridor is what the
    vehicle will actually occupy, which is the thing that can be obstructed.

    Only detected vehicles can block, so a shadow still cannot stop anything.
    """
    pos = np.asarray(pos, dtype=float)
    vel = np.asarray(vel, dtype=float)
    speed = float(np.linalg.norm(vel))
    if speed < 1e-6 or not blockers:
        return None
    forward = vel / speed
    lateral = np.array([-forward[1], forward[0]], dtype=float)

    reach = speed * horizon_s
    steps = int(np.clip(reach / max(step_px, 1e-6), 12, 90))
    dt_step = horizon_s / steps

    # Only vehicles the corridor could possibly reach. Everything beyond the
    # projection's own length plus its width is irrelevant, and discarding them
    # before the arithmetic starts is what makes this affordable in dense
    # traffic, where most of the seventy tracks on screen are nowhere near.
    margin = reach + half_width_px + 60.0
    near = [(tid, f) for tid, f in blockers
            if abs(f.cx - pos[0]) <= margin and abs(f.cy - pos[1]) <= margin]
    if not near:
        return None
    blockers = near

    offsets = np.linspace(-half_width_px, half_width_px, max(2, rays))
    best: Blockage | None = None
    for off in offsets:
        start = pos + lateral * off
        pts = np.asarray([start + vel * (k * dt_step) for k in range(steps + 1)],
                         dtype=float)
        hit = first_blockage(pts, blockers, dt=dt_step, inflate=inflate)
        if hit is not None and (best is None or hit.time_s < best.time_s):
            best = hit
    return best


def clip_to_blockage(points, sigma, blockage: Blockage | None):
    """Trim a drawn projection so it stops at whatever is in the way."""
    if blockage is None:
        return points, sigma
    end = max(2, blockage.index + 1)
    return points[:end], sigma[:end]
