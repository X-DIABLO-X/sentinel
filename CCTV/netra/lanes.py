"""Lanes inferred from where vehicles actually drive, not from painted markings.

Why infer rather than detect
----------------------------
Lane markings are the obvious thing to look for and the wrong thing to rely on.
On the footage this system is built for they are worn away, repainted at odd
offsets, buried under a monsoon sheen, hidden by the traffic itself, or simply
absent -- and on an Indian arterial the painted lanes and the used lanes are
frequently different things. A marking-detector inherits every one of those
failures, and inherits them silently.

Vehicles, on the other hand, leave the answer behind them all day. Thousands of
trajectories cross a camera's view, and they are not spread evenly: they
concentrate into bands, separated by gaps nobody drives in. Those bands are the
lanes as used, which is what an incident system actually needs to reason about.
It works on unmarked roads, it works when the markings lie, and it costs no
extra inference because the trajectories are already being computed.

How
---
1. Trajectories that actually went somewhere are kept; parked and shuffling
   vehicles carry no information about lane structure.
2. They are grouped by heading, so a two-way road produces two families that are
   never mixed together. Directions are compared circularly, so a track heading
   north-west and one heading north-east are near each other while +179 and
   -179 degrees are treated as the same direction rather than opposites.
3. Within a family, each trajectory is reduced to its mean lateral offset from
   the family's axis. Sorting those offsets and cutting them wherever the gap
   exceeds roughly half a vehicle width recovers the bands directly -- no
   cluster count has to be guessed, because the gaps declare it.
4. A band becomes a lane when enough distinct vehicles support it, which is what
   keeps an overtaking manoeuvre or one wandering driver from inventing one.

What this is not
----------------
It is not a legal lane map. It records where vehicles do drive, not where they
are permitted to, and on a road where everybody straddles the markings it will
faithfully report the straddle. That is the right failure for an incident
system -- it makes "unusual for this road" measurable, which is the question
being asked -- but it must never be presented as an authority on lane law.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else np.asarray(v, dtype=float) / n


def _circular_gap(a: float, b: float) -> float:
    """Smallest angle between two headings, in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@dataclass
class Lane:
    """One band of road that vehicles actually use."""

    lane_id: str
    centreline: np.ndarray          # (N, 2) ground points, ordered along travel
    width_px: float
    heading_deg: float
    direction_id: int
    support: int                    # distinct vehicles that used it

    @property
    def axis(self) -> np.ndarray:
        return np.array([np.cos(np.radians(self.heading_deg)),
                         np.sin(np.radians(self.heading_deg))], dtype=float)

    def offset_of(self, point) -> float:
        """Signed lateral distance from this lane's centre, in pixels."""
        p = np.asarray(point, dtype=float)
        i = int(np.argmin(np.linalg.norm(self.centreline - p, axis=1)))
        n = np.array([-self.axis[1], self.axis[0]], dtype=float)
        return float(np.dot(p - self.centreline[i], n))

    def contains(self, point, tolerance: float = 1.0) -> bool:
        return abs(self.offset_of(point)) <= 0.5 * self.width_px * tolerance


@dataclass
class LaneModel:
    """The lane structure of one camera, learned from its own traffic."""

    lanes: list[Lane] = field(default_factory=list)
    frame_shape: tuple[int, int] | None = None
    trajectories_used: int = 0

    # ------------------------------------------------------------------
    @classmethod
    def learn(cls, trajectories, frame_shape=None, vehicle_width_px: float = 60.0,
              min_travel_px: float = 60.0, direction_tol_deg: float = 35.0,
              min_support: int = 4, max_lanes_per_direction: int = 8) -> "LaneModel":
        """Recover lanes from a set of ground-point trajectories."""
        paths = []
        for tr in trajectories:
            p = np.asarray(tr, dtype=float)
            if p.ndim != 2 or len(p) < 3:
                continue
            span = float(np.linalg.norm(p[-1] - p[0]))
            if span < min_travel_px:
                continue                 # went nowhere: says nothing about lanes
            u = _unit(p[-1] - p[0])
            if u is None:
                continue
            paths.append((p, float(np.degrees(np.arctan2(u[1], u[0])))))

        model = cls(frame_shape=frame_shape, trajectories_used=len(paths))
        if len(paths) < min_support:
            return model

        # ---- group by direction of travel ------------------------------
        families: list[list] = []
        fam_head: list[float] = []
        for p, h in paths:
            placed = False
            for k, hk in enumerate(fam_head):
                if _circular_gap(h, hk) <= direction_tol_deg:
                    families[k].append((p, h))
                    # running circular mean keeps the family axis honest
                    ang = np.radians([x[1] for x in families[k]])
                    fam_head[k] = float(np.degrees(np.arctan2(
                        np.sin(ang).mean(), np.cos(ang).mean())))
                    placed = True
                    break
            if not placed:
                families.append([(p, h)])
                fam_head.append(h)

        # ---- split each family into bands ------------------------------
        gap_px = max(12.0, 0.5 * vehicle_width_px)
        for d_id, (fam, head) in enumerate(zip(families, fam_head)):
            if len(fam) < min_support:
                continue
            axis = np.array([np.cos(np.radians(head)), np.sin(np.radians(head))])
            normal = np.array([-axis[1], axis[0]])
            origin = np.mean([p.mean(axis=0) for p, _ in fam], axis=0)

            offsets = [(float(np.dot(p.mean(axis=0) - origin, normal)), p)
                       for p, _ in fam]
            offsets.sort(key=lambda r: r[0])

            bands, current = [], [offsets[0]]
            for prev, item in zip(offsets, offsets[1:]):
                if item[0] - prev[0] > gap_px:
                    bands.append(current)
                    current = [item]
                else:
                    current.append(item)
            bands.append(current)

            bands = [b for b in bands if len(b) >= min_support]
            bands.sort(key=len, reverse=True)
            for i, band in enumerate(bands[:max_lanes_per_direction]):
                centre = cls._mean_path([p for _, p in band], axis, origin)
                if centre is None:
                    continue
                spread = float(np.std([o for o, _ in band]))
                width = float(np.clip(4.0 * spread, 0.6 * vehicle_width_px,
                                      2.2 * vehicle_width_px))
                model.lanes.append(Lane(
                    lane_id=f"d{d_id}l{i}", centreline=centre, width_px=width,
                    heading_deg=head, direction_id=d_id, support=len(band)))
        return model

    @staticmethod
    def _mean_path(paths, axis, origin, samples: int = 12):
        """Average several trajectories into one centreline along the axis."""
        s_all, pts = [], []
        for p in paths:
            s = (p - origin) @ axis
            s_all.append((s.min(), s.max()))
            pts.append((s, p))
        lo = float(np.median([a for a, _ in s_all]))
        hi = float(np.median([b for _, b in s_all]))
        if hi - lo < 1e-6:
            return None
        grid = np.linspace(lo, hi, samples)
        out = []
        for g in grid:
            acc = []
            for s, p in pts:
                if s.min() <= g <= s.max():
                    acc.append([np.interp(g, s, p[:, 0]), np.interp(g, s, p[:, 1])])
            if acc:
                out.append(np.mean(acc, axis=0))
        return np.asarray(out, dtype=float) if len(out) >= 2 else None

    # ------------------------------------------------------------------
    def lane_at(self, point, tolerance: float = 1.0) -> Lane | None:
        """Which lane a ground point is in, if any."""
        best, best_off = None, None
        for lane in self.lanes:
            off = abs(lane.offset_of(point))
            if off <= 0.5 * lane.width_px * tolerance and (
                    best_off is None or off < best_off):
                best, best_off = lane, off
        return best

    def departure(self, point_from, point_to) -> tuple[Lane, Lane] | None:
        """A vehicle that has moved from one lane into another."""
        a = self.lane_at(point_from)
        b = self.lane_at(point_to)
        if a is not None and b is not None and a.lane_id != b.lane_id:
            return a, b
        return None

    def to_dict(self) -> dict:
        return {
            "trajectories_used": self.trajectories_used,
            "lanes": [{
                "id": ln.lane_id, "direction": ln.direction_id,
                "heading_deg": round(ln.heading_deg, 1),
                "width_px": round(ln.width_px, 1), "support": ln.support,
                "centreline": [[round(float(x), 1), round(float(y), 1)]
                               for x, y in ln.centreline],
            } for ln in self.lanes],
            "note": ("Lanes are inferred from observed vehicle paths, not from "
                     "painted markings. They describe where traffic does drive, "
                     "not where it is permitted to."),
        }
