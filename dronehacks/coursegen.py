"""Procedural DroneHacks-like courses: one clear route, open spaces, emergent around-paths and windows, no ceilings.

    from dronehacks.coursegen import CourseGenCfg, generate
    c = generate(seed=0)            # -> GenCourse: mesh, gates, pad, launch pose, route
    python -m dronehacks.coursegen gallery --n 20 --out out/coursegen       # top-down review sheet
    python -m dronehacks.coursegen orbit --real --seeds 1 2 3                # 3-D orbit MP4s

Route first: a straight-biased self-avoiding walk on a grid of corridor-width cells (1.05-1.4 m) from a border
start cell to an open landing area (2x2..3x3 cells, boulders, the pad). Walls go ONLY where they bound the route:
every edge between a route cell and anything that is not the next/previous route cell. Cells off the route are
plain open floor (no maze filler, no dead ends, nothing roofed except the optional round tube). Then:
  open spaces   some route cells are widened into an unused neighbour (a bay beside the corridor)
  around paths  a wall between two legs of the route (where it doubles back) may end short of the corner, so the
                route visibly wraps around a free-standing wall end, like the real course
  windows       an opening in a wall between two far-apart legs: a shortcut that clears cp 3 (+5) as on the real one
  tunnel        the round tube (the real course's only roofed element) on a straight run
  objects       boulders pushed to one side of a corridor, low hurdles (fly over), angled panels at turns
Gates: 4 lines across the corridor at 1/5..4/5 of the route (cp 0..3) + the pad (cp 4), as in rules.py. The whole
course is rotated by a random angle; the net alone bounds the arena on 60% of courses. Start: the first route cell
facing the next one (training banks add +-30 deg). NED frame, floor z = 0, walls 1.727 m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dronehacks.mesh import GROUP_ID, CourseMesh

WALL_H = 1.727


@dataclass
class CourseGenCfg:
    cells: Tuple[int, int] = (4, 6)            # grid side length range (cells per side)
    cell_m: Tuple[float, float] = (1.05, 1.4)  # cell size range: corridor width (real first corridor: 1.12 m)
    wall_t: float = 0.06                        # wall thickness (plywood + framing)
    wall_h: float = WALL_H
    net_h: float = 3.0
    margin_m: float = 0.15                      # gap between the outermost cells and the net
    route_frac: Tuple[float, float] = (0.45, 0.75)   # share of the grid the route covers
    p_straight: float = 0.65                    # route keeps its direction with this probability (long clear runs)
    p_border_walls: float = 0.4                 # else the net alone bounds route cells on the border
    p_widen: float = 0.25                       # per route cell: open its wall to an unused neighbour (open space)
    p_divider_gap: float = 0.5                  # a wall between two legs of the route is shortened at its free end
    p_window: float = 0.7                       # a window in a wall between two far-apart legs (a shortcut)
    window_wh: Tuple[float, float] = (0.7, 0.6)
    window_z: Tuple[float, float] = (0.7, 1.1)
    p_tunnel: float = 0.5                       # the round tube (the real course's only roofed element)
    tube_r: Tuple[float, float] = (0.65, 0.8)
    n_boulders: Tuple[int, int] = (0, 3)        # in corridors, pushed to one side
    boulder_r: Tuple[float, float] = (0.10, 0.22)
    boulder_h: Tuple[float, float] = (0.3, 1.2)
    p_hurdle: float = 0.3                       # low bar across the corridor (fly over); no overhead parts
    hurdle_h: Tuple[float, float] = (0.35, 0.7)
    min_passage: float = 0.75
    pad_r: float = 39 * 0.0254
    room_shapes: Tuple[Tuple[int, int], ...] = ((2, 2), (2, 3), (3, 3))   # open landing area (cells)
    room_boulders: Tuple[int, int] = (0, 4)
    n_chamfer: Tuple[int, int] = (0, 3)         # angled panels at route turns
    rotate: bool = True
    launch_yaw_tol_deg: float = 30.0            # the drone is placed pointing the way to go, within this much


@dataclass
class GenCourse:
    mesh: CourseMesh
    gates: List[Dict]                # rules.GATES format; index 4 = window (cp 3) or a dummy far away
    pad_center: np.ndarray           # (2,) NED
    pad_r: float
    launch: np.ndarray               # (3,) NED (on the floor: z = -0.08)
    launch_yaw: float                # NED heading toward the next route cell (nominal; banks add +-launch_yaw_tol)
    route: np.ndarray                # (R,3) NED polyline through the route cell centres at cruise height
    goal: np.ndarray                 # (3,) NED: above the pad centre
    info: Dict = field(default_factory=dict)


# ── geometry helpers ─────────────────────────────────────────────────────────────────────────────────────

class _MeshBuilder:
    def __init__(self):
        self.v, self.f, self.g, self.o = [], [], [], []
        self.n = 0
        self.obj = 0

    def box(self, lo, hi, group):
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                      [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
        f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int32)
        self._add(v, f, group)

    def panel(self, a, b, t, z0, z1, group):
        """A wall panel of thickness t from a to b (NED xy, any angle), between NED heights z0 < z1."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]) * t / 2
        q = [a - n, b - n, b + n, a + n]
        v = np.array([[p[0], p[1], z1] for p in q] + [[p[0], p[1], z0] for p in q], np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                      [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int32)
        self._add(v, f, group)

    def prism(self, cx, cy, r, z0, z1, group, sides=10):
        a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
        v = np.concatenate([np.c_[ring, np.full(sides, z0)], np.c_[ring, np.full(sides, z1)]]).astype(np.float32)
        f = []
        for i in range(sides):
            j = (i + 1) % sides
            f += [[i, j, sides + j], [i, sides + j, sides + i]]
        for i in range(1, sides - 1):
            f += [[0, i + 1, i], [sides, sides + i, sides + i + 1]]
        self._add(v, np.array(f, np.int32), group)

    def tube(self, p0, p1, r, zc, group, sides=16, t=0.04):
        """Hollow round tube along the horizontal segment p0 -> p1 (NED xy), axis at height zc (UP metres), inner
        radius r, wall thickness t: a ring of thin flat panels. The part below the floor is dropped."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        side = np.array([-d[1], d[0]])
        for k in range(sides):
            a0, a1 = 2 * np.pi * k / sides, 2 * np.pi * (k + 1) / sides
            pts = []
            for a in (a0, a1):
                for rr in (r, r + t):
                    off = side * rr * np.cos(a)
                    z = zc + rr * np.sin(a)
                    pts.append((off, z))
            if max(z for _, z in pts) <= 0.0:
                continue                                    # under the floor
            v = []
            for end in (p0, p1):
                for off, z in pts:
                    v.append([end[0] + off[0], end[1] + off[1], -max(z, 0.0)])
            v = np.array(v, np.float32)                     # 8 corners: [end0: i0o,i0i,i1o,i1i][end1: same]
            f = np.array([[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], [0, 4, 5], [0, 5, 1],
                          [2, 3, 7], [2, 7, 6], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5]], np.int32)
            self._add(v, f, group)

    def _add(self, v, f, group):
        self.v.append(v)
        self.f.append(f + self.n)
        self.g.append(np.full(len(f), GROUP_ID[group], np.int8))
        self.o.append(np.full(len(f), self.obj, np.int32))
        self.n += len(v)
        self.obj += 1

    def build(self) -> CourseMesh:
        return CourseMesh(np.concatenate(self.v), np.concatenate(self.f).astype(np.int32),
                          np.concatenate(self.g), np.concatenate(self.o))


def _wall_segment(mb, p0, p1, cfg, gap=None):
    """Vertical wall (NED, z down) from p0 to p1 (axis-aligned), optional rectangular opening
    gap = (centre_along, width, z_bottom_up, height)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    horiz = abs(p1[0] - p0[0]) > abs(p1[1] - p0[1])        # runs along x (north)
    t = cfg.wall_t / 2
    a0, a1 = (min(p0[0], p1[0]), max(p0[0], p1[0])) if horiz else (min(p0[1], p1[1]), max(p0[1], p1[1]))
    c = p0[1] if horiz else p0[0]

    def seg(s0, s1, zlo, zhi):                             # z in UP metres -> NED box
        if s1 - s0 < 1e-3 or zhi - zlo < 1e-3:
            return
        if horiz:
            mb.box((s0, c - t, -zhi), (s1, c + t, -zlo), 'wall')
        else:
            mb.box((c - t, s0, -zhi), (c + t, s1, -zlo), 'wall')
    if gap is None:
        seg(a0, a1, 0.0, cfg.wall_h)
        return
    gc, gw, gz, gh = gap
    seg(a0, gc - gw / 2, 0.0, cfg.wall_h)
    seg(gc + gw / 2, a1, 0.0, cfg.wall_h)
    seg(gc - gw / 2, gc + gw / 2, 0.0, gz)
    seg(gc - gw / 2, gc + gw / 2, gz + gh, cfg.wall_h)


# ── route-first layout ───────────────────────────────────────────────────────────────────────────────────

_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _route_walk(nx, ny, start, target_len, p_straight, rng, tries=400):
    """Self-avoiding walk from ``start`` that prefers going straight: a long, clear route of >= target_len cells."""
    best = None
    for _ in range(tries):
        path, seen = [start], {start}
        d = _DIRS[rng.integers(4)]
        while True:
            c = path[-1]
            opts = [dd for dd in _DIRS if (c[0] + dd[0], c[1] + dd[1]) not in seen
                    and 0 <= c[0] + dd[0] < nx and 0 <= c[1] + dd[1] < ny]
            if not opts:
                break
            if d in opts and rng.random() < p_straight:
                nd = d
            else:
                turns = [dd for dd in opts if dd != d] or opts
                nd = turns[rng.integers(len(turns))]
            d = nd
            n = (c[0] + d[0], c[1] + d[1])
            path.append(n)
            seen.add(n)
        if best is None or len(path) > len(best):
            best = path
        if len(best) >= target_len:
            return best[:max(target_len, int(len(best) * rng.uniform(0.8, 1.0)))]
    return best


def generate(seed: int, cfg: Optional[CourseGenCfg] = None, max_tries: int = 80) -> GenCourse:
    cfg = cfg or CourseGenCfg()
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        c = _try(rng, cfg)
        if c is not None:
            c.info['seed'] = seed
            return c
    raise RuntimeError(f'no valid course for seed {seed}')


def _try(rng, cfg: CourseGenCfg) -> Optional[GenCourse]:
    nx, ny = (int(v) for v in rng.integers(cfg.cells[0], cfg.cells[1] + 1, size=2))
    cs = rng.uniform(*cfg.cell_m)
    W, H = nx * cs, ny * cs
    org = np.array([-W / 2, -H / 2])
    centre = lambda c: org + (np.asarray(c, float) + 0.5) * cs   # noqa: E731
    inside = lambda c: 0 <= c[0] < nx and 0 <= c[1] < ny          # noqa: E731
    border = [(i, j) for i in range(nx) for j in range(ny) if i in (0, nx - 1) or j in (0, ny - 1)]
    start = border[rng.integers(len(border))]
    route = _route_walk(nx, ny, start, int(rng.uniform(*cfg.route_frac) * nx * ny), cfg.p_straight, rng)
    if route is None or len(route) < 11:
        return None
    # open landing area: an rw x rh block around the route's end, made only of the end cell and unused cells
    rw, rh = cfg.room_shapes[rng.integers(len(cfg.room_shapes))]
    if rng.random() < 0.5:
        rw, rh = rh, rw
    room = None
    for k in range(len(route) - 1, 5, -1):
        cx, cy = route[k]
        for ox in range(rw):
            for oy in range(rh):
                blk = {(cx - ox + i, cy - oy + j) for i in range(rw) for j in range(rh)}
                if all(inside(c) for c in blk) and not blk & set(route[:k]):
                    room = blk
                    break
            if room:
                break
        if room:
            route = route[:k + 1]
            break
    if room is None or len(route) < 10:
        return None
    idx = {c: k for k, c in enumerate(route)}
    on = set(route) | room
    # corridor topology: consecutive route cells are joined; the room is one open area; widened cells join an unused
    # neighbour (an open space beside the corridor); everything else between two cells gets a wall
    joined = {frozenset((route[k], route[k + 1])) for k in range(len(route) - 1)}
    for a_ in room:
        for b_ in room:
            if abs(a_[0] - b_[0]) + abs(a_[1] - b_[1]) == 1:
                joined.add(frozenset((a_, b_)))
    open_area = set(room)
    for c in route[1:-1]:
        if rng.random() < cfg.p_widen:
            nb = [(c[0] + d[0], c[1] + d[1]) for d in _DIRS]
            nb = [n for n in nb if inside(n) and n not in on and n not in open_area]
            if nb:
                n = nb[rng.integers(len(nb))]
                joined.add(frozenset((c, n)))
                open_area.add(n)
    region = on | open_area                                  # cells the drone may use (the rest is open floor outside)

    mb = _MeshBuilder()
    mb.box((-W / 2 - 2, -H / 2 - 2, 0.0), (W / 2 + 2, H / 2 + 2, 0.02), 'floor')
    border_walls = rng.random() < cfg.p_border_walls
    # a window: in the wall between two route cells that are far apart along the route (a shortcut, like the real one)
    window = None
    if rng.random() < cfg.p_window:
        cands = [(c, n) for c in route for d in _DIRS for n in [(c[0] + d[0], c[1] + d[1])]
                 if n in idx and idx[n] - idx[c] >= 4 and frozenset((c, n)) not in joined
                 and 0.4 * len(route) <= idx[n] <= 0.95 * len(route)]
        if cands:
            window = cands[rng.integers(len(cands))]
    walls = []
    done = set()
    for c in region:
        for d in _DIRS:
            n = (c[0] + d[0], c[1] + d[1])
            e = frozenset((c, n))
            if e in joined or e in done:
                continue
            done.add(e)
            if not inside(n) and not border_walls:
                continue                                    # the net bounds this side
            lo = org + np.array(c) * cs
            p0, p1 = {(1, 0): (lo + [cs, 0], lo + [cs, cs]), (-1, 0): (lo, lo + [0, cs]),
                      (0, 1): (lo + [0, cs], lo + [cs, cs]), (0, -1): (lo, lo + [cs, 0])}[d]
            divider = n in idx and c in idx                 # between two legs of the route
            walls.append((p0, p1, e, divider))
    for p0, p1, e, divider in walls:
        gap = None
        if window is not None and e == frozenset(window):
            along = abs(p1[0] - p0[0]) > abs(p1[1] - p0[1])
            gap = ((p0[0] + p1[0]) / 2 if along else (p0[1] + p1[1]) / 2, cfg.window_wh[0],
                   rng.uniform(*cfg.window_z), cfg.window_wh[1])
            _wall_segment(mb, p0, p1, cfg, gap)
        elif divider and rng.random() < cfg.p_divider_gap:
            # a divider ends short of the corner: the route can be seen and flown around it (an "around" path);
            # the gate order still rewards following the route
            q = np.asarray(p1, float) - (np.asarray(p1, float) - np.asarray(p0, float)) * rng.uniform(0.25, 0.45)
            _wall_segment(mb, p0, q, cfg)
        else:
            _wall_segment(mb, p0, p1, cfg)
    # round tube (only roofed element), on a straight run with a straight approach and exit
    tunnel, tube_r = [], rng.uniform(*cfg.tube_r)
    if rng.random() < cfg.p_tunnel:
        for L in (3, 2, 1):
            cands = [k0 for k0 in range(2, len(route) - L - 1)
                     if not set(route[k0 - 1:k0 + L + 1]) & room and not any(
                         frozenset((route[j], x)) in joined and x not in route
                         for j in range(k0, k0 + L) for x in [(route[j][0] + d[0], route[j][1] + d[1]) for d in _DIRS])
                     and (all(c[0] == route[k0 - 1][0] for c in route[k0 - 1:k0 + L + 1])
                          or all(c[1] == route[k0 - 1][1] for c in route[k0 - 1:k0 + L + 1]))]
            if cands:
                k0 = cands[rng.integers(len(cands))]
                tunnel = route[k0:k0 + L]
                break
    if tunnel:
        a_, b_ = centre(tunnel[0]), centre(tunnel[-1])
        k = idx[tunnel[0]]
        d = centre(route[k + 1]) - centre(route[k]) if len(tunnel) == 1 else b_ - a_
        d = d / np.linalg.norm(d)
        mb.tube(a_ - d * cs / 2, b_ + d * cs / 2, tube_r, tube_r + 0.15, 'tunnel')
    objects = []
    free_cells = [c for c in route[2:] if c not in room and c not in tunnel]
    rng.shuffle(free_cells)
    for c in free_cells[:int(rng.integers(cfg.n_boulders[0], cfg.n_boulders[1] + 1))]:
        r = rng.uniform(*cfg.boulder_r)
        if cs - cfg.wall_t - 2 * r - cfg.min_passage <= 0.05:
            continue
        lo = org + np.array(c) * cs
        off = np.array([0.0, 0.0])
        off[rng.integers(2)] = rng.choice([-1, 1]) * (cs / 2 - cfg.wall_t / 2 - r - 0.02)
        p = lo + cs / 2 + off
        mb.prism(p[0], p[1], r, -rng.uniform(*cfg.boulder_h), 0.0, 'boulder')
        objects.append(('boulder', p.tolist(), float(r)))
    padc = np.array([centre(c) for c in room]).mean(0)
    pad_r = float(min(cfg.pad_r, cs - 0.15))
    for _ in range(int(rng.integers(cfg.room_boulders[0], cfg.room_boulders[1] + 1))):
        cells = sorted(room)
        c = cells[rng.integers(len(cells))]
        r = rng.uniform(*cfg.boulder_r)
        p = org + np.array(c) * cs + rng.uniform(r + 0.05, cs - r - 0.05, 2)
        if np.linalg.norm(p - padc) < pad_r + r + 0.15 or c == route[-1]:
            continue
        mb.prism(p[0], p[1], r, -rng.uniform(*cfg.boulder_h), 0.0, 'boulder')
        objects.append(('boulder', p.tolist(), float(r)))
    turns = [k for k in range(1, len(route) - 1) if route[k] not in room and route[k] not in tunnel
             and route[k - 1][0] != route[k + 1][0] and route[k - 1][1] != route[k + 1][1]]
    rng.shuffle(turns)
    for k in turns[:int(rng.integers(cfg.n_chamfer[0], cfg.n_chamfer[1] + 1))]:
        c = np.array(route[k])
        din, dout = c - np.array(route[k - 1]), np.array(route[k + 1]) - c
        corner = centre(route[k]) + (din - dout) * cs / 2
        mb.panel(corner - din * cs * 0.38, corner + dout * cs * 0.38, cfg.wall_t, -cfg.wall_h, 0.0, 'wall')
    hurdle = None
    straight = [route[k] for k in range(2, len(route) - 2) if route[k] not in room and route[k] not in tunnel
                and (route[k - 1][0] == route[k + 1][0] or route[k - 1][1] == route[k + 1][1])]
    if straight and rng.random() < cfg.p_hurdle:
        c = straight[rng.integers(len(straight))]
        k = idx[c]
        along_x = route[k - 1][1] == route[k + 1][1]
        lo = org + np.array(c) * cs
        h = rng.uniform(*cfg.hurdle_h)
        if along_x:
            mb.box((lo[0] + cs / 2 - 0.03, lo[1], -h), (lo[0] + cs / 2 + 0.03, lo[1] + cs, 0.0), 'wall')
        else:
            mb.box((lo[0], lo[1] + cs / 2 - 0.03, -h), (lo[0] + cs, lo[1] + cs / 2 + 0.03, 0.0), 'wall')
        hurdle = (c, float(h))
    m = cfg.margin_m
    for (lo, hi) in (((-W / 2 - m - .01, -H / 2 - m, -cfg.net_h), (-W / 2 - m, H / 2 + m, 0)),
                     ((W / 2 + m, -H / 2 - m, -cfg.net_h), (W / 2 + m + .01, H / 2 + m, 0)),
                     ((-W / 2 - m, -H / 2 - m - .01, -cfg.net_h), (W / 2 + m, -H / 2 - m, 0)),
                     ((-W / 2 - m, H / 2 + m, -cfg.net_h), (W / 2 + m, H / 2 + m + .01, 0)),
                     ((-W / 2 - m, -H / 2 - m, -cfg.net_h - .01), (W / 2 + m, H / 2 + m, -cfg.net_h))):
        mb.box(lo, hi, 'net')
    mesh = mb.build()

    gates = []
    room_k = min(idx[c] for c in room if c in idx)
    for q in range(4):
        k = min(max(int(round((q + 1) / 5 * room_k)), 1), room_k - 1)
        ca, cb = centre(route[k - 1]), centre(route[k])
        mid, d = (ca + cb) / 2, (cb - ca) / np.linalg.norm(cb - ca)
        perp = np.array([-d[1], d[0]])
        A, B = mid - perp * cs / 2, mid + perp * cs / 2
        gates.append({'name': f'gate{q + 1}', 'cp': q, 'A': A, 'B': B, 't': (B - A) / np.linalg.norm(B - A),
                      'L': float(np.linalg.norm(B - A)), 'n': d})
    if window is not None:
        a_, b_ = sorted(window, key=lambda c: idx[c])
        ca, cb = centre(a_), centre(b_)
        mid, d = (ca + cb) / 2, (cb - ca) / np.linalg.norm(cb - ca)
        perp = np.array([-d[1], d[0]])
        ww = cfg.window_wh[0]
        A, B = mid - perp * ww / 2, mid + perp * ww / 2
        gates.append({'name': 'window', 'cp': 3, 'A': A, 'B': B, 't': (B - A) / np.linalg.norm(B - A),
                      'L': float(ww), 'n': d})
    else:
        A = np.array([50.0, 50.0])
        gates.append({'name': 'window(none)', 'cp': 3, 'A': A, 'B': A + [0.1, 0], 't': np.array([1.0, 0.0]),
                      'L': 0.1, 'n': np.array([0.0, 1.0])})
    theta = rng.uniform(-np.pi, np.pi) if cfg.rotate else 0.0
    Rm = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    rot = lambda p: np.asarray(p, float) @ Rm.T                       # noqa: E731
    pad = rot(padc)
    s0, s1 = rot(centre(route[0])), rot(centre(route[1]))
    yaw = float(math.atan2(s1[1] - s0[1], s1[0] - s0[0]))
    pts = [rot(centre(c)) for c in route if c not in room] + [pad]
    rt = np.array([[p[0], p[1], -0.8] for p in pts])
    mesh.vertices[:, :2] = rot(mesh.vertices[:, :2]).astype(np.float32)
    for gt in gates:
        if not gt['name'].endswith('(none)'):
            gt['A'], gt['B'], gt['t'], gt['n'] = rot(gt['A']), rot(gt['B']), rot(gt['t']), rot(gt['n'])
    info = {'cells': (nx, ny), 'cell_m': float(cs), 'route_cells': len(route), 'open_cells': len(open_area - room),
            'border_walls': bool(border_walls), 'rotation_deg': float(math.degrees(theta)), 'room': (int(rw), int(rh)),
            'launch_yaw_tol': math.radians(cfg.launch_yaw_tol_deg), 'window': window is not None,
            'tunnel': len(tunnel), 'tunnel_kind': 'round' if tunnel else None, 'hurdle': hurdle is not None,
            'beam': False, 'boulders': len(objects), 'arena_m': (float(W), float(H))}
    return GenCourse(mesh=mesh, gates=gates, pad_center=pad, pad_r=pad_r, launch=np.array([s0[0], s0[1], -0.08]),
                     launch_yaw=yaw, route=rt, goal=np.array([pad[0], pad[1], -0.3]), info=info)


# ── validation + gallery ─────────────────────────────────────────────────────────────────────────────────

def check_solvable(c: GenCourse, clearance_m: float = 0.23, res: float = 0.08, device: str = 'cpu') -> Tuple[bool, float]:
    """Is the pad reachable from the launch through 3-D free space (>= ``clearance_m`` from walls, tunnel roofs,
    boulders, hurdles, beams; between 0.2 m and the wall tops)? Hurdles must be flown over, beams under, tunnels
    through, so a fixed-altitude check is not enough. -> (ok, straight-line launch->pad distance m)."""
    from scipy import ndimage

    from dronehacks.path import build_grid
    from dronehacks.render import MeshScene
    sc = MeshScene(c.mesh.select(('wall', 'tunnel', 'boulder')), 1, device=device)
    half = float(np.abs(c.mesh.select(('net',)).vertices[:, :2]).max()) - 0.1
    grid = build_grid(sc.distance_at, np.array([-half, -half, -(WALL_H - 0.25)]), np.array([half, half, -0.2]), res)
    free = grid.clear >= clearance_m
    lab, _ = ndimage.label(free, structure=np.ones((3, 3, 3)))
    a = lab[grid.index([c.launch[0], c.launch[1], -0.3])]
    b = lab[grid.index([c.pad_center[0], c.pad_center[1], -0.3])]
    return bool(a > 0 and a == b), float(np.linalg.norm(c.launch[:2] - c.pad_center))


def plot_top(c: GenCourse, ax) -> None:
    from matplotlib.patches import Circle
    g = c.mesh.face_group
    v = c.mesh.vertices
    for grp, col, z in (('wall', '#4a4f57', 2), ('tunnel', '#3fa66b', 3), ('boulder', '#4f7fd6', 4)):
        tri = c.mesh.faces[g == GROUP_ID[grp]]
        if len(tri):
            for t in tri:
                p = v[t][:, :2]
                ax.fill(p[:, 1], p[:, 0], color=col, lw=0, zorder=z, alpha=0.9 if grp != 'tunnel' else 0.35)
    ax.add_patch(Circle((c.pad_center[1], c.pad_center[0]), c.pad_r, fill=False, ec='#2a9d3a', lw=2, zorder=6))
    ax.plot(c.route[:, 1], c.route[:, 0], color='#f2b33d', lw=1.2, ls='--', zorder=5)
    for gt in c.gates:
        if gt['name'].endswith('(none)'):
            continue
        col = '#d1495b' if gt['name'] != 'window' else '#9b5de5'
        ax.plot([gt['A'][1], gt['B'][1]], [gt['A'][0], gt['B'][0]], color=col, lw=2.5, zorder=7)
        m = (gt['A'] + gt['B']) / 2
        ax.annotate('', xy=(m[1] + 0.25 * gt['n'][1], m[0] + 0.25 * gt['n'][0]), xytext=(m[1], m[0]),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5), zorder=7)
    ax.annotate('', xy=(c.launch[1] + 0.45 * math.sin(c.launch_yaw), c.launch[0] + 0.45 * math.cos(c.launch_yaw)),
                xytext=(c.launch[1], c.launch[0]), arrowprops=dict(arrowstyle='-|>', color='black', lw=2), zorder=8)
    i = c.info
    ax.set_title(f"seed {i['seed']}  {i['cells'][0]}x{i['cells'][1]} @{i['cell_m']:.2f}m  route {i['route_cells']}"
                 f"{'  win' if i['window'] else ''}{('  ' + i['tunnel_kind'] + ' tun') if i['tunnel'] else ''}{'  hurdle' if i['hurdle'] else ''}"
                 f"{'  beam' if i['beam'] else ''}", fontsize=7)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


_COL = {'floor': (0.78, 0.74, 0.66), 'wall': (0.62, 0.64, 0.68), 'tunnel': (0.25, 0.68, 0.45),
        'boulder': (0.35, 0.52, 0.85), 'paint': (0.85, 0.25, 0.22), 'net': (0.55, 0.68, 0.85)}


def orbit_video(mesh: CourseMesh, path: str, seconds: float = 8.0, fps: int = 30, size=(640, 400),
                radius: float = 8.0, elev_deg: float = 38.0, label: str = '', marks=None, device: str = 'cpu') -> None:
    """Real-time shaded orbit (MP4, x264 CRF 18) around a course. ``marks``: [(north, east, rgb)] floor markers."""
    import os
    import shutil
    import subprocess

    import torch
    from PIL import Image, ImageDraw

    from dronehacks.mesh import GROUPS
    from dronehacks.render import MeshScene
    view = mesh.select(('floor', 'wall', 'tunnel', 'boulder', 'paint'))
    col = np.array([_COL[g] for g in GROUPS])[view.face_group]
    W, H = size
    n, chunk = int(seconds * fps), 15
    f = 0.5 * W / math.tan(math.radians(55.0) / 2)
    sc = MeshScene(view, chunk, W, H, f, f, 60.0, device=device)
    exe = shutil.which('ffmpeg') or os.environ.get('FFMPEG')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    proc = subprocess.Popen([exe, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', path], stdin=subprocess.PIPE)
    bg = np.array([0.06, 0.07, 0.10])
    target = np.array([0.0, 0.0, -0.5])
    extent = float(np.abs(view.select(('wall', 'tunnel', 'boulder')).vertices[:, :2]).max())
    radius = max(radius, 2.6 * extent)                # keep the whole arena in frame
    for c0 in range(0, n, chunk):
        idx = np.arange(c0, min(c0 + chunk, n))
        az = 2 * np.pi * idx / n + 0.6
        el = math.radians(elev_deg) + math.radians(8) * np.sin(2 * np.pi * idx / n)
        g = radius * np.cos(el)
        cam = np.stack([target[0] + g * np.cos(az), target[1] + g * np.sin(az), target[2] - radius * np.sin(el)], 1)
        d = target - cam
        pos, rpy = np.zeros((chunk, 3), np.float32), np.zeros((chunk, 3), np.float32)
        pos[:len(idx)] = cam
        rpy[:len(idx), 1] = np.arctan2(-d[:, 2], np.hypot(d[:, 0], d[:, 1]))
        rpy[:len(idx), 2] = np.arctan2(d[:, 1], d[:, 0])
        r, fc, cs = sc.render_shaded(torch.from_numpy(pos), torch.from_numpy(rpy))
        for k in range(len(idx)):
            hit = fc[k] >= 0
            rgb = col[np.maximum(fc[k], 0)] * (0.3 + 0.7 * cs[k])[..., None]
            fog = np.exp(-np.maximum(r[k], 0) / 40.0)[..., None]
            img = np.where(hit[..., None], rgb * fog + bg * (1 - fog), bg)
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            if label:
                ImageDraw.Draw(im).text((10, 8), label, fill=(235, 235, 235))
            proc.stdin.write(np.asarray(im).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit('ffmpeg failed')


def main(argv=None) -> None:
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gallery')
    g.add_argument('--n', type=int, default=20)
    g.add_argument('--seed0', type=int, default=0)
    g.add_argument('--out', default='out/coursegen')
    o = sub.add_parser('orbit', help='3-D orbit videos (MP4) of generated courses (and the real one with --real)')
    o.add_argument('--seeds', type=int, nargs='+', default=[1, 5, 12])
    o.add_argument('--real', action='store_true')
    o.add_argument('--out', default='out/coursegen')
    a = ap.parse_args(argv)
    if a.cmd == 'orbit':
        import os
        if a.real:
            real = CourseMesh.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'course.npz'))
            orbit_video(real, os.path.join(a.out, 'orbit_real.mp4'), label='real DroneHacks course')
        for sd in a.seeds:
            c = generate(sd)
            i = c.info
            orbit_video(c.mesh, os.path.join(a.out, f'orbit_seed{sd}.mp4'),
                        label=f"generated seed {sd}: {i['cells'][0]}x{i['cells'][1]} cells @{i['cell_m']:.2f} m"
                              f"{', window' if i['window'] else ''}{', ' + i['tunnel_kind'] + ' tunnel' if i['tunnel'] else ''}"
                              f"{', hurdle' if i['hurdle'] else ''}{', beam' if i['beam'] else ''}")
        print('wrote orbit videos to', a.out)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    cols = 5
    rows = math.ceil(a.n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    stats = []
    for k in range(a.n):
        c = generate(a.seed0 + k)
        ok, L = check_solvable(c)
        stats.append((ok, L))
        ax = axs.flat[k]
        plot_top(c, ax)
        ax.set_xlabel(f"{'solvable' if ok else 'BLOCKED'}", fontsize=7)
    for ax in axs.flat[a.n:]:
        ax.axis('off')
    fig.suptitle('Generated courses (top-down, north up): walls grey, tunnel roof green, boulders blue, gates red '
                 '(arrow = travel), window gate purple, route dashed, start arrow black, pad ring green', fontsize=9)
    fig.tight_layout()
    path = os.path.join(a.out, 'gallery.png')
    fig.savefig(path, dpi=110)
    print(f'wrote {path}; solvable {sum(o for o, _ in stats)}/{len(stats)}')


if __name__ == '__main__':
    main()
