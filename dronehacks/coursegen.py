"""Procedural DroneHacks courses: one clear route, no dead ends, no ceilings, elements of the real course in any
order and geometry.

    from dronehacks.coursegen import CourseGenCfg, generate
    c = generate(seed=0)            # -> GenCourse: mesh, gates, pad, launch pose, planned path
    python -m dronehacks.coursegen gallery --real --n 19 --out out/coursegen    # top-down review sheet
    python -m dronehacks.coursegen orbit --real --seeds 1 2 3                   # 3-D orbit MP4s with the path

Skeleton (the real course's): a divider splits the arena into a serpentine half (K = 2..4 spurs alternating
from the divider and the net, bays 0.95-1.35 m) and an open room split by a diagonal wall; a connector at the
divider's far end. What varies:
  order     forward (start in the first bay, land in the room: the real course) or reversed (start in the
            room, land in a wide first bay)
  tube      over the connector, along a long serpentine corridor (the serpentine is deepened to fit it), or none
  window    a shortcut through the diagonal (real), through any spur wall (so possibly the first thing met), a
            wall across a corridor that can only be passed through its window, or none
  objects   4-14 floor objects of varied shape and size (rocks, crates, cones, benches) anywhere off the walls,
            placed only where they leave a passage
  geometry  arena 5.6-6.6 m a side, bay widths, gaps, diagonal lean, deflector, short spur; mirror + 90 deg turns
Every wall is anchored to the net or another wall and every gap is on the route, so there is one way through.
Gates 0-3 are 4 of the route's openings in order; a shortcut window is gate 4, clearing the cp of the opening it
bypasses (+5, as on the real course). ``plan_course`` gives the launch -> gates -> pad path (and the window
variant) through 3-D free space; a course is kept only if it exists. NED, floor z = 0, walls 1.727 m.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from dronehacks.mesh import GROUP_ID, CourseMesh

WALL_H = 1.727
PAD_R = 39 * 0.0254
CRUISE_ALT = 0.8


@dataclass
class CourseGenCfg:
    arena_m: Tuple[float, float] = (5.6, 6.6)       # net square side (real: 6.10), each axis drawn separately
    serp_depth: Tuple[float, float] = (2.3, 3.0)    # depth of the serpentine half (real: 2.60)
    serp_depth_tube: Tuple[float, float] = (3.2, 3.8)   # when the tube goes in a serpentine corridor
    spurs: Tuple[int, int] = (2, 4)                  # real: 3
    bay_w: Tuple[float, float] = (0.95, 1.35)        # serpentine bay width (real: 1.05-1.20)
    landing_bay_w: Tuple[float, float] = (2.15, 2.5) # first bay when the order is reversed (holds the pad)
    spur_gap: Tuple[float, float] = (0.85, 1.15)     # gap at a spur's free end (real: 0.95-1.00)
    last_bay_w: Tuple[float, float] = (1.9, 2.8)     # last bay, holding the connector at its far side (real: 2.75)
    connector_w: Tuple[float, float] = (1.25, 1.6)   # real: 1.45
    p_reverse: float = 0.4                           # start in the room, land in the first bay
    tube_where: Tuple[float, float, float] = (0.35, 0.4, 0.25)   # P(connector, corridor, none)
    tunnel_len: Tuple[float, float] = (1.2, 1.8)     # real: 1.50
    diag_gap: Tuple[float, float] = (0.85, 1.15)     # diagonal top -> divider (real: 0.95)
    diag_lean: Tuple[float, float] = (-1.1, 0.4)     # bottom end's sideways offset (- = toward the pad; real: -0.8)
    window_where: Tuple[float, float, float, float] = (0.25, 0.35, 0.3, 0.1)   # P(diagonal, spur, across, none)
    window_w: Tuple[float, float] = (0.8, 0.95)      # real: 0.91
    window_z: Tuple[float, float] = (0.55, 0.9)      # sill height
    window_h: Tuple[float, float] = (0.65, 0.8)
    p_deflector: float = 0.6
    p_short_spur: float = 0.5
    n_objects: Tuple[int, int] = (4, 14)
    wall_t: float = 0.06
    wall_h: float = WALL_H
    net_h: float = 3.0
    min_clear_m: float = 0.3                         # planned path clearance a course must allow
    launch_yaw_tol_deg: float = 30.0


@dataclass
class GenCourse:
    mesh: CourseMesh
    gates: List[Dict]                # rules.GATES format; index 4 = shortcut window or a dummy far away
    pad_center: np.ndarray           # (2,) NED
    pad_r: float
    launch: np.ndarray               # (3,) NED (on the floor: z = -0.08)
    launch_yaw: float                # NED heading along the planned path (banks add +-launch_yaw_tol)
    route: np.ndarray                # (R,3) NED planned path launch -> gates -> pad
    goal: np.ndarray                 # (3,) NED: above the pad centre
    route_window: Optional[np.ndarray] = None   # the same through the shortcut window
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

    def frustum(self, cx, cy, r0, r1, h, group, sides=8, rot=0.0):
        """A cone / tapered rock standing on the floor: radius r0 at the floor, r1 at height h."""
        a = np.linspace(0, 2 * np.pi, sides, endpoint=False) + rot
        lo = np.stack([cx + r0 * np.cos(a), cy + r0 * np.sin(a), np.zeros(sides)], 1)
        hi = np.stack([cx + r1 * np.cos(a), cy + r1 * np.sin(a), np.full(sides, -h)], 1)
        v = np.concatenate([hi, lo]).astype(np.float32)
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



def _windowed_panel(mb, a, b, cfg, frac, w, z0, h):
    """Wall panel a -> b with a w x h opening centred at ``frac`` along it, sill at z0 (UP metres)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    c0, c1 = frac * L - w / 2, frac * L + w / 2
    mb.panel(a, a + d * c0, cfg.wall_t, -cfg.wall_h, 0.0, 'wall')
    mb.panel(a + d * c1, b, cfg.wall_t, -cfg.wall_h, 0.0, 'wall')
    mb.panel(a + d * c0, a + d * c1, cfg.wall_t, -z0, 0.0, 'wall')
    mb.panel(a + d * c0, a + d * c1, cfg.wall_t, -cfg.wall_h, -(z0 + h), 'wall')
    return a + d * frac * L, d


def _gate(name, cp, A, B, n, z=None):
    A, B, n = np.asarray(A, float), np.asarray(B, float), np.asarray(n, float)
    g = {'name': name, 'cp': cp, 'A': A, 'B': B, 't': (B - A) / np.linalg.norm(B - A),
         'L': float(np.linalg.norm(B - A)), 'n': n / np.linalg.norm(n)}
    if z is not None:
        g['z'] = float(z)
    return g


def _seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def _concat(meshes) -> CourseMesh:
    vs, fs, gs, os_, n = [], [], [], [], 0
    for m in meshes:
        vs.append(m.vertices)
        fs.append(m.faces + n)
        gs.append(m.face_group)
        os_.append(m.face_object)
        n += len(m.vertices)
    return CourseMesh(np.concatenate(vs).astype(np.float32), np.concatenate(fs).astype(np.int32),
                      np.concatenate(gs), np.concatenate(os_))


def generate(seed: int, cfg: Optional[CourseGenCfg] = None, max_tries: int = 60, plan: bool = True) -> GenCourse:
    cfg = cfg or CourseGenCfg()
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        lay = _layout(rng, cfg)
        if lay is None:
            continue
        c = _finish(lay, rng, cfg, plan)
        if c is not None:
            c.info['seed'] = seed
            return c
    raise RuntimeError(f'no valid course for seed {seed}')


def _layout(rng, cfg: CourseGenCfg):
    """Walls, tube, windows, pad, launch and gate candidates in the local frame (x: serpentine at +x, y: connector
    at +y). -> dict, or None to draw again."""
    U = lambda r: float(rng.uniform(*r))             # noqa: E731
    reverse = rng.random() < cfg.p_reverse
    tube_where = ('connector', 'corridor', 'none')[rng.choice(3, p=np.array(cfg.tube_where) / sum(cfg.tube_where))]
    win_where = ('diag', 'spur', 'across', 'none')[rng.choice(4, p=np.array(cfg.window_where) / sum(cfg.window_where))]
    hx, hy = U(cfg.arena_m) / 2, U(cfg.arena_m) / 2
    Dn = U(cfg.serp_depth_tube) if tube_where == 'corridor' else U(cfg.serp_depth)
    if 2 * hx - Dn < 2 * PAD_R + 0.7:                 # the room must hold the pad (forward) or the start (reverse)
        return None
    d = hx - Dn                                       # divider line x = d
    wl, wc = U(cfg.last_bay_w) - (0.5 if reverse else 0.0), U(cfg.connector_w)
    w0 = U(cfg.landing_bay_w) if reverse else None
    span = 2 * hy - wl - (w0 or 0.0)
    ks = [k for k in range(cfg.spurs[0], cfg.spurs[1] + 1)
          if cfg.bay_w[0] * (k - (1 if reverse else 0)) <= span <= cfg.bay_w[1] * (k - (1 if reverse else 0))]
    if not ks or wc > wl - 0.5:
        return None
    K = int(rng.choice(ks))
    nb = K - (1 if reverse else 0)
    w = list(rng.dirichlet(np.full(nb, 6.0)) * span) if nb else []
    if w and (min(w) < cfg.bay_w[0] - 0.05 or max(w) > cfg.bay_w[1] + 0.05):
        return None
    if reverse:
        w = [w0] + w
    w = np.array(w)
    ys = -hy + np.cumsum(w)                           # spur i at y = ys[i]
    on_div = [(K - 1 - i) % 2 == 0 for i in range(K)] # the last spur hangs off the divider: no pocket above the connector
    gaps = [U(cfg.spur_gap) for _ in range(K)]
    bay_lo = np.r_[-hy, ys[:-1]]                      # bay j spans y in [bay_lo[j], ys[j]]
    yc = hy - wc
    walls = [((d, -hy), (d, yc))]                     # divider, net to connector
    for i in range(K):
        x0, x1 = (d, hx - gaps[i]) if on_div[i] else (d + gaps[i], hx)
        walls.append(((x0, ys[i]), (x1, ys[i])))

    # where the route runs straight in bay j (between the turn zones at its two ends), and which way (forward)
    def bay_run(j):
        exit_top = on_div[j]                          # leaves bay j over (top) or under (bottom) spur j
        z_exit = gaps[j]
        z_entry = 0.9 if j == 0 else gaps[j - 1]      # bay 0: the launch zone (forward) / landing (reverse)
        lo = d + (z_entry if exit_top else z_exit)
        hi = hx - (z_exit if exit_top else z_entry)
        return lo, hi, (1.0 if exit_top else -1.0)

    mb = _MeshBuilder()
    tube = None
    if tube_where == 'corridor':
        cands = [j for j in range(0 if not reverse else 1, K) if bay_run(j)[1] - bay_run(j)[0] >= 1.1]
        if not cands:
            return None
        j = int(rng.choice(cands))
        lo, hi, _ = bay_run(j)
        L = min(hi - lo - 0.1, U(cfg.tunnel_len))
        x0 = rng.uniform(lo + 0.05, hi - 0.05 - L)
        yb = (bay_lo[j] + ys[j]) / 2
        r = (ys[j] - bay_lo[j]) / 2 - 0.04
        zc = max(r + 0.1, 0.62)
        mb.tube((x0, yb), (x0 + L, yb), r, zc, 'tunnel')
        tube = ('corridor', j)
    if tube_where == 'connector':
        tl = U(cfg.tunnel_len)
        t_hi = d + tl * rng.uniform(0.4, 0.6)
        t_lo = t_hi - tl
        r = wc / 2 - 0.03
        mb.tube((t_hi, yc + wc / 2), (t_lo, yc + wc / 2), r, r + 0.12, 'tunnel')
        exit_x = t_lo
        walls.append(((d, yc), (t_lo + 0.05, yc)))
        tube = ('connector', None)
    else:
        exit_x = d - 0.25

    # the room: diagonal from P (gB below the divider) down to the net at Q
    gB = U(cfg.diag_gap)
    yB = -hy + rng.uniform(0.45, 0.6) * 2 * hy
    lean = U(cfg.diag_lean)
    P, Q = np.array([d - gB, yB]), np.array([-hx, yB + lean])
    if yB > yc - 0.9 or Q[1] < -hy + 2 * PAD_R + 0.2:
        return None
    # the far (pad-side) region holds the pad (forward) or the launch (reverse)
    far = np.array([rng.uniform(-hx + PAD_R + 0.25, d - PAD_R - 0.35), -hy + PAD_R + rng.uniform(0.15, 0.5)])
    if _seg_dist(far, P, Q) < PAD_R + 0.45:
        return None
    deflector = rng.random() < cfg.p_deflector
    if deflector:
        Ld, ang = rng.uniform(1.1, 1.7), math.radians(rng.uniform(10, 35))
        a0 = np.array([min(exit_x, d) - 0.05, yc - 0.05])
        a1 = a0 + Ld * np.array([-math.cos(ang), -math.sin(ang)])
        if a1[0] < -hx + 1.0 or _seg_dist(a1, P, Q) < 1.0:
            deflector = False
        else:
            walls.append((tuple(a0), tuple(a1)))
    short = rng.random() < cfg.p_short_spur
    if short:
        y_lo, y_hi = far[1] + 0.5 * PAD_R, min(P[1], Q[1]) - 0.9
        room = d - far[0] - PAD_R - 0.6
        if y_hi <= y_lo or room < 0.6:
            short = False
        else:
            y_s = rng.uniform(y_lo, y_hi)
            walls.append(((d, y_s), (d - rng.uniform(0.6, min(1.1, room)), y_s)))

    # route openings in forward order: spur gaps (with any across-corridor window in its bay), connector, diagonal
    ww, wz, wh = U(cfg.window_w), U(cfg.window_z), U(cfg.window_h)
    opens = []
    window_gate, bypass = None, None
    across_j = None
    if win_where == 'across':
        cands = [j for j in range(0 if not reverse else 1, K)
                 if bay_run(j)[1] - bay_run(j)[0] >= 0.3 and not (tube and tube[0] == 'corridor' and tube[1] == j)]
        if not cands:
            win_where = 'none'
        else:
            across_j = int(rng.choice(cands))
    spur_win = int(rng.integers(K)) if win_where == 'spur' else None
    for i in range(K):
        if i == across_j:
            lo, hi, s = bay_run(i)
            xm = rng.uniform(lo + 0.1, hi - 0.1)
            c, dvec = _windowed_panel(mb, (xm, bay_lo[i]), (xm, ys[i]), cfg, rng.uniform(0.35, 0.65), ww, wz, wh)
            opens.append(_gate('window_wall', None, c - dvec * ww / 2, c + dvec * ww / 2, (s, 0.0), wz + wh / 2))
        a, b = walls[1 + i]
        if i == spur_win:                              # window near the spur's anchored end: skips its free-end gap
            anchor, free = (np.array(a), np.array(b)) if on_div[i] else (np.array(b), np.array(a))
            Ls = np.linalg.norm(free - anchor)
            if Ls < ww + 0.6:
                spur_win = None
            else:
                c, dvec = _windowed_panel(mb, anchor, free, cfg, rng.uniform(0.3 + ww / 2, Ls - ww / 2 - 0.25) / Ls,
                                          ww, wz, wh)
                window_gate = _gate('window', None, c - dvec * ww / 2, c + dvec * ww / 2, (0.0, 1.0), wz + wh / 2)
                bypass = len(opens)
                walls[1 + i] = None
        free_x = b[0] if on_div[i] else a[0]
        opens.append(_gate(f'spur{i}', None, (free_x, ys[i]), (hx if on_div[i] else d, ys[i]), (0.0, 1.0)))
    opens.append(_gate('connector', None, (exit_x, yc), (exit_x, hy), (-1.0, 0.0)))
    if win_where == 'diag':
        c, dvec = _windowed_panel(mb, Q, P, cfg, rng.uniform(0.25, 0.5), ww, wz, wh)
        nrm = np.array([-dvec[1], dvec[0]])
        nrm = -nrm if nrm[1] > 0 else nrm
        window_gate = _gate('window', None, c - dvec * ww / 2, c + dvec * ww / 2, nrm, wz + wh / 2)
        bypass = len(opens)
    else:
        walls.append((tuple(Q), tuple(P)))
    opens.append(_gate('diag_top', None, P, (d, P[1]), (0.0, -1.0)))
    for wl_ in walls:
        if wl_ is not None:
            mb.panel(wl_[0], wl_[1], cfg.wall_t, -cfg.wall_h, 0.0, 'wall')

    if reverse:                                        # fly it backwards: start where the pad was, land in bay 0
        opens = [dict(g, n=-g['n']) for g in opens[::-1]]
        if window_gate is not None:
            window_gate = dict(window_gate, n=-window_gate['n'])
            bypass = len(opens) - 1 - bypass
        launch = far
        x_closed = d if on_div[0] else hx
        pad = np.array([x_closed + (1 if on_div[0] else -1) * (PAD_R + 0.15), (-hy + ys[0]) / 2])
    else:
        pad = far
        y0 = (-hy + ys[0]) / 2
        launch = np.array([d + 0.45, y0]) if on_div[0] else np.array([hx - 0.45, y0])
    # 4 gates in route order; keep the window wall and the opening a shortcut window bypasses
    must = {k for k, g in enumerate(opens) if g['name'] == 'window_wall'} | ({bypass} if bypass is not None else set())
    rest = [k for k in range(len(opens)) if k not in must]
    pick = sorted(must | set(rng.choice(rest, size=4 - len(must), replace=False).tolist()))
    gates = []
    for cp, k in enumerate(pick):
        gates.append(dict(opens[k], cp=cp))
    if window_gate is not None and bypass in pick:
        gates.append(dict(window_gate, cp=pick.index(bypass)))
    else:
        A = np.array([50.0, 50.0])
        gates.append({'name': 'window(none)', 'cp': 3, 'A': A, 'B': A + [0.1, 0], 't': np.array([1.0, 0.0]),
                      'L': 0.1, 'n': np.array([0.0, 1.0])})
    mb.box((-hx - 2, -hy - 2, 0.0), (hx + 2, hy + 2, 0.02), 'floor')
    for x, y in (((-hx - .01, -hx), (-hy, hy)), ((hx, hx + .01), (-hy, hy)),
                 ((-hx, hx), (-hy - .01, -hy)), ((-hx, hx), (hy, hy + .01))):
        mb.box((x[0], y[0], -cfg.net_h), (x[1], y[1], 0.0), 'net')
    mb.box((-hx, -hy, -cfg.net_h - .01), (hx, hy, -cfg.net_h), 'net')
    mb.prism(pad[0], pad[1], PAD_R, -0.003, 0.0, 'paint', sides=32)
    mb.prism(launch[0], launch[1], 0.15, -0.003, 0.0, 'paint', sides=12)
    info = {'order': 'reverse' if reverse else 'forward', 'spurs': K, 'arena_m': (2 * hx, 2 * hy),
            'bay_w': [round(float(v), 2) for v in w], 'tube': tube[0] if tube else None,
            'window': win_where if (win_where != 'spur' or spur_win is not None) else 'none',
            'deflector': bool(deflector), 'short_spur': bool(short),
            'launch_yaw_tol': math.radians(cfg.launch_yaw_tol_deg)}
    return dict(mesh=mb.build(), gates=gates, pad=pad, launch=launch, hx=hx, hy=hy, info=info,
                windows=[g for g in opens + ([window_gate] if window_gate else []) if 'z' in g])


def _objects(lay, rng, cfg, n):
    """n floor objects of varied shape/size: off the walls, off the pad/launch/windows, each leaving a passage
    (hugging a wall or >= 0.9 m from it). -> list of (kind, centre, footprint radius, builder fn)."""
    from dronehacks.render import MeshScene
    sc = MeshScene(lay['mesh'].select(('wall', 'tunnel', 'net')), 1, device='cpu')
    hx, hy = lay['hx'], lay['hy']
    out = []
    for _ in range(n * 20):
        if len(out) >= n:
            break
        kind = rng.choice(['rock', 'rock', 'crate', 'cone', 'bench'])
        rot = rng.uniform(0, np.pi)
        if kind == 'rock':
            r, h = rng.uniform(0.1, 0.35), rng.uniform(0.2, 0.9)
            R, sides, r1 = r, int(rng.integers(5, 10)), r * rng.uniform(0.4, 0.9)
        elif kind == 'crate':
            a_, b_, h = rng.uniform(0.25, 0.7), rng.uniform(0.25, 0.7), rng.uniform(0.2, 0.8)
            R = 0.5 * math.hypot(a_, b_)
        elif kind == 'cone':
            r, h = rng.uniform(0.1, 0.2), rng.uniform(0.3, 0.75)
            R = r
        else:
            a_, b_, h = rng.uniform(0.8, 1.5), rng.uniform(0.25, 0.4), rng.uniform(0.35, 0.5)
            R = 0.5 * math.hypot(a_, b_)
        p = np.array([rng.uniform(-hx + R, hx - R), rng.uniform(-hy + R, hy - R)])
        dw = float(sc.distance_at(np.array([[p[0], p[1], -0.3]]))[0])
        if dw < R + 0.03 or (0.15 < dw - R < 0.75):
            continue
        if (np.linalg.norm(p - lay['pad']) < PAD_R + R + 0.05 or np.linalg.norm(p - lay['launch']) < R + 0.6
                or any(np.linalg.norm(p - (g['A'] + g['B']) / 2) < R + 0.7 for g in lay['windows'])
                or any(np.linalg.norm(p - q) < R + rq + 0.25 for _, q, rq, _ in out)):
            continue
        if kind == 'rock':
            fn = (lambda mb, p=p, r=r, r1=r1, h=h, s=sides, rot=rot: mb.frustum(p[0], p[1], r, r1, h, 'boulder', s, rot))
        elif kind == 'cone':
            fn = (lambda mb, p=p, r=r, h=h: mb.frustum(p[0], p[1], r, 0.02, h, 'boulder', 12))
        else:
            dvec = np.array([math.cos(rot), math.sin(rot)])
            fn = (lambda mb, p=p, a_=a_, b_=b_, h=h, dvec=dvec:
                  mb.panel(p - dvec * a_ / 2, p + dvec * a_ / 2, b_, -h, 0.0, 'boulder'))
        out.append((str(kind), p, R, fn))
    return out


def _finish(lay, rng, cfg, plan) -> Optional[GenCourse]:
    objs = _objects(lay, rng, cfg, int(rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1)))
    mirror = rng.random() < 0.5
    k90 = int(rng.integers(4))
    th = k90 * math.pi / 2
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ np.diag([1.0, -1.0 if mirror else 1.0])
    tf = lambda p: np.asarray(p, float) @ R.T          # noqa: E731
    gates = []
    for g in lay['gates']:
        if g['name'].endswith('(none)'):
            gates.append(g)
        else:
            gates.append(dict(g, A=tf(g['A']), B=tf(g['B']), t=tf(g['t']), n=tf(g['n'])))
    launch, pad = tf(lay['launch']), tf(lay['pad'])
    while True:
        ob = _MeshBuilder()
        for *_, fn in objs:
            fn(ob)
        mesh = _concat([lay['mesh']] + ([ob.build()] if objs else []))
        mesh.vertices[:, :2] = tf(mesh.vertices[:, :2]).astype(np.float32)
        if mirror:
            mesh.faces[:] = mesh.faces[:, ::-1]
        c = GenCourse(mesh=mesh, gates=gates, pad_center=pad, pad_r=PAD_R,
                      launch=np.array([launch[0], launch[1], -0.08]), launch_yaw=0.0, route=np.zeros((0, 3)),
                      goal=np.array([pad[0], pad[1], -0.3]),
                      info=dict(lay['info'], mirror=bool(mirror), rot_deg=k90 * 90, objects=len(objs),
                                object_kinds=sorted({k for k, *_ in objs})))
        if not plan:
            return c
        try:
            c.route, c.route_window = plan_course(c, cfg.min_clear_m)
            break
        except Exception:
            if not objs:
                return None
            objs = objs[:-1]                            # an object closed a passage: drop one and replan
    k = min(len(c.route) - 1, int(np.searchsorted(np.cumsum(np.r_[0, np.linalg.norm(np.diff(c.route[:, :2], axis=0), axis=1)]), 0.7)))
    dv = c.route[k, :2] - c.route[0, :2]
    c.launch_yaw = float(math.atan2(dv[1], dv[0]))     # facing the way the route leaves the start
    c.info['objects'] = len(objs)
    return c


# ── planned path ─────────────────────────────────────────────────────────────────────────────────────────

def plan_path_through(mesh: CourseMesh, waypoints, clear_m: float = 0.3, res: float = 0.08,
                      device: str = 'cpu') -> np.ndarray:
    """Shortest comfortable 3-D path (A*, dronehacks.path) visiting ``waypoints`` (NED xy or xyz) in order, at
    cruise height between waypoints. -> (N,3) NED polyline, corners rounded."""
    from dronehacks.path import build_grid, plan_path, round_corners, shortcut
    from dronehacks.render import MeshScene
    sc = MeshScene(mesh.select(('wall', 'tunnel', 'boulder', 'net')), 1, device=device)
    v = mesh.select(('net',)).vertices
    lo = np.array([v[:, 0].min() + 0.1, v[:, 1].min() + 0.1, -1.45])
    hi = np.array([v[:, 0].max() - 0.1, v[:, 1].max() - 0.1, -0.45])
    grid = build_grid(sc.distance_at, lo, hi, res)
    wps = [np.array([p[0], p[1], p[2] if len(p) > 2 else -CRUISE_ALT], float) for p in waypoints]
    free = grid.clear >= clear_m
    fidx = np.argwhere(free)
    if not len(fidx):
        raise RuntimeError('no free space')

    def snap(p):                                      # a waypoint too close to something: nearest free voxel
        i = np.asarray(grid.index(p))
        if free[tuple(i)]:
            return p
        j = fidx[np.argmin(((fidx - i) ** 2).sum(1))]
        if np.abs(j - i).max() * grid.res > 0.5:
            raise RuntimeError(f'waypoint {np.round(p, 2)} is boxed in')
        return grid.centre(j)
    wps = [snap(p) for p in wps]
    out = [wps[0][None]]
    for a, b in zip(wps[:-1], wps[1:]):
        seg = plan_path(grid, a, b, clear_m)
        out.append(seg[1:])
    pts = np.concatenate(out)
    pts = shortcut(sc.distance_at, pts, clear_m)
    return round_corners(sc.distance_at, pts, clear_m)


def _gate_wp(g, back=0.0):
    m = (np.asarray(g['A']) + np.asarray(g['B'])) / 2 + np.asarray(g['n']) * back
    return np.array([m[0], m[1], -float(g.get('z', CRUISE_ALT))])


def plan_course(c: 'GenCourse', clear_m: float = 0.3):
    """Launch -> cp 0..3 -> pad, plus the window-shortcut variant (None without one). Each gate is entered from
    0.3 m before it and left 0.3 m past it (at its height: windows sit above cruise), so the path really crosses it."""
    lp = np.array([c.launch[0], c.launch[1], -CRUISE_ALT])
    pad = np.array([c.pad_center[0], c.pad_center[1], -CRUISE_ALT])
    main = [lp]
    for g in c.gates[:4]:
        main += [_gate_wp(g, -0.3), _gate_wp(g, 0.3)]
    tight = any('z' in g for g in c.gates[:4])           # a window on the main route
    route = plan_path_through(c.mesh, main + [pad], min(clear_m, 0.2) if tight else clear_m)
    rw = None
    win = c.gates[4]
    if not win['name'].endswith('(none)'):
        alt = [lp]
        for g in c.gates[:4]:
            if g['cp'] == win['cp']:
                g = win
            alt += [_gate_wp(g, -0.3), _gate_wp(g, 0.3)]
        try:
            rw = plan_path_through(c.mesh, alt + [pad], min(clear_m, 0.18))
        except Exception:
            rw = None
    return route, rw


# ── review: top-down sheet + orbit video ─────────────────────────────────────────────────────────────────

def plot_top(c: GenCourse, ax, title: Optional[str] = None) -> None:
    from matplotlib.patches import Circle
    g = c.mesh.face_group
    v = c.mesh.vertices
    for grp, col, z in (('wall', '#4a4f57', 2), ('tunnel', '#3fa66b', 3), ('boulder', '#4f7fd6', 4)):
        for t in c.mesh.faces[g == GROUP_ID[grp]]:
            p = v[t][:, :2]
            ax.fill(p[:, 1], p[:, 0], color=col, lw=0, zorder=z, alpha=0.9 if grp != 'tunnel' else 0.35)
    nv = c.mesh.select(('net',)).vertices
    ax.plot([nv[:, 1].min(), nv[:, 1].max(), nv[:, 1].max(), nv[:, 1].min(), nv[:, 1].min()],
            [nv[:, 0].min(), nv[:, 0].min(), nv[:, 0].max(), nv[:, 0].max(), nv[:, 0].min()], color='#7a9cc6', lw=1)
    ax.add_patch(Circle((c.pad_center[1], c.pad_center[0]), c.pad_r, fill=False, ec='#2a9d3a', lw=2, zorder=6))
    if c.route_window is not None and len(c.route_window):
        ax.plot(c.route_window[:, 1], c.route_window[:, 0], color='#9b5de5', lw=1.3, ls='--', zorder=5)
    if len(c.route):
        ax.plot(c.route[:, 1], c.route[:, 0], color='#e63946', lw=1.8, zorder=5)
    for gt in c.gates:
        if gt['name'].endswith('(none)'):
            continue
        col = '#f4a259' if gt['name'] != 'window' else '#9b5de5'
        ax.plot([gt['A'][1], gt['B'][1]], [gt['A'][0], gt['B'][0]], color=col, lw=2.5, zorder=7)
    ax.annotate('', xy=(c.launch[1] + 0.5 * math.sin(c.launch_yaw), c.launch[0] + 0.5 * math.cos(c.launch_yaw)),
                xytext=(c.launch[1], c.launch[0]), arrowprops=dict(arrowstyle='-|>', color='black', lw=2), zorder=8)
    i = c.info
    if title is None:
        title = (f"seed {i.get('seed')} {i['order']} {i['spurs']}sp  tube:{i['tube'] or '-'}  win:{i['window']}"
                 f"  obj {i.get('objects', 0)}")
    ax.set_title(title, fontsize=7)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def real_course() -> GenCourse:
    """The real course in GenCourse form (gates from rules.py), with its planned paths."""
    import os
    from dronehacks import rules
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
    mesh = CourseMesh.load(os.path.join(root, 'course.npz'))
    import json
    lm = json.load(open(os.path.join(root, 'scene.json')))['landmarks']
    mark, pad = np.array(lm['launch_mark_ned'][:2]), np.array(lm['landing_pad_ned'][:2])
    gates = [dict(g) for g in rules.GATES]
    c = GenCourse(mesh=mesh, gates=gates, pad_center=pad, pad_r=PAD_R, launch=np.array([mark[0], mark[1], -0.08]),
                  launch_yaw=0.0, route=np.zeros((0, 3)), goal=np.array([pad[0], pad[1], -0.3]),
                  info={'order': 'forward', 'spurs': 3, 'arena_m': (6.1, 6.1), 'tube': 'connector', 'window': 'diag'})
    c.route, c.route_window = plan_course(c)
    return c


_COL = {'floor': (0.78, 0.74, 0.66), 'wall': (0.62, 0.64, 0.68), 'tunnel': (0.25, 0.68, 0.45),
        'boulder': (0.35, 0.52, 0.85), 'paint': (0.95, 0.62, 0.20), 'net': (0.55, 0.68, 0.85)}


def _path_ribbon(mb, pts, r=0.035):
    for a, b in zip(pts[:-1], pts[1:]):
        if np.linalg.norm(b[:2] - a[:2]) < 1e-4:
            continue
        mb.panel(a[:2], b[:2], 2 * r, min(a[2], b[2]) - r, max(a[2], b[2]) + r, 'paint')


def orbit_video(c, path: str, seconds: float = 10.0, fps: int = 30, size=(800, 500),
                radius: float = 8.0, elev_deg: float = 42.0, label: str = '', device: str = 'cpu') -> None:
    """Real-time shaded orbit (MP4, x264 CRF 18) of a course: walls, tube, boulders, pad, and the planned path
    (red; window variant purple). ``c``: GenCourse."""
    import os
    import shutil
    import subprocess

    import torch
    from PIL import Image, ImageDraw

    from dronehacks.mesh import GROUPS
    from dronehacks.render import MeshScene
    view = c.mesh.select(('floor', 'wall', 'tunnel', 'boulder', 'paint'))
    col = np.array([_COL[g] for g in GROUPS])[view.face_group]
    extra, ecol = [], []
    for pts, rgb in ((c.route, (0.90, 0.15, 0.20)), (c.route_window, (0.62, 0.36, 0.90))):
        if pts is None or not len(pts):
            continue
        mb = _MeshBuilder()
        _path_ribbon(mb, pts[::2] if len(pts) > 400 else pts)
        extra.append(mb.build())
        ecol.append(np.tile(rgb, (len(extra[-1].faces), 1)))
    if extra:
        vs, fs, gs, os_ = [view.vertices], [view.faces], [view.face_group], [view.face_object]
        n = len(view.vertices)
        for m in extra:
            vs.append(m.vertices)
            fs.append(m.faces + n)
            gs.append(m.face_group)
            os_.append(m.face_object)
            n += len(m.vertices)
        view = CourseMesh(np.concatenate(vs), np.concatenate(fs).astype(np.int32), np.concatenate(gs),
                          np.concatenate(os_))
        col = np.concatenate([col] + ecol)
    W, H = size
    n, chunk = int(seconds * fps), 15
    f = 0.5 * W / math.tan(math.radians(55.0) / 2)
    sc = MeshScene(view, chunk, W, H, f, f, 60.0, device=device)
    exe = shutil.which('ffmpeg') or os.environ.get('FFMPEG')
    if exe is None:
        raise SystemExit('ffmpeg not found (module load ffmpeg, or set FFMPEG)')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    proc = subprocess.Popen([exe, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                             '-movflags', '+faststart', path], stdin=subprocess.PIPE)
    bg = np.array([0.06, 0.07, 0.10])
    target = np.array([0.0, 0.0, -0.4])
    extent = float(np.abs(c.mesh.select(('wall', 'tunnel', 'boulder')).vertices[:, :2]).max())
    radius = max(radius, 2.3 * extent)
    for c0 in range(0, n, chunk):
        idx = np.arange(c0, min(c0 + chunk, n))
        az = 2 * np.pi * idx / n + 0.6
        el = math.radians(elev_deg) + math.radians(10) * np.sin(2 * np.pi * idx / n)
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
            rgb = col[np.maximum(fc[k], 0)] * (0.35 + 0.65 * cs[k])[..., None]
            fog = np.exp(-np.maximum(r[k], 0) / 40.0)[..., None]
            img = np.where(hit[..., None], rgb * fog + bg * (1 - fog), bg)
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            if label:
                dr = ImageDraw.Draw(im)
                dr.text((10, 8), label, fill=(235, 235, 235))
                dr.text((10, H - 20), 'red: planned path   purple: window shortcut   orange: pad', fill=(200, 200, 200))
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
    g.add_argument('--real', action='store_true', help='first panel = the real course')
    g.add_argument('--out', default='out/coursegen')
    o = sub.add_parser('orbit', help='3-D orbit videos (MP4) with the planned path (and the real course with --real)')
    o.add_argument('--seeds', type=int, nargs='+', default=[1, 5, 12])
    o.add_argument('--real', action='store_true')
    o.add_argument('--out', default='out/coursegen')
    a = ap.parse_args(argv)
    if a.cmd == 'orbit':
        if a.real:
            orbit_video(real_course(), os.path.join(a.out, 'orbit_real.mp4'), label='REAL DroneHacks course')
        for sd in a.seeds:
            c = generate(sd)
            i = c.info
            orbit_video(c, os.path.join(a.out, f'orbit_seed{sd}.mp4'),
                        label=f"generated seed {sd}: {i['order']} order, {i['spurs']} spurs, tube: {i['tube'] or 'none'}, "
                              f"window: {i['window']}, {i.get('objects', 0)} floor objects")
        print('wrote orbit videos to', a.out)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    courses = ([('REAL course', real_course())] if a.real else []) + [(None, generate(a.seed0 + k)) for k in range(a.n)]
    cols = 5
    rows = math.ceil(len(courses) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    for ax, (t, c) in zip(axs.flat, courses):
        plot_top(c, ax, t)
    for ax in axs.flat[len(courses):]:
        ax.axis('off')
    fig.suptitle('Courses (top-down, north up): walls grey, tube green, boulders blue, gates orange (window purple), '
                 'planned path red (window shortcut purple dashed), start arrow black, pad ring green', fontsize=9)
    fig.tight_layout()
    path = os.path.join(a.out, 'gallery.png')
    fig.savefig(path, dpi=110)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
