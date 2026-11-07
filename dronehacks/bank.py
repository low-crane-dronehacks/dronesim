"""Reset bank (start states for SGS) and the training route (numpy only).

``clearance`` is a callable (N,3)->(N,) distance to the nearest obstacle (``MeshScene.distance_at``).

* ``Route``: the planner's launch -> goal path, densely resampled. Not part of the reward; the env uses it
  only for the critic's progress feature.
* ``Bank``: N start configurations, each one an SGS cell. ``build_free_bank`` spreads them over the whole
  reachable free volume (launch pose, near goal, near route, anywhere); ``build_return_bank`` adds the
  return-leg starts (parked on the pad, climbing, crossing the barrier, final approach).
Build or rebuild them with ``python -m dronehacks.build_assets``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from dronehacks.path import build_grid, plan_path, resample, round_corners, shortcut

Clearance = Callable[[np.ndarray], np.ndarray]

BANK_LAUNCH, BANK_BODY, BANK_NEAR_GOAL, BANK_NEAR_ROUTE = 0, 1, 2, 3
#: return-phase starts (phase 1 = parked on the pad, run points banked, waiting for the return flag;
#: phase 2 = already flying back to the launch mark)
BANK_PAD, BANK_RETURN = 4, 5
BANK_CLIMB, BANK_CROSS, BANK_FINAL = 6, 7, 8        # return-leg airborne sub-types (v2 banks; v1 lumps them as BANK_RETURN)
ROUTE_STEP = 0.05


@dataclass
class Route:
    points: np.ndarray      # (R,3) NED, ROUTE_STEP apart
    s: np.ndarray           # (R,) arc length
    goal: np.ndarray        # (3,) NED goal location (success = within goal_radius)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def tangent(self) -> np.ndarray:
        t = np.gradient(self.points, axis=0)
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)

    def save(self, path: str) -> None:
        np.savez_compressed(path, points=self.points, s=self.s, goal=self.goal)

    @classmethod
    def load(cls, path: str) -> 'Route':
        with np.load(path) as z:
            return cls(z['points'].astype(np.float32), z['s'].astype(np.float32),
                       z['goal'].astype(np.float32))


@dataclass
class Bank:
    """N start configurations: pose, initial velocity/attitude, ``geo`` = geodesic distance to the goal
    around obstacles (m; for return starts the horizontal distance to the launch mark), ``phase`` = 0 run,
    1 parked on the pad, 2 flying back."""
    pos: np.ndarray         # (N,3) NED
    yaw: np.ndarray         # (N,) rad
    route_s: np.ndarray     # (N,) arc length of the nearest route point
    kind: np.ndarray        # (N,) BANK_*
    vel: Optional[np.ndarray] = None      # (N,3) NED m/s
    roll: Optional[np.ndarray] = None     # (N,) rad
    pitch: Optional[np.ndarray] = None    # (N,) rad
    geo: Optional[np.ndarray] = None      # (N,) m
    phase: Optional[np.ndarray] = None    # (N,) 0 = run, 1 = parked on the pad, 2 = returning (see BANK_PAD)

    def __post_init__(self):
        n = len(self.pos)
        if self.vel is None:
            self.vel = np.zeros((n, 3), np.float32)
        if self.roll is None:
            self.roll = np.zeros(n, np.float32)
        if self.pitch is None:
            self.pitch = np.zeros(n, np.float32)
        if self.geo is None:
            self.geo = np.full(n, np.nan, np.float32)
        if self.phase is None:
            self.phase = np.zeros(n, np.int8)

    def __len__(self) -> int:
        return len(self.pos)

    def save(self, path: str) -> None:
        np.savez_compressed(path, pos=self.pos, yaw=self.yaw, route_s=self.route_s, kind=self.kind,
                            vel=self.vel, roll=self.roll, pitch=self.pitch, geo=self.geo, phase=self.phase)

    @classmethod
    def load(cls, path: str) -> 'Bank':
        with np.load(path) as z:
            opt = {k: z[k].astype(np.float32) for k in ('vel', 'roll', 'pitch', 'geo') if k in z.files}
            if 'phase' in z.files:
                opt['phase'] = z['phase'].astype(np.int8)
            return cls(z['pos'].astype(np.float32), z['yaw'].astype(np.float32),
                       z['route_s'].astype(np.float32), z['kind'].astype(np.int8), **opt)

    @classmethod
    def concat(cls, *banks: 'Bank') -> 'Bank':
        f = lambda k: np.concatenate([getattr(b, k) for b in banks])      # noqa: E731
        return cls(f('pos'), f('yaw'), f('route_s'), f('kind'), f('vel'), f('roll'), f('pitch'), f('geo'), f('phase'))


def plan_training_route(clearance: Clearance, launch_ned, goal_ned, *, radius: float = 0.25,
                        cruise: float = 0.8, ground: float = 0.08, wall_top: float = 1.727,
                        res: float = 0.05, bounds: float = 2.9) -> Route:
    """Launch mark (on the ground) -> straight up to cruise -> planned route -> goal.

    ``goal_ned`` is the success location (already at its flight altitude); the
    route descends to it from cruise inside the planner's free space.
    """
    launch = np.asarray(launch_ned, np.float64)
    goal = np.asarray(goal_ned, np.float64)
    lo = np.array([-bounds, -bounds, -(wall_top - 0.25)])
    hi = np.array([bounds, bounds, -0.2])
    grid = build_grid(clearance, lo, hi, res)
    s_c = np.array([launch[0], launch[1], -cruise])
    g_c = np.array([goal[0], goal[1], -cruise])
    way = shortcut(clearance, plan_path(grid, s_c, g_c, radius), radius)
    pts = np.vstack([[launch[0], launch[1], -ground], way, goal])
    dense, s = resample(round_corners(clearance, pts, radius), ROUTE_STEP)
    return Route(dense.astype(np.float32), s.astype(np.float32), goal.astype(np.float32))


# ── free-space bank: starts from (nearly) the whole reachable state space ────────────────────────

def geodesic_distance(free: np.ndarray, res: float, goal_idx) -> np.ndarray:
    """Dijkstra over the 26-connected free voxels from ``goal_idx``: shortest path length [m] around
    obstacles. Unreachable / blocked voxels are +inf. ``free`` is (nx,ny,nz) bool."""
    import heapq
    shape = free.shape
    dist = np.full(shape, np.inf)
    offs = [(dx, dy, dz, res * float(np.sqrt(dx * dx + dy * dy + dz * dz)))
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    g = tuple(int(v) for v in goal_idx)
    if not free[g]:
        raise ValueError(f'goal voxel {g} is not free')
    dist[g] = 0.0
    heap = [(0.0, g)]
    nx, ny, nz = shape
    while heap:
        d, (i, j, k) = heapq.heappop(heap)
        if d > dist[i, j, k]:
            continue
        for dx, dy, dz, c in offs:
            a, b, e = i + dx, j + dy, k + dz
            if 0 <= a < nx and 0 <= b < ny and 0 <= e < nz and free[a, b, e]:
                nd = d + c
                if nd < dist[a, b, e]:
                    dist[a, b, e] = nd
                    heapq.heappush(heap, (nd, (a, b, e)))
    return dist


def geodesic_descent_heading(geo: np.ndarray, res: float, sigma_vox: float = 1.5) -> np.ndarray:
    """(nx,ny,nz) NED yaw [rad] of the horizontal direction of steepest DEcrease of the geodesic
    distance field -- i.e. the way the shortest path to the goal leaves each voxel. Unreachable voxels
    (inf) are filled with the finite maximum before smoothing so walls do not poison the gradient."""
    from scipy.ndimage import gaussian_filter
    finite = np.isfinite(geo)
    filled = np.where(finite, geo, geo[finite].max() * 1.2)
    sm = gaussian_filter(filled, sigma_vox)
    gx, gy, _ = np.gradient(sm, res)                       # d/d north, d/d east
    return np.arctan2(-gy, -gx)


def build_free_bank(route: Route, clearance: Clearance, n: int, seed: int, *, grid_lo, grid_hi,
                    res: float = 0.05, min_clearance: float = 0.23, launch_frac: float = 0.02,
                    near_goal_frac: float = 0.18, near_route_frac: float = 0.20, near_goal_m: float = 3.0,
                    route_lateral_m: float = 0.5, route_vertical_m: float = 0.3,
                    route_yaw_noise_deg: float = 35.0, moving_frac: float = 0.5,
                    vel_range=(2.0, 2.0, 1.0), look_s=(0.2, 0.4, 0.6), tilted_frac: float = 0.5,
                    tilt_sigma_deg: float = 6.0, tilt_max_deg: float = 15.0,
                    alt_range=(0.2, 1.5), heading_mix=(0.5, 0.3, 0.2), heading_sigma_deg=(45.0, 120.0)) -> Bank:
    """N starts spread over the whole free volume connected to the goal, in four categories:

    * ``launch``     ``launch_frac``: the exact launch pose (the competition start);
    * ``near_goal``  ``near_goal_frac``: free voxels within ``near_goal_m`` of the goal (GEODESIC),
                     uniform in distance (not volume) so many starts are very close to the goal;
    * ``near_route`` ``near_route_frac``: a point on the planned route (uniform in arc length) plus a
                     bounded offset, heading along the route +- noise -- the old bank's distribution;
    * ``body``       the rest: volume-uniform over ALL reachable free voxels, heading anywhere.
    ``moving_frac`` of the non-launch starts carry an initial NED velocity (uniform per axis in
    ``vel_range``) and ``tilted_frac`` an initial roll/pitch (N(0, tilt_sigma), clipped). A moving
    start is kept only if it still clears ``min_clearance`` after each ``look_s`` seconds of
    constant-velocity flight, so it is not doomed before the policy can act.
    Every start clears ``min_clearance`` (exact) and lies in the goal's connected free component, so
    it is solvable by construction. ``geo`` stores the geodesic distance to the goal.

    HEADINGS of the non-launch, non-route starts are a mixture (``heading_mix``): the geodesic descent
    direction (the way the shortest path to the goal leaves that voxel) +- N(0, 45 deg), the same +-
    N(0, 120 deg), and uniform on the circle. Fully random headings alone made the task unlearnable:
    with no goal/route input the policy cannot tell where the goal is, and floor paint is invisible to
    depth, so with a uniform heading nothing forward-flying reflexively makes progress and the sparse
    reward never bootstraps (measured: a run sat at ~10% success for 17M steps).
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(grid_lo, float), np.asarray(grid_hi, float)
    grid = build_grid(clearance, lo, hi, res)
    free = grid.clear >= min_clearance
    goal_idx = grid.index(route.goal)
    geo_grid = geodesic_distance(free, res, goal_idx)
    reach = np.isfinite(geo_grid)
    cells = np.argwhere(reach)
    if len(cells) < 1000:
        raise ValueError(f'only {len(cells)} reachable free voxels; check clearance / bounds')
    near_mask = geo_grid[reach] <= near_goal_m
    near_order = np.argsort(geo_grid[reach][near_mask])      # near-goal voxels sorted by geodesic distance
    near_cells = cells[near_mask][near_order]
    near_geo = geo_grid[reach][near_mask][near_order]

    n_launch, n_goal, n_route = (int(round(n * f)) for f in (launch_frac, near_goal_frac, near_route_frac))
    n_body = n - n_launch - n_goal - n_route
    if n_body < 0:
        raise ValueError('category fractions exceed 1')
    kinds = np.concatenate([np.full(n_launch, BANK_LAUNCH), np.full(n_goal, BANK_NEAR_GOAL),
                            np.full(n_route, BANK_NEAR_ROUTE), np.full(n_body, BANK_BODY)]).astype(np.int8)
    pool = {BANK_NEAR_GOAL: near_cells, BANK_BODY: cells}

    pos, vel = np.zeros((n, 3)), np.zeros((n, 3))
    roll, pitch, yaw = np.zeros(n), np.zeros(n), np.zeros(n)
    ridx = np.zeros(n, int)                                   # route index behind each near-route start
    tang = route.tangent()
    head = np.arctan2(tang[:, 1], tang[:, 0])
    pos[kinds == BANK_LAUNCH] = route.points[0]
    yaw[kinds == BANK_LAUNCH] = head[0]

    ok = kinds == BANK_LAUNCH
    vmax = np.asarray(vel_range, float)
    # Decide moving/stationary ONCE per start. (Redrawing it per attempt would bias accepted starts
    # toward stationary, because a moving candidate fails the look-ahead more often.)
    moving = rng.random(n) < moving_frac
    for attempt in range(100):
        if attempt == 70:                        # cannot find a clear velocity here (tight spot): start at rest
            moving[~ok] = False
        todo = np.nonzero(~ok)[0]
        if len(todo) == 0:
            break
        cand_pos = np.empty((len(todo), 3))
        for kind in (BANK_NEAR_GOAL, BANK_BODY):
            sel = kinds[todo] == kind
            if sel.any():
                if kind == BANK_NEAR_GOAL:
                    # Uniform in geodesic DISTANCE, not volume: a ball's volume grows as r^3, so volume-uniform
                    # sampling puts only ~1% of starts within 0.3 m of the goal -- and those are the early wins
                    # SGS bootstraps from. Pick a distance, then a random voxel in that shell.
                    d = rng.uniform(0.0, near_goal_m, sel.sum())
                    at = np.searchsorted(near_geo, d) + rng.integers(-40, 41, sel.sum())
                    c = near_cells[np.clip(at, 0, len(near_cells) - 1)]
                else:
                    c = pool[kind][rng.integers(0, len(pool[kind]), sel.sum())]
                cand_pos[sel] = grid.lo + c * res + rng.uniform(-0.5 * res, 0.5 * res, (sel.sum(), 3))
        sel = kinds[todo] == BANK_NEAR_ROUTE
        if sel.any():
            m = int(sel.sum())
            idx = np.clip(np.searchsorted(route.s, rng.uniform(0.0, route.length, m)), 0, len(route.s) - 1)
            ang, rad = rng.uniform(0, 2 * np.pi, m), rng.uniform(0, route_lateral_m, m)
            p = route.points[idx].astype(float)
            p[:, 0] += rad * np.cos(ang)
            p[:, 1] += rad * np.sin(ang)
            p[:, 2] = np.clip(p[:, 2] + rng.uniform(-route_vertical_m, route_vertical_m, m),
                              -alt_range[1], -alt_range[0])
            cand_pos[sel] = p
            ridx[todo[sel]] = idx
        cand_vel = np.where(moving[todo][:, None], rng.uniform(-1.0, 1.0, (len(todo), 3)) * vmax, 0.0)
        good = clearance(cand_pos) >= min_clearance
        for t in look_s:                       # constant-velocity look-ahead: still clear, still in the volume
            ahead = cand_pos + cand_vel * t
            good &= clearance(ahead) >= min_clearance
            good &= (ahead[:, 2] <= -0.15) & (ahead[:, 2] >= -1.55) & (np.abs(ahead[:, :2]) <= 2.95).all(1)
        # near-route positions must also be in the goal's component (a route offset can leave it)
        ij = np.clip(np.rint((cand_pos - grid.lo) / res).astype(int), 0, np.asarray(grid.shape) - 1)
        good &= np.isfinite(geo_grid[ij[:, 0], ij[:, 1], ij[:, 2]])
        keep = todo[good]
        pos[keep], vel[keep] = cand_pos[good], cand_vel[good]
        ok[keep] = True
    if not ok.all():
        raise ValueError(f'{int((~ok).sum())} starts could not be placed; loosen the constraints')

    non_launch = kinds != BANK_LAUNCH
    ij = np.clip(np.rint((pos - grid.lo) / res).astype(int), 0, np.asarray(grid.shape) - 1)
    desc = geodesic_descent_heading(geo_grid, res)[ij[:, 0], ij[:, 1], ij[:, 2]]
    mode = rng.choice(3, size=n, p=np.asarray(heading_mix, float) / np.sum(heading_mix))
    sig = np.deg2rad(np.asarray(heading_sigma_deg, float))
    yaw_free = np.where(mode == 0, desc + rng.normal(0.0, sig[0], n),
                        np.where(mode == 1, desc + rng.normal(0.0, sig[1], n), rng.uniform(-np.pi, np.pi, n)))
    yaw[non_launch] = yaw_free[non_launch]
    nr = kinds == BANK_NEAR_ROUTE
    noise = np.deg2rad(route_yaw_noise_deg)
    yaw[nr] = head[ridx[nr]] + np.clip(rng.normal(0.0, noise, nr.sum()), -2 * noise, 2 * noise)
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    tilt = (rng.random(n) < tilted_frac) & non_launch
    lim, sig = np.deg2rad(tilt_max_deg), np.deg2rad(tilt_sigma_deg)
    roll[tilt] = np.clip(rng.normal(0.0, sig, tilt.sum()), -lim, lim)
    pitch[tilt] = np.clip(rng.normal(0.0, sig, tilt.sum()), -lim, lim)

    ij = np.clip(np.rint((pos - grid.lo) / res).astype(int), 0, np.asarray(grid.shape) - 1)
    geo = geo_grid[ij[:, 0], ij[:, 1], ij[:, 2]]
    sub = route.points[::4].astype(float)                     # nearest route arc length (informational)
    route_s = np.empty(n)
    for c0 in range(0, n, 2048):
        d2 = ((pos[c0:c0 + 2048, None, :] - sub[None]) ** 2).sum(-1)
        route_s[c0:c0 + 2048] = route.s[::4][d2.argmin(1)]
    route_s[nr] = route.s[ridx[nr]]
    return Bank(pos.astype(np.float32), yaw.astype(np.float32), route_s.astype(np.float32), kinds,
                vel.astype(np.float32), roll.astype(np.float32), pitch.astype(np.float32),
                geo.astype(np.float32))


def gate_geodesics(route: Route, clearance: Clearance, gate_points, *, grid_lo, grid_hi, res: float = 0.05,
                   min_clearance: float = 0.23, search_m: float = 0.3) -> np.ndarray:
    """Geodesic distance [m] (around obstacles, to the goal) from each point in ``gate_points`` (G,3 NED).

    Used as the reference path length for the competition speed terms: the time a checkpoint "should"
    take from a given start is (geo_start - geo_gate) / ref_speed. A point that is not itself a free
    voxel takes the nearest reachable one within ``search_m``.
    """
    grid = build_grid(clearance, np.asarray(grid_lo, float), np.asarray(grid_hi, float), res)
    geo = geodesic_distance(grid.clear >= min_clearance, res, grid.index(route.goal))
    out = np.empty(len(gate_points))
    r = int(np.ceil(search_m / res))
    for i, p in enumerate(np.asarray(gate_points, float)):
        c = np.asarray(grid.index(p))
        lo, hi = np.maximum(c - r, 0), np.minimum(c + r + 1, np.asarray(grid.shape))
        block = geo[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        idx = np.argwhere(np.isfinite(block))
        if len(idx) == 0:
            raise ValueError(f'gate point {p.round(3)} has no reachable free voxel within {search_m} m')
        vox = lo + idx
        dist = np.linalg.norm((vox - c) * res, axis=1)
        out[i] = block[tuple(idx[dist.argmin()])]
    return out


def build_return_bank(clearance: Clearance, launch_xy, pad_xy, n: int, seed: int, *, min_clearance: float = 0.23,
                      pad_clearance: float = 0.25, pad_radius: float = 0.6, rest_alt: float = 0.08,
                      wall_top: float = 1.727, net_top: float = 3.0, arena_half: float = 2.8,
                      mix=(0.45, 0.20, 0.20, 0.15)) -> Bank:
    """Reset states for the return leg (after landing on the pad, back over the barrier to the launch mark).

    mix = (pad, climb, cross, final) fractions:
      pad    (phase 1)  parked on the pad within ``pad_radius`` of its centre, level, at rest, any heading
      climb  (phase 2)  just lifted off near the pad: 0.3-1.5 m up, rising, nose toward the mark
      cross  (phase 2)  over the barrier: between pad and mark (+1 m), above the wall tops, moving toward the mark
      final  (phase 2)  within 1.5 m of the mark, 0.15-2.5 m up, moving toward it (descent / landing approach)
    ``geo`` holds the horizontal distance to the launch mark. Every state keeps ``min_clearance`` from geometry.
    """
    rng = np.random.default_rng(seed)
    mark, pad = np.asarray(launch_xy, float)[:2], np.asarray(pad_xy, float)[:2]
    counts = np.floor(np.asarray(mix) / np.sum(mix) * n).astype(int)
    counts[0] += n - counts.sum()
    lo_xy, hi_xy = np.minimum(mark, pad) - 1.0, np.maximum(mark, pad) + 1.0
    lo_xy, hi_xy = np.maximum(lo_xy, -arena_half), np.minimum(hi_xy, arena_half)
    top_lo, top_hi = wall_top + 0.25, net_top - 0.15

    def sample(m, cat):
        if cat == 0:
            r = pad_radius * np.sqrt(rng.random(m))
            th = rng.uniform(0, 2 * np.pi, m)
            xy = pad + np.stack([r * np.cos(th), r * np.sin(th)], 1)
            alt = np.full(m, rest_alt)
        elif cat == 1:
            r = 1.0 * np.sqrt(rng.random(m))
            th = rng.uniform(0, 2 * np.pi, m)
            xy = pad + np.stack([r * np.cos(th), r * np.sin(th)], 1)
            alt = rng.uniform(0.3, 1.5, m)
        elif cat == 2:
            xy = rng.uniform(lo_xy, hi_xy, (m, 2))
            alt = rng.uniform(top_lo, top_hi, m)
        else:
            r = 1.5 * np.sqrt(rng.random(m))
            th = rng.uniform(0, 2 * np.pi, m)
            xy = mark + np.stack([r * np.cos(th), r * np.sin(th)], 1)
            alt = rng.uniform(0.15, 2.5, m)
        return np.stack([xy[:, 0], xy[:, 1], -alt], 1)

    pos, kind, phase, chunks = [], [], [], []
    for cat, cnt in enumerate(counts):
        got = np.zeros((0, 3))
        for _ in range(200):
            if len(got) >= cnt:
                break
            cand = sample(max(cnt, 64) * 2, cat)
            ok = (np.abs(cand[:, :2]) <= arena_half).all(1) & (-cand[:, 2] < net_top - 0.1)
            ok &= clearance(cand) >= (pad_clearance if cat == 0 else min_clearance)
            got = np.concatenate([got, cand[ok]])
        if len(got) < cnt:
            raise ValueError(f'return bank category {cat}: could place only {len(got)} of {cnt} starts')
        chunks.append((cat, got[:cnt]))
    vel, yaw, roll, pitch = [], [], [], []
    for cat, p in chunks:
        m = len(p)
        to_mark = mark - p[:, :2]
        head = np.arctan2(to_mark[:, 1], to_mark[:, 0])
        unit = to_mark / np.maximum(np.linalg.norm(to_mark, axis=1, keepdims=True), 1e-6)
        if cat == 0:
            v = np.zeros((m, 3))
            y = rng.uniform(-np.pi, np.pi, m)
            ro = pi_ = np.zeros(m)
        else:
            spd = {1: rng.uniform(0.0, 1.0, m), 2: rng.uniform(0.0, 3.0, m), 3: rng.uniform(0.0, 2.0, m)}[cat]
            vz = {1: -rng.uniform(0.0, 1.0, m), 2: rng.normal(0.0, 0.3, m), 3: rng.normal(0.2, 0.4, m)}[cat]
            v = np.stack([unit[:, 0] * spd, unit[:, 1] * spd, vz], 1)
            y = head + np.clip(rng.normal(0.0, 0.5, m), -1.2, 1.2)
            ro, pi_ = np.deg2rad(np.clip(rng.normal(0, 6, m), -15, 15)), np.deg2rad(np.clip(rng.normal(0, 6, m), -15, 15))
        vel.append(v); yaw.append(y); roll.append(ro); pitch.append(pi_)
        kind.append(np.full(m, (BANK_PAD, BANK_CLIMB, BANK_CROSS, BANK_FINAL)[cat]))
        phase.append(np.full(m, 1 if cat == 0 else 2))
        pos.append(p)
    pos = np.concatenate(pos)
    yaw = (np.concatenate(yaw) + np.pi) % (2 * np.pi) - np.pi
    geo = np.linalg.norm(pos[:, :2] - mark, axis=1)
    return Bank(pos.astype(np.float32), yaw.astype(np.float32), np.zeros(len(pos), np.float32),
                np.concatenate(kind).astype(np.int8), np.concatenate(vel).astype(np.float32),
                np.concatenate(roll).astype(np.float32), np.concatenate(pitch).astype(np.float32),
                geo.astype(np.float32), np.concatenate(phase).astype(np.int8))
