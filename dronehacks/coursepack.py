"""Course packs: a fixed set of procedurally generated courses (``coursegen.generate``) with everything the env
needs to train on them, in one ``.npz``.

    python -m dronehacks.coursepack build --seeds 0-9 --out assets/packs/gen10.npz          # training set
    python -m dronehacks.coursepack build --seeds 100-104 --out assets/packs/heldout5.npz   # held-out eval set
    python -m dronehacks.coursepack info assets/packs/gen10.npz

The real course (assets/course.npz, route.npz, bank_*.npz, scoring.json) is never part of a pack: it is the
generalisation test. Per course, built exactly like the real course's assets (``build_assets``):

  mesh        the generated triangles (floor / net / paint / wall / tunnel / boulder)
  gates       5 checkpoint lines in ``rules.GATES`` format; index 4 = the shortcut window, or a dummy far away
              (``window(none)``) that can never be crossed
  route       launch mark (on the ground) -> the course's planned path (``GenCourse.route``) -> goal above the pad,
              resampled every ``bank.ROUTE_STEP`` like route.npz
  bank        ``bank.build_free_bank`` with the bank_free_v3 kinds and proportions (2% launch, 18% near goal,
              20% near route, 60% anywhere). LAUNCH starts are re-drawn like v3 (within 0.30 m of the mark,
              clearance >= min_clearance) but face the planned route direction +- ``launch_yaw_tol`` (30 deg,
              ``CourseGenCfg.launch_yaw_tol_deg``) instead of any heading.
  gate_geo    geodesic distance (around obstacles, to the goal) from each gate, = scoring.json gate_geodesic_m
  sector      per bank start: the checkpoint it is working toward (= rules.sector_index on the real course).
              Computed by cutting the free voxels along every gate line, labelling the connected regions and
              naming each region after the planned route's progress through it (``label_sectors``).
  arena_half  (north, east) out-of-bounds limit = net half-size - 0.05 m (the real course: 3.049 -> env 3.0)

A course is rejected (and reported) when its bank cannot be placed, the route does not cross the four main gates
in order, or a gate cut leaks (one region holds route points of two sectors).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from dronehacks.bank import (BANK_LAUNCH, Bank, Route, ROUTE_STEP, build_free_bank, geodesic_distance)
from dronehacks.mesh import CourseMesh
from dronehacks.path import build_grid, resample

BANK_KEYS = ('pos', 'yaw', 'route_s', 'kind', 'vel', 'roll', 'pitch', 'geo', 'phase')
GATE_KEYS = ('A', 'B', 't', 'n')
WALL_TOP = 1.727
CRUISE = 0.8


@dataclass
class PackCourse:
    seed: int
    mesh: CourseMesh
    gates: List[Dict]               # rules.GATES format (name, cp, A, B, t, n, L[, z])
    pad_center: np.ndarray          # (2,)
    pad_r: float
    mark: np.ndarray                # (2,) launch mark
    launch_yaw: float
    goal: np.ndarray                # (3,)
    arena_half: np.ndarray          # (2,) north, east
    route: Route
    bank: Bank
    sector: np.ndarray              # (N,) per bank start
    gate_geo: np.ndarray            # (5,)
    info: Dict = field(default_factory=dict)


def parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ── sectors ─────────────────────────────────────────────────────────────────

def route_crossings(points: np.ndarray, gates: Sequence[Dict], tol: float = 0.1) -> np.ndarray:
    """Index of the first route point past each gate line (crossed in its direction, within its extent)."""
    out = np.full(len(gates), -1)
    xy = points[:, :2].astype(np.float64)
    for k, g in enumerate(gates):
        d = (xy - g['A']) @ g['n']
        u = (xy - g['A']) @ g['t']
        hit = np.nonzero((d[:-1] < 0) & (d[1:] >= 0) & (u[1:] >= -tol) & (u[1:] <= g['L'] + tol))[0]
        if len(hit):
            out[k] = hit[0] + 1
    return out


def label_sectors(grid, free: np.ndarray, gates: Sequence[Dict], route_pts: np.ndarray, *,
                  ext: float = 0.15):
    """-> ((nx,ny,nz) sector per voxel, info). Free voxels are cut along every real gate line (extended ``ext`` at
    both ends into the walls), the rest split into connected regions, and each region takes the sector of the route
    points inside it (sector = number of main gates 0..3 the route has crossed). Voxels in no labelled region take
    the nearest labelled voxel's sector."""
    from scipy import ndimage
    nx, ny, nz = grid.shape
    X = grid.lo[0] + grid.res * np.arange(nx)
    Y = grid.lo[1] + grid.res * np.arange(ny)
    P = np.stack(np.meshgrid(X, Y, indexing='ij'), -1)                  # (nx,ny,2)
    cut = np.zeros((nx, ny), bool)
    for g in gates:
        if str(g['name']).endswith('(none)'):
            continue
        A, t, L = np.asarray(g['A'], float), np.asarray(g['t'], float), float(g['L'])
        a = A - t * ext
        u = np.clip((P - a) @ t, 0.0, L + 2 * ext)
        cut |= np.linalg.norm(P - (a + u[..., None] * t), axis=-1) <= 0.75 * grid.res
    region = free & ~cut[:, :, None]
    lab, n = ndimage.label(region, structure=np.ones((3, 3, 3)))
    cross = route_crossings(route_pts, gates[:4])
    if (cross < 0).any() or not np.all(np.diff(cross) > 0):
        raise ValueError(f'route does not cross the main gates in order (crossing indices {cross.tolist()})')
    rsec = (np.arange(len(route_pts))[:, None] >= cross[None]).sum(1)
    ij = np.clip(np.rint((route_pts - grid.lo) / grid.res).astype(int), 0, np.asarray(grid.shape) - 1)
    rl = lab[ij[:, 0], ij[:, 1], ij[:, 2]]
    sec_of = np.full(n + 1, -1)
    conflicts = []
    for r in np.unique(rl[rl > 0]):
        vals, cnt = np.unique(rsec[rl == r], return_counts=True)
        sec_of[r] = vals[cnt.argmax()]
        if (cnt >= 3).sum() > 1:
            conflicts.append({int(v): int(c) for v, c in zip(vals, cnt)})
    sec = sec_of[lab]
    known = sec >= 0
    if not known.any():
        raise ValueError('no labelled region')
    _, idx = ndimage.distance_transform_edt(~known, return_indices=True)
    sec = sec[idx[0], idx[1], idx[2]]
    return sec, {'regions': int(n), 'route_regions': int((sec_of >= 0).sum()), 'conflicts': conflicts,
                 'route_cross_idx': cross.tolist()}


# ── one course ──────────────────────────────────────────────────────────────

def build_course(seed: int, n: int, *, bank_seed: int = 0, min_clearance: float = 0.23, res: float = 0.05,
                 launch_r: float = 0.30, goal_alt: float = 0.3, device: str = 'cpu', log=print) -> PackCourse:
    from dronehacks.coursegen import CourseGenCfg, generate
    from dronehacks.render import MeshScene
    t0 = time.time()
    c = generate(seed)
    gcfg = CourseGenCfg()
    nv = c.mesh.select(('net',)).vertices
    net_half = np.maximum(np.abs(nv[:, :2].min(0)), np.abs(nv[:, :2].max(0))).astype(np.float64)
    scene = MeshScene(c.mesh.select(('wall', 'tunnel', 'boulder')), 1, device=device)
    clr = scene.distance_at
    gh = net_half - 0.15                                             # real course: net 3.049 -> grid 2.9
    grid_lo, grid_hi = [-gh[0], -gh[1], -(WALL_TOP - 0.25)], [gh[0], gh[1], -0.2]

    # route: launch mark on the ground -> planned path -> goal above the pad (route.npz's construction)
    goal = np.array([c.pad_center[0], c.pad_center[1], -goal_alt])
    pts = np.vstack([[c.launch[0], c.launch[1], -0.08], c.route, goal])
    dense, s = resample(pts, ROUTE_STEP)
    route = Route(dense.astype(np.float32), s.astype(np.float32), goal.astype(np.float32))

    # one grid + geodesic field (the same grid build_free_bank builds internally) for sectors, launch geo, gates
    grid = build_grid(clr, np.asarray(grid_lo, float), np.asarray(grid_hi, float), res)
    free = grid.clear >= min_clearance
    geo_grid = geodesic_distance(free, res, grid.index(route.goal))

    def geo_at(p, search_m=0.3):
        r = int(np.ceil(search_m / res))
        cc = np.asarray(grid.index(p))
        lo, hi = np.maximum(cc - r, 0), np.minimum(cc + r + 1, np.asarray(grid.shape))
        block = geo_grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        idx = np.argwhere(np.isfinite(block))
        if not len(idx):
            raise ValueError(f'point {np.round(p, 3)} has no reachable free voxel within {search_m} m')
        return float(block[tuple(idx[np.linalg.norm((lo + idx - cc) * res, axis=1).argmin()])])

    gate_geo = np.zeros(len(c.gates))
    for k, g in enumerate(c.gates):
        if not str(g['name']).endswith('(none)'):                   # the dummy is never crossed: 0
            m = (np.asarray(g['A']) + np.asarray(g['B'])) / 2
            gate_geo[k] = geo_at([m[0], m[1], -float(g.get('z', CRUISE))])

    sec_grid, sinfo = label_sectors(grid, free, c.gates, route.points)
    if sinfo['conflicts']:
        raise ValueError(f'gate cut leaks: regions with route points of several sectors {sinfo["conflicts"]}')

    bank = build_free_bank(route, clr, n, bank_seed + seed, grid_lo=grid_lo, grid_hi=grid_hi, res=res,
                           min_clearance=min_clearance, xy_half=net_half - 0.1)
    # launch starts like bank_free_v3 (<= 0.30 m from the mark, clear of walls), facing the route +- tol
    rng = np.random.default_rng(10_000 + bank_seed + seed)
    li = np.nonzero(bank.kind == BANK_LAUNCH)[0]
    tol = math.radians(gcfg.launch_yaw_tol_deg)
    got = np.zeros((0, 2))
    for _ in range(200):
        if len(got) >= len(li):
            break
        r = launch_r * np.sqrt(rng.random(4 * len(li)))
        th = rng.uniform(0, 2 * np.pi, len(r))
        xy = c.launch[:2] + np.stack([r * np.cos(th), r * np.sin(th)], 1)
        ok = np.ones(len(xy), bool)
        for alt in (0.08, 0.5):
            ok &= clr(np.c_[xy, np.full(len(xy), -alt)]) >= min_clearance
        got = np.concatenate([got, xy[ok]])
    if len(got) < len(li):
        raise ValueError(f'only {len(got)} clear launch positions within {launch_r} m of the mark')
    bank.pos[li] = np.c_[got[:len(li)], np.full(len(li), -0.08)].astype(np.float32)
    yaw = c.launch_yaw + rng.uniform(-tol, tol, len(li))
    bank.yaw[li] = ((yaw + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    bank.geo[li] = np.array([geo_at(p) for p in bank.pos[li]], np.float32)

    ij = np.clip(np.rint((bank.pos - grid.lo) / res).astype(int), 0, np.asarray(grid.shape) - 1)
    sector = sec_grid[ij[:, 0], ij[:, 1], ij[:, 2]].astype(np.int8)
    if np.any(sector[li] != 0):
        raise ValueError(f'{int((sector[li] != 0).sum())} launch starts are not in sector 0')
    gsec = sec_grid[grid.index(route.goal)]
    if gsec != 4:
        raise ValueError(f'the goal is in sector {gsec}, not 4')
    info = dict(c.info, seed=seed, route_m=round(float(route.length), 2), net_half=net_half.round(3).tolist(),
                gate_names=[str(g['name']) for g in c.gates], sectors=sinfo, build_s=round(time.time() - t0, 1),
                bank_kinds={int(k): int((bank.kind == k).sum()) for k in np.unique(bank.kind)},
                sector_counts={int(k): int((sector == k).sum()) for k in range(5)})
    info = json.loads(json.dumps(info, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    log(f'[coursepack] seed {seed}: route {route.length:.1f} m, gate geo {np.round(gate_geo, 2).tolist()}, '
        f'sectors {info["sector_counts"]}, {info["build_s"]} s')
    return PackCourse(seed, c.mesh, [dict(g) for g in c.gates], np.asarray(c.pad_center, float), float(c.pad_r),
                      np.asarray(c.launch[:2], float), float(c.launch_yaw), goal, net_half - 0.05, route, bank,
                      sector, gate_geo, info)


# ── save / load ─────────────────────────────────────────────────────────────

def save_pack(path: str, courses: Sequence[PackCourse], meta: Dict) -> None:
    z = {}
    for k, c in enumerate(courses):
        p = f'c{k}_'
        z[p + 'mesh_vertices'], z[p + 'mesh_faces'] = c.mesh.vertices, c.mesh.faces
        z[p + 'mesh_group'], z[p + 'mesh_object'] = c.mesh.face_group, c.mesh.face_object
        for key in GATE_KEYS:
            z[p + 'gate_' + key] = np.array([g[key] for g in c.gates], np.float64)
        z[p + 'gate_L'] = np.array([g['L'] for g in c.gates], np.float64)
        z[p + 'gate_cp'] = np.array([g['cp'] for g in c.gates], np.int64)
        z[p + 'gate_z'] = np.array([g.get('z', np.nan) for g in c.gates], np.float64)
        z[p + 'pad'], z[p + 'mark'], z[p + 'goal'] = c.pad_center, c.mark, c.goal
        z[p + 'scalars'] = np.array([c.pad_r, c.launch_yaw, c.seed], np.float64)
        z[p + 'arena_half'] = c.arena_half
        z[p + 'route_points'], z[p + 'route_s'] = c.route.points, c.route.s
        for key in BANK_KEYS:
            z[p + 'bank_' + key] = getattr(c.bank, key)
        z[p + 'sector'], z[p + 'gate_geo'] = c.sector, c.gate_geo
    meta = dict(meta, n_courses=len(courses), seeds=[c.seed for c in courses], courses=[c.info for c in courses],
                gate_names=[[str(g['name']) for g in c.gates] for c in courses])
    z['meta'] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **z)


def load_pack(path: str) -> List[PackCourse]:
    out = []
    with np.load(path) as z:
        meta = json.loads(str(z['meta']))
        for k in range(meta['n_courses']):
            p = f'c{k}_'
            mesh = CourseMesh(z[p + 'mesh_vertices'].astype(np.float32), z[p + 'mesh_faces'].astype(np.int32),
                              z[p + 'mesh_group'].astype(np.int8), z[p + 'mesh_object'].astype(np.int32))
            gates = []
            for j, name in enumerate(meta['gate_names'][k]):
                g = {'name': name, 'cp': int(z[p + 'gate_cp'][j]), 'L': float(z[p + 'gate_L'][j])}
                g.update({key: z[p + 'gate_' + key][j].astype(np.float64) for key in GATE_KEYS})
                if np.isfinite(z[p + 'gate_z'][j]):
                    g['z'] = float(z[p + 'gate_z'][j])
                gates.append(g)
            pad_r, yaw, seed = z[p + 'scalars']
            route = Route(z[p + 'route_points'].astype(np.float32), z[p + 'route_s'].astype(np.float32),
                          z[p + 'goal'].astype(np.float32))
            bank = Bank(*(z[p + 'bank_' + key] for key in BANK_KEYS))
            out.append(PackCourse(int(seed), mesh, gates, z[p + 'pad'], float(pad_r), z[p + 'mark'], float(yaw),
                                  z[p + 'goal'].astype(np.float64), z[p + 'arena_half'], route, bank,
                                  z[p + 'sector'].astype(np.int64), z[p + 'gate_geo'], meta['courses'][k]))
    return out


def resolve_pack(path: str, assets_dir: Optional[str] = None) -> str:
    """A pack path as given, else relative to ``assets_dir``, else relative to the repo root."""
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for cand in (path, os.path.join(assets_dir or '', path), os.path.join(root, path)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f'course pack {path!r} not found (cwd, {assets_dir}, {root})')


def pack_meta(path: str) -> Dict:
    with np.load(path) as z:
        return json.loads(str(z['meta']))


# ── CLI ─────────────────────────────────────────────────────────────────────

def cmd_build(a) -> None:
    courses, rejected = [], {}
    for seed in parse_seeds(a.seeds):
        try:
            courses.append(build_course(seed, a.n, bank_seed=a.bank_seed, min_clearance=a.min_clearance,
                                        device=a.device))
        except Exception as exc:                        # noqa: BLE001 -- report and skip the course
            rejected[seed] = f'{type(exc).__name__}: {exc}'
            print(f'[coursepack] seed {seed} REJECTED: {rejected[seed]}', flush=True)
    if not courses:
        raise SystemExit('no course built')
    save_pack(a.out, courses, {'n_per_course': a.n, 'bank_seed': a.bank_seed, 'min_clearance': a.min_clearance,
                               'rejected': {str(k): v for k, v in rejected.items()}, 'source': 'coursegen.generate'})
    print(f'[coursepack] wrote {a.out}: {len(courses)} courses, seeds {[c.seed for c in courses]}, '
          f'rejected {list(rejected)}')
    cmd_info(argparse.Namespace(pack=a.out))


def cmd_info(a) -> None:
    from dronehacks.env import KIND_NAMES
    meta = pack_meta(a.pack)
    print(f"{a.pack}: {meta['n_courses']} courses, seeds {meta['seeds']}, rejected {meta.get('rejected')}")
    for c in meta['courses']:
        kinds = ' '.join(f'{KIND_NAMES[int(k)]}={v}' for k, v in c['bank_kinds'].items())
        print(f"  seed {c['seed']:4d} {c['order']:8s} spurs {c['spurs']} tube {str(c['tube']):9s} "
              f"window {c['window']:8s} route {c['route_m']:5.1f} m | {kinds} | sectors {c['sector_counts']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--seeds', required=True, help="coursegen seeds, e.g. '0-9' or '100-104' or '1,4,7'")
    b.add_argument('--out', required=True)
    b.add_argument('--n', type=int, default=16384, help='bank starts per course')
    b.add_argument('--bank-seed', type=int, default=0)
    b.add_argument('--min-clearance', type=float, default=0.23)
    b.add_argument('--device', default='cpu')
    i = sub.add_parser('info')
    i.add_argument('pack')
    a = ap.parse_args(argv)
    {'build': cmd_build, 'info': cmd_info}[a.cmd](a)


if __name__ == '__main__':
    main()
