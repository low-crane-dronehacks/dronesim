"""Layout perturbations of the DroneHacks course (framework-free: numpy only).

The organizers build the course from plywood and foam, so the real thing never matches the CAD exactly:
walls sit a few cm off and a few degrees rotated, boulders and the tunnel are nudged, painted lines and the
landing pad are not where the diagram says. ``perturb_layout`` makes one such perturbed copy of the course
at a given ``severity`` in [0, 1] (1 = the ranges below):

  walls     the 7 rigid wall assemblies (found by shared vertices): translate +-0.15 m per axis, yaw +-4 deg
            about the assembly's centre
  tunnel    the tunnel roof/side/end pieces as ONE rigid body: translate +-0.10 m, yaw +-2 deg
  boulders  each of the 5: translate +-0.10 m
  gates     the 5 dashed checkpoint lines: translate +-0.10 m, rotate +-3 deg about their midpoint
  pad/mark  landing pad centre (== the goal) and launch mark: translate +-0.10 m
Floor and net never move. Heights never change (the drone flies over walls at the same 1.727 m).

Everything that scores follows the layout: the gates, the pad ring and the goal. The collision geometry and
the depth image come from the perturbed mesh. What does NOT follow: the planned route / bank cells (their
nominal positions are reused; ``valid_start`` is how an env rejects a variant that drops a start inside a
wall) and the per-gate geodesic distances used for the reference time.

A perturbed copy is accepted only if it is still solvable (``check_solvable``): the launch is clear, a path
exists from launch to goal at cruise altitude with the bank's 0.23 m clearance, and it is neither much
longer (a passage closed / forced detour) nor much shorter (a gap opened where two walls used to meet) than
the nominal one.

Frame: NED, x north, y east.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dronehacks.mesh import CourseMesh, GROUP_ID

#: severity-1 magnitudes
WALL_SHIFT_M = 0.15
WALL_YAW_DEG = 4.0
TUNNEL_SHIFT_M = 0.10
TUNNEL_YAW_DEG = 2.0
BOULDER_SHIFT_M = 0.10
GATE_SHIFT_M = 0.10
GATE_YAW_DEG = 3.0
LANDMARK_SHIFT_M = 0.10
VARIANT_SPACING_M = 20.0        # stack_variants: spacing along +y; > the 10 m obstacle-query range + course width


@dataclass
class Body:
    name: str
    group: str
    faces: np.ndarray           # bool (F,) over CourseMesh faces
    pivot: np.ndarray           # (2,) xy centre of the body's vertices
    shift_m: float              # severity-1 translation limit
    yaw_deg: float              # severity-1 yaw limit (0 = translation only)


def _rigid_components(mesh: CourseMesh, group: str, weld: float = 1e-3) -> List[np.ndarray]:
    """Face masks of the groups of OBJ objects (of ``group``) that share a vertex position."""
    f = np.nonzero(mesh.face_group == GROUP_ID[group])[0]
    objs = np.unique(mesh.face_object[f])
    par = {int(o): int(o) for o in objs}

    def find(a: int) -> int:
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    seen: Dict[tuple, int] = {}
    for o in objs:
        for v in np.unique(mesh.faces[f[mesh.face_object[f] == o]]):
            key = tuple(np.round(mesh.vertices[v] / weld).astype(np.int64))
            if key in seen:
                par[find(int(o))] = find(seen[key])
            else:
                seen[key] = int(o)
    comps: Dict[int, List[int]] = {}
    for o in objs:
        comps.setdefault(find(int(o)), []).append(int(o))
    masks = []
    for members in sorted(comps.values(), key=lambda m: min(m)):
        masks.append((mesh.face_group == GROUP_ID[group]) & np.isin(mesh.face_object, members))
    return masks


def rigid_bodies(mesh: CourseMesh) -> List[Body]:
    """Walls (one body per connected assembly), boulders (one each) and the tunnel (all pieces together)."""
    bodies: List[Body] = []
    for i, m in enumerate(_rigid_components(mesh, 'wall')):
        bodies.append(_body(mesh, f'wall{i}', 'wall', m, WALL_SHIFT_M, WALL_YAW_DEG))
    for i, m in enumerate(_rigid_components(mesh, 'boulder')):
        bodies.append(_body(mesh, f'boulder{i}', 'boulder', m, BOULDER_SHIFT_M, 0.0))
    t = mesh.face_group == GROUP_ID['tunnel']
    if t.any():
        bodies.append(_body(mesh, 'tunnel', 'tunnel', t, TUNNEL_SHIFT_M, TUNNEL_YAW_DEG))
    return bodies


def _body(mesh, name, group, faces, shift, yaw) -> Body:
    v = mesh.vertices[np.unique(mesh.faces[faces])]
    return Body(name, group, faces, 0.5 * (v[:, :2].min(0) + v[:, :2].max(0)), shift, yaw)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass
class Layout:
    """One perturbed copy: the mesh plus everything the scoring needs to follow it."""
    severity: float
    mesh: CourseMesh
    gates: List[dict]               # same keys as rules.GATES (A, B, t, L, n, ...)
    pad_center: np.ndarray          # (2,)
    mark: np.ndarray                # (2,)
    goal: np.ndarray                # (3,) NED
    shifts: Dict[str, Tuple[float, float, float]]      # body/landmark -> (dx, dy, yaw_deg), for inspection


def perturb_layout(mesh: CourseMesh, bodies: Sequence[Body], gates: Sequence[dict], pad_center, mark, goal,
                   rng: np.random.Generator, severity: float) -> Layout:
    s = float(np.clip(severity, 0.0, 1.0))
    if s == 0.0:                                                    # exactly the nominal course (no float round trip)
        return Layout(0.0, mesh, [dict(g) for g in gates], np.asarray(pad_center, np.float64),
                      np.asarray(mark, np.float64), np.asarray(goal, np.float64), {})
    verts = mesh.vertices.copy()
    shifts: Dict[str, Tuple[float, float, float]] = {}
    for b in bodies:
        d = rng.uniform(-1, 1, 2) * b.shift_m * s
        th = np.deg2rad(rng.uniform(-1, 1) * b.yaw_deg * s) if b.yaw_deg else 0.0
        idx = np.unique(mesh.faces[b.faces])
        rel = verts[idx, :2] - b.pivot
        verts[idx, :2] = (rel @ _rot(th).T + b.pivot + d).astype(np.float32)
        shifts[b.name] = (float(d[0]), float(d[1]), float(np.rad2deg(th)))
    new_mesh = CourseMesh(verts, mesh.faces, mesh.face_group, mesh.face_object)

    new_gates = []
    for g in gates:
        d = rng.uniform(-1, 1, 2) * GATE_SHIFT_M * s
        th = np.deg2rad(rng.uniform(-1, 1) * GATE_YAW_DEG * s)
        R, mid = _rot(th), 0.5 * (g['A'] + g['B'])
        ng = dict(g)
        ng['A'] = (R @ (g['A'] - mid) + mid + d).astype(np.float64)
        ng['B'] = (R @ (g['B'] - mid) + mid + d).astype(np.float64)
        ng['t'] = R @ g['t']
        ng['n'] = R @ g['n']
        new_gates.append(ng)
        shifts['gate_' + str(g['name'])] = (float(d[0]), float(d[1]), float(np.rad2deg(th)))
    dp = rng.uniform(-1, 1, 2) * LANDMARK_SHIFT_M * s
    dm = rng.uniform(-1, 1, 2) * LANDMARK_SHIFT_M * s
    shifts['pad'], shifts['mark'] = (float(dp[0]), float(dp[1]), 0.0), (float(dm[0]), float(dm[1]), 0.0)
    goal = np.asarray(goal, np.float64).copy()
    goal[:2] += dp
    return Layout(s, new_mesh, new_gates, np.asarray(pad_center, np.float64) + dp,
                  np.asarray(mark, np.float64) + dm, goal, shifts)


# ── solvability ─────────────────────────────────────────────────────────────

def _free_slice(clearance: Callable[[np.ndarray], np.ndarray], altitude: float, half: float, res: float,
                radius: float):
    n = int(round(2 * half / res)) + 1
    ax = -half + res * np.arange(n)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    pts = np.stack([X, Y, np.full_like(X, -altitude)], -1).reshape(-1, 3)
    c = np.asarray(clearance(pts)).reshape(n, n)
    return c >= radius, n


def geodesic_length(clearance: Callable[[np.ndarray], np.ndarray], start_xy, goal_xy, *, altitude: float = 0.8,
                    half: float = 2.95, res: float = 0.05, radius: float = 0.23) -> float:
    """Shortest free path [m] from ``start_xy`` to ``goal_xy`` at ``altitude`` keeping ``radius`` of clearance
    (inf if there is none)."""
    from dronehacks.bank import geodesic_distance
    free, n = _free_slice(clearance, altitude, half, res, radius)
    idx = lambda p: tuple(int(np.clip(round((p[k] + half) / res), 0, n - 1)) for k in range(2))   # noqa: E731
    g, s = idx(goal_xy), idx(start_xy)
    if not free[g] or not free[s]:
        return float('inf')
    d = geodesic_distance(free[:, :, None], res, (g[0], g[1], 0))
    return float(d[s[0], s[1], 0])


def check_solvable(clearance: Callable[[np.ndarray], np.ndarray], launch_xy, goal_xy, nominal_len: float, *,
                   shorter: float = 0.93, longer: float = 1.30, **kw) -> Tuple[bool, str]:
    """(ok, reason). ``clearance`` is the distance-to-obstacle function of the PERTURBED mesh."""
    L = geodesic_length(clearance, launch_xy, goal_xy, **kw)
    if not np.isfinite(L):
        return False, 'no path (launch or goal blocked / passage closed)'
    if L < shorter * nominal_len:
        return False, f'shortcut opened ({L:.2f} m vs nominal {nominal_len:.2f} m)'
    if L > longer * nominal_len:
        return False, f'forced detour ({L:.2f} m vs nominal {nominal_len:.2f} m)'
    return True, 'ok'


# ── a set of variants stacked into ONE mesh ─────────────────────────────────

def stack_variants(meshes: Sequence[CourseMesh], spacing: float = VARIANT_SPACING_M
                   ) -> Tuple[CourseMesh, np.ndarray]:
    """Concatenate K meshes into one, copy k shifted by ``k * spacing`` along +y. One BVH then serves every
    variant: env e in variant k queries at ``pos + offsets[k]`` and never sees another copy (the gap is wider
    than any ray / distance query)."""
    verts, faces, grp, obj = [], [], [], []
    base = 0
    offsets = np.zeros((len(meshes), 3), np.float32)
    for k, m in enumerate(meshes):
        offsets[k, 1] = k * spacing
        verts.append(m.vertices + offsets[k])
        faces.append(m.faces + base)
        grp.append(m.face_group)
        obj.append(m.face_object)
        base += len(m.vertices)
    return CourseMesh(np.concatenate(verts), np.concatenate(faces).astype(np.int32),
                      np.concatenate(grp), np.concatenate(obj)), offsets


@dataclass
class Variants:
    """K accepted layouts; index 0 is the nominal course when ``include_nominal``."""
    layouts: List[Layout]
    severities: np.ndarray          # (K,)
    attempts: int
    rejected: Dict[str, int]

    def __len__(self) -> int:
        return len(self.layouts)


def build_variants(mesh: CourseMesh, gates: Sequence[dict], pad_center, mark, goal, *, severities: Sequence[float],
                   seed: int, clearance_factory: Callable[[CourseMesh], Callable[[np.ndarray], np.ndarray]],
                   launch_xy, max_tries: int = 60, **solve_kw) -> Variants:
    """One accepted layout per entry of ``severities`` (0 -> the unperturbed course). ``clearance_factory``
    maps a mesh to its clearance function (the exact-distance warp scene in the env)."""
    bodies = rigid_bodies(mesh)
    rng = np.random.default_rng(seed)
    obst = lambda m: m.select(('wall', 'tunnel', 'boulder'))            # noqa: E731
    nominal = geodesic_length(clearance_factory(obst(mesh)), launch_xy, goal[:2], **solve_kw)
    if not np.isfinite(nominal):
        raise ValueError('the nominal course has no launch -> goal path at this clearance')
    layouts: List[Layout] = []
    rejected: Dict[str, int] = {}
    attempts = 0
    for sev in severities:
        if sev <= 0:
            layouts.append(Layout(0.0, mesh, [dict(g) for g in gates], np.asarray(pad_center, np.float64),
                                  np.asarray(mark, np.float64), np.asarray(goal, np.float64), {}))
            continue
        for _ in range(max_tries):
            attempts += 1
            lay = perturb_layout(mesh, bodies, gates, pad_center, mark, goal, rng, sev)
            ok, why = check_solvable(clearance_factory(obst(lay.mesh)), launch_xy, lay.goal[:2], nominal,
                                     **solve_kw)
            if ok:
                layouts.append(lay)
                break
            key = why.split(' (')[0]
            rejected[key] = rejected.get(key, 0) + 1
        else:
            raise RuntimeError(f'no solvable layout at severity {sev} in {max_tries} tries: {rejected}')
    return Variants(layouts, np.asarray([lay.severity for lay in layouts], np.float32), attempts, rejected)
