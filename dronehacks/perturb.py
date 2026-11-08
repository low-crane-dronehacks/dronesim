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
NET_HALF_M = 3.05               # the net square's half width
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


def _axis(v: np.ndarray):
    """(centre, unit long axis) of a wall's xy vertices."""
    c = v.mean(0)
    w, U = np.linalg.eigh(np.cov((v - c).T))
    return c, U[:, np.argmax(w)]


def attached_walls(mesh: CourseMesh, bodies: Sequence[Body], anchor: str, touch_m: float = 0.05) -> List[str]:
    """Wall bodies that touch ``anchor`` on the nominal course (min vertex distance < touch_m)."""
    by = {b.name: b for b in bodies}
    va = mesh.vertices[np.unique(mesh.faces[by[anchor].faces])][:, :2].astype(np.float64)
    ca, ua = _axis(va)
    na = np.array([-ua[1], ua[0]])
    half_t = float(np.abs((va - ca) @ na).max())            # half thickness of the anchor
    lo, hi = float(((va - ca) @ ua).min()), float(((va - ca) @ ua).max())
    out = []
    for b in bodies:
        if b.group != 'wall' or b.name == anchor:
            continue
        vb = mesh.vertices[np.unique(mesh.faces[b.faces])][:, :2].astype(np.float64)
        along = (vb - ca) @ ua
        # touches = some vertex within (half thickness + touch_m) of the anchor's centre line, inside its extent
        # (the anchor's own vertices are only at its ends, so a vertex-to-vertex test misses T-junctions)
        near = (np.abs((vb - ca) @ na) < half_t + touch_m) & (along > lo - touch_m) & (along < hi + touch_m)
        if near.any():
            out.append(b.name)
    return out


def snap_to_anchor(verts: np.ndarray, mesh: CourseMesh, bodies: Sequence[Body], anchor: str,
                   attached: Sequence[str]) -> None:
    """In place: slide each attached wall along its own long axis until its end nearest the anchor wall lies on
    the anchor's centre line (so it physically touches it: no gap a drone could slip through)."""
    by = {b.name: b for b in bodies}
    ia = np.unique(mesh.faces[by[anchor].faces])
    ca, ua = _axis(verts[ia, :2].astype(np.float64))
    na = np.array([-ua[1], ua[0]])
    for name in attached:
        ib = np.unique(mesh.faces[by[name].faces])
        v = verts[ib, :2].astype(np.float64)
        cb, ub = _axis(v)
        if abs(float(na @ ub)) < 0.2:                 # (nearly) parallel to the anchor: nothing to slide along
            continue
        proj = (v - cb) @ ub
        ends = [cb + proj.min() * ub, cb + proj.max() * ub]
        e = min(ends, key=lambda q: abs(float(na @ (q - ca))))
        sl = -float(na @ (e - ca)) / float(na @ ub)
        verts[ib, :2] = (v + sl * ub).astype(np.float32)


def perturb_layout(mesh: CourseMesh, bodies: Sequence[Body], gates: Sequence[dict], pad_center, mark, goal,
                   rng: np.random.Generator, severity: float, scale: float = 1.0, anchor: str = '',
                   anchor_frac: float = 0.25, attached: Sequence[str] = (), gates_follow: bool = False,
                   placed: Sequence[str] = (), gaps: Sequence[tuple] = (), spans: Sequence[tuple] = ()) -> Layout:
    """``anchor``: the centre wall moves only ``anchor_frac`` x the severity-1 limits (NOT x ``scale``), and every
    wall in ``attached`` is slid back onto it afterwards (the prior: the centre line barely moves and the walls
    that meet it still meet it)."""
    s = float(np.clip(severity, 0.0, 1.0))
    k = s * float(scale)                      # body/gate magnitudes; the taped launch mark and pad keep s
    if s == 0.0:                                                    # exactly the nominal course (no float round trip)
        return Layout(0.0, mesh, [dict(g) for g in gates], np.asarray(pad_center, np.float64),
                      np.asarray(mark, np.float64), np.asarray(goal, np.float64), {})
    verts = mesh.vertices.copy()
    shifts: Dict[str, Tuple[float, float, float]] = {}
    moves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}      # body -> (R, pivot, d): p' = R(p-piv)+piv+d
    for b in bodies:
        kb = s * anchor_frac if b.name == anchor else (s if b.name in [q.split(':')[0] for q in placed] else k)
        d = rng.uniform(-1, 1, 2) * b.shift_m * kb
        th = np.deg2rad(rng.uniform(-1, 1) * b.yaw_deg * kb) if b.yaw_deg else 0.0
        idx = np.unique(mesh.faces[b.faces])
        rel = verts[idx, :2] - b.pivot
        verts[idx, :2] = (rel @ _rot(th).T + b.pivot + d).astype(np.float32)
        shifts[b.name] = (float(d[0]), float(d[1]), float(np.rad2deg(th)))
        moves[b.name] = (_rot(th), np.asarray(b.pivot, np.float64), d.astype(np.float64))
    if anchor and attached:
        before = verts.copy()
        snap_to_anchor(verts, mesh, bodies, anchor, attached)
        for b in bodies:
            if b.name in attached:
                idx = np.unique(mesh.faces[b.faces])
                R, piv, d = moves[b.name]
                moves[b.name] = (R, piv, d + (verts[idx, :2] - before[idx, :2]).mean(0).astype(np.float64))
    for q in placed:                     # a placed wall stays in front of its net and not past its limit wall
        name, side, limit = (q.split(':') + ['', ''])[:3]
        if not side:                     # a spanned wall: unscaled jitter only, no net/limit fitting
            continue
        idx = np.unique(mesh.faces[{b.name: b for b in bodies}[name].faces])
        v = verts[idx, :2]
        lim = NET_HALF_M - 0.02
        verts[idx, :2] = v + (np.minimum(0.0, lim - v.max(0)) + np.maximum(0.0, -lim - v.min(0))).astype(np.float32)
        fit_placed(verts, mesh, bodies, name, side, limit)
    for sp in spans:                              # spanned walls stay connected to their (perturbed) anchors
        span_wall(verts, mesh, bodies, *sp)
    for (ra, ob, g, piv, side, limit) in gaps:   # the measured as-built gap survives the perturbation
        enforce_gap(verts, mesh, bodies, ra, ob, g, piv, side, limit)
    new_mesh = CourseMesh(verts, mesh.faces, mesh.face_group, mesh.face_object)

    new_gates = []
    if gates_follow:          # each line moves rigidly with the wall/tunnel it is drawn against (then extend_gates)
        hosts = [b for b in bodies if b.group in ('wall', 'tunnel')]
        hv = {b.name: mesh.vertices[np.unique(mesh.faces[b.faces])][:, :2].astype(np.float64) for b in hosts}
        for g in gates:
            mid = 0.5 * (np.asarray(g['A'], np.float64) + np.asarray(g['B'], np.float64))
            seg = np.linspace(g['A'], g['B'], 9)
            host = min(hosts, key=lambda b: float(np.min(np.linalg.norm(hv[b.name][:, None] - seg[None], axis=-1))))
            R, piv, d = moves[host.name]
            ng = dict(g)
            ng['A'] = R @ (np.asarray(g['A'], np.float64) - piv) + piv + d
            ng['B'] = R @ (np.asarray(g['B'], np.float64) - piv) + piv + d
            ng['t'] = R @ np.asarray(g['t'], np.float64)
            ng['n'] = R @ np.asarray(g['n'], np.float64)
            new_gates.append(ng)
            shifts['gate_' + str(g['name'])] = (0.0, 0.0, 0.0)
    for g in ([] if gates_follow else gates):
        d = rng.uniform(-1, 1, 2) * GATE_SHIFT_M * k
        th = np.deg2rad(rng.uniform(-1, 1) * GATE_YAW_DEG * k)
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


# ── course modifications (the as-built course differs from the published map) ──

def fit_placed(verts: np.ndarray, mesh: CourseMesh, bodies: Sequence[Body], name: str, side: str, limit: str) -> None:
    """In place: keep a wall placed along the <side> net inside the net AND not past ``limit`` (e.g. wall5 along the
    north net never reaches east of wall3's far end). Shifts it back first; if it still does not fit it is shortened
    along the net (the real one is ~5 ft)."""
    by = {b.name: b for b in bodies}
    idx = np.unique(mesh.faces[by[name].faces])
    along = 0 if side in ('west', 'east') else 1
    lo_lim, hi_lim = -NET_HALF_M + 0.02, NET_HALF_M - 0.02
    if limit:
        lv = verts[np.unique(mesh.faces[by[limit].faces]), along]
        hi_lim = min(hi_lim, float(lv.max()))         # the limit wall's far end (east for wall3 rotated east)
    v = verts[idx, along].astype(np.float64)
    if v.max() > hi_lim:
        v = v - (v.max() - hi_lim)
    if v.min() < lo_lim:
        v = v + (lo_lim - v.min())
    if v.max() > hi_lim:                              # still too long: shorten it about its west/south end
        v = lo_lim + (v - v.min()) * (hi_lim - lo_lim) / max(v.max() - v.min(), 1e-6)
    verts[idx, along] = v.astype(np.float32)


def _hull(v: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    return v[ConvexHull(v + 1e-9 * np.arange(len(v))[:, None]).vertices]


def wall_gap(va: np.ndarray, vb: np.ndarray) -> float:
    """Clear distance between two walls' xy footprints (convex hulls), 0 if they touch/overlap."""
    def pt_poly(p, poly):
        a, b = poly, np.roll(poly, -1, 0)
        ab = b - a
        t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / ((ab * ab).sum(-1)[None] + 1e-12), 0, 1)
        return np.linalg.norm(p[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(1)
    ha, hb = _hull(va), _hull(vb)
    return float(min(pt_poly(ha, hb).min(), pt_poly(hb, ha).min()))


def enforce_gap(verts: np.ndarray, mesh: CourseMesh, bodies: Sequence[Body], rot: str, other: str, g: float,
                pivot: str = 'south', place_side: str = '', place_limit: str = '') -> None:
    """In place: rotate wall ``rot`` about its ``pivot`` end until its clear gap to ``other`` is ``g`` (the as-built
    course: wall3's tip is ~3 ft from wall5). Re-fits a placed ``other`` against ``rot`` after each turn."""
    by = {b.name: b for b in bodies}
    ir, io = np.unique(mesh.faces[by[rot].faces]), np.unique(mesh.faces[by[other].faces])
    v0 = verts[ir, :2].astype(np.float64).copy()
    k = {'south': (0, np.argmin), 'north': (0, np.argmax), 'west': (1, np.argmin), 'east': (1, np.argmax)}[pivot]
    piv = v0[k[1](v0[:, k[0]])]
    o0 = verts[io, :2].copy()

    def at(th):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        verts[ir, :2] = ((v0 - piv) @ R.T + piv).astype(np.float32)
        verts[io, :2] = o0
        if place_side:
            fit_placed(verts, mesh, bodies, other, place_side, place_limit)
        return wall_gap(verts[ir, :2].astype(np.float64), verts[io, :2].astype(np.float64))
    lo, hi = np.deg2rad(-40.0), np.deg2rad(40.0)       # +th turns the tip east (and down): the gap grows
    if at(lo) > g:
        return
    if at(hi) < g:
        return
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if at(mid) < g:
            lo = mid
        else:
            hi = mid
    at(hi)


def span_wall(verts: np.ndarray, mesh: CourseMesh, bodies: Sequence[Body], name: str, anchor: str, e0: float,
              corner: str, dn: float) -> None:
    """In place: lay wall ``name`` (keeping its thickness) from ``anchor``'s centre line at east = e0 to the tunnel's
    ``corner`` shifted ``dn`` north along the tunnel's west face."""
    by = {b.name: b for b in bodies}
    idx = np.unique(mesh.faces[by[name].faces])
    va = verts[np.unique(mesh.faces[by[anchor].faces]), :2].astype(np.float64)
    ca, ua = _axis(va)
    A = ca + ua * ((e0 - ca[1]) / ua[1])
    tvv = verts[np.unique(mesh.faces[mesh.face_group == GROUP_ID['tunnel']]), :2].astype(np.float64)
    Bp = np.asarray({'nw': (tvv[:, 0].max(), tvv[:, 1].min()), 'sw': (tvv[:, 0].min(), tvv[:, 1].min())}[corner], np.float64)
    Bp[0] += dn
    v = verts[idx, :2].astype(np.float64)
    c, u = _axis(v)
    proj = (v - c) @ u
    off = (v - c) @ np.array([-u[1], u[0]])
    d = Bp - A
    t = d / max(float(np.linalg.norm(d)), 1e-9)
    s01 = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-9)
    verts[idx, :2] = (A[None] + s01[:, None] * d[None] + off[:, None] * np.array([-t[1], t[0]])[None]).astype(np.float32)


def apply_mods(mesh: CourseMesh, bodies: Sequence[Body], mods: str, rng: Optional[np.random.Generator],
               info: Optional[dict] = None) -> CourseMesh:
    """'drop:wall5;rot:wall3:15:45:south' -> a modified copy. drop: the body is sunk 50 m below the floor (no render,
    no collision; face indices stay valid for the body masks). rot:<body>:<lo>:<hi>:<end>: rotate the body about its
    <end> ('south'/'north'/'east'/'west' extreme point) by an angle in [lo, hi] deg, positive = clockwise from above
    (a north tip moves east). rng=None -> the midpoint (the nominal as-built course)."""
    if not mods:
        return mesh
    by_name = {b.name: b for b in bodies}
    verts = mesh.vertices.copy()
    for spec in [s for s in mods.split(';') if s.strip()]:
        p = spec.strip().split(':')
        if p[0] == 'gapwalls':
            # gapwalls:<h> -- every wall's bottom edge raised to h m above the floor (an open slot under all walls)
            # gapwalls:<lo>:<hi> -- a random height U(lo, hi) per layout
            h = float(p[1]) if len(p) < 3 else (0.5 * (float(p[1]) + float(p[2])) if rng is None else rng.uniform(float(p[1]), float(p[2])))
            wv = np.unique(mesh.faces[mesh.face_group == GROUP_ID['wall']])
            low = wv[verts[wv, 2] > -0.02]                    # bottom vertices (NED z down; the floor is z = 0)
            verts[low, 2] = -h
            continue
        if p[0] == 'winh':
            # winh:<wall>:<sill_lo>:<sill_hi>:<top_lo>:<top_hi> -- move the window's sill (CAD 0.3048 m) and top (CAD
            # 1.2192 m) on this wall to U(lo, hi) m (the as-built window height is uncertain); runs before/after gapwalls
            wi = np.unique(mesh.faces[by_name[p[1]].faces])
            pick = (lambda lo, hi: 0.5 * (lo + hi)) if rng is None else (lambda lo, hi: rng.uniform(lo, hi))
            sill, top = pick(float(p[2]), float(p[3])), pick(float(p[4]), float(p[5]))
            z0 = -mesh.vertices[wi, 2]
            verts[wi[np.isclose(z0, 0.3048, atol=2e-3)], 2] = -sill
            verts[wi[np.isclose(z0, 1.2192, atol=2e-3)], 2] = -top
            continue
        b = by_name[p[1]]
        idx = np.unique(mesh.faces[b.faces])
        if p[0] == 'drop':
            verts[idx, 2] += 50.0
        elif p[0] == 'rot':
            lo, hi = float(p[2]), float(p[3])
            ang = 0.5 * (lo + hi) if rng is None else rng.uniform(lo, hi)
            v = verts[idx, :2].astype(np.float64)
            k = {'south': (0, np.argmin), 'north': (0, np.argmax), 'west': (1, np.argmin), 'east': (1, np.argmax)}[p[4]]
            piv = v[k[1](v[:, k[0]])]
            th = np.deg2rad(ang)                         # NED (n, e): clockwise from above = +e for a +n vector
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            verts[idx, :2] = ((v - piv) @ R.T + piv).astype(np.float32)
        elif p[0] == 'shift':
            # shift:<body>:<lo>:<hi>:<north|south|east|west>[:mark] -- translate the wall by U(lo, hi) m; walls touching
            # it slide along to stay attached; 'mark': the launch mark (and launch starts) move with it
            lo, hi = float(p[2]), float(p[3])
            dist = 0.5 * (lo + hi) if rng is None else rng.uniform(lo, hi)
            dvec = {'north': (1.0, 0.0), 'south': (-1.0, 0.0), 'east': (0.0, 1.0), 'west': (0.0, -1.0)}[p[4]]
            dv = np.asarray(dvec) * dist
            att = attached_walls(mesh, bodies, p[1])
            verts[idx, :2] = (verts[idx, :2] + dv).astype(np.float32)
            snap_to_anchor(verts, mesh, bodies, p[1], att)
            if len(p) > 5 and p[5] == 'mark' and info is not None:
                info['mark_shift'] = info.get('mark_shift', np.zeros(2)) + dv
        elif p[0] == 'move':
            # move:<body>:<dN>:<dE>:<yaw_deg> -- a fixed rigid move about the body's pivot (e.g. fitted to a scan);
            # checkpoint lines drawn against this body move with it (build_variants)
            dN, dE, yd = float(p[2]), float(p[3]), float(p[4])
            b = {x.name: x for x in bodies}[p[1]]
            piv = np.asarray(b.pivot, np.float64)
            th = np.deg2rad(yd)
            Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            verts[idx, :2] = ((verts[idx, :2].astype(np.float64) - piv) @ Rm.T + piv + np.array([dN, dE])).astype(np.float32)
            if info is not None:
                info.setdefault('moves', {})[p[1]] = (Rm, piv, np.array([dN, dE]))
        elif p[0] == 'seg':
            # seg:<body>:<n0>:<e0>:<n1>:<e1> -- lay the wall (keeping its thickness) on the segment A=(n0,e0) -> B=(n1,e1),
            # e.g. a wall measured off a scan; never snapped to the anchor wall
            v = verts[idx, :2].astype(np.float64)
            c, u = _axis(v)
            proj = (v - c) @ u
            off = (v - c) @ np.array([-u[1], u[0]])
            A, Bp = np.array([float(p[2]), float(p[3])]), np.array([float(p[4]), float(p[5])])
            dd = Bp - A
            tt = dd / max(float(np.linalg.norm(dd)), 1e-9)
            s01 = (proj - proj.min()) / max(proj.max() - proj.min(), 1e-9)
            verts[idx, :2] = (A[None] + s01[:, None] * dd[None] + off[:, None] * np.array([-tt[1], tt[0]])[None]).astype(np.float32)
        elif p[0] == 'span':
            # span:<body>:<anchor wall>:<e_lo>:<e_hi>:<tunnel corner>[:<dn_lo>:<dn_hi>] -- lay the wall from the anchor
            # wall's centre line at E ~ U(e_lo, e_hi) to the tunnel's <corner> ('nw') moved dn along its west face (as-built:
            # wall6 runs from the middle of the centre wall to the edge of the tunnel). Re-laid after perturbation.
            e0 = 0.5 * (float(p[3]) + float(p[4])) if rng is None else rng.uniform(float(p[3]), float(p[4]))
            dn = 0.0
            if len(p) > 7:
                dn = 0.5 * (float(p[6]) + float(p[7])) if rng is None else rng.uniform(float(p[6]), float(p[7]))
            span_wall(verts, mesh, bodies, p[1], p[2], e0, p[5], dn)
            if info is not None:
                info.setdefault('spans', []).append((p[1], p[2], e0, p[5], dn))
        elif p[0] == 'trim':
            # trim:<body>:<lo>:<hi>:<keep end> -- keep a U(lo, hi) fraction of the wall's length from its <keep end>
            # (the as-built course: wall6 'mostly removed', a stub left on the centre wall); < 5 % -> dropped
            f = 0.5 * (float(p[2]) + float(p[3])) if rng is None else rng.uniform(float(p[2]), float(p[3]))
            v = verts[idx, :2].astype(np.float64)
            c, u = _axis(v)
            proj = (v - c) @ u
            keep = {'south': (0, -1), 'north': (0, 1), 'west': (1, -1), 'east': (1, 1)}[p[4]]
            if np.sign(u[keep[0]]) * keep[1] < 0:
                u, proj = -u, -proj                          # u now points toward the kept end
            end = proj.max()
            if f < 0.05:
                verts[idx, 2] += 50.0
            else:
                L = proj.max() - proj.min()
                new = end - (end - proj) * f                # squeeze toward the kept end
                verts[idx, :2] = (v + ((new - proj)[:, None] * u[None])).astype(np.float32)
        elif p[0] == 'gap':
            # gap:<rot wall>:<other wall>:<lo>:<hi>[:pivot] -- turn <rot wall> about its pivot end until its clear gap
            # to <other> is U(lo, hi) m; a placed <other> is re-fitted against it
            g = 0.5 * (float(p[3]) + float(p[4])) if rng is None else rng.uniform(float(p[3]), float(p[4]))
            pl = [m.split(':') for m in mods.split(';') if m.startswith(f'place:{p[2]}:')]
            enforce_gap(verts, mesh, bodies, p[1], p[2], g, p[5] if len(p) > 5 else 'south',
                        pl[0][2] if pl else '', pl[0][7] if pl and len(pl[0]) > 7 else '')
            if info is not None:
                info.setdefault('gaps', []).append((p[1], p[2], g, p[5] if len(p) > 5 else 'south',
                                                    pl[0][2] if pl else '', pl[0][7] if pl and len(pl[0]) > 7 else ''))
        elif p[0] == 'place':
            # place:<body>:<side>:<gap_lo>:<gap_hi>:<c_lo>:<c_hi> -- stand the wall parallel to the <side> net (west/east:
            # along north; north/south: along east), <gap> m in front of it, its centre at <c> along the net
            side, glo, ghi, clo, chi = p[2], float(p[3]), float(p[4]), float(p[5]), float(p[6])
            gap = 0.5 * (glo + ghi) if rng is None else rng.uniform(glo, ghi)
            cen = 0.5 * (clo + chi) if rng is None else rng.uniform(clo, chi)
            v = verts[idx, :2].astype(np.float64)
            c, u = _axis(v)
            along = 0 if side in ('west', 'east') else 1             # the wall must run along this NED axis
            if abs(u[along]) < 0.7:                                   # rotate 90 deg about its centre
                v = (v - c) @ _rot(np.pi / 2).T + c
            half_t = 0.5 * (v[:, 1 - along].max() - v[:, 1 - along].min())
            net = {'west': -NET_HALF_M, 'east': NET_HALF_M, 'south': -NET_HALF_M, 'north': NET_HALF_M}[side]
            inward = 1.0 if side in ('west', 'south') else -1.0
            tgt = np.zeros(2)
            tgt[1 - along] = net + inward * (gap + half_t)
            tgt[along] = cen
            cur = 0.5 * (v.min(0) + v.max(0))
            verts[idx, :2] = (v + (tgt - cur)).astype(np.float32)
            if True:
                fit_placed(verts, mesh, bodies, p[1], side, p[7] if len(p) > 7 else '')
        else:
            raise ValueError(f'unknown course mod {spec!r}')
    return CourseMesh(verts, mesh.faces, mesh.face_group, mesh.face_object)


def cp0_corner_to_wall(lay: 'Layout', bodies: Sequence[Body], wall: str, corner=(3.05, -3.05)) -> 'Layout':
    """As-built scoring: checkpoint 0 (corridor end) is the line from the arena ``corner`` to the free (north) tip of
    ``wall``; it is crossed away from the launch mark."""
    b = {x.name: x for x in bodies}[wall]
    v = lay.mesh.vertices[np.unique(lay.mesh.faces[b.faces])][:, :2].astype(np.float64)
    tip = v[np.argmax(v[:, 0])]
    A = np.asarray(corner, np.float64)
    d = tip - A
    L = float(np.linalg.norm(d))
    t = d / L
    n = np.array([-t[1], t[0]])
    if float(n @ (np.asarray(lay.mark, np.float64) - A)) > 0:      # the launch mark must be BEHIND the line
        n = -n
    gates = [dict(g) for g in lay.gates]
    for g in gates:
        if g['cp'] == 0:
            g.update(A=A, B=tip, t=t, L=L, n=n)
    return Layout(lay.severity, lay.mesh, gates, lay.pad_center, lay.mark, lay.goal, dict(lay.shifts, cp0=(float(tip[0]), float(tip[1]), 0.0)))


def extend_gates(lay: 'Layout', clearance: Callable[[np.ndarray], np.ndarray], altitudes=(0.5, 0.8, 1.2),
                 step: float = 0.02, stop_m: float = 0.03, half: float = 3.05) -> 'Layout':
    """Generous checkpoints: each gate line is extended from its midpoint along itself until it meets the
    (perturbed) walls / tunnel / net on both sides, at whichever altitude gives the longest free line; never shorter
    than the drawn line. Checkpoints become generic progress lines: a perturbed wall can no longer leave a gap
    between the line's end and the wall that the drone flies through unscored."""
    gates = []
    for g in lay.gates:
        A, B = np.asarray(g['A'], np.float64), np.asarray(g['B'], np.float64)
        t = (B - A) / max(float(np.linalg.norm(B - A)), 1e-9)
        mid = 0.5 * (A + B)
        best = None
        for alt in altitudes:
            ends = []
            for sgn in (-1.0, 1.0):
                r = np.arange(0.0, 2 * half * 1.5, step)
                pts = mid[None] + sgn * r[:, None] * t[None]
                inside = (np.abs(pts) <= half).all(1)
                c = np.asarray(clearance(np.concatenate([pts, np.full((len(pts), 1), -alt)], 1).astype(np.float32)))
                hit = np.nonzero((c < stop_m) | ~inside)[0]
                ends.append(float(r[hit[0]]) if len(hit) else float(r[-1]))
            if best is None or sum(ends) > sum(best):
                best = ends
        lo = min(-best[0], float((A - mid) @ t))       # never shorter than drawn
        hi = max(best[1], float((B - mid) @ t))
        nA, nB = mid + lo * t, mid + hi * t
        ng = dict(g)
        ng.update(A=nA, B=nB, L=float(hi - lo), t=t)
        gates.append(ng)
    return Layout(lay.severity, lay.mesh, gates, lay.pad_center, lay.mark, lay.goal, lay.shifts)


# ── random boulders ─────────────────────────────────────────────────────────

BOULDER_R = (0.15, 0.35)           # horizontal semi-axes, m
BOULDER_H = (0.25, 0.60)           # height, m (the CAD ones: r 0.23, h 0.53)


def boulder_mesh(rng: np.random.Generator, xy, rings: int = 6, segs: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """An irregular low-poly dome standing on the floor at ``xy`` (NED, z up = negative). -> (verts, faces)"""
    rx, ry = rng.uniform(*BOULDER_R, 2)
    h = rng.uniform(*BOULDER_H)
    yaw = rng.uniform(0, 2 * np.pi)
    noise = 1.0 + 0.12 * rng.standard_normal((rings, segs))
    V = []
    for i in range(rings):                                  # ring i: polar angle from the equator up
        phi = 0.5 * np.pi * i / rings
        for j in range(segs):
            th = 2 * np.pi * j / segs
            r = np.cos(phi) * noise[i, j]
            x, y = rx * r * np.cos(th), ry * r * np.sin(th)
            V.append((x, y, -h * np.sin(phi) * (0.9 + 0.1 * noise[i, j])))
    V.append((0.0, 0.0, -h))                                # apex
    V = np.asarray(V, np.float64)
    c, s_ = np.cos(yaw), np.sin(yaw)
    V[:, :2] = V[:, :2] @ np.array([[c, -s_], [s_, c]]).T + np.asarray(xy, np.float64)
    F = []
    for i in range(rings - 1):
        for j in range(segs):
            a, b = i * segs + j, i * segs + (j + 1) % segs
            F += [(a, b, b + segs), (a, b + segs, a + segs)]
    apex = len(V) - 1
    top = (rings - 1) * segs
    F += [(top + j, top + (j + 1) % segs, apex) for j in range(segs)]
    return V.astype(np.float32), np.asarray(F, np.int32)


def add_random_boulders(lay: 'Layout', rng: np.random.Generator, n_range: Tuple[int, int],
                        clearance: Callable[[np.ndarray], np.ndarray], half: float = 2.8) -> 'Layout':
    """Scatter n ~ U[n_range] boulders over the floor, clear of walls/tunnel/boulders (and of each other), the
    launch mark and the pad ring. They are low (<= 0.6 m), so they never close the cruise-altitude route."""
    lo, hi = n_range
    n = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    if n == 0:
        return lay
    keep_out = [(np.asarray(lay.mark, float), 0.7), (np.asarray(lay.pad_center, float), 1.1)]
    placed: List[Tuple[np.ndarray, float]] = []
    verts, faces = [lay.mesh.vertices], [lay.mesh.faces]
    nv = len(lay.mesh.vertices)
    for _ in range(40 * n):
        if len(placed) == n:
            break
        p = rng.uniform(-half, half, 2)
        r = BOULDER_R[1] + 0.05
        if any(np.linalg.norm(p - c) < d + r for c, d in keep_out):
            continue
        if any(np.linalg.norm(p - q) < r + rq + 0.1 for q, rq in placed):
            continue
        if float(np.asarray(clearance(np.array([[p[0], p[1], -0.2]], np.float32)))[0]) < r + 0.1:
            continue
        v, f = boulder_mesh(rng, p)
        verts.append(v)
        faces.append(f + nv)
        nv += len(v)
        placed.append((p, r))
    k = len(faces) - 1
    m = lay.mesh
    nf = sum(len(f) for f in faces[1:])
    mesh = CourseMesh(np.concatenate(verts), np.concatenate(faces).astype(np.int32),
                      np.concatenate([m.face_group, np.full(nf, GROUP_ID['boulder'], np.int8)]),
                      np.concatenate([m.face_object, np.full(nf, -2, np.int32)]))
    shifts = dict(lay.shifts, random_boulders=(float(k), 0.0, 0.0))
    return Layout(lay.severity, mesh, lay.gates, lay.pad_center, lay.mark, lay.goal, shifts)


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


def move_gates(gates: Sequence[dict], mesh: CourseMesh, bodies: Sequence[Body], moves: dict) -> List[dict]:
    """Gates whose host (nearest wall/tunnel body on the unmodified course) was moved by a 'move' mod follow it."""
    if not moves:
        return [dict(g) for g in gates]
    hosts = [b for b in bodies if b.group in ('wall', 'tunnel')]
    hv = {b.name: mesh.vertices[np.unique(mesh.faces[b.faces])][:, :2].astype(np.float64) for b in hosts}
    out = []
    for g in gates:
        seg = np.linspace(g['A'], g['B'], 9)
        host = min(hosts, key=lambda b: float(np.min(np.linalg.norm(hv[b.name][:, None] - seg[None], axis=-1))))
        ng = dict(g)
        if host.name in moves:
            Rm, piv, d = moves[host.name]
            ng['A'] = Rm @ (np.asarray(g['A'], np.float64) - piv) + piv + d
            ng['B'] = Rm @ (np.asarray(g['B'], np.float64) - piv) + piv + d
            ng['t'], ng['n'] = Rm @ np.asarray(g['t'], np.float64), Rm @ np.asarray(g['n'], np.float64)
        out.append(ng)
    return out


def build_variants(mesh: CourseMesh, gates: Sequence[dict], pad_center, mark, goal, *, severities: Sequence[float],
                   seed: int, clearance_factory: Callable[[CourseMesh], Callable[[np.ndarray], np.ndarray]],
                   launch_xy, max_tries: int = 60, extra_boulders: Tuple[int, int] = (0, 0), mods: str = '',
                   scale: float = 1.0, cp0_wall: str = '', anchor_wall: str = '', gate_extend: bool = False,
                   **solve_kw) -> Variants:
    """One accepted layout per entry of ``severities`` (0 -> the unperturbed course). ``clearance_factory``
    maps a mesh to its clearance function (the exact-distance warp scene in the env)."""
    bodies = rigid_bodies(mesh)
    rng = np.random.default_rng(seed)
    obst = lambda m: m.select(('wall', 'tunnel', 'boulder'))            # noqa: E731
    base = mesh
    spanned = [m.split(':')[1] for m in mods.split(';') if m.startswith(('span:', 'seg:'))] if mods else []
    attached = [w for w in attached_walls(base, bodies, anchor_wall) if w not in spanned] if anchor_wall else []
    info0: dict = {}
    if mods:                                         # as-built course: the nominal path is measured on it
        mesh = apply_mods(base, bodies, mods, None, info0)
    launch0 = np.asarray(launch_xy, np.float64) + info0.get('mark_shift', np.zeros(2))
    gates = move_gates(gates, base, bodies, info0.get('moves', {}))       # fixed moves are the same in every layout
    nominal = geodesic_length(clearance_factory(obst(mesh)), launch0, goal[:2], **solve_kw)
    if not np.isfinite(nominal):
        raise ValueError('the nominal course has no launch -> goal path at this clearance')
    layouts: List[Layout] = []
    rejected: Dict[str, int] = {}
    attempts = 0
    for sev in severities:
        if sev <= 0:
            lay0 = Layout(0.0, mesh, [dict(g) for g in gates], np.asarray(pad_center, np.float64),
                          np.asarray(mark, np.float64) + info0.get('mark_shift', np.zeros(2)),
                          np.asarray(goal, np.float64), {})
            lay0 = cp0_corner_to_wall(lay0, bodies, cp0_wall) if cp0_wall else lay0
            if gate_extend:
                lay0 = extend_gates(lay0, clearance_factory(lay0.mesh.select(('wall', 'tunnel', 'net'))))
            layouts.append(lay0)
            continue
        for _ in range(max_tries):
            attempts += 1
            info: dict = {}
            mm = apply_mods(base, bodies, mods, rng, info) if mods else mesh
            msh = info.get('mark_shift', np.zeros(2))
            lay = perturb_layout(mm, bodies, gates, pad_center,
                                 np.asarray(mark, np.float64) + msh, goal, rng, sev, scale, anchor=anchor_wall, attached=attached,
                                 gates_follow=gate_extend,
                                 placed=[':'.join([m.split(':')[1], m.split(':')[2]] + m.split(':')[7:8])
                                         for m in mods.split(';') if m.startswith('place:')] + spanned if mods else (),
                                 gaps=info.get('gaps', ()), spans=info.get('spans', ()))
            if cp0_wall:
                lay = cp0_corner_to_wall(lay, bodies, cp0_wall)
            if extra_boulders[1] > 0:
                lay = add_random_boulders(lay, rng, extra_boulders, clearance_factory(obst(lay.mesh)))
            if gate_extend:
                lay = extend_gates(lay, clearance_factory(lay.mesh.select(('wall', 'tunnel', 'net'))))
            ok, why = check_solvable(clearance_factory(obst(lay.mesh)), lay.mark, lay.goal[:2], nominal,
                                     **solve_kw)
            if ok:
                layouts.append(lay)
                break
            key = why.split(' (')[0]
            rejected[key] = rejected.get(key, 0) + 1
        else:
            raise RuntimeError(f'no solvable layout at severity {sev} in {max_tries} tries: {rejected}')
    return Variants(layouts, np.asarray([lay.severity for lay in layouts], np.float32), attempts, rejected)
