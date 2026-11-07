"""Rebuild the files in assets/ (only needed if the organizers change the course, or to change the bank).

    python -m dronehacks.build_assets mesh <course.obj | course-obj.zip>   # course.npz + scene.json
    python -m dronehacks.build_assets route                                 # route.npz + scoring.json
    python -m dronehacks.build_assets bank --n 65536                        # bank_free_v2.npz
    python -m dronehacks.build_assets return-bank --return-n 25000          # bank_return_v2.npz

Run them in this order after a course change. All clearances are exact distances to the triangle mesh.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from dronehacks import rules
from dronehacks.bank import (BANK_CLIMB, BANK_CROSS, BANK_FINAL, BANK_PAD, Bank, Route, build_free_bank,
                             build_return_bank, gate_geodesics, plan_training_route)
from dronehacks.env import ASSETS
from dronehacks.mesh import GROUP_ID, CourseMesh, build_course, read_obj


def _paint_centre(course, names, lo_size, hi_size):
    """NED floor centre of the paint objects ``names`` (OBJ object indices)."""
    m = np.isin(course.face_object, names) & (course.face_group == GROUP_ID['paint'])
    if not m.any():
        raise SystemExit(f'landmark objects {names} not found: the layout changed; update cmd_mesh')
    pts = course.vertices[np.unique(course.faces[m])]
    size = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    if not lo_size <= size <= hi_size:
        raise SystemExit(f'landmark {names} is {size:.3f} m across, expected {lo_size}-{hi_size}: update cmd_mesh')
    c = (pts.min(0) + pts.max(0)) / 2.0
    return [round(float(c[0]), 4), round(float(c[1]), 4), 0.0]


def cmd_mesh(a) -> None:
    course = build_course(*read_obj(a.source))
    course.save(os.path.join(a.assets, 'course.npz'))
    nlo, nhi = course.select(('net',)).bounds()
    nhi[2] = min(nhi[2], 0.0)
    # Object indices are specific to course v3 (OBJ names are meaningless): the 15 cm launch dot and the
    # outermost ring of the landing pad.
    landmarks = {'launch_mark_ned': _paint_centre(course, [122, 123, 124], 0.14, 0.17),
                 'landing_pad_ned': _paint_centre(course, [155, 156, 157, 158], 1.9, 2.1),
                 'wall_top_m': rules.WALL_TOP_M}
    with open(os.path.join(a.assets, 'scene.json'), 'w') as f:
        json.dump({'display_name': 'DroneHacks 2026 course (v3)', 'mesh': 'course.npz',
                   'frame': 'NED; origin = floor centre, x = north, y = east, z = down',
                   'env_bounds': {'min': nlo.round(4).tolist(), 'max': nhi.round(4).tolist()},
                   'landmarks': landmarks}, f, indent=2)
    print(f'{len(course.faces)} triangles:', course.group_counts(), '| landmarks', landmarks)


def _setup(a):
    from dronehacks.render import MeshScene
    course = CourseMesh.load(os.path.join(a.assets, 'course.npz'))
    with open(os.path.join(a.assets, 'scene.json')) as f:
        lm = json.load(f)['landmarks']
    scene = MeshScene(course.select(('wall', 'tunnel', 'boulder')), 1, device=a.device)
    grid = dict(grid_lo=[-2.9, -2.9, -(lm['wall_top_m'] - 0.25)], grid_hi=[2.9, 2.9, -0.2])
    return scene, lm, grid


def cmd_route(a) -> None:
    scene, lm, grid = _setup(a)
    goal = np.array([lm['landing_pad_ned'][0], lm['landing_pad_ned'][1], -a.goal_altitude])
    route = plan_training_route(scene.distance_at, lm['launch_mark_ned'], goal, radius=a.route_radius,
                                cruise=a.cruise, wall_top=lm['wall_top_m'])
    route.save(os.path.join(a.assets, 'route.npz'))
    pts = np.array([[*((g['A'] + g['B']) / 2), -a.cruise] for g in rules.GATES])
    geo = gate_geodesics(route, scene.distance_at, pts, min_clearance=a.min_clearance, **grid)
    with open(os.path.join(a.assets, 'scoring.json'), 'w') as f:
        json.dump({'gates': [g['name'] for g in rules.GATES], 'gate_geodesic_m': [round(float(x), 3) for x in geo],
                   'cruise_altitude_m': a.cruise, 'min_clearance_m': a.min_clearance}, f, indent=1)
    print(f'route {route.length:.2f} m; gate geodesics to goal (m): {np.round(geo, 2).tolist()}')


def cmd_bank(a) -> None:
    scene, lm, grid = _setup(a)
    route = Route.load(os.path.join(a.assets, 'route.npz'))
    bank = build_free_bank(route, scene.distance_at, a.n, a.seed, min_clearance=a.min_clearance, **grid)
    bank.save(os.path.join(a.assets, a.name or 'bank_free_v2.npz'))
    c = scene.distance_at(bank.pos)
    print(f'{len(bank)} starts; clearance min {c.min():.3f} m; geodesic to goal median {np.nanmedian(bank.geo):.1f} m')


def cmd_return_bank(a) -> None:
    scene, lm, _ = _setup(a)
    base = Bank.load(os.path.join(a.assets, a.base))
    extra = build_return_bank(scene.distance_at, lm['launch_mark_ned'][:2], lm['landing_pad_ned'][:2], a.return_n,
                              a.seed + 1, min_clearance=a.min_clearance, pad_radius=rules.PAD_RADIUS_M - 0.4,
                              wall_top=lm['wall_top_m'])
    Bank.concat(base, extra).save(os.path.join(a.assets, a.name or 'bank_return_v2.npz'))
    print(f'{len(base)} run starts + {len(extra)} return starts (pad {(extra.kind == BANK_PAD).sum()}, climb '
          f'{(extra.kind == BANK_CLIMB).sum()}, cross {(extra.kind == BANK_CROSS).sum()}, final {(extra.kind == BANK_FINAL).sum()})')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--assets', default=ASSETS)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-clearance', type=float, default=0.23, help='start / route clearance, m')
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('mesh')
    m.add_argument('source', help='course OBJ or the organizers\' OBJ .zip')
    r = sub.add_parser('route')
    r.add_argument('--route-radius', type=float, default=0.25)
    r.add_argument('--goal-altitude', type=float, default=0.3)
    r.add_argument('--cruise', type=float, default=0.8)
    b = sub.add_parser('bank')
    b.add_argument('--n', type=int, default=65536)
    b.add_argument('--name', default=None)
    rb = sub.add_parser('return-bank')
    rb.add_argument('--base', default='bank_free_v2.npz')
    rb.add_argument('--return-n', type=int, default=25000)
    rb.add_argument('--name', default=None)
    a = ap.parse_args(argv)
    {'mesh': cmd_mesh, 'route': cmd_route, 'bank': cmd_bank, 'return-bank': cmd_return_bank}[a.cmd](a)


if __name__ == '__main__':
    main()
