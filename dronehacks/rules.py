"""DroneHacks 2026 scoring rules and checkpoint geometry (numpy only).

Geometry is copied from the organizers' "course from above" diagram (SVG, inches, origin top-left, north
up, 240 x 240 = the 20 x 20 ft net). ``svg_to_ned`` maps it into the course NED frame (origin = floor
centre, x north, y east, z down): the diagram's launch dot lands on the launch mark (0.914, -2.515) and the
pad centre on (-0.793, -2.042), both of which match the OBJ mesh independently.

Scoring per run, out of 60 before the return bonus:
  checkpoints  4, 7, 10, 7, 7 (35), cleared IN ORDER; a checkpoint reached after flying over a wall since
               the previous one earns 15% (rule 7.7)
  finished     + 20 * (fastest finish / your time), + 5 if CP4 went through the window (not Gate B)
  unfinished   + 7 * (fastest time to your last checkpoint / your time to it); 0 if no checkpoint
  return       x 1.25 if, after landing on the pad, the drone takes off by itself, flies back over the
               barrier and lands within 2 ft of the launch mark
"Fastest" is the best of all teams and is unknowable here: the env substitutes a reference time
(remaining geodesic distance / ref_speed), capped at ratio 1.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

IN = 0.0254

CP_POINTS: Tuple[float, ...] = (4.0, 7.0, 10.0, 7.0, 7.0)
FINISH_SPEED_POINTS = 20.0
UNFINISHED_SPEED_POINTS = 7.0
WINDOW_BONUS = 5.0
OVER_WALL_FACTOR = 0.15
RETURN_MULTIPLIER = 1.25
MAX_RUN_POINTS = sum(CP_POINTS) + FINISH_SPEED_POINTS + WINDOW_BONUS          # 60
WALL_TOP_M = 68 * IN                                                          # 1.727 m


def svg_to_ned(x: float, y: float) -> Tuple[float, float]:
    """Diagram inches (x right/east, y down/south, origin top-left) -> NED metres (north, east)."""
    return ((120.0 - y) * IN, (x - 120.0) * IN)


def _poly(pts: Sequence[Tuple[float, float]]) -> np.ndarray:
    return np.array([svg_to_ned(x, y) for x, y in pts], dtype=np.float64)


#: The five shaded sectors: the stretch of course that leads to checkpoint k+1 (index k).
SECTORS: List[np.ndarray] = [
    _poly([(0, 100), (44, 100), (44, 18.25), (0, 18.25)]),                                      # 1 first corridor
    _poly([(0, 18.25), (44, 18.25), (44, 100), (132, 100), (132, 0), (0, 0)]),                  # 2 maze
    _poly([(132, 100), (182, 100), (182, 132), (240, 132), (240, 0), (132, 0)]),                # 3 tunnel
    _poly([(90.6, 240), (240, 240), (240, 132), (182, 132), (182, 104), (126.17, 104),
           (126.17, 141.54), (112.32, 179.6), (90.6, 239.28)]),                                 # 4 chicane
    _poly([(0, 240), (90.6, 240), (90.6, 239.28), (112.32, 179.6), (126.17, 141.54),
           (126.17, 104), (0, 104)]),                                                           # 5 landing area
]


def _gate(name: str, cp: int, a: Tuple[float, float], b: Tuple[float, float],
          travel_svg: Tuple[float, float] = None) -> Dict:
    A, B = np.array(svg_to_ned(*a)), np.array(svg_to_ned(*b))
    t = (B - A) / np.linalg.norm(B - A)
    if travel_svg is None:                                    # perpendicular pointing toward -x (landing area)
        d = (b[0] - a[0], b[1] - a[1])
        n_svg = (d[1], -d[0])
        if n_svg[0] > 0:
            n_svg = (-n_svg[0], -n_svg[1])
        travel_svg = n_svg
    nv = np.array([-travel_svg[1], travel_svg[0]], dtype=np.float64)          # svg (dx, dy) -> NED (north, east)
    nv /= np.linalg.norm(nv)
    return {'name': name, 'cp': cp, 'A': A, 'B': B, 't': t, 'L': float(np.linalg.norm(B - A)), 'n': nv}


#: The dashed checkpoint lines. ``n`` = required direction of travel (NED); ``cp`` = checkpoint index
#: (0-based; Gate B and the window both clear checkpoint 4).
GATES: List[Dict] = [
    _gate('corridor_end', 0, (0, 18.25), (44, 18.25), (0, -1)),             # cross going north
    _gate('gate_a', 1, (132, 36.5), (132, 0), (1, 0)),                      # cross going east
    _gate('tunnel_exit', 2, (182, 132), (240, 132), (0, 1)),                # cross going south
    _gate('gate_b', 3, (126.1735, 141.5435), (126.1735, 104), (-1, 0)),     # cross going west
    _gate('window', 3, (95.3028, 226.3592), (107.6155, 192.5303)),          # cross toward the landing area
]
GATE_CP = np.array([g['cp'] for g in GATES])
WINDOW_GATE = 4

PAD_CENTER = np.array(svg_to_ned(39.6, 151.2))
PAD_RADIUS_M = 39 * IN                                                       # the dashed outer ring
LAUNCH_MARK = np.array(svg_to_ned(21, 84))
RETURN_RADIUS_M = 24 * IN                                                    # land within 2 ft of the launch mark


def _in_poly(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(len(p), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1)
        inside ^= cond
    return inside


def _dist_to_poly(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    best = np.full(len(p), np.inf)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-18), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def sector_index(xy_ned: np.ndarray) -> np.ndarray:
    """(N,2) NED -> (N,) checkpoint the drone is working toward (0 = first corridor ... 4 = landing area).
    Points on a wall strip between sectors take the nearest sector."""
    p = np.asarray(xy_ned, dtype=np.float64).reshape(-1, 2)
    out = np.full(len(p), -1, int)
    for k, poly in enumerate(SECTORS):
        m = (out < 0) & _in_poly(p, poly)
        out[m] = k
    todo = out < 0
    if todo.any():
        d = np.stack([_dist_to_poly(p[todo], poly) for poly in SECTORS], axis=1)
        out[todo] = d.argmin(1)
    return out


def run_points(cleared: Sequence[bool], over_wall: Sequence[bool], via_window: bool, finished: bool,
               t_finish: float = None, tref_finish: float = None, t_last: float = None,
               tref_last: float = None, returned: bool = False) -> float:
    """Reference scorer (pure python) for one run. ``cleared[k]``: checkpoint k cleared; ``over_wall[k]``:
    it was reached after flying over a wall. Ratios are capped at 1."""
    pts = sum(CP_POINTS[k] * (OVER_WALL_FACTOR if over_wall[k] else 1.0) for k in range(5) if cleared[k])
    if finished:
        pts += FINISH_SPEED_POINTS * min(1.0, tref_finish / max(t_finish, 1e-9))
        pts += WINDOW_BONUS if via_window else 0.0
        return pts * (RETURN_MULTIPLIER if returned else 1.0)
    if any(cleared):
        pts += UNFINISHED_SPEED_POINTS * min(1.0, tref_last / max(t_last, 1e-9))
    return pts
