"""Collision-free path planning over a clearance (distance-to-obstacle) field (numpy only).

``clearance(points) -> distances`` is ``MeshScene.distance_at`` over the exact course mesh. Frames are NED
(z down, altitude = -z).
  1. ``plan_path``: A* on a voxel grid, 26-connected, cells blocked when clearance < radius, cost = length x
     (1 + penalty below ``comfort`` m) so routes keep to the middle of corridors.
  2. ``shortcut``: line-of-sight pruning, each segment checked every 2 cm against the exact clearance.
  3. ``round_corners``: quadratic Bezier corners, shrunk until they still clear ``radius``.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

Clearance = Callable[[np.ndarray], np.ndarray]

_NEIGHBOURS = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)
_STEP_LEN = np.linalg.norm(_NEIGHBOURS, axis=1)


class NoPathError(RuntimeError):
    pass


@dataclass
class Grid:
    lo: np.ndarray            # (3,) NED min corner of voxel (0,0,0)'s centre
    res: float
    shape: Tuple[int, int, int]
    clear: np.ndarray         # (nx,ny,nz) clearance at voxel centres

    def centre(self, idx) -> np.ndarray:
        return self.lo + np.asarray(idx, dtype=np.float64) * self.res

    def index(self, p) -> Tuple[int, int, int]:
        i = np.rint((np.asarray(p, dtype=np.float64) - self.lo) / self.res).astype(int)
        return tuple(int(v) for v in np.clip(i, 0, np.asarray(self.shape) - 1))


def build_grid(clearance: Clearance, lo, hi, res: float = 0.05) -> Grid:
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    shape = tuple(int(v) for v in np.floor((hi - lo) / res).astype(int) + 1)
    axes = [lo[k] + res * np.arange(shape[k]) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return Grid(lo, res, shape, np.asarray(clearance(pts), np.float64).reshape(shape))


def plan_path(grid: Grid, start, goal, radius: float, comfort: float = 0.45,
              comfort_weight: float = 4.0) -> np.ndarray:
    """A* waypoints (N,3) from start to goal; raises NoPathError."""
    free = grid.clear >= radius
    s, g = grid.index(start), grid.index(goal)
    for name, c in (('start', s), ('goal', g)):
        if not free[c]:
            raise NoPathError(f'{name} {grid.centre(c).round(3)} has clearance '
                              f'{grid.clear[c]:.3f} m < radius {radius} m')
    shape = np.asarray(grid.shape)
    pen = 1.0 + comfort_weight * np.clip((comfort - grid.clear) / comfort, 0.0, 1.0)
    gc = np.asarray(g)

    def h(c):
        return float(np.linalg.norm(np.asarray(c) - gc))

    best = {s: 0.0}
    came = {}
    heap = [(h(s), 0.0, s)]
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == g:
            break
        if cost > best.get(cur, np.inf):
            continue
        nb = np.asarray(cur) + _NEIGHBOURS
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        for k in np.nonzero(ok)[0]:
            n = (int(nb[k, 0]), int(nb[k, 1]), int(nb[k, 2]))
            if not free[n]:
                continue
            nc = cost + _STEP_LEN[k] * 0.5 * (pen[cur] + pen[n])
            if nc < best.get(n, np.inf):
                best[n] = nc
                came[n] = cur
                heapq.heappush(heap, (nc + h(n), nc, n))
    else:
        raise NoPathError('no collision-free route between start and goal')
    chain = [g]
    while chain[-1] != s:
        chain.append(came[chain[-1]])
    return np.array([grid.centre(c) for c in reversed(chain)])


def _segment_clear(clearance: Clearance, a, b, radius: float, step: float = 0.02) -> bool:
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    pts = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
    return bool(np.all(clearance(pts) >= radius))


def shortcut(clearance: Clearance, pts: np.ndarray, radius: float) -> np.ndarray:
    """Greedy line-of-sight pruning: from each kept point jump to the farthest visible one."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_clear(clearance, pts[i], pts[j], radius):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def _bezier(a, p, b, n=24):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * p + t ** 2 * b


def round_corners(clearance: Clearance, pts: np.ndarray, radius: float,
                  fractions=(0.5, 0.35, 0.25, 0.15, 0.08)) -> np.ndarray:
    """Dense polyline: straight runs joined by the widest Bezier corners that clear ``radius``."""
    if len(pts) < 3:
        return pts
    dense = [pts[0]]
    for i in range(1, len(pts) - 1):
        p, prev, nxt = pts[i], pts[i - 1], pts[i + 1]
        chosen = None
        for f in fractions:                 # f <= 0.5 keeps each corner inside its own segments
            a = p + (prev - p) * f
            b = p + (nxt - p) * f
            curve = _bezier(a, p, b)
            if np.all(clearance(curve) >= radius):
                chosen = curve
                break
        if chosen is None:                  # no rounding clears: keep the sharp corner
            dense.append(p)
            continue
        dense.extend(chosen)
    dense.append(pts[-1])
    return np.array(dense)


def resample(pts: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform arc-length resampling -> (points, cumulative arc length)."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    pts = pts[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    grid_s = np.arange(0.0, s[-1], step)
    grid_s = np.append(grid_s, s[-1])
    out = np.stack([np.interp(grid_s, s, pts[:, k]) for k in range(3)], axis=1)
    return out, grid_s
