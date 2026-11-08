"""Build an eval course from a real scan of the course (Scaniverse/LiDAR point cloud .ply, z up, metres).

    python -m dronehacks.scan_course course_scan.ply --out assets/course_scan.npz

1. register: 2-D wall cross-sections (0.3-1.5 m) of the scan against the CAD course: exhaustive rotation (+ mirror)
   x FFT-correlation translation, then ICP. Prints the residual and writes a top-view overlay PNG.
2. voxelise the registered points (default 5 cm; voxels with >= min_pts points, dilated once to close scan gaps)
   and mesh only the exposed cube faces -> group 'wall' (collision + depth). Points from 6 cm above the floor to
   max_height; the scan's floor is replaced by the CAD floor (LiDAR barely sees the middle of it).
3. keep the CAD floor, paint and net (the see-through net model needs the net mesh; the scan barely sees netting).
Gates, pad and launch mark stay the rules' (the scan is registered into that frame).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from dronehacks.mesh import GROUP_ID, CourseMesh


def read_ply_xyz(path):
    raw = open(path, 'rb').read()
    head, _, body = raw.partition(b'end_header\n')
    lines = head.decode('ascii', 'replace').splitlines()
    n = next(int(l.split()[2]) for l in lines if l.startswith('element vertex'))
    props = [l.split() for l in lines if l.startswith('property')]
    tmap = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint8': 'u1', 'int': '<i4', 'short': '<i2',
            'ushort': '<u2'}
    dt = np.dtype([(p[2], tmap[p[1]]) for p in props if p[1] != 'list'])
    a = np.frombuffer(body[:n * dt.itemsize], dt)
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)


def sample_mesh(m: CourseMesh, groups, step=0.05, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for g in groups:
        T = m.vertices[m.faces[m.face_group == GROUP_ID[g]]].astype(np.float64)
        a = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
        k = np.maximum(1, (a / step ** 2).astype(int))
        for t, kk in zip(T, k):
            u = rng.random((kk, 2))
            s = u.sum(1) > 1
            u[s] = 1 - u[s]
            pts.append(t[0] + u[:, :1] * (t[1] - t[0]) + u[:, 1:] * (t[2] - t[0]))
    return np.concatenate(pts)


def register_2d(scan_xy, cad_ne, res=0.05, crop=None):
    """scan (x, y) -> NED (n, e): returns A (2x2, det +-1), b (2,), ICP residuals."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    Q = scan_xy
    if crop is not None:
        Q = Q[crop(Q)]
    Q = Q[np.random.default_rng(0).choice(len(Q), min(60000, len(Q)), replace=False)]
    shape, lo = (256, 256), np.array([-6.4, -6.4])

    def grid(p):
        g = np.zeros(shape)
        ij = np.floor((p - lo) / res).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1)
        return gaussian_filter(np.minimum(g, 1), 1.5)

    Fc = np.fft.rfft2(grid(cad_ne))
    qc = Q.mean(0)
    best = None
    for mirror in (False, True):
        M = np.diag([1.0, -1.0]) if mirror else np.eye(2)
        for deg in np.arange(0, 360, 1.0):
            t = np.radians(deg)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            cc = np.fft.irfft2(Fc * np.conj(np.fft.rfft2(grid((Q - qc) @ (R @ M).T))), s=shape)
            k = np.unravel_index(cc.argmax(), shape)
            if best is None or cc.max() > best[0]:
                sh = np.array([(k[0] + 128) % 256 - 128, (k[1] + 128) % 256 - 128]) * res
                best = (cc.max(), R @ M, sh)
    _, RM, sh = best
    A, b = RM, sh - RM @ qc
    X, tree = Q @ A.T + b, cKDTree(cad_ne)
    for _ in range(40):                                   # ICP (rigid, keeps the handedness found above)
        d, i = tree.query(X)
        k = d < max(np.percentile(d, 60), 0.03)
        P_, C_ = X[k], cad_ne[i[k]]
        mp, mc = P_.mean(0), C_.mean(0)
        U, _, Vt = np.linalg.svd((P_ - mp).T @ (C_ - mc))
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[1] *= -1
            Ri = Vt.T @ U.T
        ti = mc - Ri @ mp
        X = X @ Ri.T + ti
        A, b = Ri @ A, Ri @ b + ti
    # second pass: only scan points that land inside the arena (the room outside has no CAD counterpart)
    Qall = scan_xy
    for _ in range(30):
        X = Qall @ A.T + b
        Xi = X[(np.abs(X) < 3.4).all(1)]
        d, i = tree.query(Xi)
        k = d < max(np.percentile(d, 70), 0.03)
        P_, C_ = Xi[k], cad_ne[i[k]]
        mp, mc = P_.mean(0), C_.mean(0)
        U, _, Vt = np.linalg.svd((P_ - mp).T @ (C_ - mc))
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[1] *= -1
            Ri = Vt.T @ U.T
        ti = mc - Ri @ mp
        A, b = Ri @ A, Ri @ b + ti
    X = Qall @ A.T + b
    d, _ = tree.query(X[(np.abs(X) < 3.4).all(1)])
    return A, b, d


def level_floor(P, cell=0.5, z_max=0.4):
    """Fit the floor plane (robust: lowest dense layer per cell, then a least-squares plane with outlier rejection)
    and return points with z = height above that plane."""
    low = P[P[:, 2] < z_max]
    ij = np.floor(low[:, :2] / cell).astype(int)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.argsort(key)
    key, low = key[order], low[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1, len(key)]
    samples = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a >= 30:
            zz = low[a:b, 2]
            samples.append((*low[a:b, :2].mean(0), np.percentile(zz, 20)))
    S = np.array(samples)
    keep = np.ones(len(S), bool)
    for _ in range(5):
        M = np.c_[S[keep, :2], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(M, S[keep, 2], rcond=None)
        r = S[:, 2] - np.c_[S[:, :2], np.ones(len(S))] @ coef
        keep = np.abs(r) < max(0.02, 2.5 * np.std(r[keep]))
    out = P.copy()
    out[:, 2] = P[:, 2] - np.c_[P[:, :2], np.ones(len(P))] @ coef
    tilt = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
    print(f'floor plane: {len(S)} cells, tilt {tilt:.2f} deg, offset {coef[2]:+.3f} m, '
          f'residual {np.std(r[keep]) * 100:.1f} cm')
    return out


def voxel_mesh(points_ned, voxel=0.05, min_pts=2, dilate=0):
    """Exposed faces of occupied voxels -> (vertices, faces)."""
    ijk = np.floor(points_ned / voxel).astype(np.int64)
    u, cnt = np.unique(ijk, axis=0, return_counts=True)
    occ = u[cnt >= min_pts]
    if dilate:
        offs = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                         if abs(dx) + abs(dy) + abs(dz) == 1] + [(0, 0, 0)])
        occ = np.unique((occ[:, None, :] + offs[None]).reshape(-1, 3), axis=0)
    key = lambda a: (a[:, 0] * 1_000_003 + a[:, 1]) * 1_000_003 + a[:, 2]      # noqa: E731
    have = set(key(occ).tolist())
    # 6 directions: (normal axis, sign) -> the 4 corners of that face in unit-cube coordinates (outward winding)
    faces_def = {
        (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        (1, 1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)], (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]}
    V, F = [], []
    nv = 0
    for (ax, sgn), corners in faces_def.items():
        nb = occ.copy()
        nb[:, ax] += sgn
        exposed = occ[~np.isin(key(nb), np.fromiter(have, np.int64))]
        if not len(exposed):
            continue
        c = np.asarray(corners, np.float64)
        quad = (exposed[:, None, :] + c[None]) * voxel                   # (n,4,3)
        V.append(quad.reshape(-1, 3))
        base = nv + 4 * np.arange(len(exposed))[:, None]
        F.append(np.concatenate([base + [0, 1, 2], base + [0, 2, 3]]))
        nv += 4 * len(exposed)
    return np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int32), len(occ)


def build(ply, cad_path, out, voxel=0.05, min_pts=2, max_height=4.0, extent=6.5, png=None, floor_cut=0.12):
    cad = CourseMesh.load(cad_path)
    P = level_floor(read_ply_xyz(ply))
    wall = sample_mesh(cad, ('wall', 'tunnel', 'boulder', 'net'))
    wall = wall[(wall[:, 2] < -0.3) & (wall[:, 2] > -1.5)]
    mid = (P[:, 2] > 0.3) & (P[:, 2] < 1.5)
    A, b, d = register_2d(P[mid, :2], wall[:, :2])
    print(f'registration: median {np.median(d):.3f} m, {100 * (d < 0.05).mean():.0f}% within 5 cm, '
          f'{100 * (d < 0.15).mean():.0f}% within 15 cm (residuals include structure the CAD lacks)')
    ne = P[:, :2] @ A.T + b
    ned = np.column_stack([ne, -P[:, 2]])
    keep = (P[:, 2] > floor_cut) & (P[:, 2] < max_height) & (np.abs(ne) < extent).all(1)
    v, f, n_vox = voxel_mesh(ned[keep], voxel, min_pts)
    base = cad.select(('floor', 'paint', 'net'))
    mesh = CourseMesh(vertices=np.concatenate([base.vertices, v]),
                      faces=np.concatenate([base.faces, f + len(base.vertices)]),
                      face_group=np.concatenate([base.face_group, np.full(len(f), GROUP_ID['wall'], np.int8)]),
                      face_object=np.concatenate([base.face_object, np.full(len(f), -1, np.int32)]))
    mesh.save(out)
    print(f'{out}: {n_vox} voxels ({voxel * 100:.0f} cm) -> {len(f)} wall triangles, + CAD floor/paint/net')
    np.savez(os.path.splitext(out)[0] + '_registration.npz', A=A, b=b, residual_median=np.median(d))
    if png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from dronehacks import rules as cs
        fig, ax = plt.subplots(figsize=(9, 9))
        q = ned[mid]
        ax.scatter(q[:, 1], q[:, 0], s=0.3, c='tab:orange', alpha=0.4, label='scan 0.3-1.5 m')
        ax.scatter(wall[:, 1], wall[:, 0], s=0.3, c='k', alpha=0.5, label='CAD walls')
        for g in cs.GATES:
            ax.plot([g['A'][1], g['B'][1]], [g['A'][0], g['B'][0]], 'r-', lw=2)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_aspect('equal'); ax.grid(alpha=0.3)
        ax.legend(markerscale=20); ax.set_xlabel('east (m)'); ax.set_ylabel('north (m)')
        fig.savefig(png, dpi=80)
    return mesh


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ply')
    ap.add_argument('--cad', default=os.path.join(os.path.dirname(__file__), '..', 'assets', 'course.npz'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'assets', 'course_scan.npz'))
    ap.add_argument('--voxel', type=float, default=0.05)
    ap.add_argument('--min-pts', type=int, default=2)
    ap.add_argument('--png', default=None)
    a = ap.parse_args(argv)
    build(a.ply, a.cad, a.out, a.voxel, a.min_pts, png=a.png)


if __name__ == '__main__':
    main()
