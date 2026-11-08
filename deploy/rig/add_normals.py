#!/usr/bin/env python3
"""Copy the world to a scratch dir with per-vertex normals added to the OBJ meshes.

gz-physics (dartsim + ODE collision, Gazebo Harmonic) segfaults in OdeMesh::fillArrays on OBJ meshes that have no normals,
which is what export_world.py writes. usage: add_normals.py SRC_WORLD_DIR DST_DIR"""
import glob
import os
import shutil
import sys

import numpy as np

src, dst = sys.argv[1:3]
os.makedirs(dst, exist_ok=True)
for f in glob.glob(os.path.join(src, '*')):
    if not f.endswith('.obj'):
        shutil.copy(f, dst)
        continue
    V, F = [], []
    for line in open(f):
        p = line.split()
        if p and p[0] == 'v':
            V.append([float(x) for x in p[1:4]])
        elif p and p[0] == 'f':
            F.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
    V, F = np.array(V), np.array(F)
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(os.path.join(dst, os.path.basename(f)), 'w') as o:
        o.writelines('v %.6f %.6f %.6f\n' % tuple(v) for v in V)
        o.writelines('vn %.5f %.5f %.5f\n' % tuple(x) for x in n)
        o.writelines('f %d//%d %d//%d %d//%d\n' % (t[0], t[0], t[1], t[1], t[2], t[2]) for t in F + 1)
