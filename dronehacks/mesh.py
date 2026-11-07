"""The DroneHacks course as a triangle mesh (numpy only).

The organizers ship the course as an OBJ (metres, Z-up ENU, origin at the floor centre). Walls, tunnel
and boulders are kept as the exact triangles and ray traced directly (``render.py``). Everything here is
NED: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu (a proper rotation, so winding is preserved).

The OBJ's object names (mesh0..mesh183) carry no meaning, so ``classify`` labels objects from geometry.
Rebuild ``assets/course.npz`` from a new OBJ/zip with ``python -m dronehacks.build_assets mesh <obj>``.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Group ids stored per triangle. Order is part of the npz format.
GROUPS: Tuple[str, ...] = ('floor', 'net', 'paint', 'wall', 'tunnel', 'boulder')
GROUP_ID: Dict[str, int] = {g: i for i, g in enumerate(GROUPS)}

#: Groups that are physical obstacles a depth camera sees by default. ``net``
#: (the 1 mm bounding shell of the drone net) and ``paint`` (floor markings
#: 1-3 mm proud of the floor) are omitted: paint is not geometry, and a stereo
#: camera does not return the net as an opaque surface.
DEFAULT_GROUPS: Tuple[str, ...] = ('floor', 'wall', 'tunnel', 'boulder')

_PAINT_MAX_Z = 0.0035       # floor markings top out at 3 mm
_FLAT_Z = 5.0e-4            # "lies on the floor plane"
_NET_SPAN = 6.0             # net/floor shells span the whole 6.096 m net
_WALL_TOP_MIN = 0.3         # anything taller than this that is left over is wall
_BOULDER_MIN_FACES = 1000
_BOULDER_MIN_HEIGHT = 0.3
_BOUNDS_EPS = 2.0e-3


@dataclass
class ObjObject:
    """One ``o``/``g`` block of the OBJ, in the file's own (ENU) frame."""
    name: str
    faces: np.ndarray          # (F,3) int, indices into the shared vertex array

    def bounds(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = vertices[np.unique(self.faces)]
        return pts.min(axis=0), pts.max(axis=0)


@dataclass
class CourseMesh:
    """A triangle soup in NED with one group label per triangle."""
    vertices: np.ndarray       # (V,3) float32, NED metres
    faces: np.ndarray          # (F,3) int32
    face_group: np.ndarray     # (F,) int8, index into GROUPS
    face_object: np.ndarray    # (F,) int32, source OBJ object index

    def select(self, groups: Optional[Iterable[str]] = None) -> 'CourseMesh':
        """Sub-mesh holding only ``groups`` (default: DEFAULT_GROUPS)."""
        groups = DEFAULT_GROUPS if groups is None else tuple(groups)
        for g in groups:
            if g not in GROUP_ID:
                raise ValueError(f'unknown group {g!r}; choose from {GROUPS}')
        keep = np.isin(self.face_group, [GROUP_ID[g] for g in groups])
        faces = self.faces[keep]
        used, inverse = np.unique(faces, return_inverse=True)
        return CourseMesh(
            vertices=self.vertices[used],
            faces=inverse.reshape(faces.shape).astype(np.int32),
            face_group=self.face_group[keep],
            face_object=self.face_object[keep],
        )

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def group_counts(self) -> Dict[str, int]:
        return {g: int((self.face_group == i).sum()) for g, i in GROUP_ID.items()}

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, vertices=self.vertices, faces=self.faces,
            face_group=self.face_group, face_object=self.face_object,
            groups=np.array(GROUPS))

    @classmethod
    def load(cls, path: str) -> 'CourseMesh':
        with np.load(path) as z:
            if tuple(str(g) for g in z['groups']) != GROUPS:
                raise ValueError(f'{path}: group table does not match GROUPS')
            return cls(vertices=z['vertices'].astype(np.float32),
                       faces=z['faces'].astype(np.int32),
                       face_group=z['face_group'].astype(np.int8),
                       face_object=z['face_object'].astype(np.int32))


# ── OBJ parsing ─────────────────────────────────────────────────────────────

def parse_obj(lines: Iterable[str]) -> Tuple[np.ndarray, List[ObjObject]]:
    """Vertices (V,3) and the non-empty objects. n-gons are fan-triangulated."""
    verts: List[List[float]] = []
    blocks: Dict[str, List[List[int]]] = {}
    order: List[str] = []
    cur: Optional[str] = None
    for line in lines:
        if line.startswith('v '):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith(('o ', 'g ')):
            cur = line.split(None, 1)[1].strip()
            if cur not in blocks:
                blocks[cur] = []
                order.append(cur)
        elif line.startswith('f '):
            if cur is None:
                cur = '_default'
                blocks[cur] = []
                order.append(cur)
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                blocks[cur].append([idx[0], idx[k], idx[k + 1]])
    objects = [ObjObject(n, np.asarray(blocks[n], dtype=np.int64))
               for n in order if blocks[n]]
    return np.asarray(verts, dtype=np.float64), objects


def read_obj(path: str) -> Tuple[np.ndarray, List[ObjObject]]:
    """Read an OBJ from disk or from a ``.zip`` holding exactly one OBJ."""
    if path.lower().endswith('.zip'):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.obj')]
            if len(names) != 1:
                raise ValueError(f'{path}: expected one .obj, found {names}')
            with z.open(names[0]) as f:
                return parse_obj(ln.decode('utf-8', 'replace') for ln in f)
    with open(path, encoding='utf-8', errors='replace') as f:
        return parse_obj(f)


# ── classification ──────────────────────────────────────────────────────────

def _inside(lo: np.ndarray, hi: np.ndarray, box: Tuple[np.ndarray, np.ndarray],
            eps: float = _BOUNDS_EPS, xy_only: bool = False) -> bool:
    k = slice(0, 2) if xy_only else slice(0, 3)
    return bool(np.all(lo[k] >= box[0][k] - eps) and np.all(hi[k] <= box[1][k] + eps))


def classify(vertices: np.ndarray, objects: Sequence[ObjObject]) -> List[str]:
    """One group name per object, from geometry alone (ENU, Z-up input).

    Order matters; each rule only sees what the earlier ones left:
      floor    flat slab spanning the net in x AND y at z<=0 (the floor faces)
      net      any other object spanning the net (side/ceiling shells)
      boulder  tall compact dense mesh, plus its coplanar base disc
      tunnel   the tallest structure (> 1.8 m) and everything inside its XY footprint
      paint    <= 3.5 mm tall markings on the floor
      wall     everything else that rises above 0.3 m (walls, window faces)
    """
    n = len(objects)
    bounds = [o.bounds(vertices) for o in objects]
    labels: List[Optional[str]] = [None] * n

    for i, (lo, hi) in enumerate(bounds):
        ext = hi - lo
        if max(ext[0], ext[1]) > _NET_SPAN or ext[2] > 2.9:
            slab = min(ext[0], ext[1]) > _NET_SPAN and hi[2] <= _FLAT_Z
            labels[i] = 'floor' if slab else 'net'

    boulder_boxes = []
    for i, (lo, hi) in enumerate(bounds):
        ext = hi - lo
        if (labels[i] is None and len(objects[i].faces) > _BOULDER_MIN_FACES
                and ext[2] > _BOULDER_MIN_HEIGHT and max(ext[0], ext[1]) < 0.6):
            labels[i] = 'boulder'
            boulder_boxes.append((lo, hi))
    for i, (lo, hi) in enumerate(bounds):
        if labels[i] is None and hi[2] - lo[2] < _FLAT_Z and hi[2] <= _FLAT_Z:
            if any(_inside(lo, hi, b) for b in boulder_boxes):
                labels[i] = 'boulder'

    tall = [i for i, (lo, hi) in enumerate(bounds)
            if labels[i] is None and hi[2] > 1.8]
    tunnel_boxes = [bounds[i] for i in tall]
    for i in tall:
        labels[i] = 'tunnel'
    for i, (lo, hi) in enumerate(bounds):
        if labels[i] is None and any(_inside(lo, hi, b, xy_only=True)
                                     for b in tunnel_boxes):
            labels[i] = 'tunnel'

    for i, (lo, hi) in enumerate(bounds):
        if labels[i] is None and hi[2] <= _PAINT_MAX_Z:
            labels[i] = 'paint'
    for i, (lo, hi) in enumerate(bounds):
        if labels[i] is None:
            if hi[2] < _WALL_TOP_MIN:
                raise ValueError(
                    f'{objects[i].name}: unclassifiable (z {lo[2]:.4f}..{hi[2]:.4f}, '
                    f'xy {lo[:2].round(3)}..{hi[:2].round(3)})')
            labels[i] = 'wall'
    return labels  # type: ignore[return-value]


def enu_to_ned(points: np.ndarray) -> np.ndarray:
    """(…,3) ENU -> NED. x_n = y_e, y_n = x_e, z_n = -z_e."""
    p = np.asarray(points)
    return np.stack([p[..., 1], p[..., 0], -p[..., 2]], axis=-1)


def build_course(vertices: np.ndarray, objects: Sequence[ObjObject]) -> CourseMesh:
    """Classify + convert an parsed OBJ into a NED CourseMesh."""
    labels = classify(vertices, objects)
    faces = np.concatenate([o.faces for o in objects]).astype(np.int32)
    group = np.concatenate([np.full(len(o.faces), GROUP_ID[lab], dtype=np.int8)
                            for o, lab in zip(objects, labels)])
    obj_idx = np.concatenate([np.full(len(o.faces), i, dtype=np.int32)
                              for i, o in enumerate(objects)])
    return CourseMesh(vertices=enu_to_ned(vertices).astype(np.float32),
                      faces=faces, face_group=group, face_object=obj_idx)


def load_course(path: str) -> CourseMesh:
    """``.npz`` (cached CourseMesh) or ``.obj``/``.zip`` (parsed + classified)."""
    if path.lower().endswith('.npz'):
        return CourseMesh.load(path)
    return build_course(*read_obj(path))
