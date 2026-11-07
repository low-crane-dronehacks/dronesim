"""Exact triangle-mesh queries on the course (NVIDIA Warp BVH): depth images, obstacle distance, rays.

One static mesh is shared by every env (one BVH, E poses). Camera convention (NED): the camera looks along
body +x with image u -> body +y and v -> body +z; pixel centres are sampled; depth = clamp(t, near, far) /
far, a miss reads exactly 1.0 and hits closer than ``near`` are skipped (the ray restarts at ``near``).

``depth_type``: 'range' = distance along the ray; 'z' = camera-plane depth t * bx / |(bx, by, bz)|, which is
what a stereo camera (OAK-D) actually reports. The env uses 'z'.

Output buffers are persistent and reused: clone a result if you need to keep it past the next call.
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from dronehacks.mesh import CourseMesh

NO_OBSTACLE_DISTANCE = 1.0e9        # min_obstacle_distance sentinel (finite on purpose)
PIXEL_SAMPLE_OFFSET = 0.5           # sample pixel centres
DEPTH_NEAR_CLIP = 0.05
_DEPTH_TYPES = {'range': 0, 'z': 1}


@wp.kernel
def mesh_depth_kernel(
    mesh: wp.uint64,
    pos: wp.array2d(dtype=wp.float32),          # (E,3) camera origin, NED
    rpy: wp.array2d(dtype=wp.float32),          # (E,3) roll/pitch/yaw, rad
    width: int,
    height: int,
    fx: float,
    fy: float,
    near: float,
    far: float,
    pixel_offset: float,
    z_mode: int,
    out: wp.array3d(dtype=wp.float32),          # (E,H,W) normalised depth
):
    e, v, u = wp.tid()

    cu = (wp.float32(u) + pixel_offset) - 0.5 * wp.float32(width)
    cv = (wp.float32(v) + pixel_offset) - 0.5 * wp.float32(height)
    bx = 1.0
    by = cu / fx
    bz = cv / fy

    # body -> NED, ZYX
    c_r = wp.cos(rpy[e, 0])
    s_r = wp.sin(rpy[e, 0])
    c_p = wp.cos(rpy[e, 1])
    s_p = wp.sin(rpy[e, 1])
    c_y = wp.cos(rpy[e, 2])
    s_y = wp.sin(rpy[e, 2])

    dx = (c_y * c_p) * bx + (c_y * s_p * s_r - s_y * c_r) * by + (c_y * s_p * c_r + s_y * s_r) * bz
    dy = (s_y * c_p) * bx + (s_y * s_p * s_r + c_y * c_r) * by + (s_y * s_p * c_r - c_y * s_r) * bz
    dz = (-s_p) * bx + (c_p * s_r) * by + (c_p * c_r) * bz

    body_len = wp.sqrt(bx * bx + by * by + bz * bz)
    inv_len = 1.0 / wp.sqrt(dx * dx + dy * dy + dz * dz)
    d = wp.vec3(dx * inv_len, dy * inv_len, dz * inv_len)

    # restart at `near`: hits closer than the near plane are skipped, not clamped
    o = wp.vec3(pos[e, 0], pos[e, 1], pos[e, 2]) + d * near
    t = far
    q = wp.mesh_query_ray(mesh, o, d, far - near)
    if q.result:
        t = wp.clamp(near + q.t, near, far)
        if z_mode == 1:
            t = t * (bx / body_len)    # range -> camera-plane depth (hits only)
    out[e, v, u] = t / far             # a miss is exactly 1.0 in either mode


@wp.kernel
def mesh_distance_kernel(
    mesh: wp.uint64,
    pos: wp.array2d(dtype=wp.float32),
    sentinel: float,
    out: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    p = wp.vec3(pos[e, 0], pos[e, 1], pos[e, 2])
    best = sentinel
    q = wp.mesh_query_point_no_sign(mesh, p, sentinel)
    if q.result:
        best = wp.length(p - wp.mesh_eval_position(mesh, q.face, q.u, q.v))
    out[e] = best


@wp.kernel
def mesh_rays_kernel(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    far: float,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, origins[i], dirs[i], far)
    r = far
    if q.result:
        r = q.t
    out[i] = r


class MeshScene:
    """Depth images, obstacle distance and raw rays over one static mesh, batched over E poses."""

    def __init__(self, course: CourseMesh, num_envs: int, width: int = 2, height: int = 2,
                 fx: float = 1.0, fy: float = 1.0, far_clip: float = 10.0,
                 near_clip: float = DEPTH_NEAR_CLIP, depth_type: str = 'range', device: str = 'cuda:0'):
        if depth_type not in _DEPTH_TYPES:
            raise ValueError(f'depth_type must be one of {sorted(_DEPTH_TYPES)}')
        if far_clip <= near_clip:
            raise ValueError(f'far_clip ({far_clip}) must exceed near_clip ({near_clip})')
        if len(course.faces) == 0:
            raise ValueError('course has no triangles to trace')
        self.device = str(device)
        self.num_envs, self.width, self.height = int(num_envs), int(width), int(height)
        self.fx, self.fy = float(fx), float(fy)
        self.near_clip, self.far_clip = float(near_clip), float(far_clip)
        self.depth_type = depth_type

        self._points = wp.array(np.ascontiguousarray(course.vertices, np.float32), dtype=wp.vec3, device=self.device)
        self._indices = wp.array(np.ascontiguousarray(course.faces, np.int32).ravel(), dtype=wp.int32,
                                 device=self.device)
        self._mesh = wp.Mesh(points=self._points, indices=self._indices, support_winding_number=False)
        self._depth = wp.zeros((self.num_envs, self.height, self.width), dtype=wp.float32, device=self.device)
        self._distance = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)
        self._depth_view = wp.to_torch(self._depth)
        self._distance_view = wp.to_torch(self._distance)

    def render_depth(self, pos_ned, rpy_ned) -> torch.Tensor:
        """(E,3), (E,3) -> (E,H,W) normalised depth (persistent buffer)."""
        pos, rpy = self._as_warp(pos_ned, 'pos_ned'), self._as_warp(rpy_ned, 'rpy_ned')
        wp.launch(mesh_depth_kernel, dim=(self.num_envs, self.height, self.width),
                  inputs=[self._mesh.id, pos, rpy, self.width, self.height, self.fx, self.fy, self.near_clip,
                          self.far_clip, PIXEL_SAMPLE_OFFSET, _DEPTH_TYPES[self.depth_type]],
                  outputs=[self._depth], device=self.device)
        return self._depth_view

    def min_obstacle_distance(self, pos_ned) -> torch.Tensor:
        """(E,3) -> (E,) unsigned distance to the nearest triangle (persistent buffer: clone to keep it)."""
        pos = self._as_warp(pos_ned, 'pos_ned')
        wp.launch(mesh_distance_kernel, dim=self.num_envs, inputs=[self._mesh.id, pos, float(NO_OBSTACLE_DISTANCE)],
                  outputs=[self._distance], device=self.device)
        return self._distance_view

    def distance_at(self, points) -> np.ndarray:
        """(N,3) NED points -> (N,) numpy distance to the nearest triangle (any N; for planning)."""
        pts = np.ascontiguousarray(np.asarray(points, np.float32).reshape(-1, 3))
        out = wp.zeros(len(pts), dtype=wp.float32, device=self.device)
        wp.launch(mesh_distance_kernel, dim=len(pts),
                  inputs=[self._mesh.id, wp.array(pts, dtype=wp.float32, device=self.device),
                          float(NO_OBSTACLE_DISTANCE)],
                  outputs=[out], device=self.device)
        return out.numpy()

    def cast_rays(self, origins: torch.Tensor, dirs: torch.Tensor, far: float) -> torch.Tensor:
        """(N,3) NED origins + (N,3) unit directions -> (N,) hit distance (``far`` on a miss). No near skip."""
        o = origins.to(dtype=torch.float32).contiguous()
        d = dirs.to(dtype=torch.float32).contiguous()
        out = wp.zeros(o.shape[0], dtype=wp.float32, device=self.device)
        wp.launch(mesh_rays_kernel, dim=o.shape[0],
                  inputs=[self._mesh.id, wp.from_torch(o, dtype=wp.vec3), wp.from_torch(d, dtype=wp.vec3), float(far)],
                  outputs=[out], device=self.device)
        return wp.to_torch(out)

    def _as_warp(self, x, name: str):
        if tuple(x.shape) != (self.num_envs, 3):
            raise ValueError(f'{name} must be ({self.num_envs},3), got {tuple(x.shape)}')
        if isinstance(x, torch.Tensor):
            return wp.from_torch(x.to(dtype=torch.float32).contiguous(), dtype=wp.float32)
        return wp.array(np.ascontiguousarray(x, np.float32), dtype=wp.float32, device=self.device)
