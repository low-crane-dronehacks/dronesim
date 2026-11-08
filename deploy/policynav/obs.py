"""Observation / action plumbing between ROS-PX4 data and the trained policy. PURE: numpy only, no ROS.

Reproduces what the policy saw in training (dronehacks/env.py):
  depth    2-frame stack, newest first, of z-depth / far_clip on a 128x80 pinhole camera (72 x 50 deg) looking
           along body +x (image right = body right, image down = body down). The real camera's depth is
           re-projected into that virtual camera through its CameraInfo + TF, block-averaged over valid returns;
           a policy pixel with < min_valid of its source pixels valid is INVALID and reads as near (0.0): an
           invalid pixel must never look like open space (the camera is inside its 0.35 m minimum range on the
           ground at takeoff).
  proprio  [body-frame velocity / v_max (3), downward range / 1.5 (1), roll, pitch (2), previous action (A)]
  action   [forward, right, down, yaw_rate] in [-1, 1] in the yaw-aligned body frame, x v_max / yaw_rate_max
           (+ a 5th 'complete' output for handoff policies) -> PX4 NED velocity + yawspeed setpoint.
Frames: PX4 NED world, FRD body; q is PX4's vehicle_attitude (w, x, y, z), FRD body -> NED.
"""
from __future__ import annotations

import math

import numpy as np


def quat_to_mat(w, x, y, z) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_from_R(R: np.ndarray):
    """NED roll, pitch, yaw (ZYX) of an FRD->NED rotation (same as env.py's TorchQuad.rpy_ned)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


class DepthReprojector:
    """Maps the real depth image onto the training camera's grid. Built once per (CameraInfo, TF)."""

    def __init__(self, K, src_hw, R_body_cam, t_body_cam, out_hw=(80, 128), hfov_deg=72.0, vfov_deg=50.0,
                 far=5.0, min_valid=0.5, stride=1):
        """K = (fx, fy, cx, cy) of the source image; R_body_cam / t_body_cam: optical frame (x right, y down,
        z forward) -> body FLU, as TF base_link <- camera gives them."""
        fx, fy, cx, cy = K
        Hs, Ws = src_hw
        H, W = out_hw
        self.H, self.W, self.far, self.min_valid, self.stride = H, W, far, min_valid, stride
        vs, us = np.mgrid[0:Hs:stride, 0:Ws:stride]
        rays_cam = np.stack([(us + 0.5 - cx) / fx, (vs + 0.5 - cy) / fy, np.ones(us.shape)], -1).reshape(-1, 3)
        flu_to_frd = np.diag([1.0, -1.0, -1.0])
        self.R = flu_to_frd @ np.asarray(R_body_cam, float)        # optical -> body FRD
        self.t = flu_to_frd @ np.asarray(t_body_cam, float)
        self.rays = rays_cam @ self.R.T                            # point(z) = t + z * ray, body FRD
        self.fxp = 0.5 * W / math.tan(math.radians(hfov_deg) / 2.0)
        self.fyp = 0.5 * H / math.tan(math.radians(vfov_deg) / 2.0)
        self.axis_err_deg = math.degrees(math.acos(max(-1.0, min(1.0, self.R[0, 2]))))   # optical axis vs body x

    def __call__(self, depth_m: np.ndarray) -> np.ndarray:
        """(Hs, Ws) metric depth (NaN / <= 0 = no return) -> (H, W) normalised training-camera depth."""
        z = np.asarray(depth_m, np.float32)[::self.stride, ::self.stride].reshape(-1)
        ok = np.isfinite(z) & (z > 0)
        p = self.t + self.rays * np.where(ok, z, 1.0)[:, None]                         # body FRD points
        # pixel in the virtual camera through which each source point (or, if invalid, its ray) is seen
        dirs = np.where(ok[:, None], p, self.rays)
        fwd = dirs[:, 0]
        vis = fwd > 1e-6
        u = np.floor(dirs[:, 1] / np.where(vis, fwd, 1.0) * self.fxp + 0.5 * self.W).astype(np.int64)
        v = np.floor(dirs[:, 2] / np.where(vis, fwd, 1.0) * self.fyp + 0.5 * self.H).astype(np.int64)
        inside = vis & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        idx = (v * self.W + u)[inside]
        n_all = np.bincount(idx, minlength=self.H * self.W)
        okin = ok[inside]
        n_ok = np.bincount(idx[okin], minlength=self.H * self.W)
        zsum = np.bincount(idx[okin], weights=np.minimum(p[inside][okin, 0], self.far), minlength=self.H * self.W)
        mean = zsum / np.maximum(n_ok, 1)
        valid = (n_ok >= self.min_valid * np.maximum(n_all, 1)) & (n_ok > 0)
        out = np.where(valid, np.clip(mean / self.far, 0.0, 1.0), 0.0)                # invalid -> near
        return out.reshape(self.H, self.W).astype(np.float32)


class FrameStack:
    def __init__(self, n: int = 2):
        self.n, self.frames = n, None

    def reset(self):
        self.frames = None

    def push(self, img: np.ndarray) -> np.ndarray:
        """-> (n, H, W), newest first. A fresh stack is filled with its first frame (as in training)."""
        if self.frames is None:
            self.frames = [img] * self.n
        else:
            self.frames = [img] + self.frames[:-1]
        return np.stack(self.frames)


def proprio(vel_ned, q_wxyz, range_m, prev_action, v_max) -> np.ndarray:
    R = quat_to_mat(*q_wxyz)
    v_body = R.T @ np.asarray(vel_ned, float)
    roll, pitch, _ = rpy_from_R(R)
    return np.concatenate([v_body / np.asarray(v_max, float), [range_m / 1.5], [roll, pitch],
                           np.asarray(prev_action, float)]).astype(np.float32)


def action_to_setpoint(action, q_wxyz, v_max, yaw_rate_max):
    """-> (vel_ned (3,), yawspeed rad/s) for a PX4 TrajectorySetpoint. Velocity is commanded in the yaw-aligned
    frame (heading only, roll/pitch removed), exactly as the training controller interpreted it."""
    a = np.clip(np.asarray(action, float), -1.0, 1.0)
    _, _, yaw = rpy_from_R(quat_to_mat(*q_wxyz))
    f, r, d = a[0] * v_max[0], a[1] * v_max[1], a[2] * v_max[2]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * f - s * r, s * f + c * r, d]), float(a[3] * yaw_rate_max)
