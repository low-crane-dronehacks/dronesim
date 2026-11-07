"""Batched quadrotor plant in pure torch: Lee velocity controller -> rotor allocation -> motor lag -> rigid body.

  action (E,4) in [-1,1] = [v_fwd, v_right, v_down, yaw_rate] in the yaw-aligned body frame (NED axes)
    -> Lee geometric velocity controller -> body wrench
    -> rotor allocation pseudo-inverse, floored at 0
    -> per-rotor first-order thrust lag (tau 25-35 ms, drawn per episode), clamped to THRUST_RANGE
    -> thrusts -> wrench + linear/angular aero drag
    -> rigid body: semi-implicit Euler at 100 Hz, quaternion exponential update, 10 substeps per action.

Controller gains are drawn per episode (mild domain randomisation).

Frames: state lives in a Z-UP world (x = east, y = north) with FLU body axes and XYZW quaternions, because
the controller was written for that frame. ``pos_ned`` / ``vel_ned`` / ``rpy_ned`` / ``rot_ned`` convert.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

GRAVITY = 9.81
MASS = 1.348                                 # kg
I_DIAG = (0.006401, 0.006743, 0.010085)               # kg m^2
FRONT_DEADCAT_Y = 0.1596 # rotor offset along y axis, m
FRONT_DEADCAT_X = 0.0921 # rotor offset along x axis, m
REAR_DEADCAT_X = 0.1384
REAR_DEADCAT_Y = 0.1351
ROTOR_POSITIONS_B = ((+FRONT_DEADCAT_X, -FRONT_DEADCAT_Y, 0.0), (-REAR_DEADCAT_X, -REAR_DEADCAT_Y, 0.0), (-REAR_DEADCAT_X, +REAR_DEADCAT_Y, 0.0), (+FRONT_DEADCAT_X, +FRONT_DEADCAT_Y, 0.0))
ROTOR_DIRECTIONS = (-1, 1, -1, 1)
TORQUE_TO_THRUST_RATIO = 0.0116
THRUST_RANGE = (0.15, 12.6)                  # N per rotor at BATT_NOM_V (4S nominal, DYS X2807 + Gemfan 7035-2)
MOTOR_TAU_RANGE = (0.025, 0.035)             # s at BATT_NOM_V
# 4S LiPo + PX4 voltage compensation model (BAT1_R_INTERNAL + MC_BAT_SCALE_EN / HoverThrustEstimate)
BATT_NOM_V = 14.8                            # 4S nominal reference voltage (V)
BATT_CAP_AH = 2.2                            # 4S pack capacity (Ah)
BATT_R_INT_RANGE = (0.018, 0.035)            # 4S pack + XT60/lead internal resistance (Ohm), drawn per episode
BATT_SOC_INIT_RANGE = (0.35, 1.0)            # initial State of Charge in [0, 1], drawn per episode
BATT_COMP_TAU = 0.8                          # PX4 filtered voltage / hover-thrust estimator lag (s)
BATT_IDLE_I = 5.5                            # Jetson + Pixhawk + OAK-D (1.5 A) + 4x motor no-load (4.0 A)
BATT_THRUST_I_COEFF = 1.05                   # A / N^1.5 per rotor (DYS X2807 1700KV + Gemfan 7035-2 on 4S)
C_DRAG = 0.25                                # linear drag, 1/s:  F = -C_DRAG * m * v
C_ANG = 0.05                                 # angular drag, 1/s: tau = -C_ANG * I * w
YAW_RATE_SCALE = math.radians(45.0)          # internal yaw command unit (see TorchQuad._yaw_gain)
LEE_MAX_YAW_RATE = math.radians(60.0)        # controller yaw-rate ceiling, rad/s
# Lee controller gains, sampled per env per reset
K_VEL_RANGE = ((2.5, 2.5, 1.5), (3.5, 3.5, 2.0))
K_ROT_RANGE = ((1.6, 1.6, 0.25), (1.85, 1.85, 0.4))
K_ANGVEL_RANGE = ((0.4, 0.4, 0.075), (0.5, 0.5, 0.09))


# ── quaternion math (XYZW) ────────────────────────────────────────────────────────────────────────────────

@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    mag_torch = abs(mag) * torch.ones_like(other)
    return torch.copysign(mag_torch, other)


@torch.jit.script
def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_x, q_y, q_z, q_w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, :3]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 3:4] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([x, y, z, w], dim=-1).view(shape)


@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1).view(shape)


@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 0] ** 2], dim=-1),
        torch.stack([q_abs[..., 1] ** 2, m10 + m01, m02 + m20, m21 - m12], dim=-1),
        torch.stack([m10 + m01, q_abs[..., 2] ** 2, m12 + m21, m02 - m20], dim=-1),
        torch.stack([m20 + m02, m21 + m12, q_abs[..., 3] ** 2, m10 - m01], dim=-1)], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    quat = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,))
    invalid = (quat.norm(p=2, dim=-1, keepdim=True) - 1.0).abs() > 2e-5
    return torch.where(invalid, torch.full_like(quat, float('nan')), quat)


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """(E,4) XYZW body->world quaternion -> (E,3,3)."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
                        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
                        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1).reshape(*q.shape[:-1], 3, 3)


def ned_to_world(v: torch.Tensor) -> torch.Tensor:
    """(...,3) NED <-> Z-up world (x east, y north). An involution: the same map converts back."""
    return torch.stack([v[..., 1], v[..., 0], -v[..., 2]], dim=-1)


world_to_ned = ned_to_world


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], dim=-1)


def yaw_ned_to_quat_w(yaw_ned: torch.Tensor) -> torch.Tensor:
    """XYZW world quaternion of a level body with NED heading ``yaw_ned`` (world yaw = pi/2 - yaw_ned)."""
    half = 0.5 * (math.pi / 2.0 - yaw_ned)
    z = torch.zeros_like(half)
    return torch.stack([z, z, torch.sin(half), torch.cos(half)], dim=-1)


def attitude_ned_to_quat_w(roll: torch.Tensor, pitch: torch.Tensor, yaw_ned: torch.Tensor) -> torch.Tensor:
    """XYZW world quaternion for NED roll/pitch/yaw (ZYX). Body is FLU, so NED pitch is -rotation about +y."""
    z = torch.zeros_like(roll)
    qx = torch.stack([torch.sin(0.5 * roll), z, z, torch.cos(0.5 * roll)], dim=-1)
    qy = torch.stack([z, torch.sin(-0.5 * pitch), z, torch.cos(-0.5 * pitch)], dim=-1)
    return quat_mul_xyzw(yaw_ned_to_quat_w(yaw_ned), quat_mul_xyzw(qy, qx))


# ── Lee geometric velocity controller ────────────────────────────────────────────────────────────────────

class LeeVelocityController:
    """Velocity setpoint (yaw-aligned FLU frame, + yaw rate) -> body wrench [Fx, Fy, Fz, Tx, Ty, Tz]."""

    def __init__(self, num_envs: int, device: str, mass: float, max_yaw_rate: float = LEE_MAX_YAW_RATE):
        kw = dict(device=device, dtype=torch.float32)
        self.device = device
        self.max_yaw_rate = float(max_yaw_rate)
        m = torch.full((num_envs, 1), float(mass), **kw)
        self.mass = (1.0 / torch.where(m > 0, 1.0 / m, torch.zeros_like(m))).sum(dim=1)        # (E,)
        inertia = torch.zeros(num_envs, 3, 3, **kw)
        inertia[:, 0, 0], inertia[:, 1, 1], inertia[:, 2, 2] = I_DIAG
        self.inertia = 0.5 * (inertia + inertia.transpose(-1, -2))
        self.gravity = torch.tensor((0.0, 0.0, -GRAVITY), **kw).expand(num_envs, -1)
        self._rot_buf = torch.zeros(num_envs, 3, 3, **kw)
        self.K_vel_range = torch.tensor(K_VEL_RANGE, **kw).repeat(num_envs, 1, 1)
        self.K_rot_range = torch.tensor(K_ROT_RANGE, **kw).repeat(num_envs, 1, 1)
        self.K_angvel_range = torch.tensor(K_ANGVEL_RANGE, **kw).repeat(num_envs, 1, 1)
        self.K_vel = self.K_vel_range.mean(dim=1)
        self.K_rot = self.K_rot_range.mean(dim=1)
        self.K_angvel = self.K_angvel_range.mean(dim=1)

    def reset_idx(self, ids: torch.Tensor) -> None:
        for gain, rng in ((self.K_vel, self.K_vel_range), (self.K_rot, self.K_rot_range),
                          (self.K_angvel, self.K_angvel_range)):
            lo, hi = rng[ids, 0], rng[ids, 1]
            gain[ids] = torch.rand(*lo.shape, device=self.device) * (hi - lo) + lo

    def compute(self, cmd: torch.Tensor, quat_w: torch.Tensor, ang_vel_b: torch.Tensor,
                lin_vel_w: torch.Tensor) -> torch.Tensor:
        wrench = torch.zeros(cmd.shape[0], 6, device=self.device)
        _, _, yaw = euler_xyz_from_quat(quat_w)
        vehicle_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        acc = self.K_vel * (quat_apply(vehicle_quat, cmd[:, :3]) - lin_vel_w)
        forces_w = (acc - self.gravity) * self.mass.view(-1, 1)
        wrench[:, 2] = torch.sum(forces_w * matrix_from_quat(quat_w)[:, :, 2], dim=1)

        roll, pitch, yaw = euler_xyz_from_quat(quat_w)
        # desired attitude: thrust along the force, heading = current yaw
        b3 = forces_w / (torch.norm(forces_w, dim=1, keepdim=True) + 1e-12)
        tmp = torch.zeros_like(forces_w)
        tmp[:, 0] = torch.cos(yaw)
        tmp[:, 1] = torch.sin(yaw)
        b2 = torch.cross(b3, tmp, dim=1)
        b2 = b2 / (torch.norm(b2, dim=1, keepdim=True) + 1e-12)
        b1 = torch.cross(b2, b3, dim=1)
        self._rot_buf[:, :, 0], self._rot_buf[:, :, 1], self._rot_buf[:, :, 2] = b1, b2, b3
        desired_quat = quat_from_matrix(self._rot_buf)
        # yaw-rate command -> body rates
        w_des = torch.zeros((cmd.shape[0], 3), device=self.device)
        w_des[:, 0] = -torch.sin(pitch) * cmd[:, 3]
        w_des[:, 1] = torch.sin(roll) * torch.cos(pitch) * cmd[:, 3]
        w_des[:, 2] = torch.cos(roll) * torch.cos(pitch) * cmd[:, 3]
        w_des[:, 2] = torch.clamp(w_des[:, 2], -self.max_yaw_rate, self.max_yaw_rate)
        # PD on SO(3) + Coriolis feedforward
        R_err = matrix_from_quat(quat_mul(quat_inv(quat_w), desired_quat))
        skew = R_err.transpose(-1, -2) - R_err
        rot_err = 0.5 * torch.stack([-skew[:, 1, 2], skew[:, 0, 2], -skew[:, 0, 1]], dim=1)
        coriolis = torch.cross(ang_vel_b, torch.bmm(self.inertia, ang_vel_b.unsqueeze(2)).squeeze(2), dim=1)
        wrench[:, 3:6] = -self.K_rot * rot_err - self.K_angvel * (ang_vel_b - w_des) + coriolis
        return wrench


# ── plant ────────────────────────────────────────────────────────────────────────────────────────────────

_P = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))      # Z-up world <-> NED axis maps for attitude:
_Q = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))     # R_ned = P R_world Q


class TorchQuad:
    """E independent quadrotors stepped in lockstep.

    Args:
        v_max: action scale [m/s] (forward, right, down) in the yaw-aligned body frame.
        yaw_rate_max: yaw-rate action scale [rad/s].
    """

    def __init__(self, num_envs: int, device: str = 'cuda:0', physics_dt: float = 0.01, decimation: int = 10,
                 v_max=(2.0, 1.0, 1.0), yaw_rate_max: float = LEE_MAX_YAW_RATE, mass: float = MASS) -> None:
        self.num_envs, self.device = int(num_envs), str(device)
        self.dt, self.decimation, self.mass = float(physics_dt), int(decimation), float(mass)
        kw = dict(device=self.device, dtype=torch.float32)
        e = self.num_envs
        self.pos_w = torch.zeros(e, 3, **kw)
        self.quat_w = torch.zeros(e, 4, **kw)
        self.quat_w[:, 3] = 1.0
        self.lin_vel_w = torch.zeros(e, 3, **kw)
        self.ang_vel_b = torch.zeros(e, 3, **kw)

        alloc = torch.zeros(6, 4, **kw)
        pos = torch.tensor(ROTOR_POSITIONS_B, **kw)
        alloc[2] = 1.0                                                       # Fz: unit lift per rotor
        alloc[3] = pos[:, 1]                                                 # Tx = +y_i
        alloc[4] = -pos[:, 0]                                                # Ty = -x_i
        alloc[5] = torch.tensor(ROTOR_DIRECTIONS, **kw) * TORQUE_TO_THRUST_RATIO
        self._alloc = alloc
        self._alloc_pinv = torch.linalg.pinv(alloc)
        self._hover = self._alloc_pinv[:, 2] * (self.mass * GRAVITY)   # (4,) exact deadcat hover thrust
        self.applied_thrust = self._hover.unsqueeze(0).repeat(e, 1)
        self.motor_tau = torch.empty(e, 1, **kw).uniform_(*MOTOR_TAU_RANGE)
        self.batt_soc = torch.empty(e, 1, **kw).uniform_(*BATT_SOC_INIT_RANGE)
        self.batt_r_int = torch.empty(e, 1, **kw).uniform_(*BATT_R_INT_RANGE)
        self._hover_i = BATT_IDLE_I + BATT_THRUST_I_COEFF * self._hover.clamp_min(0.0).pow(1.5).sum()
        v_oc = 14.0 + 2.2 * self.batt_soc + 0.6 * self.batt_soc.pow(3)
        self.batt_v_term = (v_oc - self._hover_i * self.batt_r_int).clamp_min(11.0)
        self.batt_v_est = self.batt_v_term.clone()
        self._inertia = torch.tensor(I_DIAG, **kw)
        self._g = torch.tensor([0.0, 0.0, -GRAVITY], **kw)
        self._P = torch.tensor(_P, **kw)
        self._Q = torch.tensor(_Q, **kw)
        self.controller = LeeVelocityController(e, self.device, self.mass)
        self.v_max = torch.tensor(v_max, **kw)
        self._yaw_gain = float(yaw_rate_max) / YAW_RATE_SCALE
        
        #: set to a list to collect (E,3) NED position after every 100 Hz substep (sub-step collision checks)
        self.substep_pos = None
        #: set to a list to record (E,6) [pos_ned, rpy_ned] after every substep (videos)
        self.trace = None
        #: ground reaction: envs flagged here cannot sink below ``ground_alt`` and lose downward velocity there
        self.grounded = None
        self.ground_alt = 0.08

    def reset_idx(self, env_ids: torch.Tensor, pos_ned: torch.Tensor, yaw_ned: torch.Tensor,
                  vel_ned: Optional[torch.Tensor] = None, roll: Optional[torch.Tensor] = None,
                  pitch: Optional[torch.Tensor] = None) -> None:
        """Place ``env_ids`` at hover thrust: level and at rest unless velocity / roll / pitch are given."""
        self.pos_w[env_ids] = ned_to_world(pos_ned)
        if roll is None and pitch is None:
            self.quat_w[env_ids] = yaw_ned_to_quat_w(yaw_ned)
        else:
            z = torch.zeros_like(yaw_ned)
            self.quat_w[env_ids] = attitude_ned_to_quat_w(z if roll is None else roll, z if pitch is None else pitch,
                                                          yaw_ned)
        self.lin_vel_w[env_ids] = 0.0 if vel_ned is None else ned_to_world(vel_ned)
        self.ang_vel_b[env_ids] = 0.0
        self.applied_thrust[env_ids] = self._hover
        n = len(env_ids)
        self.motor_tau[env_ids] = torch.empty(n, 1, device=self.device).uniform_(*MOTOR_TAU_RANGE)
        self.batt_soc[env_ids] = torch.empty(n, 1, device=self.device).uniform_(*BATT_SOC_INIT_RANGE)
        self.batt_r_int[env_ids] = torch.empty(n, 1, device=self.device).uniform_(*BATT_R_INT_RANGE)
        v_oc = 14.0 + 2.2 * self.batt_soc[env_ids] + 0.6 * self.batt_soc[env_ids].pow(3)
        v_hover = (v_oc - self._hover_i * self.batt_r_int[env_ids]).clamp_min(11.0)
        self.batt_v_term[env_ids] = v_hover
        self.batt_v_est[env_ids] = v_hover
        self.controller.reset_idx(env_ids)

    @property
    def pos_ned(self) -> torch.Tensor:
        return world_to_ned(self.pos_w)

    @property
    def vel_ned(self) -> torch.Tensor:
        return world_to_ned(self.lin_vel_w)

    def rot_ned(self) -> torch.Tensor:
        """(E,3,3) body (FRD) -> NED rotation."""
        return self._P @ quat_to_rotation_matrix(self.quat_w) @ self._Q

    def rpy_ned(self) -> torch.Tensor:
        r = self.rot_ned()
        pitch = torch.asin(torch.clamp(-r[:, 2, 0], -1.0, 1.0))
        roll = torch.atan2(r[:, 2, 1], r[:, 2, 2])
        yaw = torch.atan2(r[:, 1, 0], r[:, 0, 0])
        return torch.stack((roll, pitch, yaw), dim=-1)

    def step_control(self, action: torch.Tensor) -> None:
        """Hold one (E,4) action for ``decimation`` physics steps."""
        act = action.clone()
        act[:, 3] = act[:, 3] * self._yaw_gain
        scaled = act[:, :3] * self.v_max
        cmd = torch.stack([scaled[:, 0], -scaled[:, 1], -scaled[:, 2], -act[:, 3] * YAW_RATE_SCALE], dim=1)  # NED->FLU
        for _ in range(self.decimation):
            self._physics_step(cmd)

    def _physics_step(self, cmd: torch.Tensor) -> None:
        dt = self.dt
        wrench = self.controller.compute(cmd, self.quat_w, self.ang_vel_b, self.lin_vel_w)

        # 4S LiPo current draw, terminal voltage sag (V_oc - I * R_int), and PX4 voltage-compensation filter
        i_batt = BATT_IDLE_I + BATT_THRUST_I_COEFF * self.applied_thrust.clamp_min(0.0).pow(1.5).sum(dim=-1, keepdim=True)
        v_oc = 14.0 + 2.2 * self.batt_soc + 0.6 * self.batt_soc.pow(3)
        v_term = (v_oc - i_batt * self.batt_r_int).clamp_min(11.0)
        self.batt_v_term = v_term
        self.batt_soc = (self.batt_soc - i_batt * (dt / (3600.0 * BATT_CAP_AH))).clamp_min(0.0)
        self.batt_v_est += (v_term - self.batt_v_est) * (1.0 - math.exp(-dt / BATT_COMP_TAU))

        # PX4 scales thrust commands by filtered batt_v_est, so fast I*R_int sag causes (v_term / batt_v_est)^2 drop,
        # while 100% duty-cycle motor saturation is always hard-clamped by true terminal voltage (v_term / BATT_NOM_V)^2.
        sag_gain = (v_term / self.batt_v_est).square()
        max_thrust = THRUST_RANGE[1] * (v_term / BATT_NOM_V).square()
        raw_cmd = (wrench @ self._alloc_pinv.transpose(-2, -1)).clamp_min(0.0) * sag_gain
        thrust_cmd = torch.minimum(raw_cmd.clamp_min(THRUST_RANGE[0]), max_thrust)
        tau_eff = self.motor_tau * (BATT_NOM_V / v_term)
        alpha = 1.0 - torch.exp(-dt / tau_eff)
        self.applied_thrust += (thrust_cmd - self.applied_thrust) * alpha
        w_b = self.applied_thrust @ self._alloc.transpose(-2, -1)
        drag_f_w = -C_DRAG * self.mass * self.lin_vel_w
        drag_t_b = -C_ANG * (self._inertia * self.ang_vel_b)

        rot = quat_to_rotation_matrix(self.quat_w)                          # body -> world
        acc = (rot @ w_b[:, :3].unsqueeze(-1)).squeeze(-1) / self.mass + drag_f_w / self.mass + self._g
        torque = w_b[:, 3:] + drag_t_b
        w = self.ang_vel_b
        ang_acc = (torque - torch.cross(w, self._inertia * w, dim=-1)) / self._inertia

        xy0 = self.pos_w[:, :2]
        self.lin_vel_w = self.lin_vel_w + acc * dt
        self.pos_w = self.pos_w + self.lin_vel_w * dt
        if self.grounded is not None:
            # Resting on the ground: the floor stops it sinking AND friction stops it sliding. Without the
            # friction a grounded drone could skid along the floor below the touchdown height (a parked drone
            # sits at 0.08 m, contact counts at 0.05 m) and a policy learned to taxi from the pad into the
            # barrier instead of lifting off. Only climbing frees it.
            on = self.grounded & (self.pos_w[:, 2] <= self.ground_alt)
            self.pos_w[:, 2] = torch.where(on, torch.full_like(self.pos_w[:, 2], self.ground_alt), self.pos_w[:, 2])
            self.lin_vel_w[:, 2] = torch.where(on, self.lin_vel_w[:, 2].clamp(min=0.0), self.lin_vel_w[:, 2])
            self.pos_w[:, :2] = torch.where(on.unsqueeze(-1), xy0, self.pos_w[:, :2])
            self.lin_vel_w[:, :2] = torch.where(on.unsqueeze(-1), torch.zeros_like(xy0), self.lin_vel_w[:, :2])
        self.ang_vel_b = w + ang_acc * dt
        theta = torch.linalg.vector_norm(self.ang_vel_b, dim=-1, keepdim=True) * dt
        axis = self.ang_vel_b / torch.clamp(torch.linalg.vector_norm(self.ang_vel_b, dim=-1, keepdim=True), min=1e-9)
        dq = torch.cat([axis * torch.sin(0.5 * theta), torch.cos(0.5 * theta)], dim=-1)
        q = quat_mul_xyzw(self.quat_w, dq)
        self.quat_w = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        if self.substep_pos is not None:
            self.substep_pos.append(self.pos_ned)
        if self.trace is not None:
            self.trace.append(torch.cat([self.pos_ned, self.rpy_ned()], dim=-1).clone())
