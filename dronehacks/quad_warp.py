"""Fused GPU path for ``TorchQuad.step_control`` / ``step_control_seq``: every 100 Hz substep of one control step
(Lee velocity controller -> rotor allocation -> motor lag -> rigid body) in ONE Warp launch, one thread per env.

The torch path launches a few hundred small kernels per substep, so 10 substeps were launch-overhead bound
(~60 ms per env.step at 768 envs). This is a line-by-line port of ``quad.py``'s torch code, op for op and in
the same order (including torch's ``scalar / tensor == reciprocal(tensor) * scalar``), so trajectories agree
with the torch path to float32 rounding (tests/test_fast_paths.py pins this over 100 control steps with
randomised plants, latency, the PX4 integral + tilt limit and the ground clamp).

Everything the torch path reads is read here at call time: per-env gains / motor tau / true airframe written by
``Realism.on_reset``, ``thrust_scale`` rewritten every step by ``Realism.update_thrust``, the ``vel_ki`` integral
state, ``grounded``. Nothing syncs with the CPU.
"""
from __future__ import annotations

import math

import torch
import warp as wp

wp.init()

_HALF_PI = math.pi / 2.0


@wp.func
def _cross(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@wp.func
def _euler_xyz(q: wp.vec4) -> wp.vec3:
    """quad.euler_xyz_from_quat (XYZW)."""
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = wp.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = float(0.0)
    if wp.abs(sin_pitch) >= 1.0:
        pitch = wp.sign(sin_pitch) * wp.float32(_HALF_PI)
    else:
        pitch = wp.asin(sin_pitch)
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = wp.atan2(sin_yaw, cos_yaw)
    return wp.vec3(roll, pitch, yaw)


@wp.func
def _quat_apply(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    xyz = wp.vec3(q[0], q[1], q[2])
    t = _cross(xyz, v) * 2.0
    return (v + q[3] * t) + _cross(xyz, t)


@wp.func
def _quat_mul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """quad.quat_mul (isaaclab's form)."""
    x1 = a[0]
    y1 = a[1]
    z1 = a[2]
    w1 = a[3]
    x2 = b[0]
    y2 = b[1]
    z2 = b[2]
    w2 = b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = (ww + yy) + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = (qq - ww) + (z1 - y1) * (y2 - z2)
    x = (qq - xx) + (x1 + w1) * (x2 + w2)
    y = (qq - yy) + (w1 - x1) * (y2 + z2)
    z = (qq - zz) + (z1 + y1) * (w2 - x2)
    return wp.vec4(x, y, z, w)


@wp.func
def _quat_mul_xyzw(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    """quad.quat_mul_xyzw."""
    ax = a[0]
    ay = a[1]
    az = a[2]
    aw = a[3]
    bx = b[0]
    by = b[1]
    bz = b[2]
    bw = b[3]
    return wp.vec4(((aw * bx + ax * bw) + ay * bz) - az * by,
                   ((aw * by - ax * bz) + ay * bw) + az * bx,
                   ((aw * bz + ax * by) - ay * bx) + az * bw,
                   ((aw * bw - ax * bx) - ay * by) - az * bz)


@wp.func
def _matrix_from_quat(q: wp.vec4) -> wp.mat33:
    """quad.matrix_from_quat (no normalisation; two_s = 2 / |q|^2 as torch computes it: reciprocal * 2)."""
    i = q[0]
    j = q[1]
    k = q[2]
    r = q[3]
    two_s = (1.0 / (((i * i + j * j) + k * k) + r * r)) * 2.0
    return wp.mat33(1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                    two_s * (i * j + k * r), 1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                    two_s * (i * k - j * r), two_s * (j * k + i * r), 1.0 - two_s * (i * i + j * j))


@wp.func
def _rot_from_quat_normalised(quat: wp.vec4) -> wp.mat33:
    """quad.quat_to_rotation_matrix."""
    n = wp.sqrt(((quat[0] * quat[0] + quat[1] * quat[1]) + quat[2] * quat[2]) + quat[3] * quat[3])
    n = wp.max(n, 1.0e-6)
    x = quat[0] / n
    y = quat[1] / n
    z = quat[2] / n
    w = quat[3] / n
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return wp.mat33(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
                    2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
                    2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy))


@wp.func
def _sqrt_pos(x: float) -> float:
    r = float(0.0)
    if x > 0.0:
        r = wp.sqrt(x)
    return r


@wp.func
def _quat_from_matrix(m: wp.mat33) -> wp.vec4:
    """quad.quat_from_matrix (best-conditioned candidate; NaN if it is not unit length)."""
    m00 = m[0, 0]
    m01 = m[0, 1]
    m02 = m[0, 2]
    m10 = m[1, 0]
    m11 = m[1, 1]
    m12 = m[1, 2]
    m20 = m[2, 0]
    m21 = m[2, 1]
    m22 = m[2, 2]
    a0 = _sqrt_pos(((1.0 + m00) + m11) + m22)
    a1 = _sqrt_pos(((1.0 + m00) - m11) - m22)
    a2 = _sqrt_pos(((1.0 - m00) + m11) - m22)
    a3 = _sqrt_pos(((1.0 - m00) - m11) + m22)
    # argmax, first index on ties (torch.argmax)
    best = int(0)
    bv = a0
    if a1 > bv:
        best = 1
        bv = a1
    if a2 > bv:
        best = 2
        bv = a2
    if a3 > bv:
        best = 3
        bv = a3
    c = wp.vec4(0.0)
    if best == 0:
        c = wp.vec4(m21 - m12, m02 - m20, m10 - m01, a0 * a0)
    elif best == 1:
        c = wp.vec4(a1 * a1, m10 + m01, m02 + m20, m21 - m12)
    elif best == 2:
        c = wp.vec4(m10 + m01, a2 * a2, m12 + m21, m02 - m20)
    else:
        c = wp.vec4(m20 + m02, m21 + m12, a3 * a3, m10 - m01)
    den = 2.0 * wp.max(bv, 0.1)
    qd = wp.vec4(c[0] / den, c[1] / den, c[2] / den, c[3] / den)
    nrm = wp.sqrt(((qd[0] * qd[0] + qd[1] * qd[1]) + qd[2] * qd[2]) + qd[3] * qd[3])
    if wp.abs(nrm - 1.0) > 2.0e-5:
        nan = wp.nan
        qd = wp.vec4(nan, nan, nan, nan)
    return qd


@wp.func
def _rpy_ned(quat: wp.vec4) -> wp.vec3:
    """TorchQuad.rpy_ned: R_ned = P R_world Q with the 0/+-1 maps folded in."""
    R = _rot_from_quat_normalised(quat)
    pitch = wp.asin(wp.clamp(R[2, 0], -1.0, 1.0))
    roll = wp.atan2(R[2, 1], R[2, 2])
    yaw = wp.atan2(R[0, 0], R[1, 0])
    return wp.vec3(roll, pitch, yaw)


@wp.kernel
def quad_substeps_kernel(
    n_sub: int,
    cmd: wp.array3d(dtype=wp.float32),           # (S,E,4) FLU command per substep; S == 1: held for all substeps
    pos_in: wp.array2d(dtype=wp.float32),        # (E,3) world
    quat_in: wp.array2d(dtype=wp.float32),       # (E,4) XYZW
    vel_in: wp.array2d(dtype=wp.float32),        # (E,3) world
    angv_in: wp.array2d(dtype=wp.float32),       # (E,3) body
    vint_in: wp.array2d(dtype=wp.float32),       # (E,3) velocity-loop integral
    applied: wp.array2d(dtype=wp.float32),       # (E,4) rotor thrust, updated in place
    motor_tau: wp.array2d(dtype=wp.float32),     # (E,1)
    # controller
    k_vel: wp.array2d(dtype=wp.float32),         # (E,3)
    k_rot: wp.array2d(dtype=wp.float32),
    k_angvel: wp.array2d(dtype=wp.float32),
    c_mass: wp.array(dtype=wp.float32),          # (E,)
    c_inertia: wp.array3d(dtype=wp.float32),     # (E,3,3)
    has_ki: int,
    vel_ki: wp.array2d(dtype=wp.float32),        # (E,3) if has_ki
    vel_i_limit: float,
    c_dt: float,
    has_tilt: int,
    tan_tilt: float,
    max_yaw_rate: float,
    # plant
    alloc: wp.array2d(dtype=wp.float32),         # (6,4)
    alloc_pinv: wp.array2d(dtype=wp.float32),    # (4,6)
    thrust_lo: float,
    thrust_hi: float,
    neg_dt: float,                               # -dt, as float32 (torch: (-dt) / tau)
    dt: float,
    nominal: int,                                # 1: the nominal-airframe branch of _physics_step
    mass: float,
    neg_cdrag_mass: float,                       # -c_drag * mass (python double product, then float32)
    c_drag: float,
    neg_c_ang: float,
    c_ang: float,
    inertia0: wp.vec3,
    has_tmass: int,
    true_mass: wp.array2d(dtype=wp.float32),     # (E,1)
    has_tinertia: int,
    true_inertia: wp.array2d(dtype=wp.float32),  # (E,3)
    has_dscale: int,
    drag_scale: wp.array2d(dtype=wp.float32),    # (E,1)
    has_tscale: int,
    thrust_scale: wp.array2d(dtype=wp.float32),  # (E,1)
    has_ground: int,
    grounded: wp.array(dtype=wp.bool),           # (E,)
    ground_alt: float,
    want_pose: int,
    # outputs
    pos_out: wp.array2d(dtype=wp.float32),
    quat_out: wp.array2d(dtype=wp.float32),
    vel_out: wp.array2d(dtype=wp.float32),
    angv_out: wp.array2d(dtype=wp.float32),
    vint_out: wp.array2d(dtype=wp.float32),
    sub_pos: wp.array3d(dtype=wp.float32),       # (S,E,3) NED position after every substep
    sub_pose: wp.array3d(dtype=wp.float32),      # (S,E,6) [pos_ned, rpy_ned] if want_pose
):
    e = wp.tid()
    p = wp.vec3(pos_in[e, 0], pos_in[e, 1], pos_in[e, 2])
    q = wp.vec4(quat_in[e, 0], quat_in[e, 1], quat_in[e, 2], quat_in[e, 3])
    v = wp.vec3(vel_in[e, 0], vel_in[e, 1], vel_in[e, 2])
    w = wp.vec3(angv_in[e, 0], angv_in[e, 1], angv_in[e, 2])
    vi = wp.vec3(vint_in[e, 0], vint_in[e, 1], vint_in[e, 2])
    th = wp.vec4(applied[e, 0], applied[e, 1], applied[e, 2], applied[e, 3])
    kv = wp.vec3(k_vel[e, 0], k_vel[e, 1], k_vel[e, 2])
    kr = wp.vec3(k_rot[e, 0], k_rot[e, 1], k_rot[e, 2])
    kw = wp.vec3(k_angvel[e, 0], k_angvel[e, 1], k_angvel[e, 2])
    cm = c_mass[e]
    alpha = 1.0 - wp.exp(neg_dt / motor_tau[e, 0])
    S = cmd.shape[0]

    for k in range(n_sub):
        ks = wp.min(k, S - 1)
        c0 = cmd[ks, e, 0]
        c1 = cmd[ks, e, 1]
        c2 = cmd[ks, e, 2]
        c3 = cmd[ks, e, 3]

        # ── LeeVelocityController.compute ──
        eul = _euler_xyz(q)
        yaw = eul[2]
        vq = wp.vec4(0.0, 0.0, wp.sin(yaw * 0.5), wp.cos(yaw * 0.5))
        err = _quat_apply(vq, wp.vec3(c0, c1, c2)) - v
        acc = wp.cw_mul(kv, err)
        if has_ki != 0:
            ki = wp.vec3(vel_ki[e, 0], vel_ki[e, 1], vel_ki[e, 2])
            vi = vi + err * c_dt
            for a in range(3):
                lim = vel_i_limit / wp.max(ki[a], 1.0e-6)
                vi[a] = wp.max(wp.min(vi[a], lim), -lim)
            acc = acc + wp.cw_mul(ki, vi)
        f = wp.vec3(acc[0] * cm, acc[1] * cm, (acc[2] - (-9.81)) * cm)
        if has_tilt != 0:
            h_max = wp.max(f[2], 0.0) * tan_tilt
            hn = wp.sqrt(f[0] * f[0] + f[1] * f[1])
            sc = wp.min(h_max / wp.max(hn, 1.0e-9), 1.0)
            f = wp.vec3(f[0] * sc, f[1] * sc, f[2])
        Rq = _matrix_from_quat(q)
        fz = (f[0] * Rq[0, 2] + f[1] * Rq[1, 2]) + f[2] * Rq[2, 2]

        roll = eul[0]
        pitch = eul[1]
        b3 = f / (wp.sqrt((f[0] * f[0] + f[1] * f[1]) + f[2] * f[2]) + 1.0e-12)
        tmp = wp.vec3(wp.cos(yaw), wp.sin(yaw), 0.0)
        b2 = _cross(b3, tmp)
        b2 = b2 / (wp.sqrt((b2[0] * b2[0] + b2[1] * b2[1]) + b2[2] * b2[2]) + 1.0e-12)
        b1 = _cross(b2, b3)
        Rd = wp.mat33(b1[0], b2[0], b3[0], b1[1], b2[1], b3[1], b1[2], b2[2], b3[2])
        qd = _quat_from_matrix(Rd)
        wd = wp.vec3(-wp.sin(pitch) * c3, (wp.sin(roll) * wp.cos(pitch)) * c3,
                     wp.clamp((wp.cos(roll) * wp.cos(pitch)) * c3, -max_yaw_rate, max_yaw_rate))
        nq = ((q[0] * q[0] + q[1] * q[1]) + q[2] * q[2]) + q[3] * q[3]
        nq = wp.max(nq, 1.0e-9)
        qinv = wp.vec4(-q[0] / nq, -q[1] / nq, -q[2] / nq, q[3] / nq)
        Re = _matrix_from_quat(_quat_mul(qinv, qd))
        rot_err = wp.vec3(0.5 * (-(Re[2, 1] - Re[1, 2])), 0.5 * (Re[2, 0] - Re[0, 2]), 0.5 * (-(Re[1, 0] - Re[0, 1])))
        Iw = wp.vec3((c_inertia[e, 0, 0] * w[0] + c_inertia[e, 0, 1] * w[1]) + c_inertia[e, 0, 2] * w[2],
                     (c_inertia[e, 1, 0] * w[0] + c_inertia[e, 1, 1] * w[1]) + c_inertia[e, 1, 2] * w[2],
                     (c_inertia[e, 2, 0] * w[0] + c_inertia[e, 2, 1] * w[1]) + c_inertia[e, 2, 2] * w[2])
        cor = _cross(w, Iw)
        tq = (-wp.cw_mul(kr, rot_err) - wp.cw_mul(kw, w - wd)) + cor
        wr0 = float(0.0)
        wr1 = float(0.0)

        # ── allocation + motor lag ──
        for i in range(4):
            s = ((((alloc_pinv[i, 0] * wr0 + alloc_pinv[i, 1] * wr1) + alloc_pinv[i, 2] * fz)
                  + alloc_pinv[i, 3] * tq[0]) + alloc_pinv[i, 4] * tq[1]) + alloc_pinv[i, 5] * tq[2]
            tc = wp.clamp(wp.max(s, 0.0), thrust_lo, thrust_hi)
            th[i] = th[i] + (tc - th[i]) * alpha

        # ── rigid body ──
        R = _rot_from_quat_normalised(q)
        thr = th
        m_e = mass
        I = inertia0
        cd = c_drag
        ca = c_ang
        if nominal == 0:
            if has_tscale != 0:
                thr = th * thrust_scale[e, 0]
            if has_tmass != 0:
                m_e = true_mass[e, 0]
            if has_tinertia != 0:
                I = wp.vec3(true_inertia[e, 0], true_inertia[e, 1], true_inertia[e, 2])
            if has_dscale != 0:
                cd = c_drag * drag_scale[e, 0]
                ca = c_ang * drag_scale[e, 0]
        wb = wp.vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, length=6, dtype=wp.float32)
        for j in range(6):
            wb[j] = ((thr[0] * alloc[j, 0] + thr[1] * alloc[j, 1]) + thr[2] * alloc[j, 2]) + thr[3] * alloc[j, 3]
        Fw = wp.vec3((R[0, 0] * wb[0] + R[0, 1] * wb[1]) + R[0, 2] * wb[2],
                     (R[1, 0] * wb[0] + R[1, 1] * wb[1]) + R[1, 2] * wb[2],
                     (R[2, 0] * wb[0] + R[2, 1] * wb[1]) + R[2, 2] * wb[2])
        g = wp.vec3(0.0, 0.0, -9.81)
        lin_acc = wp.vec3(0.0)
        ang_acc = wp.vec3(0.0)
        if nominal != 0:
            drag_f = v * neg_cdrag_mass
            lin_acc = (Fw / mass + drag_f / mass) + g
            Iw0 = wp.cw_mul(inertia0, w)
            torque = wp.vec3(wb[3], wb[4], wb[5]) + Iw0 * neg_c_ang
            ang_acc = wp.cw_div(torque - _cross(w, Iw0), inertia0)
        else:
            lin_acc = (Fw / m_e - v * cd) + g
            Iw1 = wp.cw_mul(I, w)
            torque = wp.vec3(wb[3], wb[4], wb[5]) - Iw1 * ca
            ang_acc = wp.cw_div(torque - _cross(w, Iw1), I)

        xy0x = p[0]
        xy0y = p[1]
        v = v + lin_acc * dt
        p = p + v * dt
        if has_ground != 0:
            if grounded[e] and p[2] <= ground_alt:
                p = wp.vec3(xy0x, xy0y, ground_alt)
                v = wp.vec3(0.0, 0.0, wp.max(v[2], 0.0))
        w = w + ang_acc * dt
        wn = wp.sqrt((w[0] * w[0] + w[1] * w[1]) + w[2] * w[2])
        theta = wn * dt
        axis = w / wp.max(wn, 1.0e-9)
        sh = wp.sin(0.5 * theta)
        dq = wp.vec4(axis[0] * sh, axis[1] * sh, axis[2] * sh, wp.cos(0.5 * theta))
        qn = _quat_mul_xyzw(q, dq)
        nn = wp.sqrt(((qn[0] * qn[0] + qn[1] * qn[1]) + qn[2] * qn[2]) + qn[3] * qn[3])
        q = qn / nn

        sub_pos[k, e, 0] = p[1]
        sub_pos[k, e, 1] = p[0]
        sub_pos[k, e, 2] = -p[2]
        if want_pose != 0:
            rpy = _rpy_ned(q)
            sub_pose[k, e, 0] = p[1]
            sub_pose[k, e, 1] = p[0]
            sub_pose[k, e, 2] = -p[2]
            sub_pose[k, e, 3] = rpy[0]
            sub_pose[k, e, 4] = rpy[1]
            sub_pose[k, e, 5] = rpy[2]

    for a in range(3):
        pos_out[e, a] = p[a]
        vel_out[e, a] = v[a]
        angv_out[e, a] = w[a]
        vint_out[e, a] = vi[a]
    for a in range(4):
        quat_out[e, a] = q[a]
        applied[e, a] = th[a]


def _f32(x, rows: int, cols: int, device) -> torch.Tensor:
    """Per-env tensor (or scalar) -> contiguous float32 (rows, cols)."""
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    return t.reshape(-1, cols).expand(rows, cols).contiguous() if t.numel() in (1, cols) else \
        t.reshape(rows, cols).contiguous()


def _wa(t: torch.Tensor):
    return wp.from_torch(t, dtype=wp.float32)


def _inertia0(quad):
    """The nominal inertia as a wp.vec3, read back from the GPU once (not every step: that is a host sync)."""
    key = quad._inertia.data_ptr()
    cached = getattr(quad, '_wp_inertia0', None)
    if cached is None or cached[0] != key:
        cached = quad._wp_inertia0 = (key, wp.vec3(*[float(x) for x in quad._inertia.tolist()]))
    return cached[1]


def supported(quad) -> bool:
    return quad.pos_w.is_cuda


def step(quad, cmd: torch.Tensor) -> None:
    """Advance ``quad`` by ``quad.decimation`` substeps. ``cmd`` (S,E,4) FLU commands, S in {1, decimation}."""
    E, dev, ctl = quad.num_envs, quad.pos_w.device, quad.controller
    n = quad.decimation
    cmd = cmd.to(torch.float32).contiguous()
    pos = quad.pos_w.contiguous()
    quat = quad.quat_w.contiguous()
    vel = quad.lin_vel_w.contiguous()
    angv = quad.ang_vel_b.contiguous()
    vint = ctl._vel_int.to(torch.float32).contiguous()
    if not quad.applied_thrust.is_contiguous():
        quad.applied_thrust = quad.applied_thrust.contiguous()
    want_pose = quad.trace is not None or quad.pose_trace is not None
    pos_o, quat_o, vel_o = torch.empty_like(pos), torch.empty_like(quat), torch.empty_like(vel)
    angv_o, vint_o = torch.empty_like(angv), torch.empty_like(vint)
    sub_pos = torch.empty(n, E, 3, device=dev, dtype=torch.float32)
    sub_pose = torch.empty(n if want_pose else 1, E, 6, device=dev, dtype=torch.float32)
    dummy = quad._wp_dummy if hasattr(quad, '_wp_dummy') else None
    if dummy is None:
        dummy = quad._wp_dummy = torch.zeros(1, 1, device=dev, dtype=torch.float32)
    tm, ti, ds, ts = quad.true_mass, quad.true_inertia, quad.drag_scale, quad.thrust_scale
    nominal = tm is None and ti is None and ds is None and ts is None
    gr = quad.grounded
    ki = ctl.vel_ki
    m = float(quad.mass)
    wp.launch(quad_substeps_kernel, dim=E, device=str(dev), inputs=[
        n, wp.from_torch(cmd, dtype=wp.float32), _wa(pos), _wa(quat), _wa(vel), _wa(angv), _wa(vint),
        _wa(quad.applied_thrust), _wa(_f32(quad.motor_tau, E, 1, dev)),
        _wa(_f32(ctl.K_vel, E, 3, dev)), _wa(_f32(ctl.K_rot, E, 3, dev)), _wa(_f32(ctl.K_angvel, E, 3, dev)),
        _wa(ctl.mass.to(torch.float32).contiguous()), _wa(ctl.inertia.to(torch.float32).contiguous()),
        int(ki is not None), _wa(_f32(ki, E, 3, dev)) if ki is not None else _wa(dummy),
        float(ctl.vel_i_limit), float(ctl.dt),
        int(ctl.max_tilt is not None), float(math.tan(ctl.max_tilt)) if ctl.max_tilt is not None else 0.0,
        float(ctl.max_yaw_rate),
        _wa(quad._alloc.contiguous()), _wa(quad._alloc_pinv.contiguous()),
        float(quad._thrust_range[0]), float(quad._thrust_range[1]), float(-quad.dt), float(quad.dt),
        int(nominal), m, float(-quad._c_drag * m), float(quad._c_drag), float(-quad._c_ang), float(quad._c_ang),
        _inertia0(quad),
        int(tm is not None), _wa(_f32(tm, E, 1, dev)) if tm is not None else _wa(dummy),
        int(ti is not None), _wa(_f32(ti, E, 3, dev)) if ti is not None else _wa(dummy),
        int(ds is not None), _wa(_f32(ds, E, 1, dev)) if ds is not None else _wa(dummy),
        int(ts is not None), _wa(_f32(ts, E, 1, dev)) if ts is not None else _wa(dummy),
        int(gr is not None), wp.from_torch(gr.contiguous(), dtype=wp.bool) if gr is not None else
        wp.from_torch(torch.zeros(1, dtype=torch.bool, device=dev), dtype=wp.bool),
        float(quad.ground_alt), int(want_pose),
    ], outputs=[_wa(pos_o), _wa(quat_o), _wa(vel_o), _wa(angv_o), _wa(vint_o),
                wp.from_torch(sub_pos, dtype=wp.float32), wp.from_torch(sub_pose, dtype=wp.float32)])
    quad.pos_w, quad.quat_w, quad.lin_vel_w, quad.ang_vel_b = pos_o, quat_o, vel_o, angv_o
    if ki is not None:
        ctl._vel_int = vint_o
    if quad.substep_pos is not None:
        quad.substep_pos.extend(sub_pos.unbind(0))
    if quad.trace is not None:
        quad.trace.extend(sub_pose.clone().unbind(0))
    if quad.pose_trace is not None:
        quad.pose_trace.extend(sub_pose.unbind(0))
