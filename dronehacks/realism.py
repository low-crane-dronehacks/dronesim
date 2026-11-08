"""Real-world effects for evaluation (and, in their cheap form, training). Everything is OFF by default.

Each effect is an independent switch so a policy can be stress-tested one effect at a time:

  plant     the true airframe differs from the nominal one the controller is tuned for: mass, inertia,
            per-motor thrust efficiency, drag, slower/faster motors, controller gains; PX4-like velocity
            integral and 45 deg tilt limit. A measured airframe goes in a plant profile (TorchQuad(params=)).
  battery   the 4S 5000 mAh LiPo: loaded voltage from state of charge and current (from the thrust being made),
            thrust capability ~ (V / V_ref)^2, charge drawn down as it flies (battery.py).
  latency   commands reach the motors late (Jetson inference + MAVLink/uXRCE + PX4 loop).
  velocity  the policy's velocity input is a VIO-like estimate: lag, white noise growing with speed, a
            random-walk bias, and dropouts (VIO lost -> the estimate freezes, then re-initialises with a new
            bias). Roll/pitch get small EKF-level noise.
  depth     OAK-D S2 stereo (passive: no IR projector, so blank surfaces lose depth; 'depth_plain' is the harsh case): depth from quantised, noisy disparity (error grows with z^2), no depth closer
            than the minimum disparity range, invalid pixels on textureless surfaces and in stereo occlusion
            shadows beside foreground edges, frame latency + 15 Hz frame age. ``render_scale`` > 1 renders at
            the sensor's resolution and downsamples like the onboard pipeline (accurate, eval); 1 applies the
            same model directly at the policy resolution (cheap, training).
  net       the boundary net is see-through: depth sees the room and random objects (people, tables) behind
            it, plus sparse returns from the net mesh itself. The net hangs on a frame of ~1.5 in (3.8 cm)
            vertical posts and horizontal rails, which stereo does see (5 posts per side; rails at the floor, the middle and the top).
  rangefinder  occasional no-return readings.
  vio_pos   (needs ``CourseEnvCfg.pos_obs``) the actor's course-frame position + yaw input is a cuVSLAM-like estimate:
            the odometry frame is initialised at the launch-mark pose, so course_pose = launch_pose o vio_odom. The
            position error INTEGRATES the same velocity-error process the ``velocity`` effect feeds the policy (lag,
            speed-dependent noise, random-walk bias, dropouts), plus the motion rotated by a slowly drifting yaw
            error; z is held near truth (rangefinder/baro fusion: the z error decays). During a dropout the
            estimate HOLDS its last value (the deploy node republishes the last odometry); at re-initialisation the
            error continues from where it was plus a re-init jump. Initial error at reset: ~1 cm / 0.5 deg at the
            launch mark (incl. liftoff starts), N(0, sigma(s)) elsewhere, sigma(route arc length s) calibrated by
            ``vio_calib.py`` flying the planned route through this same process (assets/vio_sigma.json).

Magnitudes are best guesses for the DroneHacks kit (Jetson Orin Nano, Pixhawk 6C, OAK-D S2, VL53L1X) and
should be replaced with measurements (see AGENTS.md). All draws use a private generator, so enabling an
effect never changes the nominal simulation's random stream.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dronehacks.mesh import GROUP_ID, CourseMesh

ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets'))


@dataclass
class RealismCfg:
    seed: int = 12345
    #: training: the share of episodes with the effects on grows linearly 0 -> 1 over this many env steps
    #: (constant full-strength randomisation from step 0 was observed to stop learning). 0 = always on.
    ramp_steps: int = 0
    # ── flight controller: PX4-like velocity loop (integral + tilt limit) instead of the training controller ──
    px4_loop: bool = False
    # ── plant (per-episode draws) ──
    plant: bool = False
    mass_rel: float = 0.10                     # true mass = nominal x U(1 - rel, 1 + rel)
    inertia_rel: float = 0.20
    thrust_rel: float = 0.08                   # motor/prop thrust efficiency spread
    drag_range: Tuple[float, float] = (0.6, 1.6)
    motor_tau_range: Tuple[float, float] = (0.02, 0.06)
    k_vel_range: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((1.6, 1.6, 1.2), (3.5, 3.5, 2.5))
    vel_ki: Tuple[float, float, float] = (0.4, 0.4, 2.0)       # px4_loop: MPC_XY_VEL_I_ACC / MPC_Z_VEL_I_ACC defaults
    vel_ki_random: bool = False                # px4_loop: integral gain x U(0, 1) per episode (training)
    max_tilt_deg: float = 45.0                 # px4_loop: MPC_TILTMAX_AIR default
    # ── battery: the drone's LiPo (dronehacks/battery.py, assets/battery_4s5000.json) ──
    battery: bool = False
    battery_file: str = 'battery_4s5000.json'
    soc_range: Tuple[float, float] = (0.5, 1.0)       # state of charge at takeoff
    # ── actuation latency ──
    latency: bool = False
    action_latency_s: Tuple[float, float] = (0.04, 0.12)
    # ── velocity estimate (VIO) ──
    velocity: bool = False
    vio_latency_s: Tuple[float, float] = (0.03, 0.10)
    vio_noise: float = 0.04                    # m/s, white, per axis
    vio_noise_per_mps: float = 0.03            # extra m/s of noise per m/s of speed (motion blur, fewer features)
    vio_bias_init: float = 0.05                # m/s per axis at (re)initialisation
    vio_bias_walk: float = 0.03                # m/s / sqrt(s)
    vio_dropout_per_s: float = 0.03            # rate of VIO losses
    vio_dropout_s: Tuple[float, float] = (0.2, 1.0)
    attitude_noise_deg: float = 0.5
    # ── VIO position + yaw estimate (needs CourseEnvCfg.pos_obs; uses the velocity-error model above even when the
    #    `velocity` effect itself is off) ──
    vio_pos: bool = False
    vio_pos_drift_scale: float = 1.0           # x the integrated velocity-estimate error (1 = consistent with `velocity`)
    vio_pos_noise_m: float = 0.005             # white jitter on the published position, per axis (xy)
    vio_pos_z_tau_s: float = 0.5               # z error decays with this time constant (EKF fuses rangefinder/baro)
    vio_pos_z_noise_m: float = 0.02            # white z jitter
    vio_yaw_walk_deg: float = 0.3              # yaw drift, deg / sqrt(s)
    vio_reinit_pos_m: float = 0.10             # jump at re-initialisation after a dropout, per axis (z: x 0.2)
    vio_reinit_yaw_deg: float = 2.0
    vio_launch_pos_m: float = 0.01             # initial error at a launch-mark start (placement on the mark), per axis
    vio_launch_yaw_deg: float = 0.5
    vio_sigma_file: str = 'vio_sigma.json'     # initial-error sigma vs route arc length for other starts (vio_calib.py)
    # ── depth (OAK-D S2 stereo) ──
    depth: bool = False
    render_scale: int = 5                      # 5 x 128x80 = 640x400 (the organizers' sensor preset); 1 = cheap
    sensor_width_px: int = 640                 # disparity is computed at this width
    baseline_m: float = 0.075
    max_disparity_px: float = 95.0             # standard mode -> min depth f*B/95 ~ 0.35 m at 640 px
    disparity_sigma_px: float = 0.2
    subpixel_bits: int = 3                     # disparity quantised to 1/8 px
    hole_frac: float = 0.04                    # textureless dropouts, spatially correlated
    hole_frac_max: float = 0.0                 # >0: per-episode hole fraction ~ U(hole_frac, hole_frac_max) (real SGM loses
                                               # ~45% of a dim cluttered room: 4% holes is far too clean)
    hole_scale_px: int = 12                    # blob size at policy resolution... in sensor px x render_scale/5
    occlusion: bool = True                     # invalid stereo shadows beside foreground edges
    left_band_px: float = 0.0                  # SGBM-style matchers blank the whole disparity search range at the left
                                               # edge; training randomizes the band width over 0-100 px, covering the OAK-D's 95 px disparity search range
    left_band_max_px: float = 0.0              # >0: per-episode band width ~ U(0, max) instead (training)
    left_band_hi_frac: float = 0.0             # share of episodes whose band is instead ~ U(hi_lo, hi_hi)
    left_band_hi_lo: float = 80.0
    left_band_hi_hi: float = 100.0
    edge_fill_cols: int = 0                    # deploy-side mitigation: invalid pixels in the outermost N policy columns
                                               # take the nearest valid value inward along the row (both edges)
    band_right_frac: float = 0.0               # share of episodes whose band is on the RIGHT edge instead (the drone's
                                               # VPI node shows ~14/128 invalid columns on the right, not the left)
    #: what the policy sees for an invalid pixel: 'near' | 'far' | 'hold'. 'near' is the deployment mapping: 'far'
    #: makes the floor at takeoff (inside the 0.35 m minimum range) read as open space and the launch fails.
    invalid_fill: str = 'near'
    min_valid_frac: float = 0.5                # downsampling: a block with fewer valid pixels is invalid
    depth_latency_s: Tuple[float, float] = (0.03, 0.08)
    frame_rate_hz: float = 15.0
    # ── net + room behind it ──
    net: bool = False
    clutter_variants: int = 8
    clutter_objects: Tuple[int, int] = (4, 14)
    room_gap_m: Tuple[float, float] = (0.5, 3.0)       # net -> room wall
    net_return_prob: float = 0.03              # chance a pixel looking at the net returns the net
    depth_clean_frac: float = 0.0              # training: share of episodes with CLEAN depth (other effects stay on)
    net_return_prob_max: float = 0.0           # >0: per-episode prob ~ U(0, max) instead (training robustness)
    net_bars: bool = True                      # the net's frame: vertical posts + horizontal rails (depth-visible)
    bar_d: float = 0.038                       # ~1.5 in tube
    bar_d_range: Tuple[float, float] = (0.0, 0.0)  # >0: per-bar thickness ~ U(lo, hi) instead of bar_d
    bar_drop_frac: float = 0.0                 # each bar is missing from the depth with this probability (thin bars
                                               # are what stereo drops first)
    bar_offset_m: float = 0.0                  # each bar sits up to this far off the net plane (in/out)
    post_spacing_m: Tuple[float, float] = (1.45, 1.6)   # real: 5 posts per 6.1 m side incl. corners (1.52 m)
    frame_h_m: Tuple[float, float] = (3.0, 3.1)   # top rail = net height (10 ft); rails at floor, middle, top
    # ── rangefinder ──
    rangefinder: bool = False
    rf_dropout: float = 0.02
    rf_bias_m: float = 0.0                     # constant rangefinder offset for the whole run (m, + reads high)
    rf_bias_max_m: float = 0.0                 # >0: per-episode offset ~ U(-max, +max) on top (training)


#: Named presets for ``--realism``: one per effect plus everything together. Combine with commas.
PRESETS: Dict[str, dict] = {
    'plant': dict(plant=True), 'battery': dict(battery=True), 'latency': dict(latency=True),
    'velocity': dict(velocity=True), 'depth': dict(depth=True), 'net': dict(net=True),
    'rangefinder': dict(rangefinder=True),
    'vio_pos': dict(vio_pos=True),               # NOT part of 'real': it changes the observation (needs pos_obs)
    'depth_lite': dict(depth=True, render_scale=1),
    # OAK-D S2 is PASSIVE stereo (no IR projector): bare plywood gives far more holes than textured walls
    'depth_plain': dict(depth=True, hole_frac=0.20, hole_scale_px=20),
}
PRESETS['px4'] = dict(px4_loop=True)             # the controller response differs from training
# pipeline check: full sensor-resolution depth path with every error off; must score like 'ideal'
PRESETS['depth_null'] = dict(depth=True, disparity_sigma_px=0.0, subpixel_bits=16, hole_frac=0.0, occlusion=False,
                             depth_latency_s=(0.0, 0.0), frame_rate_hz=1e6, max_disparity_px=1e6)
PRESETS['real'] = {k: v for name in ('px4', 'plant', 'battery', 'latency', 'velocity', 'depth', 'net', 'rangefinder')
                   for k, v in PRESETS[name].items()}
PRESETS['real_lite'] = dict(PRESETS['real'], render_scale=1, net_return_prob=0.0, vel_ki_random=True)   # training


def _value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    if v.startswith('(') and v.endswith(')'):
        return tuple(_value(x) for x in v[1:-1].split(';'))
    return v


def parse(spec: Optional[str], **overrides) -> Optional[RealismCfg]:
    """'depth,velocity' / 'real' / 'depth{invalid_fill=near;occlusion=false}' / '' -> RealismCfg (None when
    nothing is enabled). Braces override fields of the preset(s) before them; tuples as (a;b)."""
    if not spec and not overrides:
        return None
    kw: dict = {}
    for name in _split(spec or ''):
        name = name.strip()
        if not name:
            continue
        extra = {}
        if '{' in name:
            name, body = name.split('{', 1)
            for item in _split_items(body.rstrip('}')):
                k, v = item.split('=', 1)
                extra[k.strip()] = _value(v.strip())
        if name not in PRESETS:
            raise ValueError(f'unknown realism preset {name!r}; choose from {sorted(PRESETS)}')
        kw.update(PRESETS[name])
        kw.update(extra)
    kw.update(overrides)
    names = {f.name for f in fields(RealismCfg)}
    bad = set(kw) - names
    if bad:
        raise ValueError(f'unknown realism fields {sorted(bad)}')
    return RealismCfg(**{k: tuple(map(tuple, v)) if k == 'k_vel_range' else (tuple(v) if isinstance(v, list) else v)
                         for k, v in kw.items()})


def _split(spec: str):
    """Split on commas outside braces."""
    out, depth, cur = [], 0, ''
    for ch in spec:
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
            continue
        depth += (ch == '{') - (ch == '}')
        cur += ch
    out.append(cur)
    return out


def _split_items(body: str):
    """'a=1;b=(x;y)' -> ['a=1', 'b=(x;y)'] (semicolons inside parentheses stay)."""
    out, depth, cur = [], 0, ''
    for ch in body:
        if ch == ';' and depth == 0:
            out.append(cur)
            cur = ''
            continue
        depth += (ch == '(') - (ch == ')')
        cur += ch
    if cur:
        out.append(cur)
    return out


# ── scene: the room behind the net ───────────────────────────────────────────────────────────────────────

def _box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int32)
    return v, f


def net_frame(rng: np.random.Generator, cfg: RealismCfg, net_half: float = 3.05):
    """Posts every ``post_spacing_m`` along each side (always at the corners) and three rails (floor, centre,
    top at ``frame_h_m``): square bars of side ``bar_d`` centred on the net plane. -> boxes."""
    def tt():
        d = rng.uniform(*cfg.bar_d_range) if cfg.bar_d_range[1] > 0 else cfg.bar_d
        return d / 2
    t = cfg.bar_d / 2
    top = rng.uniform(*cfg.frame_h_m)
    rails = [t, top / 2 + rng.uniform(-0.03, 0.03), top - t]                # floor, centre, top (as measured)
    boxes = []
    for axis in (0, 1):
        for sgn in (-1, 1):
            c = sgn * net_half
            n = max(1, int(round(2 * net_half / rng.uniform(*cfg.post_spacing_m))))
            ss = np.linspace(-net_half, net_half, n + 1)
            ss[1:-1] += rng.uniform(-0.05, 0.05, n - 1)                       # posts are not perfectly even
            for s in ss:
                if rng.uniform() < cfg.bar_drop_frac:
                    continue
                t = tt(); o = rng.uniform(-cfg.bar_offset_m, cfg.bar_offset_m)
                lo, hi = [0.0, 0.0, -top], [0.0, 0.0, 0.0]
                lo[axis], hi[axis] = c + o - t, c + o + t
                lo[1 - axis], hi[1 - axis] = s - t, s + t
                boxes.append((tuple(lo), tuple(hi)))
            for z in rails:
                if rng.uniform() < cfg.bar_drop_frac:
                    continue
                t = tt(); o = rng.uniform(-cfg.bar_offset_m, cfg.bar_offset_m)
                lo, hi = [0.0, 0.0, -z - t], [0.0, 0.0, -z + t]
                lo[axis], hi[axis] = c + o - t, c + o + t
                lo[1 - axis], hi[1 - axis] = -net_half, net_half
                boxes.append((tuple(lo), tuple(hi)))
    return boxes


def room_clutter(rng: np.random.Generator, cfg: RealismCfg, net_half: float = 3.05) -> CourseMesh:
    """A random room around the net (NED, z down): floor beyond the net, four walls, people/tables/chairs."""
    boxes = [((-9.0, -9.0, 0.0), (9.0, 9.0, 0.02))]                         # floor outside the net (slightly low)
    for axis in (0, 1):
        for sgn in (-1, 1):
            d = net_half + rng.uniform(*cfg.room_gap_m)
            lo, hi = [-9.0, -9.0, -4.0], [9.0, 9.0, 0.0]
            lo[axis], hi[axis] = (d, d + 0.2) if sgn > 0 else (-d - 0.2, -d)
            boxes.append((tuple(lo), tuple(hi)))
    if cfg.net_bars:                                                        # the net's frame, in the net plane
        boxes += net_frame(rng, cfg, net_half)
    for _ in range(rng.integers(*cfg.clutter_objects, endpoint=True)):
        kind = rng.choice(['person', 'table', 'chair', 'cart'])
        w, dpt, h = {'person': (0.45, 0.30, rng.uniform(1.55, 1.9)), 'table': (rng.uniform(0.8, 1.8), 0.75, 0.75),
                     'chair': (0.5, 0.5, rng.uniform(0.45, 0.95)), 'cart': (0.6, 0.9, 1.0)}[kind]
        r = net_half + 0.15 + rng.uniform(0.0, cfg.room_gap_m[0] + 1.0)
        th = rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(th), r * np.sin(th)])
        c = np.clip(c, -8.0, 8.0)
        boxes.append(((c[0] - w / 2, c[1] - dpt / 2, -h), (c[0] + w / 2, c[1] + dpt / 2, 0.0)))
    verts, faces, base = [], [], 0
    for lo, hi in boxes:
        v, f = _box(lo, hi)
        verts.append(v)
        faces.append(f + base)
        base += len(v)
    faces = np.concatenate(faces)
    return CourseMesh(np.concatenate(verts), faces, np.full(len(faces), GROUP_ID['wall'], np.int8),
                      np.full(len(faces), -1, np.int32))


def concat_meshes(a: CourseMesh, b: CourseMesh) -> CourseMesh:
    return CourseMesh(np.concatenate([a.vertices, b.vertices]), np.concatenate([a.faces, b.faces + len(a.vertices)]),
                      np.concatenate([a.face_group, b.face_group]), np.concatenate([a.face_object, b.face_object]))


# ── runtime ──────────────────────────────────────────────────────────────────────────────────────────────

class VioVelocity:
    """The VIO velocity-error process, per env: lag, white noise growing with speed, a random-walk bias, dropouts
    (estimate frozen) that end with a re-initialised bias. Shared by the ``velocity`` effect (the observed velocity)
    and ``vio_pos`` (whose position drift integrates this same error). ``step`` once per control step."""

    def __init__(self, cfg: RealismCfg, num_envs: int, device, dt: float, gen: torch.Generator):
        self.cfg, self.E, self.dev, self.dt, self.gen = cfg, num_envs, device, dt, gen
        kw = dict(device=device, dtype=torch.float32)
        self.v_hist = None                                  # (Hn, E, 3) true NED velocity, newest first
        self.lat = torch.zeros(num_envs, **kw)
        self.bias = torch.zeros(num_envs, 3, **kw)
        self.drop_left = torch.zeros(num_envs, **kw)        # seconds of dropout remaining
        self.v_est = torch.zeros(num_envs, 3, **kw)
        self.lost = torch.zeros(num_envs, dtype=torch.bool, device=device)    # lost during the last step
        self.ended = torch.zeros(num_envs, dtype=torch.bool, device=device)   # re-initialised at the end of it

    def _u(self, n, lo, hi, *shape):
        return torch.rand(n, *shape, device=self.dev, generator=self.gen) * (hi - lo) + lo

    def _n(self, *shape):
        return torch.randn(*shape, device=self.dev, generator=self.gen)

    def reset(self, ids: torch.Tensor, vel_ned: torch.Tensor) -> None:
        c, n = self.cfg, len(ids)
        self.lat[ids] = self._u(n, *c.vio_latency_s)
        self.bias[ids] = self._n(n, 3) * c.vio_bias_init
        self.drop_left.index_fill_(0, ids, 0.0)
        if self.v_hist is not None:
            self.v_hist[:, ids] = vel_ned[ids]
        self.v_est[ids] = vel_ned[ids]
        self.lost.index_fill_(0, ids, False)
        self.ended.index_fill_(0, ids, False)
    def step(self, true_vel_ned: torch.Tensor) -> torch.Tensor:
        """-> (E,3) the estimate for every env (no realism mask applied)."""
        c = self.cfg
        n_h = int(math.ceil(c.vio_latency_s[1] / self.dt)) + 2
        if self.v_hist is None:
            self.v_hist = true_vel_ned.unsqueeze(0).repeat(n_h, 1, 1)
        self.v_hist = torch.cat([true_vel_ned.unsqueeze(0), self.v_hist[:-1]], 0)
        x = self.lat / self.dt                                                       # steps back, fractional
        i0 = x.floor().long().clamp(max=n_h - 2)
        w = (x - i0).unsqueeze(-1)
        ar = torch.arange(self.E, device=self.dev)
        v_lag = self.v_hist[i0, ar] * (1 - w) + self.v_hist[i0 + 1, ar] * w
        self.bias += self._n(self.E, 3) * c.vio_bias_walk * math.sqrt(self.dt)
        speed = torch.linalg.vector_norm(v_lag, dim=-1, keepdim=True)
        est = v_lag + self.bias + self._n(self.E, 3) * (c.vio_noise + c.vio_noise_per_mps * speed)
        start = (self.drop_left <= 0) & (torch.rand(self.E, device=self.dev, generator=self.gen)
                                         < c.vio_dropout_per_s * self.dt)
        self.drop_left = torch.where(start, self._u(self.E, *c.vio_dropout_s), self.drop_left)
        lost = self.drop_left > 0
        self.drop_left = (self.drop_left - self.dt).clamp(min=0)
        ended = lost & (self.drop_left <= 0)
        self.bias = torch.where(ended.unsqueeze(-1), self._n(self.E, 3) * c.vio_bias_init, self.bias)   # re-init
        self.v_est = torch.where(lost.unsqueeze(-1), self.v_est, est)                 # frozen while lost
        self.lost, self.ended = lost, ended
        return self.v_est


#: the RealismCfg fields assets/vio_sigma.json was calibrated with (vio_calib.py)
VIO_PARAM_FIELDS = ('vio_latency_s', 'vio_noise', 'vio_noise_per_mps', 'vio_bias_init', 'vio_bias_walk',
                    'vio_dropout_per_s', 'vio_dropout_s', 'vio_pos_drift_scale', 'vio_pos_z_tau_s', 'vio_yaw_walk_deg',
                    'vio_reinit_pos_m', 'vio_reinit_yaw_deg', 'vio_launch_pos_m', 'vio_launch_yaw_deg')


def load_vio_sigma(path: str, cfg: Optional[RealismCfg] = None) -> dict:
    with open(path) as f:
        d = json.load(f)
    if cfg is not None:
        cur = {k: list(v) if isinstance(v, tuple) else v for k, v in ((k, getattr(cfg, k)) for k in VIO_PARAM_FIELDS)}
        diff = sorted(k for k in VIO_PARAM_FIELDS if d.get('params', {}).get(k) != cur[k])
        if diff:
            import warnings
            warnings.warn(f'{path} was calibrated with different VIO parameters ({diff}): '
                          'rerun python -m dronehacks.vio_calib with these settings')
    return d


class Realism:
    """Per-env state of every enabled effect. Owned by ``CourseEnv``; all tensors on the env device."""

    def __init__(self, cfg: RealismCfg, num_envs: int, device, control_dt: float, nominal_mass: float,
                 nominal_inertia, depth_hw: Tuple[int, int], fx: float, far_clip: float):
        self.cfg, self.E, self.dev, self.dt = cfg, num_envs, torch.device(device), control_dt
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(cfg.seed)
        kw = dict(device=self.dev, dtype=torch.float32)
        E = num_envs
        self.m0, self.I0 = float(nominal_mass), torch.tensor(nominal_inertia, **kw)
        self.H, self.W = depth_hw
        self.far = far_clip
        # plant / battery
        self.net_p = torch.full((E, 1, 1), float(cfg.net_return_prob), **kw)   # per-env net return prob
        self.true_mass = torch.full((E, 1), self.m0, **kw)
        self.true_inertia = self.I0.repeat(E, 1)
        self.drag_scale = torch.ones(E, 1, **kw)
        self.eff = torch.ones(E, 1, **kw)
        self.soc = torch.ones(E, **kw)
        self.pack = None
        if cfg.battery:
            from dronehacks.battery import LiPoPack
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', cfg.battery_file)
            self.pack = LiPoPack.from_json(here)
            self._soc_x = torch.tensor(self.pack.soc_curve, **kw)
            self._ocv_y = torch.tensor(self.pack.ocv_curve, **kw) * self.pack.cells
        # latency: action history, newest first
        self.n_hist = max(2, int(math.ceil(cfg.action_latency_s[1] / control_dt)) + 2)
        self.act_hist = None
        self.act_lat = torch.zeros(E, **kw)
        # VIO velocity (draws on self.gen, exactly as before the refactor)
        self.vio = VioVelocity(cfg, E, self.dev, control_dt, self.gen)
        # VIO position + yaw: its own generator, so adding vio_pos leaves every other effect's stream unchanged
        if cfg.vio_pos:
            self.gen_pos = torch.Generator(device=self.dev)
            self.gen_pos.manual_seed(cfg.seed + 7919)
            # with the velocity effect off, the same error model runs privately for the position estimate
            self.vio_pos_vel = self.vio if cfg.velocity else VioVelocity(cfg, E, self.dev, control_dt, self.gen_pos)
            self.pe = torch.zeros(E, 3, **kw)               # position error while tracking (NED)
            self.ye = torch.zeros(E, **kw)                  # yaw error
            self.p_hold = torch.zeros(E, 3, **kw)           # last published pose (held through dropouts)
            self.y_hold = torch.zeros(E, **kw)
            self.vio_ok = torch.ones(E, dtype=torch.bool, device=self.dev)
            self.vio_sigma = None
            path = os.path.join(ASSETS, cfg.vio_sigma_file)
            if os.path.exists(path):
                self.vio_sigma = load_vio_sigma(path, cfg)
                t = self.vio_sigma['table']
                self._sig_s = torch.tensor(t['s'], **kw)
                self._sig = torch.tensor([t['xy'], t['z'], t['yaw_deg']], **kw)          # (3, B)
        # depth
        self.depth_lat = torch.zeros(E, **kw)
        self.prev_depth = None
        s = cfg.render_scale
        self.f_sensor = fx * cfg.sensor_width_px / self.W   # focal at the sensor's width, px
        self.fx_render = fx * s
        self.fB = self.f_sensor * cfg.baseline_m
        self.z_min = self.fB / cfg.max_disparity_px
        #: per-env: are the effects on this episode (all True unless ramping)
        self.mask = torch.ones(E, dtype=torch.bool, device=self.dev)
        self.depth_on = torch.ones(E, dtype=torch.bool, device=self.dev)    # per-env: stereo model on (vs clean)
        self.band = torch.full((E, 1, 1), float(cfg.left_band_px), **kw)
        self.hole = torch.full((E,), float(cfg.hole_frac), **kw)   # per-env hole fraction (hole_frac_max)
        self.rf_bias = torch.zeros(E, **kw)                  # per-env constant rangefinder offset (rf_bias_max_m)   # per-env left band (sensor px)
        self.ki = torch.zeros(E, 3, **kw)
        self.fast = os.environ.get('DRONEHACKS_TORCH_STEREO', '0') != '1'    # fused Warp stereo model on CUDA

    def draw_mask(self, ids: torch.Tensor, env_steps: int) -> None:
        if self.cfg.ramp_steps > 0 and len(ids):
            p = min(1.0, env_steps / self.cfg.ramp_steps)
            self.mask[ids] = torch.rand(len(ids), device=self.dev, generator=self.gen) < p

    def _const(self, values) -> torch.Tensor:
        """A small config tuple as a device tensor, uploaded once (a fresh torch.tensor per reset is a host sync)."""
        cache = self.__dict__.setdefault('_consts', {})
        key = tuple(values)
        if key not in cache:
            cache[key] = torch.tensor(key, device=self.dev)
        return cache[key]

    # random helpers on the private generator
    def _u(self, n, lo, hi, *shape):
        return torch.rand(n, *shape, device=self.dev, generator=self.gen) * (hi - lo) + lo

    def _n(self, *shape):
        return torch.randn(*shape, device=self.dev, generator=self.gen)

    # ── resets ──
    def on_reset(self, ids: torch.Tensor, plant) -> None:
        c, n = self.cfg, len(ids)
        if n == 0:
            return
        on = self.mask[ids].unsqueeze(-1)                    # (n,1): effects on for this episode
        if c.hole_frac_max > c.hole_frac:
            self.hole[ids] = self._u(n, c.hole_frac, c.hole_frac_max, 1).view(n)
        if c.rf_bias_max_m > 0:
            self.rf_bias[ids] = self._u(n, -c.rf_bias_max_m, c.rf_bias_max_m, 1).view(n)
        pick = lambda x, nominal: torch.where(on, x, nominal)  # noqa: E731
        if c.plant:
            self.true_mass[ids] = pick(self.m0 * self._u(n, 1 - c.mass_rel, 1 + c.mass_rel, 1), torch.full_like(on, self.m0, dtype=torch.float32))
            self.true_inertia[ids] = pick(self.I0 * self._u(n, 1 - c.inertia_rel, 1 + c.inertia_rel, 3), self.I0.expand(n, 3))
            self.drag_scale[ids] = pick(self._u(n, *c.drag_range, 1), torch.ones(n, 1, device=self.dev))
            self.eff[ids] = pick(self._u(n, 1 - c.thrust_rel, 1 + c.thrust_rel, 1), torch.ones(n, 1, device=self.dev))
            plant.motor_tau[ids] = pick(self._u(n, *c.motor_tau_range, 1), plant.motor_tau[ids])
            lo, hi = (self._const(x) for x in c.k_vel_range)
            plant.controller.K_vel[ids] = pick(self._u(n, 0.0, 1.0, 3) * (hi - lo) + lo, plant.controller.K_vel[ids])
        if c.px4_loop:
            ki = self._const(c.vel_ki).expand(n, 3)
            if c.vel_ki_random:
                ki = ki * self._u(n, 0.0, 1.0, 1)
            self.ki[ids] = ki * on
        if c.left_band_max_px > 0:
            self.band[ids] = self._u(n, 0.0, c.left_band_max_px, 1).view(n, 1, 1)
            if c.left_band_hi_frac > 0:
                hi = torch.rand(n, device=self.dev, generator=self.gen) < c.left_band_hi_frac
                self.band[ids] = torch.where(hi.view(n, 1, 1), self._u(n, c.left_band_hi_lo, c.left_band_hi_hi, 1).view(n, 1, 1),
                                             self.band[ids])
            if c.band_right_frac > 0:   # encoded as a NEGATIVE threshold: invalid where u_sensor > -band (right edge)
                rt = torch.rand(n, device=self.dev, generator=self.gen) < c.band_right_frac
                thr = -(float(c.sensor_width_px) - self.band[ids]).clamp(min=1e-3)
                self.band[ids] = torch.where(rt.view(n, 1, 1), thr, self.band[ids])
        if c.depth_clean_frac > 0:
            self.depth_on[ids] = torch.rand(n, device=self.dev, generator=self.gen) >= c.depth_clean_frac
        if c.net_return_prob_max > 0:
            self.net_p[ids] = self._u(n, 0.0, c.net_return_prob_max, 1).view(n, 1, 1)
        if c.battery:
            self.soc[ids] = self._u(n, *c.soc_range)
        if c.latency:
            self.act_lat[ids] = self._u(n, *c.action_latency_s) * on[:, 0]
            if self.act_hist is not None:
                self.act_hist.index_fill_(1, ids, 0.0)
        if c.velocity:
            self.vio.reset(ids, plant.vel_ned)
        if c.depth:
            self.depth_lat[ids] = self._u(n, *c.depth_latency_s)

    def configure_plant(self, plant) -> None:
        c = self.cfg
        if c.plant:
            plant.true_mass, plant.true_inertia, plant.drag_scale = self.true_mass, self.true_inertia, self.drag_scale
        if c.px4_loop:
            plant.controller.vel_ki = self.ki            # per env, set at reset (0 where the effects are off)
            plant.controller.max_tilt = math.radians(c.max_tilt_deg)
        if c.plant or c.battery:
            plant.thrust_scale = self.eff.clone()

    # ── per step: plant ──
    def _ocv(self, soc):
        x, y = self._soc_x, self._ocv_y
        i = torch.clamp(torch.searchsorted(x, soc.clamp(0, 1)), 1, len(x) - 1)
        w = (soc.clamp(0, 1) - x[i - 1]) / (x[i] - x[i - 1])
        return y[i - 1] + w * (y[i] - y[i - 1])

    def update_thrust(self, plant, t_episode: torch.Tensor) -> None:
        if self.cfg.plant or self.cfg.battery:
            s = self.eff
            if self.cfg.battery:
                p = self.pack
                thrust = (plant.applied_thrust * (plant.thrust_scale if plant.thrust_scale is not None else 1.0)).sum(-1)
                v = self._ocv(self.soc)
                for _ in range(2):                                   # current depends on the voltage it sags to
                    i = p.hover_power_w * (thrust.clamp(min=0) / p.hover_thrust_n) ** 1.5 / v.clamp(min=1.0)
                    v = self._ocv(self.soc) - i * p.r_pack_ohm
                f = ((v / p.v_ref) ** 2).clamp(0.3, 1.15).unsqueeze(-1)
                s = s * torch.where(self.mask.unsqueeze(-1), f, torch.ones_like(f))
                self.soc = (self.soc - i * self.dt / (p.capacity_mah * 3.6)).clamp(0, 1)
            plant.thrust_scale = s

    def substep_actions(self, action: torch.Tensor, decimation: int) -> Optional[torch.Tensor]:
        """(S,E,A) action in force at each physics substep, or None without latency. Substep k of this control
        step runs at t + k*sub_dt; the command in force then is the newest one issued at least ``latency`` ago."""
        if not self.cfg.latency:
            return None
        if self.act_hist is None:
            self.act_hist = torch.zeros(self.n_hist, *action.shape, device=self.dev)
        self.act_hist = torch.cat([action.unsqueeze(0), self.act_hist[:-1]], 0)
        sub_dt = self.dt / decimation
        k = torch.arange(decimation, device=self.dev, dtype=torch.float32).unsqueeze(1)          # (S,1)
        back = torch.ceil((self.act_lat.unsqueeze(0) - k * sub_dt) / self.dt - 1e-6).clamp(0, self.n_hist - 1).long()
        return self.act_hist[back, torch.arange(self.E, device=self.dev).unsqueeze(0)]          # (S,E,A)

    # ── per step: observations ──
    def velocity(self, true_vel_ned: torch.Tensor) -> torch.Tensor:
        """VIO-like NED velocity estimate (call once per control step)."""
        if not self.cfg.velocity:
            return true_vel_ned
        v_est = self.vio.step(true_vel_ned)
        return torch.where(self.mask.unsqueeze(-1), v_est, true_vel_ned)

    # ── VIO position + yaw (vio_pos) ──
    def vio_sigma_at(self, s: torch.Tensor) -> torch.Tensor:
        """(n,) flown arc length -> (3, n) calibrated initial-error sigma: xy per axis (m), z (m), yaw (deg).
        Linear in the table, the fitted sqrt(a + b s + c s^2) curve beyond its end."""
        if self.vio_sigma is None:
            raise FileNotFoundError(f'vio_pos needs assets/{self.cfg.vio_sigma_file} for non-launch starts: '
                                    'run python -m dronehacks.vio_calib')
        ss, sig = self._sig_s, self._sig
        x = s.clamp(min=float(ss[0]))
        i = torch.clamp(torch.searchsorted(ss, x), 1, len(ss) - 1)
        w = ((x - ss[i - 1]) / (ss[i] - ss[i - 1])).clamp(0, 1)
        out = sig[:, i - 1] * (1 - w) + sig[:, i] * w
        fit = self.vio_sigma['fit']
        beyond = s > ss[-1]
        if beyond.any():
            ext = torch.stack([torch.sqrt((fit[k][0] + fit[k][1] * s + fit[k][2] * s * s).clamp(min=0))
                               for k in ('xy', 'z', 'yaw_deg')])
            out = torch.where(beyond, torch.maximum(ext, sig[:, -1:]), out)
        return out

    def vio_pos_reset(self, ids: torch.Tensor, start_s: torch.Tensor, launch: torch.Tensor, pos_ned: torch.Tensor,
                      yaw: torch.Tensor, vel_ned: torch.Tensor) -> None:
        """Initial estimate error for fresh episodes. ``start_s`` (n,) = route arc length (m) the VIO has flown to
        reach this start; ``launch`` (n,) = starts on/just above the launch mark (odometry just initialised there).
        ``pos_ned``/``yaw``/``vel_ned`` are the full (E,...) true state after the plant reset."""
        c, n = self.cfg, len(ids)
        if not c.vio_pos or n == 0:
            return
        if not c.velocity:
            self.vio_pos_vel.reset(ids, vel_ned)
        g = self.gen_pos
        if bool(launch.all()):
            sig = torch.zeros(3, n, device=self.dev)
        else:
            sig = self.vio_sigma_at(start_s)
        lp = torch.tensor([c.vio_launch_pos_m, c.vio_launch_pos_m, c.vio_launch_yaw_deg], device=self.dev)
        sig = torch.where(launch.unsqueeze(0), lp.unsqueeze(1), sig)
        e = torch.randn(n, 4, device=self.dev, generator=g)
        self.pe[ids] = e[:, :3] * torch.stack([sig[0], sig[0], sig[1]], -1)
        self.ye[ids] = e[:, 3] * torch.deg2rad(sig[2])
        self.p_hold[ids] = pos_ned[ids] + self.pe[ids]
        self.y_hold[ids] = yaw[ids] + self.ye[ids]
        self.vio_ok.index_fill_(0, ids, True)
    def vio_pose(self, pos_ned: torch.Tensor, vel_ned: torch.Tensor, yaw: torch.Tensor):
        """Once per control step, AFTER ``velocity()`` (with the velocity effect on, this reuses that step's
        estimate). -> (p_hat (E,3) NED, yaw_hat (E,), ok (E,) bool: VIO tracking). Envs whose realism is off this
        episode (ramp) get the true pose."""
        c, dt, g, E = self.cfg, self.dt, self.gen_pos, self.E
        m = self.vio_pos_vel
        v_est = m.v_est if c.velocity else m.step(vel_ned)
        dv = (v_est - vel_ned) * c.vio_pos_drift_scale
        adv = ~m.lost                                                    # tracking: the error evolves
        rot = self.ye.unsqueeze(-1) * torch.stack([-vel_ned[:, 1], vel_ned[:, 0]], -1)   # motion seen in a yawed frame
        d = torch.cat([dv[:, :2] + rot, dv[:, 2:]], -1) * dt
        decay = torch.tensor([1.0, 1.0, math.exp(-dt / max(c.vio_pos_z_tau_s, 1e-6))], device=self.dev)
        r = torch.randn(E, 8, device=self.dev, generator=g)              # fixed draw count per step
        self.pe = torch.where(adv.unsqueeze(-1), self.pe * decay + d, self.pe)
        self.ye = torch.where(adv, self.ye + r[:, 0] * math.radians(c.vio_yaw_walk_deg) * math.sqrt(dt), self.ye)
        jump = r[:, 1:4] * torch.tensor([1.0, 1.0, 0.2], device=self.dev) * c.vio_reinit_pos_m
        self.pe = torch.where(m.ended.unsqueeze(-1), self.pe + jump, self.pe)               # re-initialisation
        self.ye = torch.where(m.ended, self.ye + r[:, 4] * math.radians(c.vio_reinit_yaw_deg), self.ye)
        jit = r[:, 5:8] * torch.tensor([c.vio_pos_noise_m, c.vio_pos_noise_m, c.vio_pos_z_noise_m], device=self.dev)
        self.p_hold = torch.where(m.lost.unsqueeze(-1), self.p_hold, pos_ned + self.pe + jit)  # held while lost
        self.y_hold = torch.where(m.lost, self.y_hold, yaw + self.ye)
        self.vio_ok = ~m.lost
        on = self.mask
        return (torch.where(on.unsqueeze(-1), self.p_hold, pos_ned), torch.where(on, self.y_hold, yaw),
                self.vio_ok | ~on)

    def attitude(self, rp: torch.Tensor) -> torch.Tensor:
        if not self.cfg.velocity:
            return rp
        return rp + self._n(*rp.shape) * math.radians(self.cfg.attitude_noise_deg) * self.mask.unsqueeze(-1)

    def rangefinder(self, r: torch.Tensor, rf_max: float) -> torch.Tensor:
        if not self.cfg.rangefinder:
            return r
        if self.cfg.rf_bias_m != 0.0 or self.cfg.rf_bias_max_m > 0:
            b = self.cfg.rf_bias_m + (self.rf_bias if self.cfg.rf_bias_max_m > 0 else 0.0)
            r = torch.where(self.mask, (r + b).clamp(0.0, rf_max), r)
        lost = (torch.rand(r.shape, device=self.dev, generator=self.gen) < self.cfg.rf_dropout) & self.mask
        return torch.where(lost, torch.full_like(r, rf_max), r)

    def depth_lag_substeps(self, decimation: int) -> torch.Tensor:
        """(E,) how many physics substeps old each env's depth frame is (latency + frame age at 15 Hz)."""
        age = self.depth_lat + torch.rand(self.E, device=self.dev, generator=self.gen) / self.cfg.frame_rate_hz
        return torch.round(age / (self.dt / decimation)).long() * self.mask

    def corrupt_depth(self, z_m: torch.Tensor, net_z_m: Optional[torch.Tensor],
                      far_scale: Optional[float] = None, net_rays=None) -> torch.Tensor:
        """(E,Hr,Wr) true z-depth in metres at render resolution (far clip = miss) -> (E,H,W) normalised depth
        as the policy would receive it from the stereo pipeline. With ``far_scale`` the inputs are NORMALISED
        depth (metres = x far_scale), e.g. the renderer's own buffers, read without a copy. ``net_rays`` =
        (net MeshScene, pos, rpy) in place of ``net_z_m`` (needs ``far_scale``): the fused path casts the net only
        where a return fires; the torch path renders it in full.

        On CUDA this runs the fused Warp kernels (stereo_warp.py; same model, per-pixel randoms from a
        counter-based RNG instead of ``self.gen``). ``self.fast = False`` or DRONEHACKS_TORCH_STEREO=1 forces the
        torch reference below."""
        if self.fast and z_m.is_cuda:
            try:
                from dronehacks import stereo_warp
            except ImportError:
                self.fast = False
            else:
                return self._edge_fill(stereo_warp.corrupt_depth(self, z_m, net_z_m, 1.0 if far_scale is None else far_scale,
                                                                 net_rays=net_rays))
        if net_rays is not None and net_z_m is None:
            sc, pos, rpy = net_rays
            net_z_m = sc.render_depth(pos, rpy)
        if far_scale is not None:
            z_m = z_m * far_scale
            net_z_m = None if net_z_m is None else net_z_m * far_scale
        return self._edge_fill(self._corrupt_depth_torch(z_m, net_z_m))

    def _edge_fill(self, out: torch.Tensor) -> torch.Tensor:
        """Fill invalid pixels in the outermost ``edge_fill_cols`` columns with the nearest valid value inward (per
        row), as a deploy node can do on the 80x128 grid; they then count as valid (``last_bad``)."""
        n = int(self.cfg.edge_fill_cols)
        if n <= 0:
            return out
        out, bad = out.clone(), self.last_bad.clone()
        W = out.shape[-1]
        for cols in (range(W - n, W), range(n - 1, -1, -1)):          # right edge left-to-right, left edge right-to-left
            for c in cols:
                src = c - 1 if c >= W - n else c + 1
                b = bad[..., c] & ~bad[..., src]
                out[..., c] = torch.where(b, out[..., src], out[..., c])
                bad[..., c] = bad[..., c] & bad[..., src]
        self.last_bad = bad
        return out

    def _corrupt_depth_torch(self, z_m: torch.Tensor, net_z_m: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.cfg
        E, Hr, Wr = z_m.shape
        s = Wr / self.W                                                       # render px per policy px
        f_r = self.f_sensor * Wr / c.sensor_width_px                          # focal at render resolution
        z_clean = z_m
        miss = z_m >= self.far * 0.999
        if net_z_m is not None:                                               # sparse net returns
            hit = (torch.rand(z_m.shape, device=self.dev, generator=self.gen) < self.net_p) & (net_z_m < z_m)
            z_m = torch.where(hit, net_z_m, z_m)
            miss = miss & ~hit
        # disparity in SENSOR pixels: noise + quantisation
        d = self.fB / z_m.clamp(min=1e-3)
        d = d + self._n(*d.shape) * c.disparity_sigma_px
        q = 2 ** c.subpixel_bits
        d = torch.round(d * q) / q
        invalid = (d > c.max_disparity_px) | (d <= 0)
        z = self.fB / d.clamp(min=1e-6)
        # textureless holes: thresholded low-frequency noise
        if c.hole_frac > 0:
            gh, gw = max(2, int(Hr / (c.hole_scale_px * s))), max(2, int(Wr / (c.hole_scale_px * s)))
            field_ = F.interpolate(torch.rand(E, 1, gh, gw, device=self.dev, generator=self.gen), size=(Hr, Wr),
                                   mode='bilinear', align_corners=False)[:, 0]
            invalid |= field_ < (self.hole.view(E, 1, 1) if c.hole_frac_max > c.hole_frac else c.hole_frac)
        # stereo occlusion (left camera reference, right camera at +baseline): pixel u is occluded when a pixel
        # v > u to its right is closer and lands left of it in the right image: v - d_v <= u - d_u
        if c.occlusion:
            d_r = (self.fB / z_m.clamp(min=1e-3)) * (Wr / c.sensor_width_px)         # disparity in render px
            u = torch.arange(Wr, device=self.dev, dtype=torch.float32)
            xr = u - d_r                                                             # right-image column
            suffix = torch.flip(torch.cummin(torch.flip(xr, [-1]), dim=-1).values, [-1])
            right_min = torch.cat([suffix[..., 1:], torch.full_like(suffix[..., :1], float('inf'))], -1)
            invalid |= (right_min <= xr - 0.5) | (xr < 0)        # occluded, or no match: the left border band
        if c.left_band_px > 0 or c.left_band_max_px > 0:                     # no-overlap band at the left edge
            u_s = torch.arange(Wr, device=self.dev, dtype=torch.float32) * (c.sensor_width_px / Wr)
            u_s = u_s.view(1, 1, -1)
            invalid = invalid | torch.where(self.band >= 0, u_s < self.band, u_s > -self.band)
        valid = ~invalid & ~miss
        z = torch.where(valid, z, torch.zeros_like(z))
        # downsample to the policy grid: mean of valid pixels; mostly-invalid blocks are invalid
        if Wr != self.W or Hr != self.H:
            k = (Hr // self.H, Wr // self.W)
            vf = F.avg_pool2d(valid.float().unsqueeze(1), k)[:, 0]
            zs = F.avg_pool2d(z.unsqueeze(1), k)[:, 0] / vf.clamp(min=1e-6)
            mf = F.avg_pool2d(miss.float().unsqueeze(1), k)[:, 0]
            valid_l = vf >= c.min_valid_frac
            miss_l = (mf > 0.5) & ~valid_l
        else:
            zs, valid_l, miss_l = z, valid, miss
        out = (zs / self.far).clamp(0.0, 1.0)
        out = torch.where(miss_l, torch.ones_like(out), out)
        bad = ~valid_l & ~miss_l
        if c.invalid_fill == 'near':
            fill = torch.zeros_like(out)
        elif c.invalid_fill == 'hold' and self.prev_depth is not None:
            fill = self.prev_depth
        else:
            fill = torch.ones_like(out)
        out = torch.where(bad, fill, out)
        on = self.mask & self.depth_on
        if not bool(on.all()):                                                # ramping / depth_clean_frac: clean depth
            zc = z_clean if (Wr == self.W and Hr == self.H) else F.avg_pool2d(z_clean.unsqueeze(1), (Hr // self.H, Wr // self.W))[:, 0]
            out = torch.where(on.view(-1, 1, 1), out, (zc / self.far).clamp(0.0, 1.0))
            bad = bad & on.view(-1, 1, 1)
        self.last_bad = bad                         # (E,H,W) invalid pixels of this frame (the env's invalid mask)
        self.prev_depth = out
        return out
