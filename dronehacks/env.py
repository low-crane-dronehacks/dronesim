"""Vectorised DroneHacks course env (rsl_rl VecEnv): depth + rangefinder -> velocity commands, scored by the rules.

Dynamics are ``quad.TorchQuad``; depth, collisions and the rangefinder are exact ray/distance queries on the
course mesh (``render.MeshScene``). No physics engine, so thousands of envs run on one GPU.

Task (one episode = one competition run)
-----------------------------------------
Start from a reset-bank state drawn by SGS. Reward = competition points / 60, paid when they are earned:
checkpoints in order (4/7/10/7/7, x0.15 if reached after flying over a wall), and on landing inside the
pad's outer ring after the last checkpoint: + 20 x speed ratio + 5 if CP4 went through the window. An
unfinished run that ends gets + 7 x speed ratio for its last checkpoint. Episodes end on the pad landing,
collision (``collision_radius`` to a wall/tunnel/boulder), any other ground contact, leaving the net, or
timeout. Every touch is a crash here (the real rules allow touches with a +5 s penalty).

Return leg, three ways (``return_mode``):
  'handoff'  the RUN policy of a two-policy setup. action[4] > 0 = "complete": disarm. Landing on the pad parks
             the drone; disarming there finishes the run (finish points timed at the disarm, as the rules say:
             land AND disarm) and ends the episode. Disarming in the air is a crash; never disarming leaves the
             run unfinished. No flag observation. A separate return policy then flies the return leg.
  'return'   the RETURN policy: starts only from return-leg states (pad: released at once), 4 actions, reward
             = the x1.25 bonus on a completed return. Never flies the run.
  'joint'    (= ``return_phase=True``, older) one policy does both, described below.
Joint return leg: after the pad landing the episode CONTINUES with the drone parked.
action[4] > 0 is the return flag (terminal if given before the landing; once given it latches and is
observed). It releases the drone; landing within 2 ft of the launch mark and holding 3 s pays the x1.25
bonus, i.e. 0.25 x the run's points. The reward stays sparse: nothing else is paid on the way back.

Observations (rsl_rl groups)
----------------------------
``policy``   (E, 2, H, W)  depth stack (z-depth / far_clip, newest first) from the 72x50 deg OAK-D-like camera
``proprio``  (E, P)        body-frame velocity / v_max (3), rangefinder / 1.5 (1), roll, pitch (2), previous
                           action (4, or 5 + the latched return flag with the return leg); with ``pos_obs``
                           + course-frame position estimate / 3 m (NED, 3), sin / cos of the yaw estimate (2), VIO
                           tracking flag (1): the true pose, or a VIO-like estimate with realism ``vio_pos``
``critic``   (E, C)        privileged state for the value function only (position, velocity, attitude, rates,
                           thrusts, previous action, goal vector, route progress, clearance, ...)
Without ``pos_obs`` the actor gets NO position or goal information: it has to navigate from what it sees.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from tensordict import TensorDict

from dronehacks import perturb as cp
from dronehacks import rules as cs
from dronehacks.bank import BANK_LAUNCH, Bank, Route
from dronehacks.mesh import CourseMesh
from dronehacks.quad import LEE_MAX_YAW_RATE, TorchQuad
from dronehacks.realism import Realism, RealismCfg, concat_meshes, room_clutter
from dronehacks.render import MeshScene
from dronehacks.sgs import SGSSampler

ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets'))
CONTROL_DT = 0.1
ACTION_DIM = 4
KIND_NAMES = ('launch', 'body', 'near_goal', 'near_route', 'pad', 'return', 'climb', 'cross', 'final',
              'liftoff')  # == bank.BANK_*; 'liftoff' (9) = a launch cell started in the air (launch_liftoff_frac)
LIFTOFF_KIND = 9
CLEARANCE_OBS_MAX = 3.0              # critic clearance input saturates here (m)


@dataclass
class CourseEnvCfg:
    num_envs: int = 2048
    device: str = 'cuda:0'
    seed: int = 0
    assets_dir: str = ASSETS
    bank_file: str = 'bank_free_v3.npz'
    course_file: str = 'course.npz'          # geometry in assets_dir (course_scan.npz: the real course, scanned)
    # camera: OAK-D S2 mono pair (72 x 50 deg), downsampled
    depth_width: int = 128
    depth_height: int = 80
    hfov_deg: float = 72.0
    vfov_deg: float = 50.0
    far_clip: float = 5.0
    near_clip: float = 0.05
    frame_stack: int = 2
    v_max_start: Optional[Tuple[float, float, float]] = None  # speed curriculum: action scale anneals from this
    v_max_anneal_steps: int = 0          # ... to v_max over this many env steps (observations keep v_max_start's scale)
    critic_full: bool = False            # asymmetric critic: + next checkpoint (one-hot, body-frame vector to its target),
                                         # episode time, and the hidden randomized physics (mass, thrust eff, drag, motor tau,
                                         # actuation / depth latency, battery, velocity-loop ki)
    critic_depth: bool = False           # asymmetric critic: + CLEAN depth of the original geometry (walls, tunnel,
                                         # boulders and the net drawn SOLID) at the current pose, min-pooled to 10x16
    obs_pool: int = 1                    # >1: pool the policy depth by k (nearest VALID return per cell; a cell is
                                         # invalid only if < 25% of it is valid) -- very low-res input
    invalid_mask: bool = False           # policy image gets frame_stack extra channels: 1 = invalid depth pixel
    net_in_depth: bool = True            # the net is drawn as opaque room walls
    # downward VL53L1X-style rangefinder: 5 rays over 27 deg, nearest wins; reads wall tops/boulders under it
    rf_max: float = 4.0
    rf_min: float = 0.04
    rf_half_angle_deg: float = 13.5
    rf_noise_abs: float = 0.005          # sigma = abs + frac * range
    rf_noise_frac: float = 0.01
    # task
    collision_radius: float = 0.20
    max_episode_s: float = 45.0
    wall_top: float = cs.WALL_TOP_M
    arena_half: float = 3.0
    net_top: float = 3.0                 # net ceiling; above it = out of bounds
    ref_speed: float = 4.0               # m/s stand-in for the fastest team (reference time = geodesic / ref_speed)
    cp_margin: float = 0.15              # "whole drone crosses the line": centre must be this far past it
    cmd_speed_cap: float = 0.0           # >0: clamp the commanded |v| (m/s) like the deploy interface (eval)
    cmd_yaw_cap: float = 0.0             # >0: clamp the commanded yaw rate (rad/s)
    launch_yaw_centers_deg: Tuple[float, ...] = (-90.0, 0.0)  # real launch: facing the west net (-90) or up the corridor (0)
    launch_yaw_max_deg: float = 180.0      # keep only launch starts within this of the centre (180 = all headings)
    start_heading_max_deg: float = 180.0   # other starts: keep only those within this of the route direction there
    cp_shaping: float = 0.0              # potential-based shaping: reward per m of approach to the NEXT checkpoint's
                                         # target (30 cm past its line midpoint; the pad centre for the landing)
    skip_terminal: bool = False          # training: crossing a LATER checkpoint's gate (one was skipped) ends the episode as a crash
    touchdown_alt: float = 0.05          # centre height at ground contact (a parked drone sits at 0.08)
    land_max_speed: float = 1.0          # |vz| and |v_xy| limit for a valid landing, m/s
    # ground reaction: a drone that starts on the ground (launch pose, pad) cannot sink below rest_alt until it
    # has been above lift_alt; only then is ground contact terminal / a landing
    rest_alt: float = 0.08
    lift_alt: float = 0.3
    # return leg
    return_phase: bool = False           # = return_mode 'joint' (kept for older configs)
    return_mode: str = 'none'            # 'none' | 'handoff' | 'return' | 'joint' (see module docstring)
    return_radius: float = cs.RETURN_RADIUS_M
    return_hold_s: float = 3.0
    return_start_points: float = 50.0    # run points assumed banked for return-leg bank starts
    land_max_tilt: float = 0.35          # rad, roll/pitch limit for a valid return touchdown
    # layout perturbation (perturb.py): K pre-built layouts stacked in one BVH, one drawn per reset
    perturb: float = 0.0                 # max severity in [0,1]; 0 = the exact course
    perturb_variants: int = 16
    anchor_wall: str = ''                # perturbation prior: this centre wall barely moves; walls touching it stay touching
    gate_extend: bool = False            # generous checkpoints: every line extended to the (perturbed) walls/net
    launch_idle_steps: int = 0           # launch starts: the drone sits on the floor this many steps while the policy
                                         # runs (its commands ignored), e.g. to pre-load the GRU / frame stack
    idle_prev_action: str = 'policy'     # what the policy sees as its previous action while idling: policy | zero
    cp0_wall: str = ''                   # as-built scoring: checkpoint 0 = line from the NW corner to this wall's tip ('wall3')
    perturb_scale: float = 1.0          # multiplies the severity-1 wall/tunnel/boulder/gate ranges (2.5: walls +-0.375 m, +-10 deg)
    window_gate: Optional[Tuple[float, float, float, float]] = None   # (n0, e0, n1, e1): the window checkpoint line
                                         # moved onto a re-laid window (e.g. course_w0scan.npz); crossing toward -E
    course_mods: str = ''                # as-built changes, e.g. 'drop:wall5;rot:wall3:15:45:south' (perturb.apply_mods)
    extra_boulders: Tuple[int, int] = (0, 0)  # perturbed layouts: + n ~ U[lo, hi] random low-poly boulders
    perturb_mode: str = 'fixed'          # 'fixed': all K at `perturb`; 'spread': 0..perturb, layout 0 nominal
    perturb_seed: int = 0
    perturb_ramp_steps: int = 0          # 'spread': ramp allowed severity over this many env steps
    perturb_start_clearance: float = 0.15
    # course pack (coursepack.py): train on its K generated courses instead of the real course. Each bank cell
    # belongs to one course and an env flies that course; '' = the real course (assets/course.npz etc.)
    course_pack: str = ''
    # plant
    v_max: tuple = (5.0, 3.0, 2.0)       # forward, right, down (m/s), yaw-aligned body frame
    yaw_rate_max: float = LEE_MAX_YAW_RATE
    plant_params: Optional[dict] = None  # nominal airframe override (a measured drone), see quad.PLANT_PARAMS
    # real-world effects (realism.py); None = the ideal sim the policies were trained in
    realism: Optional[RealismCfg] = None
    #: actor proprio += position / 3 m, sin/cos yaw, VIO-ok flag (6). True pose unless realism has vio_pos. Changes
    #: the proprio size: new runs only.
    pos_obs: bool = False
    # SGS
    sgs_target: float = 0.5
    sgs_concentration: float = 1.0
    sgs_history: int = 100
    sgs_temperature: float = 2.0
    sgs_eps: float = 1e-4
    sgs_uniform: bool = False            # ablation: uniform bank sampling
    launch_frac: float = 0.0             # share of training resets forced onto a launch-mark start (0 = SGS alone)
    launch_liftoff_frac: float = 0.0     # of those, the share started just after takeoff: in the air over the mark
    liftoff_alt: Tuple[float, float] = (0.15, 0.7)     # m above the floor
    liftoff_vz: Tuple[float, float] = (0.0, 0.8)       # climb rate, m/s up
    liftoff_vxy: float = 0.2             # horizontal speed sigma, m/s
    # logging
    metrics_path: Optional[str] = None
    metrics_interval_s: float = 30.0


class CourseEnv:
    """rsl_rl VecEnv (duck-typed)."""

    OUTCOMES = ('timeout', 'success', 'collision', 'floor', 'over_wall', 'oob', 'early_flag')

    def __init__(self, cfg: CourseEnvCfg) -> None:
        self.cfg = cfg
        mode = cfg.return_mode if cfg.return_mode != 'none' else ('joint' if cfg.return_phase else 'none')
        if mode not in ('none', 'handoff', 'return', 'joint'):
            raise ValueError(f'unknown return_mode {mode!r}')
        self._mode = mode
        self._rp = mode != 'none'                                     # phase machinery (parking etc.) in use
        self.num_actions = ACTION_DIM + 1 if mode in ('joint', 'handoff') else ACTION_DIM
        e, dev = cfg.num_envs, cfg.device
        self.num_envs, self.device = e, torch.device(dev)
        self.max_episode_length = int(round(cfg.max_episode_s / CONTROL_DT))
        self.episode_length_buf = torch.zeros(e, dtype=torch.long, device=dev)
        torch.manual_seed(cfg.seed)
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(cfg.seed)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)      # noqa: E731

        # ── geometry: K layouts (1 = the exact course) stacked into one mesh ─────────
        rc = cfg.realism
        if isinstance(rc, dict):
            rc = cfg.realism = RealismCfg(**rc)
        if rc is not None and rc.vio_pos and not cfg.pos_obs:
            raise ValueError('realism vio_pos models the position input: it needs CourseEnvCfg.pos_obs (--pos-obs)')
        self._pack = bool(cfg.course_pack)
        if self._pack:
            # a pack is K generated courses, each with its own gates / pad / mark / route / bank. Everything that
            # edits or perturbs THE real course has no meaning here: refuse it rather than silently ignore it
            if mode != 'none':
                raise ValueError(f'course_pack supports return_mode none only (got {mode!r}): packs have no '
                                 'return-leg banks')
            bad = [n for n, on in (('perturb', cfg.perturb > 0), ('course_mods', bool(cfg.course_mods)),
                                   ('gate_extend', cfg.gate_extend), ('cp0_wall', bool(cfg.cp0_wall)),
                                   ('anchor_wall', bool(cfg.anchor_wall)),
                                   ('extra_boulders', tuple(cfg.extra_boulders) != (0, 0)),
                                   ('course_file', cfg.course_file != 'course.npz'),
                                   ('window_gate', bool(cfg.window_gate)),
                                   ('arena_half', cfg.arena_half != 3.0),
                                   ('launch_yaw_max_deg', cfg.launch_yaw_max_deg < 180)) if on]
            if bad:
                raise ValueError(f'course_pack cannot be combined with {", ".join(bad)}: those edit or perturb the '
                                 'real course (pack courses are fixed; each has its own net size and launch heading)')
            from dronehacks.coursepack import load_pack, resolve_pack
            pack = load_pack(resolve_pack(cfg.course_pack, cfg.assets_dir))
            lays = [cp.Layout(0.0, c.mesh, c.gates, np.asarray(c.pad_center, np.float64),
                              np.asarray(c.mark, np.float64), np.asarray(c.goal, np.float64), {}) for c in pack]
            if any(len(lay.gates) != len(cs.GATES) for lay in lays):
                raise ValueError(f'course_pack: every course needs {len(cs.GATES)} gates (rules.GATES layout)')
            self.pack_seeds = [c.seed for c in pack]
            self.layout_info = {'course_pack': cfg.course_pack, 'seeds': self.pack_seeds}
            mark_xy = None
            net_halves = [np.asarray(c.arena_half, np.float64) + 0.05 for c in pack]   # arena_half = net - 0.05
        else:
            course = CourseMesh.load(os.path.join(cfg.assets_dir, cfg.course_file))
            route = Route.load(os.path.join(cfg.assets_dir, 'route.npz'))
            with open(os.path.join(cfg.assets_dir, 'scene.json')) as f:
                mark_xy = np.asarray(json.load(f)['landmarks']['launch_mark_ned'][:2], np.float64)
            K = max(1, int(cfg.perturb_variants)) if cfg.perturb > 0 else 1
            if cfg.perturb_mode not in ('fixed', 'spread'):
                raise ValueError("perturb_mode must be 'fixed' or 'spread'")
            sev = ([cfg.perturb] * K if cfg.perturb_mode == 'fixed' else
                   [cfg.perturb * k / max(K - 1, 1) for k in range(K)]) if K > 1 else [0.0]
            if rc is not None and rc.net and K == 1:          # nominal course, but K different rooms behind the net
                K = max(1, rc.clutter_variants)
                sev = [0.0] * K
            variants = cp.build_variants(
                course, self._gates(), cs.PAD_CENTER, mark_xy, np.asarray(route.goal, np.float64), severities=sev,
                seed=cfg.perturb_seed, launch_xy=mark_xy, extra_boulders=tuple(cfg.extra_boulders), mods=cfg.course_mods,
                scale=cfg.perturb_scale, cp0_wall=cfg.cp0_wall, anchor_wall=cfg.anchor_wall, gate_extend=cfg.gate_extend,
                clearance_factory=lambda m: MeshScene(m, 1, device='cpu').distance_at)
            lays = variants.layouts
            self.layout_info = {'severities': [float(x) for x in variants.severities], 'attempts': variants.attempts,
                                'rejected': variants.rejected}
            net_halves = [3.05] * len(lays)
        self.layouts = lays                               # (video.py draws the flown layout, e.g. as-built mods)
        self._n_lay = len(lays)                           # distinct layouts (pack: courses), before the net copy
        # ramped training with the see-through net: every layout twice, net drawn (effects off) / see-through
        self._net_split = len(lays) if (rc is not None and rc.net and rc.ramp_steps > 0) else 0
        if self._net_split:
            lays = lays + lays
            net_halves = net_halves + net_halves

        def stacked(grp):
            return cp.stack_variants([lay.mesh.select(grp) for lay in lays])

        w, h = cfg.depth_width, cfg.depth_height
        fx = 0.5 * w / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        fy = 0.5 * h / math.tan(math.radians(cfg.vfov_deg) / 2.0)
        self._net_scene = None
        if rc is not None and rc.net:                     # see-through net: depth sees the room behind it
            rng = np.random.default_rng(rc.seed)
            mesh, offs = cp.stack_variants([
                lay.mesh.select(('floor', 'wall', 'tunnel', 'boulder', 'net')) if i < self._net_split else
                concat_meshes(lay.mesh.select(('floor', 'wall', 'tunnel', 'boulder')), room_clutter(rng, rc, net_halves[i]))
                for i, lay in enumerate(lays)])
        else:
            groups = ('floor', 'wall', 'tunnel', 'boulder') + (('net',) if cfg.net_in_depth else ())
            mesh, offs = stacked(groups)
        s_r = rc.render_scale if rc is not None and rc.depth else 1
        self._depth = MeshScene(mesh, e, w * s_r, h * s_r, fx * s_r, fy * s_r, cfg.far_clip, cfg.near_clip,
                                depth_type='z', device=dev)
        if rc is not None and rc.net and rc.depth and (rc.net_return_prob > 0 or rc.net_return_prob_max > 0):
            self._net_scene = MeshScene(stacked(('net',))[0], e, w * s_r, h * s_r, fx * s_r, fy * s_r, cfg.far_clip,
                                        cfg.near_clip, depth_type='z', device=dev)
        self._obst = MeshScene(stacked(('wall', 'tunnel', 'boulder'))[0], e, device=dev, bvh_leaf_size=1)  # point queries
        self._crit_scene = (MeshScene(stacked(('floor', 'wall', 'tunnel', 'boulder', 'net'))[0], e, w, h, fx, fy,
                                      cfg.far_clip, cfg.near_clip, depth_type='z', device=dev)
                            if cfg.critic_depth else None)
        # per-layout scoring geometry, gathered per env at reset (_assign_variant)
        self._v_off, self._v_sev = t(offs), t([lay.severity for lay in lays])
        self._v_goal = t([lay.goal for lay in lays])
        self._v_pad = t([lay.pad_center for lay in lays])
        self._v_mark = t([lay.mark for lay in lays])
        # launch starts sit on the taped mark: a layout whose mark moved (a 'shift:...:mark' course mod, e.g. the centre
        # wall moved north) moves its launch-type bank starts by the same offset
        self._v_lshift = t([np.asarray(lay.mark, np.float64) - (lay.mark if mark_xy is None else mark_xy) for lay in lays])
        self._shifted_launch = not self._pack and any(m.startswith('shift:') and m.endswith(':mark') for m in (cfg.course_mods or '').split(';'))
        self._v_gP0 = t([[g['A'] for g in lay.gates] for lay in lays])             # (K,G,2)
        self._v_gt = t([[g['t'] for g in lay.gates] for lay in lays])
        self._v_gn = t([[g['n'] for g in lay.gates] for lay in lays])
        self._v_gL = t([[g['L'] for g in lay.gates] for lay in lays])              # (K,G)
        G = len(cs.GATES)
        self._var = torch.zeros(e, dtype=torch.long, device=dev)
        self._off = self._v_off[0].repeat(e, 1)                                    # (E,3) layout offset in the stack
        self._goal = self._v_goal[0].repeat(e, 1)
        self._pad_c = self._v_pad[0].repeat(e, 1)
        self._mark = self._v_mark[0].repeat(e, 1)
        self._gP0 = self._v_gP0[0].view(G, 1, 1, 2).repeat(1, 1, e, 1)             # (G,1,E,2)
        self._gt = self._v_gt[0].view(G, 1, 1, 2).repeat(1, 1, e, 1)
        self._gn = self._v_gn[0].view(G, 1, 1, 2).repeat(1, 1, e, 1)
        self._gL = self._v_gL[0].view(G, 1, 1).repeat(1, 1, e)                     # (G,1,E)
        self._env_steps = 0

        # ── reset bank + route ─────────────────────────────────────────────────
        if self._pack:
            # one bank / route per course, concatenated; each cell carries its course id, each course its own slice
            # of the route arrays [r0, r1) so the progress feature stays on the env's own course
            bank = Bank.concat(*[c.bank for c in pack])
            nr = np.array([len(c.route.points) for c in pack])
            r0 = np.concatenate([[0], np.cumsum(nr)[:-1]])
            route = Route(np.concatenate([c.route.points for c in pack]), np.concatenate([c.route.s for c in pack]),
                          np.zeros(3, np.float32))
            course_id = np.concatenate([np.full(len(c.bank), k) for k, c in enumerate(pack)])
            ridx = np.concatenate([r0[k] + np.minimum(np.searchsorted(c.route.s, c.bank.route_s), nr[k] - 1)
                                   for k, c in enumerate(pack)])
            sector = np.concatenate([c.sector for c in pack])
            self._v_r0, self._v_r1 = t(r0, torch.long), t(r0 + nr, torch.long)
            self._v_rlen = t([c.route.length for c in pack])
            self._v_ggeo = t([c.gate_geo for c in pack])                              # (K,G)
            self._v_gcp = t([[g['cp'] for g in c.gates] for c in pack], torch.long)    # (K,G)
            self._v_arena = t([c.arena_half for c in pack])                            # (K,2)
            self._r0, self._r1 = self._v_r0[0].repeat(e), self._v_r1[0].repeat(e)
            self._rlen = self._v_rlen[0].repeat(e)
            self._arena = self._v_arena[0].repeat(e, 1)
        else:
            bank = Bank.load(os.path.join(cfg.assets_dir, cfg.bank_file))
        if mode in ('handoff', 'return'):                 # run policy: run starts only; return policy: return starts
            keep = (bank.phase == 0) if mode == 'handoff' else (bank.phase > 0)
            if not keep.any():
                raise ValueError(f"return_mode {mode!r}: bank {cfg.bank_file} has no {'run' if mode == 'handoff' else 'return-leg'} starts")
            bank = Bank(*(getattr(bank, k)[keep] for k in ('pos', 'yaw', 'route_s', 'kind', 'vel', 'roll', 'pitch', 'geo', 'phase')))
        if cfg.launch_yaw_max_deg < 180 or cfg.start_heading_max_deg < 180:
            wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi                      # noqa: E731
            launch = bank.kind == 0
            keep = np.ones(len(bank.kind), bool)
            near = np.zeros(len(bank.kind), bool)
            for c in cfg.launch_yaw_centers_deg:
                near |= np.abs(wrap(bank.yaw - np.radians(c))) <= np.radians(cfg.launch_yaw_max_deg)
            keep &= ~launch | near
            # (a pack's route is every course's route concatenated: index it per course, not by arc length)
            idx = ridx if self._pack else np.clip(np.searchsorted(route.s, bank.route_s), 0, len(route.s) - 1)
            tan = route.tangent()[idx]
            along = np.arctan2(tan[:, 1], tan[:, 0])                                 # NED heading of the route there
            keep &= launch | (np.abs(wrap(bank.yaw - along)) <= np.radians(cfg.start_heading_max_deg))
            print(f'[env] heading filter: kept {int(keep[launch].sum())}/{int(launch.sum())} launch, '
                  f'{int(keep[~launch].sum())}/{int((~launch).sum())} other starts', flush=True)
            bank = Bank(*(getattr(bank, k)[keep] for k in ('pos', 'yaw', 'route_s', 'kind', 'vel', 'roll', 'pitch', 'geo', 'phase')))
            if self._pack:                                # the per-cell course arrays follow the bank
                course_id, ridx, sector = course_id[keep], ridx[keep], sector[keep]
        if self._pack:
            self._bank_course, self._bank_ridx = t(course_id, torch.long), t(ridx, torch.long)
            self._course = torch.zeros(e, dtype=torch.long, device=dev)          # the course each env flies
        self._route_pts, self._route_s = t(route.points), t(route.s)
        self._route_len = float(route.length)
        self._bank_pos, self._bank_yaw = t(bank.pos), t(bank.yaw)
        self._bank_s, self._bank_kind = t(bank.route_s), t(bank.kind, torch.long)
        self._bank_vel, self._bank_roll, self._bank_pitch = t(bank.vel), t(bank.roll), t(bank.pitch)
        self._bank_geo = t(bank.geo)
        self._bank_phase = t(bank.phase, torch.long)
        if torch.isnan(self._bank_geo).any():
            raise ValueError('the bank needs geodesic distances (bank_free_v2.npz or newer)')
        if self._rp:
            if mode == 'joint' and not (self._bank_phase > 0).any():
                raise ValueError('return_phase needs a bank with return-leg starts (bank_return_v2.npz)')
            self._land_factor = torch.ones(e, device=dev)                 # handoff: checkpoint-5 factor at landing
            self._phase = torch.zeros(e, dtype=torch.long, device=dev)     # 0 run, 1 parked on pad, 2 returning, 3 parked at mark
            self._start_phase = torch.zeros(e, dtype=torch.long, device=dev)
            self._fin_run = torch.zeros(e, dtype=torch.bool, device=dev)
            self._park_pos = torch.zeros(e, 3, device=dev)
            self._park_yaw = torch.zeros(e, device=dev)
            self._hold = torch.zeros(e, dtype=torch.long, device=dev)
            self._hold_steps = int(round(cfg.return_hold_s / CONTROL_DT))
        self._launch_cells = torch.nonzero(self._bank_kind == 0).flatten()
        self.sgs = SGSSampler(len(bank), history=cfg.sgs_history, target=cfg.sgs_target,
                              concentration=cfg.sgs_concentration, eps=cfg.sgs_eps,
                              temperature=cfg.sgs_temperature, device=dev)

        # ── plant + per-env state ──────────────────────────────────────────────
        self.plant = TorchQuad(e, dev, v_max=cfg.v_max, yaw_rate_max=cfg.yaw_rate_max, params=cfg.plant_params)
        self.plant.speed_cap = cfg.cmd_speed_cap or None
        self.plant.yaw_cap = cfg.cmd_yaw_cap or None
        self.real = None
        self._pose_hist = None
        if rc is not None:
            self.real = Realism(rc, e, dev, CONTROL_DT, self.plant.mass, self.plant.params['inertia'], (h, w), fx,
                                cfg.far_clip)
            self.real.configure_plant(self.plant)
        self._cell = torch.zeros(e, dtype=torch.long, device=dev)
        self._liftoff = torch.zeros(e, dtype=torch.bool, device=dev)
        self._route_idx = torch.zeros(e, dtype=torch.long, device=dev)
        self._prev_action = torch.zeros(e, self.num_actions, device=dev)
        self._idle = torch.zeros(e, dtype=torch.long, device=dev)            # launch_idle_steps left
        self._idle_pose = torch.zeros(e, 6, device=dev)                      # pos(3), yaw, roll, pitch
        k = max(1, int(cfg.obs_pool))
        if h % k or w % k:
            raise ValueError(f'obs_pool {k} must divide the depth image {h}x{w}')
        self._stack = torch.ones(e, cfg.frame_stack, h // k, w // k, device=dev)
        self._mstack = torch.zeros(e, cfg.frame_stack, h // k, w // k, device=dev) if cfg.invalid_mask else None
        self._stack_pos = 0
        n = cfg.frame_stack
        self._gather = torch.tensor([[(p - k) % n for k in range(n)] for p in range(n)], dtype=torch.long, device=dev)
        self._prime_ids = torch.arange(e, device=dev)
        self._prime_pending = True
        self._path = torch.zeros(e, device=dev)            # path length flown this episode, m
        self._last_pos = torch.zeros(e, 3, device=dev)
        self._eval_q: Optional[torch.Tensor] = None      # see set_eval_queue
        self._eval_ptr = torch.zeros(e, dtype=torch.long, device=dev)
        self.finished: list = []
        self.finished_points: list = []
        self.finished_fin: list = []
        self._v_scale = torch.tensor(cfg.v_max_start or cfg.v_max, device=dev)   # obs scale fixed through a speed anneal
        self._grounded = torch.zeros(e, dtype=torch.bool, device=dev)  # resting on the ground, not yet lifted off
        self.plant.grounded, self.plant.ground_alt = self._grounded, cfg.rest_alt
        self._early = torch.zeros(e, dtype=torch.bool, device=dev)    # return flag given before the pad landing
        self._rf = MeshScene(stacked(('floor', 'wall', 'tunnel', 'boulder'))[0], 1, far_clip=cfg.rf_max, device=dev)
        a = math.radians(cfg.rf_half_angle_deg)
        self._rf_dirs = t([[0.0, 0.0, 1.0]] + [[math.sin(a) * sx, math.sin(a) * sy, math.cos(a)]
                                               for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))])   # body FRD

        # ── competition scoring state ──────────────────────────────────────────
        self._ep_points = torch.zeros(e, device=dev)
        self._next_cp = torch.zeros(e, dtype=torch.long, device=dev)       # checkpoint being worked toward (4 = landing)
        self._start_cp = torch.zeros(e, dtype=torch.long, device=dev)
        self._cleared = torch.zeros(e, 5, dtype=torch.bool, device=dev)
        self._flag = torch.zeros(e, dtype=torch.bool, device=dev)          # flew over a wall since the last checkpoint
        self._ever_over = torch.zeros(e, dtype=torch.bool, device=dev)
        self._via = torch.zeros(e, dtype=torch.bool, device=dev)           # CP4 went through the window
        self._ncl = torch.zeros(e, dtype=torch.long, device=dev)
        self._t_last = torch.zeros(e, device=dev)
        self._shape_d = torch.zeros(e, device=dev)
        self._shape_cp = torch.full((e,), -1, dtype=torch.long, device=dev)
        self._start_geo = torch.zeros(e, device=dev)
        self._last_geo = torch.zeros(e, device=dev)
        if self._pack:
            self._gcp = self._v_gcp[0].view(G, 1).repeat(1, e)                     # (G,E): per-course gate cps
        else:
            with open(os.path.join(cfg.assets_dir, 'scoring.json')) as f:
                self._gate_geo = t(json.load(f)['gate_geodesic_m'])
            self._gcp = t(cs.GATE_CP, torch.long)
        self._cp_pts = t(cs.CP_POINTS)
        self._pad_r = float(cs.PAD_RADIUS_M)
        self._gate_tol = 0.1                                                     # slack along the line, m
        self._bank_sector = t(sector if self._pack else cs.sector_index(bank.pos[:, :2]), torch.long)

        self._reset_idx(torch.arange(e, device=dev))
        self._reset_stats()
        self._last_log = time.time()
        self.proprio_dim = int(self._proprio().shape[1])
        self.critic_dim = int(self._critic(self._clearance()).shape[1])

    # ── rsl_rl VecEnv ───────────────────────────────────────────────────────

    def get_observations(self) -> TensorDict:
        return self._observations()

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        a = torch.clamp(actions, -1.0, 1.0)
        idle = (self._idle > 0) if cfg.launch_idle_steps > 0 else None
        a_policy = a
        if idle is not None:                                     # idling on the floor: the command is ignored
            a = torch.where(idle.unsqueeze(-1), torch.zeros_like(a), a)
        early = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        disarm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._rp:
            ph0 = self._phase.clone()                             # phase the action was chosen in
            if self._mode in ('joint', 'handoff'):
                flag = a[:, 4] > 0
                early = (ph0 == 0) & flag                         # flag / disarm before the pad landing: terminal
                self._early = early
                if self._mode == 'joint':
                    go = (ph0 == 1) & flag                        # return flag, given while parked on the pad
                    self._phase[go] = 2
                    self._grounded[go] = True
                else:
                    disarm = (ph0 == 1) & flag                    # handoff: "complete" = disarm on the pad
            ph0 = self._phase.clone()
        self.plant.substep_pos = []
        seq = None
        if self.real is not None:
            self.real.update_thrust(self.plant, self.episode_length_buf.float() * CONTROL_DT)
            seq = self.real.substep_actions(a[:, :ACTION_DIM], self.plant.decimation)
            if self.real.cfg.depth:
                if self._pose_hist is None:
                    self._pose_hist = torch.cat([self.plant.pos_ned, self.plant.rpy_ned()], -1).unsqueeze(0).repeat(30, 1, 1)
                self.plant.pose_trace = []
        if seq is None:
            self.plant.step_control(a[:, :ACTION_DIM])
        else:
            self.plant.step_control_seq(seq)
        sub = torch.stack(self.plant.substep_pos)                 # (S,E,3) NED after every 100 Hz substep
        self.plant.substep_pos = None
        if idle is not None and bool(idle.any()):                 # hold the idling drones exactly on their start pose
            iid = torch.nonzero(idle).flatten()
            ip = self._idle_pose[iid]
            self.plant.reset_idx(iid, ip[:, :3], ip[:, 3], torch.zeros_like(ip[:, :3]), ip[:, 4], ip[:, 5])
            sub[:, iid] = ip[:, :3].unsqueeze(0)
            self._idle[iid] -= 1
        if self.plant.pose_trace is not None:                     # depth latency: keep ~0.3 s of substep poses
            self._pose_hist = torch.cat([self._pose_hist, torch.stack(self.plant.pose_trace)])[-30:]
            self.plant.pose_trace = None
        if self._rp:
            parked = (ph0 == 1) | (ph0 == 3)                      # disarmed on the ground: hold the pose exactly
            sub[:, parked] = self._park_pos[parked]
            pid = torch.nonzero(parked).flatten()
            if len(pid):
                self.plant.reset_idx(pid, self._park_pos[pid], self._park_yaw[pid])
        if idle is not None:
            pa = a_policy if cfg.idle_prev_action == 'policy' else torch.zeros_like(a_policy)
            self._prev_action = torch.where(idle.unsqueeze(-1), pa, a).clone()
        else:
            self._prev_action = a.clone()
        self.episode_length_buf += 1
        self._env_steps += self.num_envs
        if cfg.v_max_anneal_steps > 0 and cfg.v_max_start is not None:
            f = min(1.0, self._env_steps / cfg.v_max_anneal_steps)
            if f < 1.0 or not getattr(self, '_v_anneal_done', False):
                v = [a + (b - a) * f for a, b in zip(cfg.v_max_start, cfg.v_max)]
                self.plant.v_max.copy_(torch.tensor(v, device=self.plant.v_max.device))
                self._v_anneal_done = f >= 1.0
        self._advance_route_idx()

        pts = torch.cat([self._last_pos.unsqueeze(0), sub], dim=0)
        prev = pts[:-1]                                           # position before each substep
        self._path += torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=-1).sum(0)
        self._last_pos = sub[-1].clone()

        # Checks run over ALL substeps: at 5 m/s a control step is 0.5 m, enough to jump a wall.
        alt = -sub[..., 2]
        self._grounded &= ~(alt > cfg.lift_alt).any(0)
        # min_obstacle_distance returns a persistent buffer: clone per substep or every row aliases the last
        # one launch over every substep, search capped just above the radius (the collision test needs nothing more)
        S_ = sub.shape[0]
        clearance = self._obst.min_distance_batch(sub + self._off.unsqueeze(0), cfg.collision_radius + 0.3).view(S_, -1).amin(0)
        collision = clearance < cfg.collision_radius
        if self._pack:                                            # per-course net size
            oob = (sub[..., :2].abs() > self._arena).any(-1).any(0) | (alt > cfg.net_top).any(0)
        else:
            oob = (sub[..., :2].abs() > cfg.arena_half).any(-1).any(0) | (alt > cfg.net_top).any(0)
        pts_step, success, floor = self._score_step(sub, prev, alt)
        if self._mode == 'handoff':
            pts_step = pts_step + self._disarm(disarm)
            collision = collision & ~(self._phase == 1)                       # parked on the pad
            success = disarm
            crash = (collision | floor | oob | early) & ~success
            sgs_success = success
        elif self._rp:
            ret_ok, ret_floor, ret_r = self._return_step(sub, prev, alt, ph0)
            collision = collision & ~((self._phase == 1) | (self._phase == 3))    # disarmed on the ground
            floor = floor | ret_floor
            sgs_success = torch.where(self._start_phase == 0, self._fin_run, ret_ok)
            crash = (collision | floor | oob | early) & ~ret_ok
            success = ret_ok                                    # with a return leg, success = completed return
        else:
            crash = (collision | floor | oob) & ~success
            sgs_success = success
        terminated = success | crash
        truncated = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        done = terminated | truncated
        # unfinished run: 7 x (reference time to the last checkpoint / your time to it); 0 if none reached
        unf = done & ~success & (self._ncl > 0) & ((self._phase <= 1) if self._mode == 'handoff' else
                                                   (self._phase == 0) if self._rp else True)
        tref = ((self._start_geo - self._last_geo) / cfg.ref_speed).clamp(min=0.5)
        extra = cs.UNFINISHED_SPEED_POINTS * torch.clamp(tref / self._t_last.clamp(min=0.1), max=1.0) * unf
        self._ep_points += extra
        reward = (pts_step + extra) / cs.MAX_RUN_POINTS
        if cfg.cp_shaping > 0:
            reward = reward + cfg.cp_shaping * self._cp_shaping(sub[-1, :, :2])
        if self._mode in ('joint', 'return'):
            reward = reward + ret_r

        extras = {'time_outs': truncated.clone()}
        ids = torch.nonzero(done).flatten()
        if len(ids):
            sid = ids[~self._liftoff[ids]]                # liftoff episodes would make launch cells look easy
            self.sgs.record(self._cell[sid], sgs_success[sid])
            self._accumulate(ids, sgs_success, collision, floor, oob, truncated, success)
            if self._eval_q is not None:
                self._record_finished(ids, success, collision, floor, oob)
            self._reset_idx(ids)
        self._maybe_log()
        return self._observations(), reward, done.to(torch.long), extras

    # ── scoring ─────────────────────────────────────────────────────────────

    def _score_step(self, sub: torch.Tensor, prev: torch.Tensor, alt: torch.Tensor):
        """Advance every env's checkpoint state over this step's substeps -> (points, landed, bad ground contact).

        Checkpoint k clears when the centre crosses its line (``cp_margin`` past it, within its extent) in the
        required direction while it is the NEXT one. Flying over a wall cuts the next checkpoint to 15%. Ground
        contact is the landing if inside the pad's outer ring, after CP4 and slow; otherwise a crash."""
        cfg = self.cfg
        S, E = sub.shape[0], sub.shape[1]
        ar = torch.arange(E, device=self.device)
        run = (self._phase == 0) if self._rp else torch.ones(E, dtype=torch.bool, device=self.device)
        t_now = self.episode_length_buf.float() * CONTROL_DT
        rel = sub[..., :2].unsqueeze(0) - self._gP0                       # (G,S,E,2)
        prel = prev[..., :2].unsqueeze(0) - self._gP0
        d = (rel * self._gn).sum(-1) - cfg.cp_margin
        dp = (prel * self._gn).sum(-1) - cfg.cp_margin
        u = (rel * self._gt).sum(-1)
        crossed = ((dp < 0) & (d >= 0) & (u >= -self._gate_tol) & (u <= self._gL + self._gate_tol)).any(1) & run   # (G,E)

        over = (alt > cfg.wall_top).any(0) & run
        self._flag |= over
        self._ever_over |= over
        nc = self._next_cp
        gcp = self._gcp if self._gcp.dim() == 2 else self._gcp.unsqueeze(1)   # (G,E) with a pack, else (G,1)
        skip = (crossed & (gcp > nc.unsqueeze(0))).any(0) if cfg.skip_terminal else torch.zeros_like(run)
        hit = crossed & (gcp == nc.unsqueeze(0))                           # (G,E): a gate of the NEXT checkpoint
        cleared = hit.any(0)
        factor = torch.where(self._flag, torch.full_like(t_now, cs.OVER_WALL_FACTOR), torch.ones_like(t_now))
        pts = self._cp_pts[nc.clamp(max=4)] * factor * cleared
        gidx = hit.long().argmax(0)
        # through the window, not the main-route gate of the same checkpoint (gate 3 on the real course)
        self._via |= cleared & hit[cs.WINDOW_GATE] & ~hit[:cs.WINDOW_GATE].any(0)
        ggeo = self._v_ggeo[self._course, gidx] if self._pack else self._gate_geo[gidx]
        self._last_geo = torch.where(cleared, ggeo, self._last_geo)
        self._t_last = torch.where(cleared, t_now, self._t_last)
        # == _cleared[ar[cleared], nc[cleared]] = True without the boolean-mask host sync (a cleared env's nc <= 3)
        self._cleared[ar, nc.clamp(max=4)] |= cleared
        self._ncl += cleared.long()
        self._flag &= ~cleared
        nc = nc + cleared.long()
        self._next_cp = nc

        touch = alt <= cfg.touchdown_alt                                   # (S,E)
        has = touch.any(0) & run
        idx = touch.long().argmax(0)
        ps, pp = sub[idx, ar], prev[idx, ar]
        v = (ps - pp) / (CONTROL_DT / S)
        slow = (v[:, 2].abs() <= cfg.land_max_speed) & (torch.linalg.vector_norm(v[:, :2], dim=-1) <= cfg.land_max_speed)
        in_pad = torch.linalg.vector_norm(ps[:, :2] - self._pad_c, dim=-1) <= (self._pad_r - cfg.cp_margin)
        landed = has & in_pad & (nc == 4) & slow
        factor = torch.where(self._flag, torch.full_like(t_now, cs.OVER_WALL_FACTOR), torch.ones_like(t_now))
        tref = (self._start_geo / cfg.ref_speed).clamp(min=1.0)
        fin = (self._cp_pts[4] * factor
               + cs.FINISH_SPEED_POINTS * torch.clamp(tref / t_now.clamp(min=0.1), max=1.0)
               + cs.WINDOW_BONUS * self._via.float())
        if self._mode == 'handoff':                 # landing alone is not a finish: the disarm completes it
            self._land_factor = torch.where(landed, factor, self._land_factor)
        else:
            pts = pts + fin * landed
            self._cleared[:, 4] |= landed
            self._ncl += landed.long()
            self._t_last = torch.where(landed, t_now, self._t_last)
        self._ep_points += pts
        if self._rp:
            self._fin_run |= landed
            self._phase[landed] = 1                                    # park: the episode goes on (return leg)
            lid = torch.nonzero(landed).flatten()
            if len(lid):
                self._park_pos[lid] = torch.cat([ps[lid, :2], torch.full_like(ps[lid, :1], -cfg.rest_alt)], dim=1)
                self._park_yaw[lid] = self.plant.rpy_ned()[lid, 2]
                self.plant.reset_idx(lid, self._park_pos[lid], self._park_yaw[lid])
            return pts, landed & False, (has & ~landed) | skip
        return pts, landed, (has & ~landed) | skip

    def _disarm(self, disarm: torch.Tensor) -> torch.Tensor:
        """handoff: finish points for envs that disarmed on the pad this step, timed at the disarm."""
        cfg = self.cfg
        ar = torch.arange(self.num_envs, device=self.device)
        t_now = self.episode_length_buf.float() * CONTROL_DT
        tref = (self._start_geo / cfg.ref_speed).clamp(min=1.0)
        fin = (self._cp_pts[4] * self._land_factor
               + cs.FINISH_SPEED_POINTS * torch.clamp(tref / t_now.clamp(min=0.1), max=1.0)
               + cs.WINDOW_BONUS * self._via.float()) * disarm
        self._cleared[ar[disarm], 4] = True
        self._ncl += disarm.long()
        self._t_last = torch.where(disarm, t_now, self._t_last)
        self._ep_points += fin
        return fin

    def _return_step(self, sub, prev, alt, ph0):
        """Return-leg bookkeeping (``ph0`` = phase while this step's action ran) -> (returned, bad contact, reward)."""
        cfg = self.cfg
        E, S = sub.shape[1], sub.shape[0]
        ar = torch.arange(E, device=self.device)
        flying = ph0 == 2
        touch = alt <= cfg.touchdown_alt
        has = touch.any(0) & flying
        idx = touch.long().argmax(0)
        ps, pp = sub[idx, ar], prev[idx, ar]
        v = (ps - pp) / (CONTROL_DT / S)
        slow = (v[:, 2].abs() <= cfg.land_max_speed) & (torch.linalg.vector_norm(v[:, :2], dim=-1) <= cfg.land_max_speed)
        rpy = self.plant.rpy_ned()
        level = (rpy[:, :2].abs() <= cfg.land_max_tilt).all(-1)
        good = has & slow & level & (torch.linalg.vector_norm(ps[:, :2] - self._mark, dim=-1) <= cfg.return_radius)
        lid = torch.nonzero(good).flatten()
        self._phase[good] = 3
        self._hold[good] = 0
        if len(lid):
            self._park_pos[lid] = torch.cat([ps[lid, :2], torch.full_like(ps[lid, :1], -cfg.rest_alt)], dim=1)
            self._park_yaw[lid] = rpy[lid, 2]
            self.plant.reset_idx(lid, self._park_pos[lid], self._park_yaw[lid])
        holding = ph0 == 3
        self._hold += holding.long()
        done_ok = holding & (self._hold >= self._hold_steps)
        reward = done_ok.float() * (cs.RETURN_MULTIPLIER - 1.0) * self._ep_points / cs.MAX_RUN_POINTS
        self._ep_points = torch.where(done_ok, self._ep_points * cs.RETURN_MULTIPLIER, self._ep_points)
        return done_ok, has & ~good, reward

    def _gates(self) -> list:
        """rules.GATES, with the window line moved to ``cfg.window_gate`` when set (normal toward -E, as in the CAD)."""
        gates = [dict(g) for g in cs.GATES]
        if self.cfg.window_gate:
            n0, e0, n1, e1 = map(float, self.cfg.window_gate)
            A, B = np.array([n0, e0]), np.array([n1, e1])
            t = (B - A) / np.linalg.norm(B - A)
            n = np.array([-t[1], t[0]])
            n = -n if n[1] > 0 else n
            assert gates[cs.WINDOW_GATE]['name'] == 'window'
            gates[cs.WINDOW_GATE].update(A=A, B=B, t=t, L=float(np.linalg.norm(B - A)), n=n)
        return gates

    def _rangefinder(self) -> torch.Tensor:
        """(E,) downward ToF reading [m]: nearest hit over a 27 deg cone of rays along body down (tilts with
        the drone; reads the wall top / boulder / tunnel when flying over one)."""
        cfg, p = self.cfg, self.plant
        K = self._rf_dirs.shape[0]
        dirs = (p.rot_ned().unsqueeze(1) @ self._rf_dirs.view(1, K, 3, 1)).squeeze(-1)    # (E,K,3)
        o = (p.pos_ned + self._off).unsqueeze(1).expand(-1, K, -1)
        r = self._rf.cast_rays(o.reshape(-1, 3), dirs.reshape(-1, 3), cfg.rf_max).view(-1, K).amin(1)
        noisy = r + torch.randn_like(r) * (cfg.rf_noise_abs + cfg.rf_noise_frac * r)
        return torch.where(r < cfg.rf_max, noisy, r).clamp(cfg.rf_min, cfg.rf_max)

    # ── evaluation ──────────────────────────────────────────────────────────

    def set_eval_queue(self, queue: torch.Tensor) -> None:
        """Fly a FIXED list of start cells instead of SGS: ``queue`` (E, m) bank indices, env i flies ``queue[i]``
        in order. Each of the first m episodes per env is appended to ``self.finished`` as
        (env, k, cell, outcome, ep_len), so the result is not biased toward short episodes."""
        assert queue.shape[0] == self.num_envs
        self._eval_q = queue.to(self.device, torch.long)
        self._eval_ptr.zero_()
        self.finished, self.finished_points, self.finished_fin = [], [], []
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def _record_finished(self, ids, success, collision, floor, oob) -> None:
        k = self._eval_ptr[ids] - 1                       # episode index that just ended
        keep = k < self._eval_q.shape[1]
        outcome = torch.zeros(len(ids), dtype=torch.long, device=self.device)   # timeout
        for code, flag in ((6, self._early), (5, oob), (3, floor), (2, collision)):
            outcome = torch.where(flag[ids], torch.full_like(outcome, code), outcome)
        outcome = torch.where(success[ids], torch.ones_like(outcome), outcome)
        self.finished.append(torch.stack([ids, k, self._cell[ids], outcome, self.episode_length_buf[ids]])[:, keep])
        self.finished_points.append(self._ep_points[ids][keep])
        if self._rp:
            self.finished_fin.append(torch.stack([self._start_phase[ids], self._fin_run[ids].long(),
                                                  (self._phase[ids] >= 2).long()])[:, keep])

    def eval_done(self) -> bool:
        return bool((self._eval_ptr > self._eval_q.shape[1]).all())

    # ── resets ──────────────────────────────────────────────────────────────

    def _assign_variant(self, ids: torch.Tensor, k: torch.Tensor) -> None:
        self._var[ids] = k
        self._off[ids] = self._v_off[k]
        self._goal[ids] = self._v_goal[k]
        self._pad_c[ids] = self._v_pad[k]
        self._mark[ids] = self._v_mark[k]
        self._gP0[:, 0, ids] = self._v_gP0[k].transpose(0, 1)
        self._gt[:, 0, ids] = self._v_gt[k].transpose(0, 1)
        self._gn[:, 0, ids] = self._v_gn[k].transpose(0, 1)
        self._gL[:, 0, ids] = self._v_gL[k].transpose(0, 1)
        if self._pack:                                   # per-course arrays: layout k is course k % n (net copy)
            c = k % self._n_lay
            self._course[ids] = c
            self._gcp[:, ids] = self._v_gcp[c].transpose(0, 1)
            self._r0[ids], self._r1[ids], self._rlen[ids] = self._v_r0[c], self._v_r1[c], self._v_rlen[c]
            self._arena[ids] = self._v_arena[c]

    def _choose_variant(self, ids: torch.Tensor, cells: torch.Tensor) -> None:
        """Each resetting env draws a layout; one that puts the start within ``perturb_start_clearance`` of a
        wall is redrawn (up to 6 times), then the nominal layout 0 is used."""
        cfg = self.cfg
        if self._pack:                                   # the drawn cell's course (its see-through-net copy if on)
            k = self._bank_course[cells]
            if self._net_split:
                k = k + self.real.mask[ids].long() * self._net_split
            self._assign_variant(ids, k)
            return
        if len(self._v_sev) == 1:
            return
        allowed = torch.arange(len(self._v_sev), device=self.device)
        if cfg.perturb_ramp_steps > 0 and cfg.perturb_mode == 'spread':
            level = cfg.perturb * min(1.0, self._env_steps / cfg.perturb_ramp_steps)
            allowed = torch.nonzero(self._v_sev <= level + 1e-6).flatten()
        S = self._net_split
        if S:                                         # pick among the first S, then the net copy matching the mask
            allowed = allowed[allowed < S]
            on = self.real.mask[ids].long() * S
        draw = lambda n: allowed[torch.randint(len(allowed), (n,), device=self.device, generator=self._gen)] + (on if S else 0)   # noqa: E731
        k = draw(len(ids))
        start = self._bank_pos[cells].clone()
        is_l = (self._bank_kind[cells] == BANK_LAUNCH).unsqueeze(-1)
        for r in range(7):
            if self._shifted_launch:
                start[:, :2] = self._bank_pos[cells][:, :2] + torch.where(is_l, self._v_lshift[k], torch.zeros_like(self._v_lshift[k]))
            # only the resetting envs, search capped just past the threshold: the same test, far cheaper
            bad = self._obst.distance_capped(start + self._v_off[k], cfg.perturb_start_clearance + 0.01) \
                < cfg.perturb_start_clearance
            if not bool(bad.any()):
                break
            k = torch.where(bad, (on if S else torch.zeros_like(k)) if r == 6 else draw(len(ids)), k)
        self._assign_variant(ids, k)

    def _cp_target(self, pos2: torch.Tensor) -> torch.Tensor:
        """(E,2) target of the next checkpoint: 30 cm past the nearest of its gate lines' midpoints (the pad centre
        for the landing)."""
        mid = self._gP0[:, 0] + self._gt[:, 0] * (0.5 * self._gL[:, 0]).unsqueeze(-1)            # (G,E,2)
        tgt = mid + self._gn[:, 0] * (self.cfg.cp_margin + 0.15)
        d_g = torch.linalg.vector_norm(tgt - pos2.unsqueeze(0), dim=-1)                        # (G,E)
        nc = self._next_cp
        gcp = self._gcp if self._gcp.dim() == 2 else self._gcp.view(-1, 1)        # (G,E) with a pack, else (G,1)
        d_g = torch.where(gcp == nc.view(1, -1), d_g, torch.full_like(d_g, float('inf')))
        best = tgt[d_g.argmin(0), torch.arange(pos2.shape[0], device=pos2.device)]
        return torch.where((nc >= 4).unsqueeze(-1), self._pad_c[:, :2], best)

    def _cp_shaping(self, pos2: torch.Tensor) -> torch.Tensor:
        """(E,) decrease of the distance to the next checkpoint's target since the last step (0 when the target
        changed this step or the episode just started). Slow flight turns corners tight and never reaches a
        line it has no reason to approach; this gives it one without changing which behaviour is optimal."""
        tgt = self._cp_target(pos2)
        d = torch.linalg.vector_norm(tgt - pos2, dim=-1)
        nc = self._next_cp
        same = nc == self._shape_cp
        out = torch.where(same & torch.isfinite(d), self._shape_d - d, torch.zeros_like(d))
        self._shape_d, self._shape_cp = d.clone(), nc.clone()
        return out

    def _reset_idx(self, ids: torch.Tensor) -> None:
        cfg = self.cfg
        n = len(ids)
        if self._eval_q is not None:
            k = self._eval_ptr[ids].clamp(max=self._eval_q.shape[1] - 1)
            cells = self._eval_q[ids, k]
            self._eval_ptr[ids] += 1
        elif cfg.sgs_uniform:
            cells = torch.randint(0, len(self._bank_pos), (n,), device=self.device, generator=self._gen)
        else:
            cells = self.sgs.sample(n, generator=self._gen)
        may_lift = self._eval_q is None and cfg.launch_frac > 0 and len(self._launch_cells) > 0   # else no .any() sync
        if may_lift:
            force = torch.rand(n, device=self.device, generator=self._gen) < cfg.launch_frac
            pick = torch.randint(0, len(self._launch_cells), (n,), device=self.device, generator=self._gen)
            cells = torch.where(force, self._launch_cells[pick], cells)
            lift = force & (torch.rand(n, device=self.device, generator=self._gen) < cfg.launch_liftoff_frac)
        else:
            lift = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._cell[ids] = cells
        self._liftoff[ids] = lift
        p0, v0 = self._bank_pos[cells].clone(), self._bank_vel[cells].clone()
        if may_lift and lift.any():                     # just lifted off the mark: low, rising, same heading
            m = int(lift.sum())
            u = lambda r: r[0] + (r[1] - r[0]) * torch.rand(m, device=self.device, generator=self._gen)   # noqa: E731
            p0[lift, 2] = -u(cfg.liftoff_alt)
            v0[lift, 2] = -u(cfg.liftoff_vz)
            v0[lift, :2] = torch.randn(m, 2, device=self.device, generator=self._gen) * cfg.liftoff_vxy
        if self.real is not None:
            self.real.draw_mask(ids, self._env_steps)
        self._choose_variant(ids, cells)
        if self._shifted_launch:
            is_l = (self._bank_kind[cells] == BANK_LAUNCH).unsqueeze(-1)
            p0[:, :2] = p0[:, :2] + torch.where(is_l, self._v_lshift[self._var[ids]], torch.zeros_like(p0[:, :2]))
        self.plant.reset_idx(ids, p0, self._bank_yaw[cells], v0, self._bank_roll[cells], self._bank_pitch[cells])
        if cfg.launch_idle_steps > 0:
            on_floor = (self._bank_kind[cells] == BANK_LAUNCH) & ~lift
            self._idle[ids] = torch.where(on_floor, torch.full_like(self._idle[ids], cfg.launch_idle_steps), torch.zeros_like(self._idle[ids]))
            self._idle_pose[ids] = torch.cat([p0, self._bank_yaw[cells].unsqueeze(-1), self._bank_roll[cells].unsqueeze(-1),
                                              self._bank_pitch[cells].unsqueeze(-1)], -1)
        if self.real is not None:
            self.real.on_reset(ids, self.plant)
            if self.real.cfg.vio_pos:                            # VIO pose error: tiny at the mark, sigma(s) elsewhere
                kind = self._bank_kind[cells]
                s_fl = self._bank_s[cells]
                ph = self._bank_phase[cells]
                if bool((ph > 0).any()):                         # return leg: the whole route + the way back so far
                    back = torch.linalg.vector_norm(p0[:, :2] - self._pad_c[ids, :2], dim=-1)
                    s_fl = torch.where(ph > 0, self._route_len + back, s_fl)
                p = self.plant
                self.real.vio_pos_reset(ids, s_fl, (kind == 0) | lift, p.pos_ned, p.rpy_ned()[:, 2], p.vel_ned)
            if self._pose_hist is not None:                       # a fresh episode has no older frames
                self._pose_hist[:, ids] = torch.cat([self.plant.pos_ned, self.plant.rpy_ned()], -1)[ids]
        if self._pack:                                  # the cell's own course's slice of the concatenated route
            self._route_idx[ids] = self._bank_ridx[cells]
        else:
            self._route_idx[ids] = torch.clamp(torch.searchsorted(self._route_s, self._bank_s[cells]),
                                               max=len(self._route_s) - 1)
        self.episode_length_buf.index_fill_(0, ids, 0)
        self._path.index_fill_(0, ids, 0.0)
        self._last_pos[ids] = p0
        self._prev_action.index_fill_(0, ids, 0.0)
        self._grounded[ids] = -p0[:, 2] <= cfg.rest_alt + 0.05
        self._ep_points.index_fill_(0, ids, 0.0)
        self._ever_over.index_fill_(0, ids, False)
        sec = self._bank_sector[cells]
        if self._rp:
            ph = self._bank_phase[cells]
            if self._mode == 'return':                  # the return policy is in charge from the start: released
                ph = torch.where(ph == 1, torch.full_like(ph, 2), ph)
                self._land_factor.index_fill_(0, ids, 1.0)
            elif self._mode == 'handoff':
                self._land_factor.index_fill_(0, ids, 1.0)
            self._phase[ids] = ph
            self._start_phase[ids] = ph
            self._fin_run.index_fill_(0, ids, False)
            self._hold.index_fill_(0, ids, 0)
            self._park_pos[ids] = self._bank_pos[cells]
            self._park_yaw[ids] = self._bank_yaw[cells]
            self._ep_points[ids] = (ph > 0).float() * cfg.return_start_points
            sec = torch.where(ph > 0, torch.full_like(sec, 5), sec)
        self._next_cp[ids] = sec
        self._start_cp[ids] = sec
        self._cleared.index_fill_(0, ids, False)
        self._flag.index_fill_(0, ids, False)
        self._via.index_fill_(0, ids, False)
        self._ncl.index_fill_(0, ids, 0)
        self._t_last.index_fill_(0, ids, 0.0)
        self._shape_cp.index_fill_(0, ids, -1)
        self._start_geo[ids] = self._bank_geo[cells]
        self._last_geo[ids] = self._bank_geo[cells]
        self._prime_ids = torch.cat([self._prime_ids, ids]).unique() if self._prime_pending else ids
        self._prime_pending = True

    # ── observations ────────────────────────────────────────────────────────

    def _advance_route_idx(self, back: int = 10, ahead: int = 60) -> None:
        """Nearest route point in a window around the last one (critic progress feature)."""
        r = self._route_pts.shape[0]
        offs = torch.arange(-back, ahead + 1, device=self.device)
        if self._pack:                                   # stay inside this env's course's slice of the route
            win = torch.maximum(torch.minimum(self._route_idx.unsqueeze(1) + offs, self._r1.unsqueeze(1) - 1),
                                self._r0.unsqueeze(1))
        else:
            win = torch.clamp(self._route_idx.unsqueeze(1) + offs, 0, r - 1)           # (E,W)
        d = torch.linalg.vector_norm(self._route_pts[win] - self.plant.pos_ned.unsqueeze(1), dim=-1)
        self._route_idx = win.gather(1, d.argmin(dim=1, keepdim=True)).squeeze(1)

    def _clearance(self) -> torch.Tensor:
        # the critic clamps at CLEARANCE_OBS_MAX: a search capped just past it gives the same input, much cheaper
        return self._obst.min_obstacle_distance(self.plant.pos_ned + self._off, CLEARANCE_OBS_MAX + 1e-3)

    def _proprio(self) -> torch.Tensor:
        p = self.plant
        if self.real is None:
            v_body = (p.rot_ned().transpose(1, 2) @ p.vel_ned.unsqueeze(-1)).squeeze(-1)
            parts = [v_body / self._v_scale, self._rangefinder().unsqueeze(-1) / 1.5, p.rpy_ned()[:, :2],
                     self._prev_action]
        else:                                                    # estimated velocity / attitude, lossy rangefinder
            v_body = (p.rot_ned().transpose(1, 2) @ self.real.velocity(p.vel_ned).unsqueeze(-1)).squeeze(-1)
            rf = self.real.rangefinder(self._rangefinder(), self.cfg.rf_max)
            parts = [v_body / self._v_scale, rf.unsqueeze(-1) / 1.5, self.real.attitude(p.rpy_ned()[:, :2]),
                     self._prev_action]
        if self._mode == 'joint':
            parts.append((self._phase >= 2).float().unsqueeze(-1))        # the latched return flag
        if self.cfg.pos_obs:                                              # course-frame pose (VIO) estimate
            yaw = p.rpy_ned()[:, 2]
            if self.real is not None and self.real.cfg.vio_pos:
                pos, yaw, ok = self.real.vio_pose(p.pos_ned, p.vel_ned, yaw)
            else:
                pos, ok = p.pos_ned, torch.ones_like(yaw, dtype=torch.bool)
            parts += [pos / 3.0, torch.sin(yaw).unsqueeze(-1), torch.cos(yaw).unsqueeze(-1), ok.float().unsqueeze(-1)]
        return torch.cat(parts, dim=-1)

    def _critic(self, clearance: torch.Tensor) -> torch.Tensor:
        p = self.plant
        rot, pos = p.rot_ned(), p.pos_ned
        to_goal = self._goal - pos
        dist = torch.linalg.vector_norm(to_goal, dim=-1, keepdim=True)
        goal_body = (rot.transpose(1, 2) @ (to_goal / dist.clamp(min=1e-6)).unsqueeze(-1)).squeeze(-1)
        progress = self._route_s[self._route_idx] / (self._rlen if self._pack else self._route_len)
        return torch.cat([
            pos / 3.0, p.vel_ned / 2.0, p.rpy_ned(), p.ang_vel_b / 3.0,
            p.applied_thrust / (p.mass * 9.81 / 4.0) - 1.0, self._prev_action,
            to_goal / 10.0, dist / 10.0, progress.unsqueeze(-1), clearance.clamp(max=CLEARANCE_OBS_MAX).unsqueeze(-1) / 3.0,
            goal_body, (dist / 3.0).clamp(max=1.0)] + (
            [(self._mark - pos[:, :2]) / 3.0, self._phase.float().unsqueeze(-1) / 3.0] if self._rp else []) + (
            [self._critic_depth(pos, p.rpy_ned())] if self._crit_scene is not None else []) + (
            [self._critic_full(pos, rot)] if self.cfg.critic_full else []), dim=-1)

    def _critic_full(self, pos, rot) -> torch.Tensor:
        """(E,20) privileged state the actor cannot observe directly: next checkpoint (one-hot of 6), the body-frame
        vector to its target + distance, episode time, and the hidden randomized physics."""
        E, dev = pos.shape[0], pos.device
        nc = self._next_cp.clamp(0, 5)
        onehot = torch.nn.functional.one_hot(nc, 6).float()
        rel = torch.cat([self._cp_target(pos[:, :2]) - pos[:, :2], torch.zeros(E, 1, device=dev)], -1)
        rel_b = (rot.transpose(1, 2) @ rel.unsqueeze(-1)).squeeze(-1)[:, :2]
        dist = torch.linalg.vector_norm(rel[:, :2], dim=-1, keepdim=True)
        t = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(-1)
        z = torch.zeros(E, 1, device=dev)
        r = self.real
        hid = [self.plant.motor_tau * 30.0 - 1.0]
        if r is not None:
            hid += [r.true_mass / r.m0 - 1.0, r.eff - 1.0, r.drag_scale - 1.0, r.soc.view(E, 1),
                    r.act_lat.view(E, 1) * 10.0, r.depth_lat.view(E, 1) * 10.0, r.ki.mean(-1, keepdim=True)]
        else:
            hid += [z] * 7
        return torch.cat([onehot, rel_b / 3.0, dist / 3.0, t] + hid, -1)

    def _critic_depth(self, pos, rpy) -> torch.Tensor:
        """(E,160): clean z-depth of the original geometry (net solid, no stereo effects, no latency), nearest return
        per 8x8 cell of the 80x128 image -> 10x16, flattened. The critic only; the actor never sees it."""
        import torch.nn.functional as F
        d = self._crit_scene.render_depth(pos + self._off, rpy)
        return (-F.max_pool2d(-d.unsqueeze(1), 8)).flatten(1)

    def _observations(self) -> TensorDict:
        p = self.plant
        if self.real is not None and self.real.cfg.depth:
            depth = self._real_depth()
        else:
            depth = self._depth.render_depth(p.pos_ned + self._off, p.rpy_ned())
        bad = None
        if self._mstack is not None:
            bad = getattr(self.real, 'last_bad', None) if (self.real is not None and self.real.cfg.depth) else None
        if self.cfg.obs_pool > 1:
            depth, bad = self._pool_depth(depth, bad)
        self._stack_pos = (self._stack_pos + 1) % self._stack.shape[1]
        self._stack[:, self._stack_pos] = depth
        if self._mstack is not None:
            m = bad.float() if bad is not None else torch.zeros_like(depth)
            self._mstack[:, self._stack_pos] = m
        if self._prime_pending:                      # a fresh episode's stack is filled with its first frame
            self._stack[self._prime_ids] = depth[self._prime_ids].unsqueeze(1)
            if self._mstack is not None:
                self._mstack[self._prime_ids] = m[self._prime_ids].unsqueeze(1)
            self._prime_pending = False
        img = self._stack.index_select(1, self._gather[self._stack_pos])    # fresh tensor (rsl_rl keeps a reference)
        if self._mstack is not None:                 # [depth frames..., mask frames...], newest first in each half
            img = torch.cat([img, self._mstack.index_select(1, self._gather[self._stack_pos])], dim=1)
        return TensorDict({'policy': img, 'proprio': self._proprio(), 'critic': self._critic(self._clearance())},
                          batch_size=[self.num_envs])

    def _pool_depth(self, depth: torch.Tensor, bad):
        """(E,H,W) normalised depth (0 = invalid/near) -> (E,H/k,W/k): the nearest valid return of each k x k cell;
        a cell with < 25% valid pixels is invalid (0). Min-pooling keeps thin obstacles; averaging would invent
        phantom near obstacles next to holes, and 'any hole -> near' would black out the image."""
        import torch.nn.functional as F
        k = int(self.cfg.obs_pool)
        valid = depth > 0
        x = torch.where(valid, depth, torch.full_like(depth, 2.0)).unsqueeze(1)
        vmin = -F.max_pool2d(-x, k)[:, 0]
        vf = F.avg_pool2d(valid.float().unsqueeze(1), k)[:, 0]
        ok = vf >= 0.25
        out = torch.where(ok, vmin.clamp(max=1.0), torch.zeros_like(vmin))
        return out, (~ok if bad is not None else None)

    def _real_depth(self) -> torch.Tensor:
        """Stereo depth as the policy would get it: rendered at the frame's (older) pose, at sensor resolution."""
        p = self.plant
        now = torch.cat([p.pos_ned, p.rpy_ned()], -1)
        if self._pose_hist is None:
            self._pose_hist = now.unsqueeze(0).repeat(30, 1, 1)
        lag = self.real.depth_lag_substeps(p.decimation)
        idx = (self._pose_hist.shape[0] - 1 - lag).clamp(min=0)
        pose = self._pose_hist[idx, torch.arange(self.num_envs, device=self.device)]
        pos, rpy = pose[:, :3] + self._off, pose[:, 3:]
        far = self.cfg.far_clip
        # normalised render buffers (persistent; the stereo model reads them before the next render)
        z = self._depth.render_depth(pos, rpy)
        net = (self._net_scene, pos, rpy) if self._net_scene is not None else None    # cast where a return fires
        return self.real.corrupt_depth(z, None, far_scale=far, net_rays=net)

    # ── metrics ─────────────────────────────────────────────────────────────

    def _reset_stats(self) -> None:
        self._stat = {k: 0 for k in ('episodes', 'success', 'collision', 'floor', 'oob', 'timeout', 'steps',
                                     'over_flag', 'via', 'run_eps', 'early_flag', 'finished', 'parked', 'flagged',
                                     'returned', 'ret_collision', 'ret_floor', 'ret_oob', 'ret_timeout')}
        self._stat.update(sev_sum=0.0, speed_sum=0.0, points_sum=0.0, t_fin=0.0, cp_started=[0] * 5, cp_cleared=[0] * 5)
        self._stat_kind = {k: [0, 0, 0, 0, 0, 0] for k in KIND_NAMES}   # episodes, successes, collision, floor, oob, timeout
        if self._pack:
            self._stat_course = torch.zeros(3, self._n_lay, device=self.device)    # episodes, successes, points

    def _accumulate(self, ids, success, collision, floor, oob, truncated, returned) -> None:
        """Add the finished episodes ``ids`` to the running stats. Every count/sum is computed on the GPU and
        brought to the host in ONE transfer (it used to be ~50 separate syncs, ~20 ms per step at 768 envs)."""
        s = self._stat
        sx = success[ids]
        crash = ~sx
        cnt = lambda m: m.sum()                                                   # noqa: E731
        el = self.episode_length_buf[ids]
        cl, st = self._cleared[ids], self._start_cp[ids]
        ints = [('success', cnt(sx))] + [(k, cnt(m[ids] & crash)) for k, m in
                                          (('collision', collision), ('floor', floor), ('oob', oob))]
        ints += [('timeout', cnt(truncated[ids])), ('steps', el.sum()), ('over_flag', cnt(self._ever_over[ids])),
                 ('via', cnt(self._via[ids] & sx))]
        ints += [(('cp_started', k), cnt(st <= k)) for k in range(5)]
        ints += [(('cp_cleared', k), cnt(cl[:, k])) for k in range(5)]
        if self._rp:
            ph = self._start_phase[ids]
            fin = self._fin_run[ids]
            parked = (ph > 0) | fin                                # episodes that reached (or began at) the pad
            flagged = parked & (self._phase[ids] >= 2)
            ret = returned[ids]
            ints += [('run_eps', cnt(ph == 0)), ('finished', cnt(fin & (ph == 0))), ('parked', cnt(parked)),
                     ('flagged', cnt(flagged)), ('returned', cnt(ret)), ('early_flag', cnt(self._early[ids] & ~ret))]
            fl = flagged & ~ret
            ints += [('ret_' + k, cnt(fl & m[ids])) for k, m in
                     (('collision', collision), ('floor', floor), ('oob', oob), ('timeout', truncated))]
        floats = [('sev_sum', self._v_sev[self._var[ids]].sum()),
                  ('speed_sum', (self._path[ids] / (el.clamp(min=1) * CONTROL_DT)).sum()),
                  ('points_sum', self._ep_points[ids].sum()),
                  ('t_fin', torch.where(sx, self._t_last[ids], torch.zeros_like(self._t_last[ids])).sum())]
        if self._pack:                                   # per-course episodes / successes / points (stays on the GPU)
            self._stat_course.index_add_(1, self._course[ids], torch.stack([torch.ones_like(self._ep_points[ids]),
                                                                            sx.float(), self._ep_points[ids]]))
        kinds = torch.where(self._liftoff[ids], torch.full_like(self._bank_kind[self._cell[ids]], LIFTOFF_KIND),
                            self._bank_kind[self._cell[ids]])
        onehot = kinds.unsqueeze(1) == torch.arange(len(KIND_NAMES), device=kinds.device).unsqueeze(0)   # (n,K)
        cols = torch.stack([torch.ones_like(sx), sx] + [crash & c[ids] for c in (collision, floor, oob, truncated)], 1)
        kind_rows = (onehot.unsqueeze(2) & cols.unsqueeze(1)).sum(0)                                     # (K,6)
        vals = torch.cat([torch.stack([v.to(torch.float64) for _, v in ints + floats]),
                          kind_rows.reshape(-1).to(torch.float64)]).tolist()                           # one sync
        s['episodes'] += len(ids)
        for (k, _), v in zip(ints, vals):
            if isinstance(k, tuple):
                s[k[0]][k[1]] += int(v)
            else:
                s[k] += int(v)
        for (k, _), v in zip(floats, vals[len(ints):]):
            s[k] += v
        kr = vals[len(ints) + len(floats):]
        for code, name in enumerate(KIND_NAMES):
            row = self._stat_kind[name]
            for j in range(6):
                row[j] += int(kr[code * 6 + j])

    def pop_stats(self) -> dict:
        """Aggregate over episodes finished since the last call (host sync)."""
        s = self._stat
        n = max(s['episodes'], 1)
        nan = float('nan')
        out = {'episodes': s['episodes'], 'success_rate': s['success'] / n, 'collision_rate': s['collision'] / n,
               'floor_rate': s['floor'] / n, 'oob_rate': s['oob'] / n, 'timeout_rate': s['timeout'] / n,
               'mean_points': s['points_sum'] / n, 'mean_ep_len': s['steps'] / n, 'mean_speed': s['speed_sum'] / n,
               'layout_severity': s['sev_sum'] / n, 'over_wall_flight_rate': s['over_flag'] / n,
               'window_share': s['via'] / s['success'] if s['success'] else nan,
               'mean_finish_s': s['t_fin'] / s['success'] if s['success'] else nan}
        for k in range(5):
            out[f'cp{k + 1}_rate'] = s['cp_cleared'][k] / s['cp_started'][k] if s['cp_started'][k] else nan
        for name, (eps, succ, *fails) in self._stat_kind.items():
            if eps:
                out[f'success_{name}'] = succ / eps
                out[f'episodes_{name}'] = eps
                if name in ('pad', 'return', 'climb', 'cross', 'final'):
                    for cause, v in zip(('collision', 'floor', 'oob', 'timeout'), fails):
                        out[f'fail_{name}_{cause}'] = v / eps
        if self._rp:
            pk = s['parked']
            out['finish_rate'] = s['finished'] / s['run_eps'] if s['run_eps'] else nan
            out['flag_rate'] = s['flagged'] / pk if pk else nan
            out['return_rate'] = s['returned'] / pk if pk else nan
            out['early_flag_rate'] = s['early_flag'] / n
            for k in ('collision', 'floor', 'oob', 'timeout'):
                out['ret_fail_' + k] = s['ret_' + k] / pk if pk else nan
        if self.real is not None:
            out['realism_on'] = float(self.real.mask.float().mean())
        if self._pack:
            sc = self._stat_course.cpu().numpy()
            for k, seed in enumerate(self.pack_seeds):
                if sc[0, k]:
                    out[f'success_course{seed}'] = float(sc[1, k] / sc[0, k])
                    out[f'points_course{seed}'] = float(sc[2, k] / sc[0, k])
        out.update(self.sgs.stats())
        self._reset_stats()
        return out

    def _maybe_log(self) -> None:
        if self.cfg.metrics_path is None or time.time() - self._last_log < self.cfg.metrics_interval_s:
            return
        self._last_log = time.time()
        rec = self.pop_stats()
        rec['t'] = time.time()
        rec['env_steps'] = self._env_steps
        os.makedirs(os.path.dirname(os.path.abspath(self.cfg.metrics_path)), exist_ok=True)
        with open(self.cfg.metrics_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def close(self) -> None:
        pass
