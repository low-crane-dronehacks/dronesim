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
                           action (4, or 5 + the latched return flag with the return leg)
``critic``   (E, C)        privileged state for the value function only (position, velocity, attitude, rates,
                           thrusts, previous action, goal vector, route progress, clearance, ...)
The actor gets NO position or goal information: it has to navigate from what it sees.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from dronehacks import perturb as cp
from dronehacks import rules as cs
from dronehacks.bank import Bank, Route
from dronehacks.mesh import CourseMesh
from dronehacks.quad import LEE_MAX_YAW_RATE, TorchQuad
from dronehacks.render import MeshScene
from dronehacks.sgs import SGSSampler

ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets'))
CONTROL_DT = 0.1
ACTION_DIM = 4
KIND_NAMES = ('launch', 'body', 'near_goal', 'near_route', 'pad', 'return', 'climb', 'cross', 'final')  # == bank.BANK_*


@dataclass
class CourseEnvCfg:
    num_envs: int = 2048
    device: str = 'cuda:0'
    seed: int = 0
    assets_dir: str = ASSETS
    bank_file: str = 'bank_free_v2.npz'
    # camera: OAK-D S2 mono pair (72 x 50 deg), downsampled
    depth_width: int = 128
    depth_height: int = 80
    hfov_deg: float = 72.0
    vfov_deg: float = 50.0
    far_clip: float = 5.0
    near_clip: float = 0.05
    frame_stack: int = 2
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
    perturb_mode: str = 'fixed'          # 'fixed': all K at `perturb`; 'spread': 0..perturb, layout 0 nominal
    perturb_seed: int = 0
    perturb_ramp_steps: int = 0          # 'spread': ramp allowed severity over this many env steps
    perturb_start_clearance: float = 0.15
    # plant
    v_max: tuple = (5.0, 3.0, 2.0)       # forward, right, down (m/s), yaw-aligned body frame
    yaw_rate_max: float = LEE_MAX_YAW_RATE
    # SGS
    sgs_target: float = 0.5
    sgs_concentration: float = 1.0
    sgs_history: int = 100
    sgs_temperature: float = 2.0
    sgs_eps: float = 1e-4
    sgs_uniform: bool = False            # ablation: uniform bank sampling
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
        course = CourseMesh.load(os.path.join(cfg.assets_dir, 'course.npz'))
        route = Route.load(os.path.join(cfg.assets_dir, 'route.npz'))
        with open(os.path.join(cfg.assets_dir, 'scene.json')) as f:
            mark_xy = np.asarray(json.load(f)['landmarks']['launch_mark_ned'][:2], np.float64)
        K = max(1, int(cfg.perturb_variants)) if cfg.perturb > 0 else 1
        if cfg.perturb_mode not in ('fixed', 'spread'):
            raise ValueError("perturb_mode must be 'fixed' or 'spread'")
        sev = ([cfg.perturb] * K if cfg.perturb_mode == 'fixed' else
               [cfg.perturb * k / max(K - 1, 1) for k in range(K)]) if K > 1 else [0.0]
        variants = cp.build_variants(
            course, cs.GATES, cs.PAD_CENTER, mark_xy, np.asarray(route.goal, np.float64), severities=sev,
            seed=cfg.perturb_seed, launch_xy=mark_xy,
            clearance_factory=lambda m: MeshScene(m, 1, device='cpu').distance_at)
        lays = variants.layouts
        self.layout_info = {'severities': [float(x) for x in variants.severities], 'attempts': variants.attempts,
                            'rejected': variants.rejected}

        def stacked(grp):
            return cp.stack_variants([lay.mesh.select(grp) for lay in lays])

        w, h = cfg.depth_width, cfg.depth_height
        fx = 0.5 * w / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        fy = 0.5 * h / math.tan(math.radians(cfg.vfov_deg) / 2.0)
        groups = ('floor', 'wall', 'tunnel', 'boulder') + (('net',) if cfg.net_in_depth else ())
        mesh, offs = stacked(groups)
        self._depth = MeshScene(mesh, e, w, h, fx, fy, cfg.far_clip, cfg.near_clip, depth_type='z', device=dev)
        self._obst = MeshScene(stacked(('wall', 'tunnel', 'boulder'))[0], e, device=dev)
        # per-layout scoring geometry, gathered per env at reset (_assign_variant)
        self._v_off, self._v_sev = t(offs), t(variants.severities)
        self._v_goal = t([lay.goal for lay in lays])
        self._v_pad = t([lay.pad_center for lay in lays])
        self._v_mark = t([lay.mark for lay in lays])
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
        bank = Bank.load(os.path.join(cfg.assets_dir, cfg.bank_file))
        if mode in ('handoff', 'return'):                 # run policy: run starts only; return policy: return starts
            keep = (bank.phase == 0) if mode == 'handoff' else (bank.phase > 0)
            if not keep.any():
                raise ValueError(f"return_mode {mode!r}: bank {cfg.bank_file} has no {'run' if mode == 'handoff' else 'return-leg'} starts")
            bank = Bank(*(getattr(bank, k)[keep] for k in ('pos', 'yaw', 'route_s', 'kind', 'vel', 'roll', 'pitch', 'geo', 'phase')))
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
        self.sgs = SGSSampler(len(bank), history=cfg.sgs_history, target=cfg.sgs_target,
                              concentration=cfg.sgs_concentration, eps=cfg.sgs_eps,
                              temperature=cfg.sgs_temperature, device=dev)

        # ── plant + per-env state ──────────────────────────────────────────────
        self.plant = TorchQuad(e, dev, v_max=cfg.v_max, yaw_rate_max=cfg.yaw_rate_max)
        self._cell = torch.zeros(e, dtype=torch.long, device=dev)
        self._route_idx = torch.zeros(e, dtype=torch.long, device=dev)
        self._prev_action = torch.zeros(e, self.num_actions, device=dev)
        self._stack = torch.ones(e, cfg.frame_stack, h, w, device=dev)
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
        self._v_scale = torch.tensor(cfg.v_max, device=dev)
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
        self._start_geo = torch.zeros(e, device=dev)
        self._last_geo = torch.zeros(e, device=dev)
        with open(os.path.join(cfg.assets_dir, 'scoring.json')) as f:
            self._gate_geo = t(json.load(f)['gate_geodesic_m'])
        self._gcp = t(cs.GATE_CP, torch.long)
        self._cp_pts = t(cs.CP_POINTS)
        self._pad_r = float(cs.PAD_RADIUS_M)
        self._gate_tol = 0.1                                                     # slack along the line, m
        self._bank_sector = t(cs.sector_index(bank.pos[:, :2]), torch.long)

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
        self.plant.step_control(a[:, :ACTION_DIM])
        sub = torch.stack(self.plant.substep_pos)                 # (S,E,3) NED after every 100 Hz substep
        self.plant.substep_pos = None
        if self._rp:
            parked = (ph0 == 1) | (ph0 == 3)                      # disarmed on the ground: hold the pose exactly
            sub[:, parked] = self._park_pos[parked]
            pid = torch.nonzero(parked).flatten()
            if len(pid):
                self.plant.reset_idx(pid, self._park_pos[pid], self._park_yaw[pid])
        self._prev_action = a.clone()
        self.episode_length_buf += 1
        self._env_steps += self.num_envs
        self._advance_route_idx()

        pts = torch.cat([self._last_pos.unsqueeze(0), sub], dim=0)
        prev = pts[:-1]                                           # position before each substep
        self._path += torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=-1).sum(0)
        self._last_pos = sub[-1].clone()

        # Checks run over ALL substeps: at 5 m/s a control step is 0.5 m, enough to jump a wall.
        alt = -sub[..., 2]
        self._grounded &= ~(alt > cfg.lift_alt).any(0)
        # min_obstacle_distance returns a persistent buffer: clone per substep or every row aliases the last
        clearance = torch.stack([self._obst.min_obstacle_distance(p + self._off).clone() for p in sub]).amin(0)
        collision = clearance < cfg.collision_radius
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
        if self._mode in ('joint', 'return'):
            reward = reward + ret_r

        extras = {'time_outs': truncated.clone()}
        ids = torch.nonzero(done).flatten()
        if len(ids):
            self.sgs.record(self._cell[ids], sgs_success[ids])
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
        hit = crossed & (self._gcp.unsqueeze(1) == nc.unsqueeze(0))        # (G,E): a gate of the NEXT checkpoint
        cleared = hit.any(0)
        factor = torch.where(self._flag, torch.full_like(t_now, cs.OVER_WALL_FACTOR), torch.ones_like(t_now))
        pts = self._cp_pts[nc.clamp(max=4)] * factor * cleared
        gidx = hit.long().argmax(0)
        self._via |= cleared & hit[cs.WINDOW_GATE] & ~hit[3]
        self._last_geo = torch.where(cleared, self._gate_geo[gidx], self._last_geo)
        self._t_last = torch.where(cleared, t_now, self._t_last)
        self._cleared[ar[cleared], nc[cleared]] = True
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
            self._cleared[ar[landed], 4] = True
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
            return pts, landed & False, has & ~landed
        return pts, landed, has & ~landed

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

    def _choose_variant(self, ids: torch.Tensor, cells: torch.Tensor) -> None:
        """Each resetting env draws a layout; one that puts the start within ``perturb_start_clearance`` of a
        wall is redrawn (up to 6 times), then the nominal layout 0 is used."""
        cfg = self.cfg
        if len(self._v_sev) == 1:
            return
        allowed = torch.arange(len(self._v_sev), device=self.device)
        if cfg.perturb_ramp_steps > 0 and cfg.perturb_mode == 'spread':
            level = cfg.perturb * min(1.0, self._env_steps / cfg.perturb_ramp_steps)
            allowed = torch.nonzero(self._v_sev <= level + 1e-6).flatten()
        draw = lambda n: allowed[torch.randint(len(allowed), (n,), device=self.device, generator=self._gen)]   # noqa: E731
        k = draw(len(ids))
        start = self._bank_pos[cells]
        for r in range(7):
            pos = self.plant.pos_ned.clone()
            pos[ids] = start + self._v_off[k]
            bad = self._obst.min_obstacle_distance(pos)[ids] < cfg.perturb_start_clearance
            if not bool(bad.any()):
                break
            k = torch.where(bad, torch.zeros_like(k) if r == 6 else draw(len(ids)), k)
        self._assign_variant(ids, k)

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
        self._cell[ids] = cells
        self._choose_variant(ids, cells)
        self.plant.reset_idx(ids, self._bank_pos[cells], self._bank_yaw[cells], self._bank_vel[cells],
                             self._bank_roll[cells], self._bank_pitch[cells])
        self._route_idx[ids] = torch.clamp(torch.searchsorted(self._route_s, self._bank_s[cells]),
                                           max=len(self._route_s) - 1)
        self.episode_length_buf[ids] = 0
        self._path[ids] = 0.0
        self._last_pos[ids] = self._bank_pos[cells]
        self._prev_action[ids] = 0.0
        self._grounded[ids] = -self._bank_pos[cells][:, 2] <= cfg.rest_alt + 0.05
        self._ep_points[ids] = 0.0
        self._ever_over[ids] = False
        sec = self._bank_sector[cells]
        if self._rp:
            ph = self._bank_phase[cells]
            if self._mode == 'return':                  # the return policy is in charge from the start: released
                ph = torch.where(ph == 1, torch.full_like(ph, 2), ph)
                self._land_factor[ids] = 1.0
            elif self._mode == 'handoff':
                self._land_factor[ids] = 1.0
            self._phase[ids] = ph
            self._start_phase[ids] = ph
            self._fin_run[ids] = False
            self._hold[ids] = 0
            self._park_pos[ids] = self._bank_pos[cells]
            self._park_yaw[ids] = self._bank_yaw[cells]
            self._ep_points[ids] = (ph > 0).float() * cfg.return_start_points
            sec = torch.where(ph > 0, torch.full_like(sec, 5), sec)
        self._next_cp[ids] = sec
        self._start_cp[ids] = sec
        self._cleared[ids] = False
        self._flag[ids] = False
        self._via[ids] = False
        self._ncl[ids] = 0
        self._t_last[ids] = 0.0
        self._start_geo[ids] = self._bank_geo[cells]
        self._last_geo[ids] = self._bank_geo[cells]
        self._prime_ids = torch.cat([self._prime_ids, ids]).unique() if self._prime_pending else ids
        self._prime_pending = True

    # ── observations ────────────────────────────────────────────────────────

    def _advance_route_idx(self, back: int = 10, ahead: int = 60) -> None:
        """Nearest route point in a window around the last one (critic progress feature)."""
        r = self._route_pts.shape[0]
        offs = torch.arange(-back, ahead + 1, device=self.device)
        win = torch.clamp(self._route_idx.unsqueeze(1) + offs, 0, r - 1)           # (E,W)
        d = torch.linalg.vector_norm(self._route_pts[win] - self.plant.pos_ned.unsqueeze(1), dim=-1)
        self._route_idx = win.gather(1, d.argmin(dim=1, keepdim=True)).squeeze(1)

    def _clearance(self) -> torch.Tensor:
        return self._obst.min_obstacle_distance(self.plant.pos_ned + self._off)

    def _proprio(self) -> torch.Tensor:
        p = self.plant
        v_body = (p.rot_ned().transpose(1, 2) @ p.vel_ned.unsqueeze(-1)).squeeze(-1)
        parts = [v_body / self._v_scale, self._rangefinder().unsqueeze(-1) / 1.5, p.rpy_ned()[:, :2], self._prev_action]
        if self._mode == 'joint':
            parts.append((self._phase >= 2).float().unsqueeze(-1))        # the latched return flag
        return torch.cat(parts, dim=-1)

    def _critic(self, clearance: torch.Tensor) -> torch.Tensor:
        p = self.plant
        rot, pos = p.rot_ned(), p.pos_ned
        to_goal = self._goal - pos
        dist = torch.linalg.vector_norm(to_goal, dim=-1, keepdim=True)
        goal_body = (rot.transpose(1, 2) @ (to_goal / dist.clamp(min=1e-6)).unsqueeze(-1)).squeeze(-1)
        progress = self._route_s[self._route_idx] / self._route_len
        return torch.cat([
            pos / 3.0, p.vel_ned / 2.0, p.rpy_ned(), p.ang_vel_b / 3.0,
            p.applied_thrust / (p.mass * 9.81 / 4.0) - 1.0, self._prev_action,
            to_goal / 10.0, dist / 10.0, progress.unsqueeze(-1), clearance.clamp(max=3.0).unsqueeze(-1) / 3.0,
            goal_body, (dist / 3.0).clamp(max=1.0)] + (
            [(self._mark - pos[:, :2]) / 3.0, self._phase.float().unsqueeze(-1) / 3.0] if self._rp else []), dim=-1)

    def _observations(self) -> TensorDict:
        p = self.plant
        depth = self._depth.render_depth(p.pos_ned + self._off, p.rpy_ned())
        self._stack_pos = (self._stack_pos + 1) % self._stack.shape[1]
        self._stack[:, self._stack_pos] = depth
        if self._prime_pending:                      # a fresh episode's stack is filled with its first frame
            self._stack[self._prime_ids] = depth[self._prime_ids].unsqueeze(1)
            self._prime_pending = False
        img = self._stack.index_select(1, self._gather[self._stack_pos])    # fresh tensor (rsl_rl keeps a reference)
        return TensorDict({'policy': img, 'proprio': self._proprio(), 'critic': self._critic(self._clearance())},
                          batch_size=[self.num_envs])

    # ── metrics ─────────────────────────────────────────────────────────────

    def _reset_stats(self) -> None:
        self._stat = {k: 0 for k in ('episodes', 'success', 'collision', 'floor', 'oob', 'timeout', 'steps',
                                     'over_flag', 'via', 'run_eps', 'early_flag', 'finished', 'parked', 'flagged',
                                     'returned', 'ret_collision', 'ret_floor', 'ret_oob', 'ret_timeout')}
        self._stat.update(sev_sum=0.0, speed_sum=0.0, points_sum=0.0, t_fin=0.0, cp_started=[0] * 5, cp_cleared=[0] * 5)
        self._stat_kind = {k: [0, 0, 0, 0, 0, 0] for k in KIND_NAMES}   # episodes, successes, collision, floor, oob, timeout

    def _accumulate(self, ids, success, collision, floor, oob, truncated, returned) -> None:
        s = self._stat
        s['episodes'] += len(ids)
        s['sev_sum'] += float(self._v_sev[self._var[ids]].sum())
        s['success'] += int(success[ids].sum())
        crash = ~success
        for k, m in (('collision', collision), ('floor', floor), ('oob', oob)):
            s[k] += int((m & crash)[ids].sum())
        s['timeout'] += int(truncated[ids].sum())
        s['steps'] += int(self.episode_length_buf[ids].sum())
        s['speed_sum'] += float((self._path[ids] / (self.episode_length_buf[ids].clamp(min=1) * CONTROL_DT)).sum())
        s['points_sum'] += float(self._ep_points[ids].sum())
        cl, st = self._cleared[ids], self._start_cp[ids]
        for k in range(5):
            s['cp_started'][k] += int((st <= k).sum())
            s['cp_cleared'][k] += int(cl[:, k].sum())
        s['over_flag'] += int(self._ever_over[ids].sum())
        s['via'] += int((self._via[ids] & success[ids]).sum())
        s['t_fin'] += float(self._t_last[ids][success[ids]].sum())
        if self._rp:
            ph = self._start_phase[ids]
            parked = (ph > 0) | self._fin_run[ids]                 # episodes that reached (or began at) the pad
            s['run_eps'] += int((ph == 0).sum())
            s['finished'] += int((self._fin_run[ids] & (ph == 0)).sum())
            s['parked'] += int(parked.sum())
            s['flagged'] += int((parked & (self._phase[ids] >= 2)).sum())
            s['returned'] += int(returned[ids].sum())
            s['early_flag'] += int((self._early & ~returned)[ids].sum())
            fl = parked & (self._phase[ids] >= 2) & ~returned[ids]
            for k, m in (('collision', collision), ('floor', floor), ('oob', oob), ('timeout', truncated)):
                s['ret_' + k] += int((fl & m[ids]).sum())
        kinds = self._bank_kind[self._cell[ids]]
        for code, name in enumerate(KIND_NAMES):
            m = kinds == code
            if m.any():
                bad = m & ~success[ids]
                row = self._stat_kind[name]
                row[0] += int(m.sum())
                row[1] += int((m & success[ids]).sum())
                for j, c in enumerate((collision, floor, oob, truncated)):
                    row[2 + j] += int((bad & c[ids]).sum())

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
