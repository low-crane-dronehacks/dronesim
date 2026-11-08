import math

import pytest
import torch

from dronehacks.env import CourseEnv, CourseEnvCfg
from dronehacks.realism import PRESETS, Realism, RealismCfg, parse


def _real(**kw):
    return Realism(RealismCfg(**kw), 4, 'cpu', 0.1, 1.5, (0.012, 0.012, 0.02), (80, 128), 88.1, 5.0)


def test_presets_parse():
    assert parse('') is None
    rc = parse('depth,velocity')
    assert rc.depth and rc.velocity and not rc.plant
    assert parse('real').net and parse('real_lite').render_scale == 1
    with pytest.raises(ValueError):
        parse('nope')
    assert set(PRESETS) >= {'plant', 'battery', 'latency', 'velocity', 'depth', 'net', 'rangefinder', 'real'}


def test_latency_delays_commands():
    r = _real(latency=True, action_latency_s=(0.05, 0.05))
    r.act_lat[:] = 0.05
    seq = r.substep_actions(torch.ones(4, 4), 10)
    assert seq.shape == (10, 4, 4)
    assert (seq[:5] == 0).all() and (seq[5:] == 1).all()             # 50 ms of the previous (zero) command


def test_velocity_estimate_is_noisy_but_close():
    r = _real(velocity=True)
    v = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
    r.on_reset(torch.arange(4), type('P', (), {'vel_ned': v})())
    est = torch.stack([r.velocity(v) for _ in range(50)])
    err = (est - v).abs()
    assert err.max() > 1e-3 and err.mean() < 0.3


def test_depth_invalid_below_min_range_and_far_stays_far():
    r = _real(depth=True, render_scale=1, hole_frac=0.0, occlusion=False, invalid_fill='near')
    z = torch.full((4, 80, 128), 2.0)
    z[:, :, :10] = 0.2                                                  # closer than the 0.35 m stereo minimum
    z[:, :, -10:] = 5.0                                                 # far clip = nothing there
    out = r.corrupt_depth(z, None)
    assert (out[:, :, :10] == 0).all()                                  # invalid -> 'near' fill
    assert (out[:, :, -10:] == 1).all()
    mid = out[:, :, 40:80] * 5.0
    assert (mid - 2.0).abs().mean() < 0.1 and mid.std() > 0             # noisy but unbiased at 2 m


def test_env_runs_with_everything_cheap():
    env = CourseEnv(CourseEnvCfg(num_envs=4, device='cpu', depth_width=32, depth_height=20, realism=parse('real_lite')))
    obs = env.get_observations()
    for _ in range(5):
        obs, *_ = env.step(torch.zeros(4, 4))
    assert all(torch.isfinite(obs[k]).all() for k in ('policy', 'proprio', 'critic'))


# ── VIO position input (pos_obs + realism vio_pos) ───────────────────────────────────────────────────────

def _env(spec='', n=4, **kw):
    return CourseEnv(CourseEnvCfg(num_envs=n, device='cpu', depth_width=32, depth_height=20, seed=5,
                                  realism=parse(spec, seed=5) if spec else None, **kw))


def _true_pose_err(env, obs):
    p = env.plant
    pp = obs['proprio'][:, -6:]
    pos_err = (pp[:, :3] * 3.0 - p.pos_ned).norm(dim=-1)
    yaw = torch.atan2(pp[:, 3], pp[:, 4])
    yaw_err = torch.remainder(yaw - p.rpy_ned()[:, 2] + math.pi, 2 * math.pi) - math.pi
    return pos_err, yaw_err.abs()


def test_pos_obs_off_keeps_the_proprio_layout_and_vio_pos_needs_it():
    assert _env().proprio_dim == 10
    assert _env('velocity').proprio_dim == 10
    assert _env(pos_obs=True).proprio_dim == 16
    with pytest.raises(ValueError):
        _env('vio_pos')
    assert 'vio_pos' in parse('real_lite,vio_pos').__dict__ and parse('real_lite,vio_pos').vio_pos
    assert not parse('real').vio_pos                                   # 'real' keeps the old observation


def test_pos_obs_without_vio_is_the_true_pose():
    env = _env(pos_obs=True)
    obs = env.get_observations()
    pe, ye = _true_pose_err(env, obs)
    assert pe.max() < 1e-5 and ye.max() < 1e-5 and (obs['proprio'][:, -1] == 1).all()


def test_vio_pos_leaves_every_other_effect_unchanged():
    """vio_pos draws on its own generator: real_lite's depth/velocity/plant streams are bit-identical with it."""
    def roll(spec):                    # one env at a time: the plant's gain draws use the global torch RNG
        env, g, out = _env(spec, pos_obs=True), torch.Generator().manual_seed(0), []
        for _ in range(20):
            o, r, d, _ = env.step(torch.rand(4, 4, generator=g) * 2 - 1)
            out.append((o['policy'], o['critic'], o['proprio'][:, :10], r, d))
        return out
    for x, y in zip(roll('real_lite'), roll('real_lite,vio_pos')):
        assert all(torch.equal(u, v) for u, v in zip(x, y))


def test_vio_launch_starts_are_near_perfect():
    env = _env('vio_pos', n=64, pos_obs=True, launch_frac=1.0, launch_liftoff_frac=0.5)
    env._reset_idx(torch.arange(64))
    assert env._liftoff.any() and (~env._liftoff).any()
    obs = env.get_observations()
    _, ye = _true_pose_err(env, obs)
    exy = (obs['proprio'][:, -6:-4] * 3.0 - env.plant.pos_ned[:, :2]).norm(dim=-1)
    assert exy.max() < 0.06 and exy.mean() < 0.025                     # ~1 cm/axis placement + one step of drift
    assert ye.max() < math.radians(3.0)


def test_vio_mid_course_spread_matches_the_calibration():
    n = 4000
    r = Realism(RealismCfg(vio_pos=True), n, 'cpu', 0.1, 1.5, (0.012, 0.012, 0.02), (80, 128), 88.1, 5.0)
    z3 = torch.zeros(n, 3)
    for s, launch in ((10.0, False), (2.0, False), (10.0, True)):
        r.vio_pos_reset(torch.arange(n), torch.full((n,), s), torch.full((n,), launch), z3, torch.zeros(n), z3)
        sig = r.vio_sigma_at(torch.tensor([s]))[:, 0]
        want = 0.01 if launch else float(sig[0])
        got = float(r.pe[:, :2].std())
        assert abs(got / want - 1) < 0.1, (s, launch, got, want)
        assert abs(math.degrees(float(r.ye.std())) / (0.5 if launch else float(sig[2])) - 1) < 0.1
    assert r.vio_sigma_at(torch.tensor([30.0]))[0, 0] > r.vio_sigma_at(torch.tensor([14.0]))[0, 0]   # extrapolates


def test_vio_drift_grows_with_distance_flown():
    from dronehacks.vio_calib import calibrate
    res = calibrate(RealismCfg(vio_pos=True, vio_sigma_file='__none__'), runs=256, speed=(1.5, 3.0))
    s, xy = torch.tensor(res['table']['s']), torch.tensor(res['table']['xy'])
    at = lambda x: float(xy[(s - x).abs().argmin()])                   # noqa: E731
    assert at(0) < 0.02 < at(2) < at(5) < at(10) < at(14)
    assert 0.2 < at(14) < 1.0                                          # in the range the shipped table has


def test_vio_dropout_holds_the_estimate_then_reinitialises():
    n = 8
    r = Realism(RealismCfg(vio_pos=True), n, 'cpu', 0.1, 1.5, (0.012, 0.012, 0.02), (80, 128), 88.1, 5.0)
    pos, vel = torch.zeros(n, 3), torch.tensor([[2.0, 0.0, 0.0]]).repeat(n, 1)
    r.vio_pos_reset(torch.arange(n), torch.zeros(n), torch.ones(n, dtype=torch.bool), pos, torch.zeros(n), vel)
    r.vio_pose(pos, vel, torch.zeros(n))
    r.vio_pos_vel.drop_left[:] = 0.35                                  # VIO lost for 4 steps
    held = []
    for k in range(6):
        pos = pos + vel * 0.1
        p_hat, _, ok = r.vio_pose(pos, vel, torch.zeros(n))
        held.append((p_hat.clone(), ok.clone()))
    assert all(not h[1].any() for h in held[:4]) and held[5][1].all()
    assert all(torch.equal(held[0][0], h[0]) for h in held[:4])        # frozen while lost
    assert ((held[5][0] - pos).norm(dim=-1) < 0.5).all()              # back to tracking (+ re-init jump)


def test_vio_ramp_off_envs_get_the_true_pose():
    r = Realism(RealismCfg(vio_pos=True, ramp_steps=10**9), 4, 'cpu', 0.1, 1.5, (0.012, 0.012, 0.02), (80, 128),
                88.1, 5.0)
    r.draw_mask(torch.arange(4), 0)
    assert not r.mask.any()
    pos = torch.randn(4, 3)
    r.vio_pos_reset(torch.arange(4), torch.full((4,), 10.0), torch.zeros(4, dtype=torch.bool), pos, torch.zeros(4),
                    torch.zeros(4, 3))
    p_hat, y_hat, ok = r.vio_pose(pos, torch.zeros(4, 3), torch.zeros(4))
    assert torch.equal(p_hat, pos) and (y_hat == 0).all() and ok.all()
