import numpy as np
import pytest
import torch

from dronehacks.bank import BANK_LAUNCH, BANK_PAD
from dronehacks.env import CourseEnv, CourseEnvCfg


def _env(**kw):
    cfg = dict(num_envs=8, device='cpu', seed=0, depth_width=32, depth_height=20)
    cfg.update(kw)
    return CourseEnv(CourseEnvCfg(**cfg))


def _queue(env, kind):
    cell = int(np.nonzero(env._bank_kind.numpy() == kind)[0][0])
    env.set_eval_queue(torch.full((env.num_envs, 1), cell))


@pytest.fixture(scope='module')
def env():
    return _env()


def test_observation_shapes(env):
    obs = env.get_observations()
    assert obs['policy'].shape == (8, 2, 20, 32)
    assert obs['proprio'].shape == (8, env.proprio_dim) and env.proprio_dim == 10
    assert obs['policy'].min() >= 0 and obs['policy'].max() <= 1
    o, r, d, extras = env.step(torch.zeros(8, 4))
    assert r.shape == (8,) and d.shape == (8,) and 'time_outs' in extras


def test_launch_descent_is_not_a_crash(env):
    """Ground reaction: commanding down on the launch pad leaves the drone parked, not crashed."""
    _queue(env, BANK_LAUNCH)
    for _ in range(20):
        _, _, done, _ = env.step(torch.tensor([[0., 0., 1., 0.]]).repeat(8, 1))
        assert not done.any()
    assert (-env.plant.pos_ned[:, 2] >= env.cfg.rest_alt - 1e-5).all()


def test_climb_from_launch(env):
    _queue(env, BANK_LAUNCH)
    for _ in range(10):
        env.step(torch.tensor([[0., 0., -0.5, 0.]]).repeat(8, 1))
    assert (-env.plant.pos_ned[:, 2] > 0.5).all()


def test_return_flag_before_landing_is_terminal():
    e = _env(num_envs=2, return_phase=True, bank_file='bank_return_v3.npz', max_episode_s=90.0)
    _queue(e, BANK_LAUNCH)
    _, _, done, _ = e.step(torch.tensor([[0., 0., 0., 0., 1.]]).repeat(2, 1))
    assert done.all()
    assert int(e.finished[-1][3, 0]) == CourseEnv.OUTCOMES.index('early_flag')


def test_pad_start_waits_for_flag_then_releases():
    e = _env(num_envs=2, return_phase=True, bank_file='bank_return_v3.npz', max_episode_s=90.0)
    _queue(e, BANK_PAD)
    start = e.plant.pos_ned.clone()
    for _ in range(5):                                    # parked: climbing without the flag does nothing
        e.step(torch.tensor([[0., 0., -1., 0., -1.]]).repeat(2, 1))
    assert torch.allclose(e.plant.pos_ned, start, atol=1e-5)
    for _ in range(8):                                    # flag on: it lifts off
        e.step(torch.tensor([[0., 0., -1., 0., 1.]]).repeat(2, 1))
    assert (-e.plant.pos_ned[:, 2] > 0.5).all()
    assert e.get_observations()['proprio'][:, -1].eq(1).all()      # latched flag is observed


def test_handoff_disarm_finishes_the_run_and_flag_in_the_air_is_a_crash():
    e = _env(num_envs=2, return_mode='handoff')
    assert e.num_actions == 5 and e.proprio_dim == 11            # 5 previous actions, no flag observation
    assert (e._bank_phase == 0).all()                            # run starts only
    _queue(e, BANK_LAUNCH)
    _, _, done, _ = e.step(torch.tensor([[0., 0., -0.5, 0., 1.]]).repeat(2, 1))
    assert done.all() and int(e.finished[-1][3, 0]) == CourseEnv.OUTCOMES.index('early_flag')
    # parked on the pad (as after a landing): disarming pays the finish and ends the episode as a success
    _queue(e, BANK_LAUNCH)
    e._phase[:] = 1
    e._next_cp[:] = 4
    e._park_pos[:] = e.plant.pos_ned
    _, r, done, _ = e.step(torch.tensor([[0., 0., 0., 0., 1.]]).repeat(2, 1))
    assert done.all() and (r > 0).all() and int(e.finished[-1][3, 0]) == CourseEnv.OUTCOMES.index('success')


def test_return_policy_starts_released_and_has_no_flag():
    e = _env(num_envs=2, return_mode='return', bank_file='bank_return_v3.npz')
    assert e.num_actions == 4 and e.proprio_dim == 10
    assert (e._bank_phase > 0).all()                             # return-leg starts only
    _queue(e, BANK_PAD)
    assert (e._phase == 2).all() and e._grounded.all()           # released, resting on the pad
    for _ in range(8):
        e.step(torch.tensor([[0., 0., -1., 0.]]).repeat(2, 1))
    assert (-e.plant.pos_ned[:, 2] > 0.5).all()
