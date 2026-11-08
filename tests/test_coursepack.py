import math
import os

import numpy as np
import pytest
import torch

from dronehacks.bank import BANK_LAUNCH
from dronehacks.coursepack import load_pack
from dronehacks.env import ASSETS, CourseEnv, CourseEnvCfg

PACK = os.path.join(ASSETS, 'packs', 'gen10.npz')
pytestmark = pytest.mark.skipif(not os.path.exists(PACK), reason='no course pack (python -m dronehacks.coursepack build)')


@pytest.fixture(scope='module')
def pack():
    return load_pack(PACK)


@pytest.fixture(scope='module')
def env():
    return CourseEnv(CourseEnvCfg(num_envs=16, device='cpu', seed=0, depth_width=32, depth_height=20,
                                  course_pack=PACK, sgs_uniform=True))


def _check_envs(env, pack):
    for i in range(env.num_envs):
        k = int(env._var[i])
        c = pack[k]
        assert int(env._bank_course[env._cell[i]]) == k                       # the drawn cell's course
        assert np.allclose(env._off[i].numpy(), env._v_off[k].numpy())
        assert np.allclose(env._pad_c[i].numpy(), c.pad_center, atol=1e-5)
        assert np.allclose(env._goal[i].numpy(), c.goal, atol=1e-5)
        assert np.allclose(env._mark[i].numpy(), c.mark, atol=1e-5)
        assert np.allclose(env._gP0[:, 0, i].numpy(), np.array([g['A'] for g in c.gates]), atol=1e-5)
        assert np.allclose(env._gn[:, 0, i].numpy(), np.array([g['n'] for g in c.gates]), atol=1e-5)
        assert env._gcp[:, i].tolist() == [g['cp'] for g in c.gates]
        assert np.allclose(env._arena[i].numpy(), c.arena_half, atol=1e-5)
        assert int(env._r0[i]) <= int(env._route_idx[i]) < int(env._r1[i])


def test_pack_layout(pack):
    assert len(pack) == 10 and len({c.seed for c in pack}) == 10
    for c in pack:
        launch = c.bank.kind == BANK_LAUNCH
        assert launch.sum() > 0 and (c.sector[launch] == 0).all()
        d = (c.bank.yaw[launch] - c.launch_yaw + np.pi) % (2 * np.pi) - np.pi
        assert np.abs(d).max() <= math.radians(30) + 1e-5                  # facing the planned route +- 30 deg
        assert np.linalg.norm(c.bank.pos[launch, :2] - c.mark, axis=1).max() <= 0.3 + 1e-5
        assert (c.sector >= 0).all() and (c.sector <= 4).all() and len(c.gate_geo) == 5


def test_every_env_flies_its_own_course(env, pack):
    """Starts are clear on the env's OWN layout and its scoring geometry is that course's, after resets too."""
    _check_envs(env, pack)
    clear = env._clearance()
    assert (clear > env.cfg.collision_radius).all()
    g = torch.Generator().manual_seed(0)
    seen = set(env._var.tolist())
    for _ in range(40):
        a = torch.rand(env.num_envs, 4, generator=g) * 2 - 1
        obs, r, d, _ = env.step(a)
        assert torch.isfinite(obs['critic']).all() and torch.isfinite(r).all()
        seen |= set(env._var.tolist())
        # reset envs have just been placed: their (new) course geometry must match
        _check_envs(env, pack)
    assert len(seen) >= 5
    stats = env.pop_stats()
    assert stats['episodes'] > 0 and any(k.startswith('success_course') for k in stats)


def test_pack_rejects_return_modes():
    with pytest.raises(ValueError, match='course_pack'):
        CourseEnv(CourseEnvCfg(num_envs=2, device='cpu', depth_width=32, depth_height=20, course_pack=PACK,
                               return_mode='handoff'))


def test_planned_route_scores_every_checkpoint_and_lands(pack):
    """Fed through the scorer as substep positions, each course's planned route clears cp 1-4 and lands on its pad:
    gates, sectors, pad and the launch cell agree with each other."""
    env = CourseEnv(CourseEnvCfg(num_envs=1, device='cpu', depth_width=32, depth_height=20, course_pack=PACK))
    for k, c in enumerate(pack):
        cell = int(np.nonzero((env._bank_course.numpy() == k) & (env._bank_kind.numpy() == BANK_LAUNCH))[0][0])
        env.set_eval_queue(torch.tensor([[cell]]))
        last = c.route.points[-1].astype(np.float64)
        descent = np.stack([np.full(60, last[0]), np.full(60, last[1]), np.linspace(last[2], -0.04, 60)], 1)
        path = torch.tensor(np.concatenate([c.route.points, descent]), dtype=torch.float32)
        prev, landed = path[0], False
        for i in range(1, len(path), 10):
            sub = path[i:i + 10].unsqueeze(1)
            env.episode_length_buf += 1
            _, land, bad = env._score_step(sub, torch.cat([prev.view(1, 1, 3), sub[:-1]]), -sub[..., 2])
            prev = sub[-1, 0]
            assert not bool(bad.any())
            if bool(land.any()):
                landed = True
                break
        assert landed and env._cleared[0].all(), (c.seed, env._cleared[0].tolist())


def test_pack_rejects_real_course_edits():
    for kw in (dict(course_mods='drop:wall5'), dict(gate_extend=True), dict(cp0_wall='wall3'), dict(perturb=0.5),
               dict(extra_boulders=(1, 2)), dict(arena_half=2.85)):
        with pytest.raises(ValueError, match='course_pack'):
            CourseEnv(CourseEnvCfg(num_envs=2, device='cpu', depth_width=32, depth_height=20, course_pack=PACK, **kw))


@pytest.mark.parametrize('ramp', [0, 1000])
def test_pack_with_realism_recipe(pack, ramp):
    """The training recipe on a pack: full realism (stereo depth, holes, see-through net + room per course), speed
    curriculum, skip_terminal, cp_shaping, invalid mask, critic_full, launch / liftoff starts and the start-heading
    filter. With a realism ramp every course is stacked twice (net drawn / see-through) and an env must still fly
    its own course's copy."""
    from dronehacks.realism import parse
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rc = parse('real{hole_frac_max=0.45;edge_fill_cols=16;net_return_prob_max=0.15}', seed=0, ramp_steps=ramp)
    env = CourseEnv(CourseEnvCfg(num_envs=32, device=dev, seed=1, course_pack=PACK, realism=rc, skip_terminal=True,
                                 cp_shaping=0.02, v_max=(1.2, 0.6, 0.6), v_max_start=(5.0, 3.0, 2.0),
                                 v_max_anneal_steps=10_000, invalid_mask=True, critic_full=True, launch_frac=0.3,
                                 launch_liftoff_frac=0.5, start_heading_max_deg=120.0))
    n = len(pack)
    assert env._n_lay == n and env._v_off.shape[0] == (2 * n if ramp else n)
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(30):
        obs, r, d, _ = env.step(torch.rand(env.num_envs, 4, generator=g, device=dev) * 2 - 1)
        assert torch.isfinite(r).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(obs['proprio']).all()
        assert obs['policy'].shape[1] == 2 * env.cfg.frame_stack
        c = env._course
        assert (env._var % n == c).all() and (env._bank_course[env._cell] == c).all()
        assert ((env._route_idx >= env._r0) & (env._route_idx < env._r1)).all()
        if ramp:                                    # the see-through copy exactly when the effects are on
            assert ((env._var >= n) == env.real.mask).all()
    stats = env.pop_stats()
    assert stats['episodes'] > 0 and any(k.startswith('success_course') for k in stats)
