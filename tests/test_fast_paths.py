"""The fused GPU paths (dronehacks/quad_warp.py, dronehacks/stereo_warp.py) against the torch reference paths.

Deterministic parts only: same initial state + actions -> same trajectory; noise-free stereo -> same image.
Skipped without CUDA (the fast paths only run on the GPU).
"""
import math

import pytest
import torch

from dronehacks.quad import TorchQuad
from dronehacks.realism import Realism, parse

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA GPU')
DEV = 'cuda:0'


def _plant_pair(spec, E=256, grounded=False):
    """Two identical plants + realism states (fast, reference), reset to the same random start."""
    out = []
    for fast in (True, False):
        torch.manual_seed(3)
        q = TorchQuad(E, DEV, v_max=(5.0, 3.0, 2.0))
        q.fast = fast
        r = None
        if spec:
            r = Realism(parse(spec, seed=11), E, DEV, 0.1, q.mass, q.params['inertia'], (80, 128), 88.0, 5.0)
            r.configure_plant(q)
        g = torch.Generator(device=DEV).manual_seed(5)
        ids = torch.arange(E, device=DEV)
        pos = torch.randn(E, 3, device=DEV, generator=g) * torch.tensor([2.0, 2.0, 0.3], device=DEV)
        pos[:, 2] -= 1.0
        yaw = torch.rand(E, device=DEV, generator=g) * 2 * math.pi - math.pi
        vel = torch.randn(E, 3, device=DEV, generator=g)
        roll = torch.randn(E, device=DEV, generator=g) * 0.2
        pitch = torch.randn(E, device=DEV, generator=g) * 0.2
        if grounded:                                          # half the envs parked on the floor
            pos[::2, 2] = -0.08
            vel[::2] = 0.0
            roll[::2] = 0.0
            pitch[::2] = 0.0
            q.grounded = torch.zeros(E, dtype=torch.bool, device=DEV)
            q.grounded[::2] = True
            q.ground_alt = 0.08
        q.reset_idx(ids, pos, yaw, vel, roll, pitch)
        if r is not None:
            r.on_reset(ids, q)
        out.append((q, r))
    return out


def _fly(q, r, steps, seed=7, trace=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    E = q.num_envs
    traj, subs = [], []
    if trace:
        q.trace = []
    for t in range(steps):
        a = torch.rand(E, 4, device=DEV, generator=g) * 2 - 1
        q.substep_pos = []
        seq = None
        if r is not None:
            r.update_thrust(q, torch.full((E,), t * 0.1, device=DEV))
            seq = r.substep_actions(a, q.decimation)
        if seq is None:
            q.step_control(a)
        else:
            q.step_control_seq(seq)
        subs.append(torch.stack(q.substep_pos))
        q.substep_pos = None
        traj.append(torch.cat([q.pos_w, q.quat_w, q.lin_vel_w, q.ang_vel_b, q.applied_thrust], -1))
    out = torch.stack(traj), torch.cat(subs)
    if trace:
        out = out + (torch.stack(q.trace),)
        q.trace = None
    return out


@cuda
@pytest.mark.parametrize('spec', ['', 'plant,latency', 'plant,latency,px4,battery'])
def test_fused_physics_matches_torch(spec):
    (qf, rf), (qt, rt) = _plant_pair(spec)
    tf, sf = _fly(qf, rf, 100)
    tt, st = _fly(qt, rt, 100)
    assert torch.isfinite(tt).all()
    assert (tf - tt).abs().max() < 1e-4, (tf - tt).abs().max()
    assert (sf - st).abs().max() < 1e-4
    # integral state too
    if rf is not None and rf.cfg.px4_loop:
        assert (qf.controller._vel_int - qt.controller._vel_int).abs().max() < 1e-4


@cuda
def test_fused_physics_ground_clamp_and_trace():
    (qf, rf), (qt, rt) = _plant_pair('plant', grounded=True)
    tf, sf, trf = _fly(qf, rf, 30, trace=True)
    tt, st, trt = _fly(qt, rt, 30, trace=True)
    assert (tf - tt).abs().max() < 1e-4
    assert (trf - trt).abs().max() < 1e-4
    assert (sf[:, ::2, 2] <= -0.08 + 1e-6).all()            # NED z: parked envs never sink below rest height


# ── stereo model ─────────────────────────────────────────────────────────────────────────────────────────────

FAR = 5.0


def _scene(E, Hr, Wr, seed=0):
    """Normalised z-depth with foreground boxes (occlusion edges), misses, and pixels inside the min range."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    v = torch.linspace(0, 1, Hr, device=DEV).view(1, Hr, 1)
    u = torch.linspace(0, 1, Wr, device=DEV).view(1, 1, Wr)
    z = 2.5 + 2.0 * u * v + 0.3 * torch.rand(E, 1, 1, device=DEV, generator=g)           # background, metres
    z = z.expand(E, Hr, Wr).clone()
    for _ in range(6):
        r0 = torch.randint(0, Hr - 20, (E,), device=DEV, generator=g)
        c0 = torch.randint(0, Wr - 40, (E,), device=DEV, generator=g)
        hh = torch.randint(10, Hr // 2, (E,), device=DEV, generator=g)
        ww = torch.randint(10, Wr // 3, (E,), device=DEV, generator=g)
        d = 0.2 + 2.0 * torch.rand(E, device=DEV, generator=g)
        rr = torch.arange(Hr, device=DEV).view(1, Hr, 1)
        cc = torch.arange(Wr, device=DEV).view(1, 1, Wr)
        box = ((rr >= r0.view(E, 1, 1)) & (rr < (r0 + hh).view(E, 1, 1)) & (cc >= c0.view(E, 1, 1))
               & (cc < (c0 + ww).view(E, 1, 1)))
        z = torch.where(box, torch.minimum(z, d.view(E, 1, 1)), z)
    z[:, : Hr // 6, Wr // 2:] = FAR                                                       # sky: misses
    net = torch.where(torch.rand(E, Hr, Wr, device=DEV, generator=g) < 0.5, 1.2 + 0.0 * z, torch.full_like(z, FAR))
    return (z / FAR).clamp(max=1.0), (net / FAR).clamp(max=1.0)


def _stereo_pair(E=8, scale=5, **cfg):
    out = []
    for fast in (True, False):
        c = parse('depth', seed=3, render_scale=scale, **cfg)
        r = Realism(c, E, DEV, 0.1, 1.5, (0.012, 0.012, 0.02), (80, 128), 88.0, FAR)
        r.fast = fast
        r._n = lambda *s: torch.zeros(*s, device=DEV)     # sigma = 0: keep the hole grid draw aligned on r.gen
        if c.left_band_max_px > 0:
            r.band = torch.linspace(0, c.left_band_max_px, E, device=DEV).view(E, 1, 1)
        out.append(r)
    return out


def _compare(rf, rt, scale=5, net=False, frames=2, max_bad_mismatch=0):
    E = rf.E
    for k in range(frames):
        z, nz = _scene(E, 80 * scale, 128 * scale, seed=k)
        of = rf.corrupt_depth(z, nz if net else None, far_scale=FAR)
        ot = rt.corrupt_depth(z, nz if net else None, far_scale=FAR)
        assert of.shape == ot.shape == (E, 80, 128)
        assert (of - ot).abs().max() < 1e-5, (of - ot).abs().max()
        assert int((rf.last_bad != rt.last_bad).sum()) <= max_bad_mismatch
        assert 0 < float(rt.last_bad.float().mean()) < 1 or not rt.cfg.occlusion
    return of, ot


@cuda
@pytest.mark.parametrize('scale', [5, 1])
def test_fused_stereo_null_matches_torch(scale):
    """depth_null: the full sensor-resolution pipeline with every error off."""
    _compare(*_stereo_pair(scale=scale, disparity_sigma_px=0.0, subpixel_bits=16, hole_frac=0.0, occlusion=False,
                           max_disparity_px=1e6), scale=scale)


@cuda
@pytest.mark.parametrize('scale', [5, 1])
@pytest.mark.parametrize('fill', ['near', 'far', 'hold'])
def test_fused_stereo_deterministic_parts_match_torch(scale, fill):
    """Quantisation, min range, occlusion, left band, holes (same coarse grid), pooling, fills, misses."""
    rf, rt = _stereo_pair(scale=scale, disparity_sigma_px=0.0, left_band_max_px=100.0, invalid_fill=fill)
    _compare(rf, rt, scale=scale, frames=3, max_bad_mismatch=2)


@cuda
@pytest.mark.parametrize('scale', [5, 1])
def test_fused_stereo_right_band_matches_torch_and_lands_right(scale):
    """band < 0 encodes a RIGHT-edge band (invalid past -band sensor px), as the drone's VPI node produces."""
    rf, rt = _stereo_pair(scale=scale, disparity_sigma_px=0.0, left_band_max_px=100.0, occlusion=False, hole_frac=0.0,
                          max_disparity_px=1e6)
    E = rf.E
    w = torch.linspace(0, 100.0, E, device=DEV).view(E, 1, 1)
    for r in (rf, rt):
        r.band = torch.where(torch.arange(E, device=DEV).view(E, 1, 1) % 2 == 1, -(640.0 - w).clamp(min=1e-3), w)
    _compare(rf, rt, scale=scale, frames=2, max_bad_mismatch=2)
    z, _ = _scene(E, 80 * scale, 128 * scale, seed=0)
    rt.corrupt_depth(z, None, far_scale=FAR)
    bad = rt.last_bad.float().mean(1)                   # (E, 128) invalid share per column
    e = E - 1                                           # right band, width 100 px of 640 -> ~20 of 128 columns
    # (the scene's sky rows are misses, not 'bad', so a fully banded column reads ~0.83)
    assert bad[e, -15:].min() > 0.8 and bad[e, :15].max() < 0.3, bad[e]
    l = E - 2                                           # same width as a LEFT band: mirrored
    assert bad[l, :15].min() > 0.8 and bad[l, -15:].max() < 0.3, bad[l]


@cuda
def test_band_right_frac_draws_right_bands():
    q = TorchQuad(512, DEV, v_max=(5.0, 3.0, 2.0))
    c = parse('depth', seed=3, left_band_max_px=100.0, band_right_frac=0.5)
    r = Realism(c, 512, DEV, 0.1, q.mass, q.params['inertia'], (80, 128), 88.0, FAR)
    r.on_reset(torch.arange(512, device=DEV), q)
    neg = float((r.band < 0).float().mean())
    assert 0.35 < neg < 0.65, neg
    assert float(r.band[r.band < 0].min()) >= -640.0 and float(r.band[r.band < 0].max()) <= -540.0 + 1e-3


@cuda
@pytest.mark.parametrize('p', [0.0, 1.0])
def test_fused_stereo_net_returns_and_clean_override(p):
    """Net returns at prob 0 / 1 are deterministic; envs with the stereo model off get clean pooled depth.
    (No holes here: the torch path draws its net uniforms on r.gen before the hole grid, the fused one does not.)"""
    rf, rt = _stereo_pair(disparity_sigma_px=0.0, net_return_prob=p, net=True, hole_frac=0.0)
    for r in (rf, rt):
        r.depth_on[::3] = False
        r.mask[1::4] = False
    _compare(rf, rt, net=True, max_bad_mismatch=2)


@cuda
@pytest.mark.parametrize('p', [0.0, 1.0])
def test_fused_stereo_net_rays_on_demand_match_full_net_render(p):
    """Casting the net only where a return fires == rendering it in full (torch path) for p in {0, 1}."""
    from types import SimpleNamespace
    import numpy as np
    from dronehacks.render import MeshScene
    E, s = 4, 5
    verts, faces = [], []
    for i, y in enumerate(np.linspace(-1.5, 1.5, 7)):                   # vertical strips 1.5 m ahead (NED x)
        b = len(verts)
        verts += [(1.5, y, -2.0), (1.5, y + 0.2, -2.0), (1.5, y + 0.2, 1.0), (1.5, y, 1.0)]
        faces += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    net = MeshScene(SimpleNamespace(vertices=np.array(verts, np.float32), faces=np.array(faces, np.int32)), E,
                    128 * s, 80 * s, 88.0 * s, 88.0 * s, FAR, 0.05, depth_type='z', device=DEV)
    g = torch.Generator(device=DEV).manual_seed(1)
    pos = torch.randn(E, 3, device=DEV, generator=g) * 0.2
    rpy = torch.randn(E, 3, device=DEV, generator=g) * 0.1
    rf, rt = _stereo_pair(E=E, disparity_sigma_px=0.0, net_return_prob=p, net=True, hole_frac=0.0)
    z, _ = _scene(E, 80 * s, 128 * s)
    of = rf.corrupt_depth(z, None, far_scale=FAR, net_rays=(net, pos, rpy))
    ot = rt.corrupt_depth(z, None, far_scale=FAR, net_rays=(net, pos, rpy))
    assert (of - ot).abs().max() < 1e-5
    assert int((rf.last_bad != rt.last_bad).sum()) <= 2
    if p == 1.0:                                                       # the net strips really are in the image
        assert (ot - rt._corrupt_depth_torch(z * FAR, None)).abs().max() > 0.1


@cuda
def test_hole_frac_max_per_env_matches_torch_and_scales():
    """Per-episode hole fraction: fused == torch, and more holes for envs with a larger draw."""
    rf, rt = _stereo_pair(disparity_sigma_px=0.0, occlusion=False, hole_frac=0.04, hole_frac_max=0.5, max_disparity_px=1e6)
    E = rf.E
    h = torch.linspace(0.04, 0.5, E, device=DEV)
    for r in (rf, rt):
        r.hole = h.clone()
    _compare(rf, rt, frames=2, max_bad_mismatch=4)
    z, _ = _scene(E, 400, 640, seed=0)
    rt.corrupt_depth(z, None, far_scale=FAR)
    bad = rt.last_bad.float().mean((1, 2))
    assert float(bad[-1]) > float(bad[0]) + 0.2, bad
