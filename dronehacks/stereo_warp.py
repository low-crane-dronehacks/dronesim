"""Fused GPU path for ``Realism.corrupt_depth`` (the OAK-D stereo model): three Warp launches instead of ~60 torch
ops over sensor-resolution (E,400,640) tensors.

  kernels 1-2 (only with ``occlusion``): the right-to-left running minimum of the right-image column
           xr = u - d(u), at every policy-block boundary: per-block minima (one thread per block), then a
           right-to-left scan over the blocks of each row.
  kernel 3 (one thread per POLICY pixel): for the kh x kw render pixels of its block, sparse net returns,
           disparity noise + quantisation, min range, the textureless-hole field (bilinear upsampling of the same
           coarse random grid, drawn from ``Realism.gen`` exactly as the torch path draws it), stereo occlusion
           (continuing the running minimum leftwards through the block), the left band; then the
           valid-mask average pool with ``min_valid_frac``, the miss rule, ``invalid_fill``, and the clean-depth
           override for envs whose stereo model is off (``mask & depth_on``).

Deterministic parts are op-for-op the torch code (incl. torch's ``scalar / t == reciprocal(t) * scalar`` and
``t / scalar == t * (1 / scalar)``). The per-pixel randoms (net-return uniform, disparity Gaussian) come from a
counter-based Warp RNG instead of ``Realism.gen``: same distributions, different streams. The block sum runs
right-to-left instead of avg_pool2d's left-to-right, so pooled depth can differ by float32 rounding.
"""
from __future__ import annotations

from typing import Optional

import torch
import warp as wp

wp.init()

_FILL = {'near': 0, 'far': 1, 'hold': 2}


@wp.func
def _hole_field(grid: wp.array3d(dtype=wp.float32), e: int, r: int, u: int, sh: float, sw: float) -> float:
    """F.interpolate(grid, (Hr, Wr), mode='bilinear', align_corners=False) at (r, u)."""
    gh = grid.shape[1]
    gw = grid.shape[2]
    hr = sh * (float(r) + 0.5) - 0.5
    if hr < 0.0:
        hr = 0.0
    wr = sw * (float(u) + 0.5) - 0.5
    if wr < 0.0:
        wr = 0.0
    h1 = int(hr)
    w1 = int(wr)
    h1p = int(0)
    if h1 < gh - 1:
        h1p = 1
    w1p = int(0)
    if w1 < gw - 1:
        w1p = 1
    h1l = hr - float(h1)
    h0l = 1.0 - h1l
    w1l = wr - float(w1)
    w0l = 1.0 - w1l
    return h0l * (w0l * grid[e, h1, w1] + w1l * grid[e, h1, w1 + w1p]) + \
        h1l * (w0l * grid[e, h1 + h1p, w1] + w1l * grid[e, h1 + h1p, w1 + w1p])


@wp.struct
class NetRays:
    """The net-only scene, cast on demand: only pixels whose net-return draw fires need its depth."""
    mesh: wp.uint64
    pos: wp.array2d(dtype=wp.float32)            # (E,3) camera origin, NED (incl. the layout offset)
    rpy: wp.array2d(dtype=wp.float32)
    width: int
    height: int
    fx: float
    fy: float
    near: float
    far: float
    pixel_offset: float
    z_mode: int


@wp.func
def _net_depth(nr: NetRays, e: int, v: int, u: int) -> float:
    """render.mesh_depth_kernel for one pixel (normalised depth, miss = 1), op for op."""
    cu = (wp.float32(u) + nr.pixel_offset) - 0.5 * wp.float32(nr.width)
    cv = (wp.float32(v) + nr.pixel_offset) - 0.5 * wp.float32(nr.height)
    bx = 1.0
    by = cu / nr.fx
    bz = cv / nr.fy
    c_r = wp.cos(nr.rpy[e, 0])
    s_r = wp.sin(nr.rpy[e, 0])
    c_p = wp.cos(nr.rpy[e, 1])
    s_p = wp.sin(nr.rpy[e, 1])
    c_y = wp.cos(nr.rpy[e, 2])
    s_y = wp.sin(nr.rpy[e, 2])
    dx = (c_y * c_p) * bx + (c_y * s_p * s_r - s_y * c_r) * by + (c_y * s_p * c_r + s_y * s_r) * bz
    dy = (s_y * c_p) * bx + (s_y * s_p * s_r + c_y * c_r) * by + (s_y * s_p * c_r - c_y * s_r) * bz
    dz = (-s_p) * bx + (c_p * s_r) * by + (c_p * c_r) * bz
    body_len = wp.sqrt(bx * bx + by * by + bz * bz)
    inv_len = 1.0 / wp.sqrt(dx * dx + dy * dy + dz * dz)
    d = wp.vec3(dx * inv_len, dy * inv_len, dz * inv_len)
    o = wp.vec3(nr.pos[e, 0], nr.pos[e, 1], nr.pos[e, 2]) + d * nr.near
    t = nr.far
    q = wp.mesh_query_ray(nr.mesh, o, d, nr.far - nr.near)
    if q.result:
        t = wp.clamp(nr.near + q.t, nr.near, nr.far)
        if nr.z_mode == 1:
            t = t * (bx / body_len)
    return t / nr.far


@wp.func
def _true_z(z_n: wp.array3d(dtype=wp.float32), net_n: wp.array3d(dtype=wp.float32), has_net: int,
            net_p: wp.array(dtype=wp.float32), e: int, r: int, u: int, far: float, un: float) -> wp.vec3:
    """-> (z after sparse net returns [m], net hit (0/1), z clean [m]); ``un`` = the pixel's net uniform.
    ``net_n`` is only read where ``un`` fires (with on-demand net rays it holds nothing anywhere else)."""
    zc = z_n[e, r, u] * far
    zm = zc
    hit = float(0.0)
    if has_net != 0:
        if un < net_p[e]:
            nz = net_n[e, r, u] * far
            if nz < zm:
                zm = nz
                hit = 1.0
    return wp.vec3(zm, hit, zc)


@wp.kernel
def net_candidates_kernel(
    nr: NetRays, net_p: wp.array(dtype=wp.float32), seed: int, Hr: int, Wr: int, cap: int,
    count: wp.array(dtype=wp.int32),             # (E*Hr,) zeroed
    lst: wp.array2d(dtype=wp.int32),             # (E*Hr, cap) columns whose net-return draw fires
    net_buf: wp.array3d(dtype=wp.float32),       # (E,Hr,Wr) written only at those columns
):
    e, r, b = wp.tid()
    row = e * Hr + r
    for j in range(8):
        u = b * 8 + j
        if u < Wr:
            state = wp.rand_init(seed, row * Wr + u)
            if wp.randf(state) < net_p[e]:
                slot = wp.atomic_add(count, row, 1)
                if slot < cap:
                    lst[row, slot] = u
                else:                                       # list full (never at the configured rates): cast here
                    net_buf[e, r, u] = _net_depth(nr, e, r, u)


@wp.kernel
def net_rays_kernel(nr: NetRays, Hr: int, count: wp.array(dtype=wp.int32), lst: wp.array2d(dtype=wp.int32),
                    net_buf: wp.array3d(dtype=wp.float32)):
    row, slot = wp.tid()
    if slot < count[row]:
        u = lst[row, slot]
        e = row // Hr
        r = row - e * Hr
        net_buf[e, r, u] = _net_depth(nr, e, r, u)


@wp.kernel
def stereo_blockmin_kernel(
    z_n: wp.array3d(dtype=wp.float32), net_n: wp.array3d(dtype=wp.float32), has_net: int,
    net_p: wp.array(dtype=wp.float32), far: float, fB: float, r_px: float, seed: int, kw: int,
    bmin: wp.array3d(dtype=wp.float32),          # (E,Hr,Wb): min of xr = u - d_r(u) over each kw-column block
):
    e, r, b = wp.tid()
    Hr = z_n.shape[1]
    Wr = z_n.shape[2]
    m = float(wp.inf)
    for j in range(kw):
        u = b * kw + j
        if u < Wr:
            un = float(1.0)
            if has_net != 0:
                state = wp.rand_init(seed, (e * Hr + r) * Wr + u)
                un = wp.randf(state)
            t = _true_z(z_n, net_n, has_net, net_p, e, r, u, far, un)
            d_r = ((1.0 / wp.max(t[0], 1.0e-3)) * fB) * r_px
            m = wp.min(m, float(u) - d_r)
    bmin[e, r, b] = m


@wp.kernel
def stereo_suffix_kernel(bmin: wp.array3d(dtype=wp.float32)):
    """In place: block b -> min of xr over all columns RIGHT of block b (inf for the last)."""
    e, r = wp.tid()
    run = float(wp.inf)
    Wb = bmin.shape[2]
    for i in range(Wb):
        b = Wb - 1 - i
        x = bmin[e, r, b]
        bmin[e, r, b] = run
        run = wp.min(run, x)


@wp.kernel
def stereo_kernel(
    z_n: wp.array3d(dtype=wp.float32),           # (E,Hr,Wr) normalised render depth (miss = 1)
    net_n: wp.array3d(dtype=wp.float32),         # (E,Hr,Wr) net-only render, if has_net
    has_net: int,
    net_p: wp.array(dtype=wp.float32),           # (E,)
    far: float,
    miss_thr: float,
    fB: float,
    sigma: float,
    q: float,
    inv_q: float,
    max_disp: float,
    has_hole: int,
    grid: wp.array3d(dtype=wp.float32),          # (E,gh,gw) coarse hole field
    hole_frac: float,
    g_sh: float,
    g_sw: float,
    occlusion: int,
    suffix: wp.array3d(dtype=wp.float32),       # (E,Hr,Wb) min xr right of each block
    r_px: float,                                 # render px per sensor px (occlusion)
    has_band: int,
    band: wp.array(dtype=wp.float32),            # (E,) sensor px
    s_px: float,                                 # sensor px per render px (band)
    kh: int,
    kw: int,
    pooled: int,
    min_valid: float,
    fill_mode: int,
    has_prev: int,
    prev: wp.array3d(dtype=wp.float32),
    on: wp.array(dtype=wp.bool),                 # (E,) stereo model on
    seed: int,
    out: wp.array3d(dtype=wp.float32),           # (E,H,W)
    bad_out: wp.array3d(dtype=wp.bool),
):
    e, vo, wo = wp.tid()
    Hr = z_n.shape[1]
    Wr = z_n.shape[2]
    n_valid = int(0)
    n_miss = int(0)
    s_z = float(0.0)
    s_zc = float(0.0)
    zpix = float(0.0)
    zcpix = float(0.0)
    for rr in range(kh):
        r = vo * kh + rr
        run = float(wp.inf)
        if occlusion != 0:
            run = suffix[e, r, wo]
        for j in range(kw):
            u = wo * kw + kw - 1 - j                       # right to left: the occlusion minimum runs leftwards
            state = wp.rand_init(seed, (e * Hr + r) * Wr + u)
            un = float(1.0)
            if has_net != 0:
                un = wp.randf(state)                       # first draw: the same uniform the block-min kernel used
            t = _true_z(z_n, net_n, has_net, net_p, e, r, u, far, un)
            zm = t[0]
            zc = t[2]
            miss = zc >= miss_thr and t[1] == 0.0
            d = (1.0 / wp.max(zm, 1.0e-3)) * fB
            if sigma > 0.0:
                d = d + wp.randn(state) * sigma
            d = wp.rint(d * q) * inv_q
            invalid = d > max_disp or d <= 0.0
            z = (1.0 / wp.max(d, 1.0e-6)) * fB
            if has_hole != 0:
                if _hole_field(grid, e, r, u, g_sh, g_sw) < hole_frac:
                    invalid = True
            if occlusion != 0:
                d_r = ((1.0 / wp.max(zm, 1.0e-3)) * fB) * r_px
                xr = float(u) - d_r
                if run <= xr - 0.5 or xr < 0.0:
                    invalid = True
                run = wp.min(run, xr)
            if has_band != 0:
                bw = band[e]                     # >= 0: left band width; < 0: right band, invalid past -band
                if bw >= 0.0:
                    if float(u) * s_px < bw:
                        invalid = True
                elif float(u) * s_px > -bw:
                    invalid = True
            valid = (not invalid) and (not miss)
            if valid:
                n_valid += 1
                s_z += z
                zpix = z
            else:
                zpix = 0.0
            if miss:
                n_miss += 1
            s_zc += zc
            zcpix = zc
    inv_far = 1.0 / far
    zs = float(0.0)
    valid_l = False
    miss_l = False
    zclean = float(0.0)
    if pooled != 0:
        kk = float(kh * kw)
        vf = float(n_valid) / kk
        zs = (s_z / kk) / wp.max(vf, 1.0e-6)
        valid_l = vf >= min_valid
        miss_l = (float(n_miss) / kk > 0.5) and (not valid_l)
        zclean = s_zc / kk
    else:
        zs = zpix
        valid_l = n_valid > 0
        miss_l = n_miss > 0
        zclean = zcpix
    o = wp.clamp(zs * inv_far, 0.0, 1.0)
    if miss_l:
        o = 1.0
    bad = (not valid_l) and (not miss_l)
    if bad:
        if fill_mode == 0:
            o = 0.0
        elif fill_mode == 2 and has_prev != 0:
            o = prev[e, vo, wo]
        else:
            o = 1.0
    if not on[e]:
        o = wp.clamp(zclean * inv_far, 0.0, 1.0)
        bad = False
    out[e, vo, wo] = o
    bad_out[e, vo, wo] = bad


def corrupt_depth(real, z_n: torch.Tensor, net_n: Optional[torch.Tensor], far: float, net_rays=None) -> torch.Tensor:
    """``Realism.corrupt_depth`` from NORMALISED render depth (the MeshScene buffers, not cloned): z_m = z_n * far.
    ``net_rays`` = (MeshScene, pos (E,3), rpy (E,3)) instead of ``net_n``: the net is ray cast only at the pixels
    whose net-return draw fires (a few %; compacted per row first so the casts are dense), with the render
    kernel's exact ray math, so the result equals rendering it in full."""
    c = real.cfg
    dev = z_n.device
    E, Hr, Wr = z_n.shape
    H, W = real.H, real.W
    pooled = int(Wr != W or Hr != H)
    kh, kw = (Hr // H, Wr // W) if pooled else (1, 1)
    s = Wr / W
    z_n = z_n.to(torch.float32).contiguous()
    has_net = net_n is not None
    net_n = net_n.to(torch.float32).contiguous() if has_net else z_n
    real._wp_calls = getattr(real, '_wp_calls', 0) + 1
    seed = (int(c.seed) * 1000003 + real._wp_calls * 7919) % 2147483647
    f32 = torch.float32
    net_p = real.net_p.reshape(E).to(f32).contiguous()
    if net_rays is not None and not has_net:             # net depth only where a return fires: compact, then cast
        sc, pos, rpy = net_rays
        assert (sc.height, sc.width) == (Hr, Wr), 'net scene must render at the depth resolution'
        from dronehacks.render import PIXEL_SAMPLE_OFFSET, _DEPTH_TYPES
        nr = NetRays()
        nr.mesh = sc._mesh.id
        nr.pos = wp.from_torch(pos.to(f32).contiguous(), dtype=wp.float32)
        nr.rpy = wp.from_torch(rpy.to(f32).contiguous(), dtype=wp.float32)
        nr.width, nr.height, nr.fx, nr.fy = sc.width, sc.height, sc.fx, sc.fy
        nr.near, nr.far = sc.near_clip, sc.far_clip
        nr.pixel_offset, nr.z_mode = PIXEL_SAMPLE_OFFSET, _DEPTH_TYPES[sc.depth_type]
        p_max = max(float(c.net_return_prob), float(c.net_return_prob_max))
        cap = int(min(Wr, Wr * p_max * 2 + 32))
        buf = getattr(real, '_wp_net_buf', None)
        if buf is None or buf.shape != z_n.shape:
            buf = real._wp_net_buf = torch.empty_like(z_n)
        count = torch.zeros(E * Hr, device=dev, dtype=torch.int32)
        lst = torch.empty(E * Hr, cap, device=dev, dtype=torch.int32)
        wi = lambda t: wp.from_torch(t, dtype=wp.int32)  # noqa: E731
        wp.launch(net_candidates_kernel, dim=(E, Hr, (Wr + 7) // 8), device=str(dev),
                  inputs=[nr, wp.from_torch(net_p), seed, Hr, Wr, cap],
                  outputs=[wi(count), wi(lst), wp.from_torch(buf)])
        wp.launch(net_rays_kernel, dim=(E * Hr, cap), device=str(dev),
                  inputs=[nr, Hr, wi(count), wi(lst), wp.from_torch(buf)])
        net_n, has_net = buf, True
    has_net_flag = int(has_net)
    dummy3 = torch.zeros(1, 1, 1, device=dev)
    grid = dummy3
    if c.hole_frac > 0:                                  # the torch path's exact draw on the private generator
        gh, gw = max(2, int(Hr / (c.hole_scale_px * s))), max(2, int(Wr / (c.hole_scale_px * s)))
        grid = torch.rand(E, 1, gh, gw, device=dev, generator=real.gen)[:, 0].contiguous()
        if c.hole_frac_max > c.hole_frac:            # per-env threshold hole_e == field * (hole_frac / hole_e) < hole_frac
            grid = (grid * (c.hole_frac / real.hole.clamp(min=1e-6)).view(E, 1, 1)).contiguous()
    fB = float(real.fB)
    r_px = Wr / c.sensor_width_px
    suffix = dummy3
    wa = lambda t: wp.from_torch(t, dtype=wp.float32)  # noqa: E731
    if c.occlusion:                                      # running min of xr from the right, at block granularity
        Wb = (Wr + kw - 1) // kw
        suffix = torch.empty(E, Hr, Wb, device=dev, dtype=f32)
        wp.launch(stereo_blockmin_kernel, dim=(E, Hr, Wb), device=str(dev),
                  inputs=[wa(z_n), wa(net_n), has_net_flag, wa(net_p), float(far), fB, float(r_px), seed, kw],
                  outputs=[wa(suffix)])
        wp.launch(stereo_suffix_kernel, dim=(E, Hr), device=str(dev), inputs=[wa(suffix)])
    has_band = c.left_band_px > 0 or c.left_band_max_px > 0
    band = real.band.reshape(E).to(f32).contiguous()
    fill = _FILL.get(c.invalid_fill, 1)
    prev = real.prev_depth
    has_prev = int(fill == 2 and prev is not None)
    prev_t = prev.to(f32).contiguous() if has_prev else dummy3
    on = (real.mask & real.depth_on).contiguous()
    q = 2 ** c.subpixel_bits
    out = torch.empty(E, H, W, device=dev, dtype=f32)
    bad = torch.empty(E, H, W, device=dev, dtype=torch.bool)
    wp.launch(stereo_kernel, dim=(E, H, W), device=str(dev), inputs=[
        wa(z_n), wa(net_n), has_net_flag, wa(net_p), float(far), float(real.far * 0.999), fB,
        float(c.disparity_sigma_px), float(q), float(1.0 / q), float(c.max_disparity_px),
        int(c.hole_frac > 0), wa(grid), float(c.hole_frac),
        float(grid.shape[1]) / float(Hr) if c.hole_frac > 0 else 1.0,
        float(grid.shape[2]) / float(Wr) if c.hole_frac > 0 else 1.0,
        int(c.occlusion), wa(suffix), float(r_px), int(has_band), wa(band), float(c.sensor_width_px / Wr),
        kh, kw, pooled, float(c.min_valid_frac), fill, has_prev, wa(prev_t),
        wp.from_torch(on, dtype=wp.bool), seed,
    ], outputs=[wa(out), wp.from_torch(bad, dtype=wp.bool)])
    real.last_bad = bad
    real.prev_depth = out
    return out
