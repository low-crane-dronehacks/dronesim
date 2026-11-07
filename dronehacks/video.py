"""Render a checkpoint's flight: FPV depth (left) + live top-down map (right), real time.

    python -m dronehacks.video checkpoints/comp_s2.pt --out out/comp_s2.mp4
    python -m dronehacks.video runs/ret/checkpoints/model_3051.pt --start final --pick success --out out/ret.mp4

``--start`` picks the bank start type (launch = the competition run; pad / climb / cross / final = return-leg
starts). ``--pick`` flies starts of that type in order until one ends that way (any / success / failure).
The map shows the pad ring (green) and the launch mark with its 2 ft return circle (yellow).

Needs ffmpeg (on PATH or $FFMPEG), pillow and matplotlib. Writes <out>.json with outcome / time / speed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess

import numpy as np
import torch

from dronehacks import rules
from dronehacks.bank import BANK_CLIMB, BANK_CROSS, BANK_FINAL, BANK_LAUNCH, BANK_PAD

STARTS = {'launch': BANK_LAUNCH, 'pad': BANK_PAD, 'climb': BANK_CLIMB, 'cross': BANK_CROSS, 'final': BANK_FINAL}
PHASES = ('run', 'parked on pad', 'returning', 'landed at mark')
from dronehacks.evaluate import load_policy
from dronehacks.mesh import CourseMesh
from dronehacks.render import MeshScene

FPS = 30


def _ffmpeg() -> str:
    exe = shutil.which('ffmpeg') or os.environ.get('FFMPEG')
    if not exe:
        raise SystemExit('ffmpeg not found: put it on PATH or set $FFMPEG')
    return exe


def _focal(size: int, fov_deg: float) -> float:
    return 0.5 * size / math.tan(math.radians(fov_deg) / 2.0)


def _map_base(course: CourseMesh, device: str, px: int, half: float):
    """Top-down image (PIL): wall / tunnel / boulder footprints and floor paint."""
    from PIL import Image
    gn, ge = np.meshgrid(np.linspace(half, -half, px), np.linspace(-half, half, px), indexing='ij')

    def mask(groups, z, thresh=0.012):
        sc = MeshScene(course.select(groups), 1, device=device)
        pts = np.stack([gn.ravel(), ge.ravel(), np.full(gn.size, z)], axis=1)
        return (sc.distance_at(pts) < thresh).reshape(px, px)

    img = np.zeros((px, px, 3), np.float32) + np.array([0.09, 0.10, 0.14])
    img[mask(('paint',), -0.003, 0.004)] = (0.55, 0.18, 0.17)
    img[mask(('boulder',), -0.25)] = (0.30, 0.45, 0.80)
    img[mask(('tunnel',), -0.9)] = (0.20, 0.62, 0.40)
    img[mask(('wall',), -0.9)] = (0.80, 0.82, 0.86)
    return Image.fromarray((img * 255).astype(np.uint8))


def write_fpv(path: str, course: CourseMesh, t, pos, rpy, speed, device: str, width=640, height=400,
              hfov=72.0, vfov=50.0, far=6.0, circles=(), labels=None) -> None:
    """FPV depth + top-down map video of a (T,) trajectory sampled at FPS (``pos``/``rpy`` (T,3) NED)."""
    import matplotlib
    from PIL import Image, ImageDraw
    cmap = matplotlib.colormaps['turbo_r']
    chunk = 30
    scene = MeshScene(course.select(('floor', 'wall', 'tunnel', 'boulder', 'net')), chunk, width, height,
                      _focal(width, hfov), _focal(height, vfov), far, depth_type='z', device=device)
    clr = MeshScene(course.select(('wall', 'tunnel', 'boulder')), 1, device=device).distance_at(pos)
    half, MAP = 3.15, height
    base = _map_base(course, device, MAP, half)
    sc = MAP / (2 * half)
    to_px = lambda n, e: ((e + half) * sc, (half - n) * sc)      # noqa: E731
    flown = [to_px(p[0], p[1]) for p in pos[::3]]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    proc = subprocess.Popen([_ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{width + MAP}x{height}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                             '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                            stdin=subprocess.PIPE)
    T = len(t)
    for c0 in range(0, T, chunk):
        n = min(chunk, T - c0)
        pp, rr = np.zeros((chunk, 3), np.float32), np.zeros((chunk, 3), np.float32)
        pp[:n], rr[:n] = pos[c0:c0 + n], rpy[c0:c0 + n]
        d = scene.render_depth(torch.from_numpy(pp).to(device), torch.from_numpy(rr).to(device)).cpu().numpy()
        for k in range(n):
            i = c0 + k
            rgb = (cmap(np.clip(d[k], 0, 1))[..., :3] * 255).astype(np.uint8)
            rgb[d[k] >= 1.0] = (12, 12, 16)                                  # beyond far clip
            left = Image.fromarray(rgb)
            dr = ImageDraw.Draw(left)
            dr.rectangle([0, 0, width, 22], fill=(0, 0, 0))
            dr.text((8, 5), f't {t[i]:5.1f}s  v {speed[i]:.2f} m/s  alt {-pos[i, 2]:.2f} m  clearance {clr[i]:.2f} m'
                    + (f'   {labels[i]}' if labels is not None else ''), fill=(235, 235, 235))
            right = base.copy()
            dm = ImageDraw.Draw(right)
            for (cn, ce, rad, col) in circles:
                px, py = to_px(cn, ce)
                dm.ellipse([px - rad * sc, py - rad * sc, px + rad * sc, py + rad * sc], outline=col, width=2)
            if i // 3 > 1:
                dm.line(flown[:i // 3], fill=(255, 255, 255), width=3)
            cx, cy = to_px(pos[i, 0], pos[i, 1])
            for s_ in (-1, 1):                                               # horizontal FOV edges
                ang = rpy[i, 2] + s_ * math.radians(hfov / 2)
                dm.line([(cx, cy), (cx + 0.55 * sc * math.sin(ang), cy - 0.55 * sc * math.cos(ang))],
                        fill=(120, 220, 255), width=1)
            dm.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 70, 70))
            frame = Image.new('RGB', (width + MAP, height))
            frame.paste(left, (0, 0))
            frame.paste(right, (width, 0))
            proc.stdin.write(np.asarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit('ffmpeg failed')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('checkpoint')
    ap.add_argument('--out', default='out/policy.mp4')
    ap.add_argument('--seed', type=int, default=1000, help='varies the rotor time constants / controller gains')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--start', choices=sorted(STARTS), default='launch')
    ap.add_argument('--pick', choices=('any', 'success', 'failure'), default='any')
    ap.add_argument('--skip', type=int, default=0, help='skip this many qualifying episodes first')
    ap.add_argument('--max-tries', type=int, default=64)
    a = ap.parse_args(argv)

    env, policy = load_policy(a.checkpoint, 1, a.device, a.seed, perturb=0.0)
    cells = np.nonzero(env._bank_kind.cpu().numpy() == STARTS[a.start])[0]
    if not len(cells):
        raise SystemExit(f'the bank has no {a.start} starts (return starts need a return-phase checkpoint)')
    rng = np.random.default_rng(a.seed)
    skip = a.skip
    for attempt in range(a.max_tries):
        cell = int(cells[0] if a.start == 'launch' else rng.choice(cells))
        tr, phases, outcome, points = _fly(env, policy, cell)
        ok = outcome == 'success'
        if a.pick == 'any' or ok == (a.pick == 'success'):
            if skip == 0:
                break
            skip -= 1
    else:
        raise SystemExit(f'no {a.pick} episode in {a.max_tries} {a.start} starts')
    ts = np.arange(len(tr)) * env.plant.dt
    t = np.arange(0.0, ts[-1], 1.0 / FPS)
    pos = np.stack([np.interp(t, ts, tr[:, k]) for k in range(3)], axis=1).astype(np.float32)
    rpy = np.stack([np.interp(t, ts, np.unwrap(tr[:, 3 + k])) for k in range(3)], axis=1).astype(np.float32)
    speed = np.linalg.norm(np.gradient(pos, 1.0 / FPS, axis=0), axis=1)
    print(f'{a.checkpoint}: {a.start} start (cell {cell}, try {attempt + 1}): {outcome} after {ts[-1]:.1f} s, '
          f'{points:.1f} points')
    course = CourseMesh.load(os.path.join(env.cfg.assets_dir, 'course.npz'))
    mark, pad = env._mark[0].cpu().numpy(), env._pad_c[0].cpu().numpy()
    circles = [(pad[0], pad[1], rules.PAD_RADIUS_M, (80, 220, 120)),
               (mark[0], mark[1], rules.RETURN_RADIUS_M, (255, 210, 60))]
    step_of = np.minimum((t / 0.1).astype(int), len(phases) - 1)
    labels = [f'{PHASES[phases[k]]}' + (f'  -> {outcome}' if k == len(phases) - 1 else '') for k in step_of]
    write_fpv(a.out, course, t, pos, rpy, speed, a.device, circles=circles, labels=labels)
    path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as f:
        json.dump({'checkpoint': a.checkpoint, 'start': a.start, 'cell': cell, 'outcome': outcome, 'points': points, 'duration_s': float(ts[-1]),
                   'path_length_m': path_len, 'mean_speed_ms': path_len / max(float(t[-1]), 1e-6),
                   'max_speed_ms': float(speed.max())}, f, indent=1)


def _fly(env, policy, cell: int):
    """One deterministic episode from bank ``cell`` -> (trace (S,6) at 100 Hz, phase per step, outcome, points).
    While the drone is parked (disarmed on the pad / at the mark) the trace holds the parked pose."""
    env.set_eval_queue(torch.tensor([[cell]]))
    with torch.inference_mode():
        policy.reset()
    env.plant.trace = []
    first = torch.cat([env.plant.pos_ned, env.plant.rpy_ned()], dim=-1).clone()
    obs = env.get_observations()
    outcome, points, phases = 'timeout', 0.0, []
    for _ in range(env.max_episode_length):
        ph = int(env._phase[0]) if env._rp else 0
        with torch.inference_mode():
            act = policy(obs)
        obs, _, dones, _ = env.step(act.clone())
        if env._rp and ph in (1, 3):                      # parked during this step: hold the pose in the video
            pk = torch.cat([env._park_pos[0], torch.zeros(2, device=env.device), env._park_yaw[0:1]]).view(1, 6)
            env.plant.trace[-env.plant.decimation:] = [pk.clone() for _ in range(env.plant.decimation)]
        phases.append(ph)
        with torch.inference_mode():
            policy.reset(dones)
        if int(dones[0]):
            outcome = env.OUTCOMES[int(env.finished[-1][3, 0])]
            points = float(env.finished_points[-1][0])
            break
    tr = torch.cat([first] + env.plant.trace, dim=0).cpu().numpy()
    env.plant.trace = None
    return tr, phases, outcome, points


if __name__ == '__main__':
    main()
