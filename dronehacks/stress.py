"""Stress-test a checkpoint against real-world effects: each effect on its own, then all together.

    python -m dronehacks.stress checkpoints/comp_s2.pt
    python -m dronehacks.stress checkpoints/comp_s2.pt --presets ideal,depth,real --episodes 256

For every preset it runs the launch condition (the competition run) and free-space starts, and prints points /
success / collision against the ideal sim. Writes <checkpoint>_stress_s<seed>.json. See realism.py for what
each preset does.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from dronehacks.bank import BANK_BODY, BANK_LAUNCH
from dronehacks.evaluate import load_plant, load_policy, run_queue, summarise
from dronehacks.realism import parse

DEFAULT = 'ideal,plant,battery,latency,velocity,rangefinder,net,depth,depth_plain,real'


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('checkpoint')
    ap.add_argument('--presets', default=DEFAULT, help="comma list; 'ideal' = no effects, 'a+b' combines presets")
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--envs', type=int, default=256)
    ap.add_argument('--episodes', type=int, default=512, help='launch episodes per preset')
    ap.add_argument('--cells', type=int, default=1024, help='free-space starts per preset (multiple of --envs)')
    ap.add_argument('--perturb', type=float, default=0.0, help='also move walls/gates by this layout severity')
    ap.add_argument('--plant', default=None, help='airframe profile json for the TRUE drone (e.g. the measured kit)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    results = {}
    for preset in a.presets.split(','):
        rc = None if preset == 'ideal' else parse(preset.replace('+', ','), seed=a.seed)
        env, policy = load_policy(a.checkpoint, a.envs, a.device, a.seed, realism=rc, perturb=a.perturb,
                                  perturb_mode='fixed', perturb_seed=777, perturb_ramp_steps=0,
                                  **({'plant_params': load_plant(a.plant)} if a.plant else {}))
        kind = env._bank_kind.cpu().numpy()
        rng = np.random.default_rng(a.seed)
        e = env.num_envs
        launch = np.nonzero(kind == BANK_LAUNCH)[0]
        rec, pts = run_queue(env, policy, torch.as_tensor(rng.choice(launch, size=(e, math.ceil(a.episodes / e)))))
        row = {'launch': summarise(rec[:, :a.episodes], pts[:a.episodes])}
        free = np.nonzero(kind == BANK_BODY)[0]
        rec, pts = run_queue(env, policy, torch.as_tensor(rng.choice(free, size=a.cells).reshape(e, -1)))
        row['free'] = summarise(rec, pts)
        results[preset] = row
        L, Fr = row['launch'], row['free']
        print(f"{preset:16s} launch {L['mean_points']:5.1f} pts  finish {L['success'] * 100:5.1f}%  "
              f"collide {L['collision'] * 100:5.1f}%  floor {L['floor'] * 100:5.1f}% | free {Fr['mean_points']:5.1f} pts  "
              f"success {Fr['success'] * 100:5.1f}%", flush=True)
        del env, policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = a.out or os.path.splitext(a.checkpoint)[0] + f'_stress_s{a.seed}' + (f'_p{a.perturb:g}' if a.perturb else '') + '.json'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'checkpoint': a.checkpoint, 'seed': a.seed, 'perturb': a.perturb, 'results': results}, f, indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
