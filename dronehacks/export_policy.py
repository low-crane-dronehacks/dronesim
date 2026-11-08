"""Export a checkpoint's actor to a NumPy .npz for deploy/policynav (no torch needed at runtime).

    python -m dronehacks.export_policy checkpoints/comp_s2.pt --out deploy/policynav/comp_s2.npz

Checks that the NumPy forward pass matches rsl_rl's deterministic policy on random inputs, step by step
through the GRU, and stores the env settings the node needs (v_max, image size, actions, proprio layout).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

from dronehacks.evaluate import checkpoint_config
from dronehacks.policy import CNNGRUModel
from dronehacks.train import train_cfg


def build_actor(path: str):
    cfg = checkpoint_config(path)
    mode = cfg.get('return_mode', 'none')
    if mode == 'none' and cfg.get('return_phase'):
        mode = 'joint'
    n_act = 5 if mode in ('joint', 'handoff') else 4
    n_prop = 3 + 1 + 2 + n_act + (1 if mode == 'joint' else 0)
    H, W = cfg.get('depth_height', 80), cfg.get('depth_width', 128)
    obs = TensorDict({'policy': torch.zeros(1, cfg.get('frame_stack', 2), H, W), 'proprio': torch.zeros(1, n_prop)},
                     batch_size=[1])
    a = train_cfg(1)['actor']
    a.pop('class_name')
    actor = CNNGRUModel(obs, {'actor': ['policy', 'proprio']}, 'actor', n_act, **a)
    sd = torch.load(path, map_location='cpu', weights_only=False)['actor_state_dict']
    actor.load_state_dict(sd, strict=False)
    actor.eval()
    meta = {'mode': mode, 'num_actions': n_act, 'proprio_dim': n_prop, 'depth_hw': [H, W],
            'frame_stack': cfg.get('frame_stack', 2), 'v_max': list(cfg.get('v_max', (5, 3, 2))),
            'v_obs_scale': list(cfg.get('v_max_start') or cfg.get('v_max', (5, 3, 2))),   # proprio velocity scale
            # deploy preprocessing the policy was trained with (deploy reads it from metadata; 0 = legacy invalid-near)
            'edge_fill_cols': int((cfg.get('realism') or {}).get('edge_fill_cols', 0)) if isinstance(cfg.get('realism'), dict) else 0,
            'yaw_rate_max': cfg.get('yaw_rate_max', 1.0471975511965976), 'far_clip': cfg.get('far_clip', 5.0),
            'hfov_deg': cfg.get('hfov_deg', 72.0), 'vfov_deg': cfg.get('vfov_deg', 50.0),
            'rf_max': cfg.get('rf_max', 4.0), 'source': os.path.basename(path)}
    return actor, meta


def export(actor, meta) -> dict:
    out = {}
    convs = [m for m in actor.cnns['policy'] if isinstance(m, torch.nn.Conv2d)]
    mods = list(actor.cnns['policy'])
    for i, c in enumerate(convs):
        j = mods.index(c)
        pool = any(isinstance(m, torch.nn.MaxPool2d) for m in mods[j + 1:j + 3])
        out[f'conv{i}_w'], out[f'conv{i}_b'] = c.weight.detach().numpy(), c.bias.detach().numpy()
        out[f'conv{i}_stride'], out[f'conv{i}_pad'] = np.array(c.stride), np.array(c.padding)
        out[f'conv{i}_pool'] = np.array(pool)
    g = actor.rnn.rnn
    out['gru_w_ih'], out['gru_w_hh'] = g.weight_ih_l0.detach().numpy(), g.weight_hh_l0.detach().numpy()
    out['gru_b_ih'], out['gru_b_hh'] = g.bias_ih_l0.detach().numpy(), g.bias_hh_l0.detach().numpy()
    lins = [m for m in actor.mlp if isinstance(m, torch.nn.Linear)]
    for i, l in enumerate(lins):
        out[f'mlp{i}_w'], out[f'mlp{i}_b'] = l.weight.detach().numpy(), l.bias.detach().numpy()
    meta = dict(meta, n_conv=len(convs), n_mlp=len(lins))
    out['meta'] = np.array(json.dumps(meta))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('checkpoint')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    actor, meta = build_actor(a.checkpoint)
    np.savez_compressed(a.out, **export(actor, meta))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deploy', 'policynav'))
    from policy_np import NumpyPolicy
    npol = NumpyPolicy(a.out)
    g = torch.Generator().manual_seed(0)
    H, W = meta['depth_hw']
    worst = 0.0
    with torch.inference_mode():
        actor.reset()
        for _ in range(20):                                   # 20 steps: exercises the GRU memory too
            d = torch.rand(1, meta['frame_stack'], H, W, generator=g)
            p = torch.randn(1, meta['proprio_dim'], generator=g) * 0.5
            ref = actor(TensorDict({'policy': d, 'proprio': p}, batch_size=[1]))[0].numpy()
            got = npol(d[0].numpy(), p[0].numpy())
            worst = max(worst, float(np.abs(ref - got).max()))
    print(f'wrote {a.out} ({os.path.getsize(a.out) // 1024} KB); numpy vs torch max |diff| over 20 steps: {worst:.2e}')
    if worst > 1e-4:
        raise SystemExit('numpy forward pass does not match torch')


if __name__ == '__main__':
    main()
