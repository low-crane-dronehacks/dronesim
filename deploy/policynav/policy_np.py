"""The trained CNN+GRU actor in plain NumPy (no torch): runs on the companion / Jetson CPU.

Weights come from ``python -m dronehacks.export_policy <checkpoint>`` (an .npz). The forward pass is the
deterministic policy (Gaussian mean), identical to rsl_rl's ``act_inference`` (checked by the exporter).
"""
from __future__ import annotations

import json

import numpy as np


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _conv2d(x, w, b, stride, pad):
    """x (C,H,W), w (O,C,kh,kw) -> (O,H',W'), zero padding (ph, pw)."""
    ph, pw = pad
    x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    C, H, W = x.shape
    O, _, kh, kw = w.shape
    sh, sw = stride
    Ho, Wo = (H - kh) // sh + 1, (W - kw) // sw + 1
    cols = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::sh, ::sw][:, :Ho, :Wo]
    return np.einsum('chwij,ocij->ohw', cols, w, optimize=True) + b[:, None, None]


def _maxpool3s2p1(x):
    x = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    C, H, W = x.shape
    Ho, Wo = (H - 3) // 2 + 1, (W - 3) // 2 + 1
    win = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))[:, ::2, ::2][:, :Ho, :Wo]
    return win.max(axis=(-1, -2))


class NumpyPolicy:
    """policy = NumpyPolicy('comp_s2.npz'); a = policy(depth_stack (2,H,W), proprio (P,)); policy.reset()"""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z['meta']))
        self.conv = [(z[f'conv{i}_w'], z[f'conv{i}_b'], tuple(z[f'conv{i}_stride']), tuple(z[f'conv{i}_pad']),
                      bool(z[f'conv{i}_pool'])) for i in range(self.meta['n_conv'])]
        self.gru = {k: z['gru_' + k] for k in ('w_ih', 'w_hh', 'b_ih', 'b_hh')}
        self.mlp = [(z[f'mlp{i}_w'], z[f'mlp{i}_b']) for i in range(self.meta['n_mlp'])]
        self.h = np.zeros(self.gru['w_hh'].shape[1], np.float32)

    def reset(self):
        self.h[:] = 0.0

    def __call__(self, depth: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        x = np.asarray(depth, np.float32)
        for w, b, s, p, pool in self.conv:
            x = _elu(_conv2d(x, w, b, s, p))
            if pool:
                x = _maxpool3s2p1(x)
        feat = np.concatenate([np.asarray(proprio, np.float32), x.reshape(-1)])     # 1-D groups first, then CNN
        g = self.gru
        gi, gh = g['w_ih'] @ feat + g['b_ih'], g['w_hh'] @ self.h + g['b_hh']
        H = self.h.shape[0]
        r = _sigmoid(gi[:H] + gh[:H])
        u = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        self.h = ((1.0 - u) * n + u * self.h).astype(np.float32)
        y = self.h
        for i, (w, b) in enumerate(self.mlp):
            y = w @ y + b
            if i < len(self.mlp) - 1:
                y = _elu(y)
        return y
