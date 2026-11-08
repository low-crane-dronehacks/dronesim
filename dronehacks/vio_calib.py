"""Calibrate the VIO pose-error spread vs distance flown -> assets/vio_sigma.json (realism ``vio_pos``).

    python -m dronehacks.vio_calib                       # 4096 flights, 1.5-3 m/s cruise
    python -m dronehacks.vio_calib --speed 1.5 3.0 --runs 8192

The VIO odometry frame is initialised at the launch-mark pose, so its error is ~0 at launch and grows with the
distance flown. A reset anywhere else on the course must start with the error a drone would have accumulated
getting there. This flies the planned route (assets/route.npz) from the launch mark many times at realistic speeds
(cruise ~ U(speed range), 2 m/s^2 acceleration from rest, nose along the route's horizontal tangent) through the
SAME error process the env uses (``Realism.vio_pose`` with the ``vio_pos`` defaults or the given overrides) and
records the TRACKING error (the error the estimate continues from; a dropout's held output is not a start state)
as a function of route arc length s, minus the latency part (p(t - lat) - p(t), which the process regenerates from
the motion after a reset). Output: per-0.5 m table of per-axis xy sigma, z sigma and yaw sigma (deg),
plus a sigma^2 = a + b s + c s^2 fit used beyond the end of the route (return-leg starts).

The table belongs to the error-model parameters it was made with (stored under 'params'): re-run this after
changing any vio_* field. ``Realism`` warns if they differ.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

import numpy as np
import torch

from dronehacks.bank import Route
from dronehacks.realism import ASSETS, VIO_PARAM_FIELDS as PARAM_FIELDS, Realism, RealismCfg, parse


def route_flights(route: Route, n: int, speed: tuple, accel: float, dt: float, seed: int):
    """-> s (T,n), pos (T,n,3), vel (T,n,3), yaw (T,n): n flights along the route, until each reaches its end."""
    rng = np.random.default_rng(seed)
    cruise = rng.uniform(*speed, n)
    pts, rs = route.points.astype(np.float64), route.s.astype(np.float64)
    tan = np.gradient(pts, axis=0)
    k = 10                                                     # +-0.5 m window: smooth headings
    hxy = np.stack([np.convolve(tan[:, i], np.ones(2 * k + 1), 'same') for i in (0, 1)], 1)
    ok = np.linalg.norm(hxy, axis=1) > 1e-3 * (2 * k + 1) * 0.05
    head = np.arctan2(hxy[:, 1], hxy[:, 0])
    first = int(np.argmax(ok))
    head[:first] = head[first]                                 # vertical takeoff: the first defined heading
    head = np.unwrap(head)
    s, v = np.zeros(n), np.zeros(n)
    S = [s.copy()]
    while (s < rs[-1]).any():
        v = np.minimum(cruise, v + accel * dt)
        s = np.minimum(s + v * dt, rs[-1])
        S.append(s.copy())
    S = np.stack(S)                                            # (T+1, n)
    pos = np.stack([np.interp(S, rs, pts[:, i]) for i in range(3)], -1)
    vel = np.diff(pos, axis=0, append=pos[-1:]) / dt           # the velocity over the coming step
    yaw = np.interp(S, rs, head)
    return S, pos, vel, yaw


def calibrate(cfg: RealismCfg, runs: int, speed: tuple, accel: float = 2.0, bin_m: float = 0.5, seed: int = 0,
              device: str = 'cpu') -> dict:
    route = Route.load(os.path.join(ASSETS, 'route.npz'))
    dt = 0.1
    S, pos, vel, yaw = route_flights(route, runs, speed, accel, dt, seed)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)        # noqa: E731
    r = Realism(cfg, runs, device, dt, 1.5, (0.01, 0.01, 0.02), (80, 128), 88.0, 5.0)
    ids = torch.arange(runs, device=device)
    r.vio_pos_reset(ids, t(S[0]), torch.ones(runs, dtype=torch.bool, device=device), t(pos[0]), t(yaw[0]), t(vel[0]))
    edges = np.arange(0.0, route.length + bin_m, bin_m)
    nb = len(edges) - 1
    acc = np.zeros((nb, 5))                                    # sum x^2+y^2, z^2, yaw^2, count, out xy^2
    done = np.zeros(runs, bool)
    lat = r.vio_pos_vel.lat.cpu().numpy() / dt                 # VIO latency, steps
    ar = np.arange(runs)
    for k in range(len(S)):
        p_t, v_t, y_t = t(pos[k]), t(vel[k]), t(yaw[k])
        if cfg.velocity:
            r.velocity(v_t)
        p_hat, _, _ = r.vio_pose(p_t, v_t, y_t)
        live = ~done
        b = np.clip(np.searchsorted(edges, S[k], side='right') - 1, 0, nb - 1)
        pe, ye = r.pe.cpu().numpy(), r.ye.cpu().numpy()
        # the lagged-velocity part of the error is the pose's latency (p(t - lat) - p(t)), not drift: the process
        # regenerates it from the motion after any reset, so it is not part of the initial error. Remove it (xy).
        f = np.clip(k - lat, 0, None)
        i0 = np.floor(f).astype(int)
        w = (f - i0)[:, None]
        i1 = np.minimum(i0 + 1, len(S) - 1)
        p_lag = pos[i0, ar] * (1 - w) + pos[i1, ar] * w
        pe = pe.copy()
        pe[:, :2] -= (p_lag - pos[k])[:, :2] * cfg.vio_pos_drift_scale
        out = (p_hat - p_t).cpu().numpy()
        np.add.at(acc, b[live], np.stack([pe[live, 0] ** 2 + pe[live, 1] ** 2, pe[live, 2] ** 2, ye[live] ** 2,
                                          np.ones(live.sum()), out[live, 0] ** 2 + out[live, 1] ** 2], 1))
        done |= S[k] >= route.length
    cnt = np.maximum(acc[:, 3], 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    keep = acc[:, 3] > 0
    xy = np.sqrt(acc[:, 0] / cnt / 2)[keep]
    z = np.sqrt(acc[:, 1] / cnt)[keep]
    yw = np.degrees(np.sqrt(acc[:, 2] / cnt))[keep]
    out_xy = np.sqrt(acc[:, 4] / cnt / 2)[keep]
    sm = mid[keep]
    # table starts at s=0 with the launch error
    s_tab = np.concatenate([[0.0], sm])
    tab = {'s': s_tab.round(3).tolist(),
           'xy': np.concatenate([[cfg.vio_launch_pos_m], xy]).round(5).tolist(),
           'z': np.concatenate([[cfg.vio_launch_pos_m], z]).round(5).tolist(),
           'yaw_deg': np.concatenate([[cfg.vio_launch_yaw_deg], yw]).round(4).tolist()}
    A = np.stack([np.ones_like(sm), sm, sm ** 2], 1)
    fit = {k: np.linalg.lstsq(A, np.asarray(v) ** 2, rcond=None)[0].tolist() for k, v in
           (('xy', xy), ('z', z), ('yaw_deg', yw))}
    return {'table': tab, 'fit': fit, 'held_output_xy': out_xy.round(5).tolist(),
            'params': {f: list(v) if isinstance(v := getattr(cfg, f), tuple) else v for f in PARAM_FIELDS},
            'flights': {'runs': runs, 'speed_mps': list(speed), 'accel_mps2': accel, 'route_m': route.length,
                        'seed': seed},
            'note': 'sigma of the VIO TRACKING error vs route arc length s (m); xy is per axis. '
                    'fit: sigma^2 = a + b s + c s^2 (used beyond the table). Written by dronehacks/vio_calib.py.'}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--runs', type=int, default=4096)
    ap.add_argument('--speed', type=float, nargs=2, default=(1.5, 3.0), metavar=('LO', 'HI'), help='cruise m/s')
    ap.add_argument('--accel', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--realism', default='vio_pos', help="error-model preset/overrides, e.g. 'vio_pos{vio_bias_init=0.03}'")
    ap.add_argument('--out', default=os.path.join(ASSETS, 'vio_sigma.json'))
    a = ap.parse_args(argv)
    cfg = dataclasses.replace(parse(a.realism, seed=a.seed), vio_sigma_file='__none__')
    res = calibrate(cfg, a.runs, tuple(a.speed), a.accel, seed=a.seed)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    t = res['table']
    for s0 in (0, 2, 5, 10, 14):
        i = int(np.argmin(np.abs(np.asarray(t['s']) - s0)))
        print(f"s={t['s'][i]:5.2f} m  sigma xy {t['xy'][i] * 100:5.1f} cm/axis  z {t['z'][i] * 100:4.1f} cm  "
              f"yaw {t['yaw_deg'][i]:4.2f} deg")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
