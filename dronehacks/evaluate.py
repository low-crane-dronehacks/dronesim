"""Deterministic evaluation of a checkpoint (never trust the training curve: it is stochastic and SGS-weighted).

    python -m dronehacks.evaluate checkpoints/comp_s2.pt
    python -m dronehacks.evaluate runs/my_run/checkpoints/model_2000.pt --perturb 0.5

Conditions, each a FIXED queue of start cells per env (so fast episodes are not over-counted):
  launch    the exact launch pose: the competition run (only the rotor time constants / gains vary)
  route     starts near the planned route
  free      starts anywhere in the reachable volume
  return    (return-leg policies) pad / climb / cross / final starts
Actions are the policy mean; the GRU state is reset on done. GPU evals are not bit-reproducible run to run:
treat differences under ~4 points as noise unless repeated with other seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from dronehacks.bank import BANK_BODY, BANK_CLIMB, BANK_CROSS, BANK_FINAL, BANK_LAUNCH, BANK_NEAR_ROUTE, BANK_PAD
from dronehacks.env import CourseEnv, CourseEnvCfg


def checkpoint_config(path: str) -> dict:
    """The env config a checkpoint was trained with: runs/<id>/config.json, or a <ckpt>.json sidecar."""
    for cand in (os.path.splitext(path)[0] + '.json',
                 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), 'config.json')):
        if os.path.exists(cand):
            with open(cand) as f:
                return json.load(f)['env']
    raise FileNotFoundError(f'no config for {path} (expected {os.path.splitext(path)[0]}.json or ../config.json)')


def load_policy(path: str, num_envs: int, device: str, seed: int, **env_overrides):
    """-> (env, deterministic policy). Builds the env the checkpoint was trained with, plus overrides."""
    from rsl_rl.runners import OnPolicyRunner

    from dronehacks.train import train_cfg
    saved = checkpoint_config(path)
    fields = {f for f in CourseEnvCfg.__dataclass_fields__}
    ec = {k: v for k, v in saved.items() if k in fields}
    ec.pop('assets_dir', None)
    ec.update(num_envs=num_envs, device=device, seed=seed, metrics_path=None, v_max=tuple(ec.get('v_max', (5, 3, 2))))
    ec.update(env_overrides)
    env = CourseEnv(CourseEnvCfg(**ec))
    runner = OnPolicyRunner(env, train_cfg(num_envs), log_dir=None, device=device)
    runner.load(path, load_cfg={'actor': True, 'critic': True, 'optimizer': False, 'iteration': False, 'rnd': False},
                map_location=device)
    return env, runner.get_inference_policy(device=device)


@torch.no_grad()
def run_queue(env: CourseEnv, policy, queue: torch.Tensor):
    """Fly ``queue`` (E, m) -> (records (5,N): env, k, cell, outcome, ep_len; points (N,))."""
    policy.reset()                              # no stale GRU memory from the previous condition
    env.set_eval_queue(queue)
    obs = env.get_observations()
    for _ in range(env.max_episode_length * (queue.shape[1] + 1) + 5):
        with torch.inference_mode():
            act = policy(obs)
        obs, _, dones, _ = env.step(act)
        with torch.inference_mode():
            policy.reset(dones)                 # inside inference_mode: the hidden state is an inference tensor
        if env.eval_done():
            break
    rec = torch.cat(env.finished, 1).cpu().numpy() if env.finished else np.zeros((5, 0), np.int64)
    pts = torch.cat(env.finished_points).cpu().numpy() if env.finished_points else np.zeros(0)
    return rec, pts


def summarise(rec: np.ndarray, pts: np.ndarray) -> dict:
    n = rec.shape[1]
    out = {'episodes': int(n), 'mean_points': float(pts.mean()) if n else float('nan')}
    out.update({k: float((rec[3] == i).mean()) if n else float('nan') for i, k in enumerate(CourseEnv.OUTCOMES)})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('checkpoint')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--episodes', type=int, default=512, help='launch / return episodes per condition')
    ap.add_argument('--cells', type=int, default=2048, help='route / free-space starts (multiple of --envs)')
    ap.add_argument('--perturb', type=float, default=0.0, help='evaluate on perturbed layouts of this severity')
    ap.add_argument('--perturb-variants', type=int, default=16)
    ap.add_argument('--perturb-seed', type=int, default=777, help='fixed, so every checkpoint sees the same layouts')
    ap.add_argument('--out', default=None, help='json path (default: next to the checkpoint)')
    a = ap.parse_args(argv)

    env, policy = load_policy(a.checkpoint, a.envs, a.device, a.seed, perturb=a.perturb,
                              perturb_variants=a.perturb_variants, perturb_mode='fixed', perturb_seed=a.perturb_seed,
                              perturb_ramp_steps=0)
    kind = env._bank_kind.cpu().numpy()
    rng = np.random.default_rng(a.seed)
    e = env.num_envs
    per_env = math.ceil(a.episodes / e)
    pools = {'launch': (BANK_LAUNCH,)}
    if env._rp:
        pools.update(pad=(BANK_PAD,), climb=(BANK_CLIMB,), cross=(BANK_CROSS,), final=(BANK_FINAL,))
    result = {'checkpoint': a.checkpoint, 'seed': a.seed, 'perturb': a.perturb}
    for name, kinds in pools.items():
        pool = np.nonzero(np.isin(kind, kinds))[0]
        if len(pool):
            rec, pts = run_queue(env, policy, torch.as_tensor(rng.choice(pool, size=(e, per_env))))
            result[name] = summarise(rec[:, :a.episodes], pts[:a.episodes])
    if a.cells % e:
        raise SystemExit(f'--cells ({a.cells}) must be a multiple of --envs ({e})')
    for name, k in (('route', BANK_NEAR_ROUTE), ('free', BANK_BODY)):
        pool = np.nonzero(kind == k)[0]
        if not len(pool):                                  # a return-only policy has no run starts
            continue
        cells = rng.choice(pool, size=a.cells, replace=len(pool) < a.cells)
        rec, pts = run_queue(env, policy, torch.as_tensor(cells.reshape(e, -1)))
        result[name] = summarise(rec, pts)

    for name, r in result.items():
        if isinstance(r, dict):
            print(f"{name:7s} points {r['mean_points']:5.1f}/60  success {r['success'] * 100:5.1f}%  "
                  f"collision {r['collision'] * 100:5.1f}%  floor {r['floor'] * 100:5.1f}%  "
                  f"timeout {r['timeout'] * 100:5.1f}%  (n={r['episodes']})")
    out = a.out or os.path.splitext(a.checkpoint)[0] + f'_eval_s{a.seed}' + (f'_p{a.perturb:g}' if a.perturb else '') + '.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
