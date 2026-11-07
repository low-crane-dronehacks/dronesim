"""Train a course policy: rsl_rl PPO, CNN+GRU actor, asymmetric critic.

    python -m dronehacks.train --run-id my_run --num-envs 2048 --total-steps 100000000

Writes runs/<run-id>/ (or --runs-dir): config.json, metrics.jsonl (one line per ~30 s) and checkpoints/model_<iter>.pt.
Re-running the same run id resumes from its newest checkpoint.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import re
import time

# The recurrent update allocates a large padded copy of the rollout; without expandable segments the caching
# allocator fragments and OOMs. Must be set before torch initialises CUDA.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from dronehacks.env import CourseEnv, CourseEnvCfg  # noqa: E402

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
OBS_GROUPS = {'actor': ['policy', 'proprio'], 'critic': ['critic']}


def train_cfg(num_envs: int, run_id: str = 'run', total_steps: int = 100_000_000, seed: int = 0,
              steps_per_env: int = 24, epochs: int = 5, lr: float = 3e-4, gamma: float = 0.99,
              entropy_coef: float = 0.0, init_std: float = 0.3, min_traj_per_mb: int = 256,
              save_interval: int = 25, hidden: int = 128) -> dict:
    """The ``train_cfg`` dict for rsl_rl's OnPolicyRunner (it mutates it: build a fresh one each time)."""
    mb = next(n for n in range(max(1, num_envs // min_traj_per_mb), 0, -1) if num_envs % n == 0)  # must divide envs
    return {
        'class_name': 'OnPolicyRunner',
        'num_steps_per_env': steps_per_env,
        'max_iterations': max(1, math.ceil(total_steps / (steps_per_env * num_envs))),
        'save_interval': save_interval,
        'experiment_name': run_id,
        'logger': {'class_name': 'dronehacks.policy:NullLogWriter'},
        'seed': seed,
        'clip_actions': 1.0,
        'obs_groups': {k: list(v) for k, v in OBS_GROUPS.items()},
        'algorithm': {
            'class_name': 'PPO', 'num_learning_epochs': epochs, 'num_mini_batches': mb,
            'clip_param': 0.2, 'gamma': gamma, 'lam': 0.95, 'value_loss_coef': 1.0, 'entropy_coef': entropy_coef,
            'learning_rate': lr, 'max_grad_norm': 1.0, 'use_clipped_value_loss': True,
            'schedule': 'adaptive', 'desired_kl': 0.01, 'normalize_advantage_per_mini_batch': False,
            'rnd_cfg': None, 'symmetry_cfg': None,
        },
        'actor': {
            'class_name': 'dronehacks.policy:CNNGRUModel',
            'hidden_dims': [hidden, hidden], 'activation': 'elu', 'obs_normalization': False,
            'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': init_std, 'std_type': 'log'},
            'cnn_cfg': {'output_channels': [16, 32, 64], 'kernel_size': [5, 3, 3], 'stride': [4, 2, 2],
                        'padding': 'zeros', 'max_pool': [False, False, True], 'activation': 'elu', 'flatten': True},
            'rnn_type': 'gru', 'rnn_hidden_dim': hidden, 'rnn_num_layers': 1,
        },
        'critic': {'class_name': 'MLPModel', 'hidden_dims': [128, 128], 'activation': 'elu', 'obs_normalization': True},
    }


def latest_checkpoint(ckpt_dir: str):
    best, best_it = None, -1
    for path in glob.glob(os.path.join(ckpt_dir, 'model_*.pt')):
        m = re.fullmatch(r'model_(\d+)\.pt', os.path.basename(path))
        if m and int(m.group(1)) > best_it:
            best, best_it = path, int(m.group(1))
    return best


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--runs-dir', default=RUNS, help='parent of the run directory (lab PCs sync only out/ or logs/)')
    ap.add_argument('--num-envs', type=int, default=2048, help='2048 needs ~32 GB VRAM; use 1024 on 24 GB')
    ap.add_argument('--total-steps', type=int, default=100_000_000, help='env steps summed over envs')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bank', default=None, help='reset bank in assets/ (default bank_free_v2.npz, '
                                                  'bank_return_v2.npz with --return-phase)')
    ap.add_argument('--return-phase', action='store_true', help="= --return-mode joint (one policy does both legs)")
    ap.add_argument('--return-mode', choices=('none', 'handoff', 'return', 'joint'), default='none',
                    help="handoff: run policy whose 5th action is 'complete' (disarm on the pad); "
                         'return: separate return-leg policy; joint: one policy for both')
    ap.add_argument('--max-episode-s', type=float, default=None, help='default 45, 90 with --return-phase')
    ap.add_argument('--v-max', type=float, nargs=3, default=(5.0, 3.0, 2.0), metavar=('FWD', 'RIGHT', 'DOWN'))
    ap.add_argument('--perturb', type=float, default=0.0, help='layout perturbation severity in [0,1]')
    ap.add_argument('--perturb-variants', type=int, default=32)
    ap.add_argument('--perturb-mode', choices=('fixed', 'spread'), default='spread')
    ap.add_argument('--perturb-ramp-steps', type=int, default=0)
    ap.add_argument('--uniform', action='store_true', help='uniform bank sampling instead of SGS')
    ap.add_argument('--init-std', type=float, default=0.3)
    ap.add_argument('--entropy-coef', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--init-checkpoint', default=None, help='warm start actor+critic (fresh runs train better here)')
    ap.add_argument('--save-interval', type=int, default=25)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse(argv)
    from rsl_rl.runners import OnPolicyRunner

    from dronehacks.policy import patch_rsl_rl
    patch_rsl_rl()
    run_dir = os.path.join(a.runs_dir, a.run_id)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    mode = 'joint' if a.return_phase else a.return_mode
    cfg = CourseEnvCfg(
        num_envs=a.num_envs, device=a.device, seed=a.seed, v_max=tuple(a.v_max),
        bank_file=a.bank or ('bank_return_v2.npz' if mode in ('joint', 'return') else 'bank_free_v2.npz'),
        return_mode=mode, max_episode_s=a.max_episode_s or (90.0 if mode == 'joint' else 45.0),
        perturb=a.perturb, perturb_variants=a.perturb_variants, perturb_mode=a.perturb_mode,
        perturb_ramp_steps=a.perturb_ramp_steps, sgs_uniform=a.uniform,
        metrics_path=os.path.join(run_dir, 'metrics.jsonl'))
    tcfg = train_cfg(a.num_envs, a.run_id, a.total_steps, a.seed, lr=a.lr, gamma=a.gamma,
                     entropy_coef=a.entropy_coef, init_std=a.init_std, save_interval=a.save_interval)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'args': vars(a), 'env': dataclasses.asdict(cfg), 'train': tcfg}, f, indent=1, default=str)

    env = CourseEnv(cfg)
    print(f'[train] {a.run_id}: {a.num_envs} envs, bank {cfg.bank_file} ({len(env._bank_pos)} starts), '
          f'{tcfg["max_iterations"]} iterations', flush=True)
    runner = OnPolicyRunner(env, tcfg, log_dir=ckpt_dir, device=a.device)
    resume = latest_checkpoint(ckpt_dir)
    if resume:
        print(f'[train] resuming from {resume}', flush=True)
        runner.load(resume)
        runner.current_learning_iteration += 1
    elif a.init_checkpoint:
        print(f'[train] warm start from {a.init_checkpoint}', flush=True)
        runner.load(a.init_checkpoint, load_cfg={'actor': True, 'critic': True, 'optimizer': False,
                                                 'iteration': False, 'rnd': False}, map_location=a.device)
    remaining = tcfg['max_iterations'] - runner.current_learning_iteration
    if remaining <= 0:
        print('[train] budget already met')
        return
    t0 = time.time()
    runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    print(f'[train] done in {(time.time() - t0) / 60:.1f} min; checkpoints in {ckpt_dir}', flush=True)


if __name__ == '__main__':
    main()
