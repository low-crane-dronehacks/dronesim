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
from dronehacks.realism import parse as parse_realism  # noqa: E402

RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
OBS_GROUPS = {'actor': ['policy', 'proprio'], 'critic': ['critic']}


def _cnn_cfg(pool: int) -> dict:
    """Small CNNs for pooled depth (80x128 / pool): both end at 32 x 5 x 8 = 1280 features."""
    if pool == 4:                                    # 20 x 32
        return {'output_channels': [16, 32, 32], 'kernel_size': [3, 3, 3], 'stride': [1, 2, 2], 'padding': 'zeros',
                'max_pool': [False, False, False], 'activation': 'elu', 'flatten': True}
    if pool == 8:                                    # 10 x 16
        return {'output_channels': [16, 32], 'kernel_size': [3, 3], 'stride': [1, 2], 'padding': 'zeros',
                'max_pool': [False, False], 'activation': 'elu', 'flatten': True}
    raise ValueError(f'no CNN preset for obs_pool {pool} (use 4 or 8)')


def train_cfg(num_envs: int, run_id: str = 'run', total_steps: int = 100_000_000, seed: int = 0,
              steps_per_env: int = 24, epochs: int = 5, lr: float = 3e-4, gamma: float = 0.99,
              entropy_coef: float = 0.0, init_std: float = 0.3, min_traj_per_mb: int = 256,
              save_interval: int = 25, hidden: int = 128, obs_pool: int = 1) -> dict:
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
            'cnn_cfg': _cnn_cfg(obs_pool) if obs_pool > 1 else {'output_channels': [16, 32, 64], 'kernel_size': [5, 3, 3], 'stride': [4, 2, 2],
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
    ap.add_argument('--runs-dir', default=RUNS, help='parent of the run directory (default runs/)')
    ap.add_argument('--num-envs', type=int, default=2048, help='2048 needs ~32 GB VRAM; use 1024 on 24 GB')
    ap.add_argument('--total-steps', type=int, default=100_000_000, help='env steps summed over envs')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--launch-frac', type=float, default=0.0, help='share of resets forced onto launch-mark starts')
    ap.add_argument('--invalid-mask', action='store_true', help='add invalid-depth mask channels to the policy image (warm starts zero-pad the first conv)')
    ap.add_argument('--extra-boulders', type=int, nargs=2, default=(0, 0), metavar=('LO', 'HI'), help='perturbed layouts: add LO..HI random boulders each')
    ap.add_argument('--arena-half', type=float, default=None, help='OOB half-width for the drone CENTRE (net at 3.05 m; 2.85 = net - 0.2 m radius)')
    ap.add_argument('--cp-shaping', type=float, default=0.0, help='reward per m of approach to the next checkpoint target (potential-based)')
    ap.add_argument('--launch-yaw-max', type=float, default=180.0, help='deg: keep launch starts within this of west (facing the net) or north (up the corridor)')
    ap.add_argument('--start-heading-max', type=float, default=180.0, help='deg: keep other starts within this of the route direction')
    ap.add_argument('--cp-margin', type=float, default=None, help='m past a checkpoint line the centre must be (default 0.15 = whole drone); <0 counts it early (curriculum)')
    ap.add_argument('--v-max-start', type=float, nargs=3, default=None, help='speed curriculum: start at this action scale and anneal to --v-max')
    ap.add_argument('--v-max-anneal-steps', type=int, default=0)
    ap.add_argument('--critic-full', action='store_true', help='asymmetric critic: + next checkpoint, episode time, hidden randomized physics')
    ap.add_argument('--critic-depth', action='store_true', help='asymmetric critic: + clean depth of the original geometry (net solid), 10x16')
    ap.add_argument('--anchor-wall', default='', help='perturbation prior: this centre wall barely moves and walls touching it stay attached (e.g. wall1)')
    ap.add_argument('--gate-extend', action='store_true', help='generous checkpoints: extend every line to the perturbed walls/net')
    ap.add_argument('--cp0-wall', default='', help="as-built scoring: checkpoint 0 = line from the NW corner to this wall's tip (e.g. wall3)")
    ap.add_argument('--perturb-scale', type=float, default=1.0, help='scale the wall/tunnel/boulder/gate perturbation ranges (2.5 = walls +-0.375 m, +-10 deg)')
    ap.add_argument('--course-file', default='course.npz', help='geometry in assets/ (e.g. course_w0scan.npz: wall 0 re-laid on the scan)')
    ap.add_argument('--window-gate', type=float, nargs=4, default=None, metavar=('N0', 'E0', 'N1', 'E1'), help='window checkpoint line (with a re-laid window)')
    ap.add_argument('--course-mods', default='', help="as-built course changes, e.g. 'drop:wall5;rot:wall3:15:45:south'")
    ap.add_argument('--obs-pool', type=int, default=1, help='pool the 80x128 policy depth by 4 (20x32) or 8 (10x16): nearest valid return per cell')
    ap.add_argument('--skip-terminal', action='store_true', help='end the episode (as a crash) when a later checkpoint gate is crossed with one skipped')
    ap.add_argument('--launch-liftoff-frac', type=float, default=0.0, help='of the forced launch starts, the share started just airborne (0.15-0.7 m, rising)')
    ap.add_argument('--bank', default=None, help='reset bank in assets/ (default bank_free_v3.npz, '
                                                  'bank_return_v3.npz with a return mode)')
    ap.add_argument('--return-phase', action='store_true', help="= --return-mode joint (one policy does both legs)")
    ap.add_argument('--return-mode', choices=('none', 'handoff', 'return', 'joint'), default='none',
                    help="handoff: run policy whose 5th action is 'complete' (disarm on the pad); "
                         'return: separate return-leg policy; joint: one policy for both')
    ap.add_argument('--max-episode-s', type=float, default=None, help='default 45, 90 with --return-phase')
    ap.add_argument('--v-max', type=float, nargs=3, default=(5.0, 3.0, 2.0), metavar=('FWD', 'RIGHT', 'DOWN'))
    ap.add_argument('--yaw-rate-max', type=float, default=None, help='rad/s (default: the env default, 60 deg/s)')
    ap.add_argument('--perturb', type=float, default=0.0, help='layout perturbation severity in [0,1]')
    ap.add_argument('--perturb-variants', type=int, default=32)
    ap.add_argument('--perturb-mode', choices=('fixed', 'spread'), default='spread')
    ap.add_argument('--perturb-ramp-steps', type=int, default=0)
    ap.add_argument('--uniform', action='store_true', help='uniform bank sampling instead of SGS')
    ap.add_argument('--realism', default='', help="real-world effects during training, e.g. 'real_lite' (realism.py)")
    ap.add_argument('--realism-ramp-steps', type=int, default=30_000_000,
                    help='share of episodes with the effects on grows 0 -> 1 over this many env steps (0 = always)')
    ap.add_argument('--pos-obs', action='store_true',
                    help="actor proprio += course-frame position + yaw estimate + VIO flag (true pose; VIO-like with "
                         "--realism ...,vio_pos). New runs only: changes the proprio size")
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
    # multi-GPU (torchrun --nproc_per_node=N -m dronehacks.train ...): one env + policy copy per GPU, rsl_rl
    # all-reduces the gradients. Rank 0 alone writes config / metrics / checkpoints; ranks get different seeds.
    world, rank, local = (int(os.getenv(k, d)) for k, d in (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')))
    if world > 1:
        import torch
        a.device = f'cuda:{local}'
        torch.cuda.set_device(local)
        a.seed = a.seed + 1000 * rank
    mode = 'joint' if a.return_phase else a.return_mode
    cfg = CourseEnvCfg(
        num_envs=a.num_envs, device=a.device, seed=a.seed, v_max=tuple(a.v_max), **({'yaw_rate_max': a.yaw_rate_max} if a.yaw_rate_max else {}),
        bank_file=a.bank or ('bank_return_v3.npz' if mode in ('joint', 'return') else 'bank_free_v3.npz'),
        return_mode=mode, max_episode_s=a.max_episode_s or (90.0 if mode == 'joint' else 45.0),
        perturb=a.perturb, perturb_variants=a.perturb_variants, perturb_mode=a.perturb_mode,
        perturb_ramp_steps=a.perturb_ramp_steps, sgs_uniform=a.uniform, launch_frac=a.launch_frac, launch_liftoff_frac=a.launch_liftoff_frac, skip_terminal=a.skip_terminal, course_mods=a.course_mods, course_file=a.course_file, **({'window_gate': tuple(a.window_gate)} if a.window_gate else {}), perturb_scale=a.perturb_scale, cp0_wall=a.cp0_wall, anchor_wall=a.anchor_wall, gate_extend=a.gate_extend, critic_depth=a.critic_depth, v_max_anneal_steps=a.v_max_anneal_steps, **({'v_max_start': tuple(a.v_max_start)} if a.v_max_start else {}), critic_full=a.critic_full, obs_pool=a.obs_pool, **({'cp_margin': a.cp_margin} if a.cp_margin is not None else {}), launch_yaw_max_deg=a.launch_yaw_max, start_heading_max_deg=a.start_heading_max, cp_shaping=a.cp_shaping, **({'arena_half': a.arena_half} if a.arena_half else {}), extra_boulders=tuple(a.extra_boulders), invalid_mask=a.invalid_mask,
        pos_obs=a.pos_obs,
        realism=parse_realism(a.realism, seed=a.seed, ramp_steps=a.realism_ramp_steps) if a.realism else None,
        metrics_path=os.path.join(run_dir, 'metrics.jsonl') if rank == 0 else None)
    tcfg = train_cfg(a.num_envs, a.run_id, a.total_steps, a.seed, lr=a.lr, gamma=a.gamma,
                     entropy_coef=a.entropy_coef, init_std=a.init_std, save_interval=a.save_interval,
                     obs_pool=a.obs_pool)
    if world > 1:                                     # the step budget is global: fewer iterations per rank
        tcfg['max_iterations'] = max(1, math.ceil(a.total_steps / (tcfg['num_steps_per_env'] * a.num_envs * world)))
    if rank == 0:
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump({'args': vars(a), 'env': dataclasses.asdict(cfg), 'train': tcfg, 'world_size': world}, f,
                      indent=1, default=str)

    env = CourseEnv(cfg)
    print(f'[train] {a.run_id}: {a.num_envs} envs, bank {cfg.bank_file} ({len(env._bank_pos)} starts), '
          f'{tcfg["max_iterations"]} iterations', flush=True)
    runner = OnPolicyRunner(env, tcfg, log_dir=ckpt_dir if rank == 0 else None, device=a.device)
    resume = latest_checkpoint(ckpt_dir)
    if resume:
        print(f'[train] resuming from {resume}', flush=True)
        runner.load(resume)
        runner.current_learning_iteration += 1
    elif cfg.return_mode in ('handoff', 'joint'):
        # The 5th output ends the episode when it is > 0 at the wrong time (in the air). From a fresh policy (mean 0,
        # std 0.3) that happens on half the steps, episodes last 1-2 steps, and the recurrent update pads thousands
        # of tiny trajectories until it runs out of GPU memory. Start the flag output well below 0; it still learns
        # to raise it when completing pays off.
        import torch
        last = [m for m in runner.alg.actor.mlp if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.bias[4] = -1.0
        print('[train] flag output bias initialised to -1', flush=True)
    if not resume and a.init_checkpoint:
        print(f'[train] warm start from {a.init_checkpoint}', flush=True)
        init = a.init_checkpoint
        if a.invalid_mask:                           # widen the first conv: new mask channels start with zero weight
            import torch as _t
            ck = _t.load(init, map_location='cpu', weights_only=False)
            w = ck['actor_state_dict']['cnns.policy.0.weight']
            want = 2 * env.cfg.frame_stack
            if w.shape[1] < want:
                ck['actor_state_dict']['cnns.policy.0.weight'] = _t.cat(
                    [w, _t.zeros(w.shape[0], want - w.shape[1], *w.shape[2:])], dim=1)
                init = os.path.join(ckpt_dir, f'init_mask_expanded_r{rank}.pt')
                _t.save(ck, init)
                print(f'[train] first conv widened {w.shape[1]} -> {want} channels (zeros)', flush=True)
        if a.critic_depth or a.critic_full:
            print('[train] critic gets clean depth: warm start loads the ACTOR only (the critic input changed)', flush=True)
        runner.load(init, load_cfg={'actor': True, 'critic': not (a.critic_depth or a.critic_full), 'optimizer': False,
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
