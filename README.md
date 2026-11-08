# dronesim

GPU simulator and reinforcement-learning training for **DroneHacks 2026**, the Stanford indoor drone race.
The policies trained here fly the real drone through the onboard stack in
[low-crane-dronehacks/deploy](https://github.com/low-crane-dronehacks/deploy).

**Read [docs/APPROACH.md](docs/APPROACH.md) first:** the approach, why it generalizes to new courses, the
domain randomization, and the evaluation results of the current policies.

## Approach in one screen

- **Simulator.** Thousands of quadrotors in parallel on one GPU, in PyTorch. Depth images, obstacle distances
  and the rangefinder are ray traced against a triangle mesh of the course with NVIDIA Warp
  (`dronehacks/render.py`). The course is data: the organizers' published map, a scan-registered as-built
  variant, randomly perturbed layouts (`dronehacks/perturb.py`) or procedurally generated courses
  (`dronehacks/coursegen.py`).
- **Plant.** A Lee geometric velocity controller, rotor allocation, first-order motor lag and drag
  (`dronehacks/quad.py`). The policy commands body-frame velocity + yaw rate, which is how the real drone is
  flown (PX4 Offboard velocity setpoints).
- **Policy.** PPO (rsl_rl), CNN + GRU actor on a 2-frame depth stack (128 x 80) plus proprioception (velocity
  estimate, attitude, rangefinder, previous action). There is **no position, map or goal input**; the policy
  navigates from what it sees. The critic gets privileged state (asymmetric actor-critic).
- **Reward.** The competition score itself (points / 60, paid when earned; `dronehacks/rules.py`), plus an
  optional potential-based term toward the next checkpoint (`--cp-shaping`).
- **Curriculum.** Success Guided Sampling over a bank of start states generated from the course geometry
  (`dronehacks/bank.py`, `dronehacks/sgs.py`): starts are drawn where recent success is near 50%.
- **Robustness.** Stereo-depth, plant, latency, state-estimate and rangefinder randomization
  (`dronehacks/realism.py`, `--realism`), layout perturbation (`--perturb`) and a speed curriculum
  (`--v-max-start` / `--v-max-anneal-steps`). Details in [docs/APPROACH.md](docs/APPROACH.md).

**Deployed policy.** Policies are exported to NumPy (`python -m dronehacks.export_policy`) and run on the drone
by the deploy repo. On deploy `main`, the drone launch (`launch/skybeta.launch.py`) defaults to
`policies/c5_s164_1575.npz` (the "deployed c5_s164" column in docs/APPROACH.md); the branch
`policy/holes-v5h3-s196` switches that default to `policies/holes_v5h3_s196_3255.npz` (the "s196" column).
See `docs/POLICIES.md` in the deploy repo for every exported policy and the observation/action contract.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[video,dev]"        # torch, warp-lang, rsl-rl-lib 5.4.1, tensordict, numpy, scipy
pytest                                  # CPU; GPU-only tests are skipped without CUDA
```

A CUDA GPU is needed for training. Everything else also runs on CPU (`--device cpu`), just slowly.

## Use

```bash
# train (2048 envs needs ~32 GB VRAM; use --num-envs 1024 on 24 GB). Default budget: 100M env steps.
python -m dronehacks.train --run-id my_run
python -m dronehacks.train --run-id my_dr --realism real_lite --perturb 1.0      # + sensor/plant DR, moved walls
python -m dronehacks.train --run-id my_ret --return-phase                        # + the return leg

# evaluate deterministically (launch run, near-route starts, anywhere starts)
python -m dronehacks.evaluate runs/my_run/checkpoints/<model>.pt
python -m dronehacks.evaluate <ckpt> --perturb 0.5                   # on randomly moved walls/gates
python -m dronehacks.evaluate <ckpt> --course course_scan.npz        # on the scanned as-built course
python -m dronehacks.evaluate <ckpt> --realism real                  # every real-world effect on
python -m dronehacks.evaluate <ckpt> --realism depth,velocity --plant assets/plant_nominal.json
python -m dronehacks.stress <ckpt>                                   # each real-world effect alone, then all

# watch it fly (FPV depth + top-down map; needs ffmpeg)
python -m dronehacks.video <ckpt> --out out/my_run.mp4

# export for the drone (NumPy only; verifies the forward pass against torch)
python -m dronehacks.export_policy <ckpt> --out my_policy.npz

# procedurally generated courses: top-down review sheet
python -m dronehacks.coursegen gallery --n 20 --out out/coursegen
```

Runs go to `runs/<run-id>/` (`metrics.jsonl`, `checkpoints/`, `config.json`). Re-running a run id resumes it.
Judge policies with `dronehacks.evaluate`, never the training curve.

Real-world effects (`dronehacks/realism.py`, all off by default): `plant` (true airframe differs from nominal,
PX4-like velocity loop), `battery` (thrust sag), `latency` (late commands), `velocity` (VIO estimate: lag, noise,
bias, dropouts), `depth` (OAK-D-like stereo at 640x400: noise, min range, holes, occlusion shadows, edge bands,
latency), `net` (see-through boundary net with a random room behind it), `rangefinder` (noise, bias, dropouts).
`real` = all of them; `real_lite` / `depth_lite` = cheaper versions for training. `vio_pos` (+ `train.py
--pos-obs`) adds a VIO-like course-frame pose input (calibrated by `python -m dronehacks.vio_calib`); the
policies in docs/APPROACH.md do not use it.

## Baseline checkpoints in this repo

The `.pt` files in `checkpoints/` are the **first-generation baselines** (ideal sim, published map), kept as
small reproducible examples. They are not the current policies; those are described in docs/APPROACH.md and
shipped as `.npz` in the deploy repo. Each `.pt` is an rsl_rl state dict; its `.json` holds the training config.

Points out of 60, `python -m dronehacks.evaluate`: mean of seeds 1000 and 2000, 512 launch episodes,
2048 starts per other condition.

| checkpoint | launch run | near-route starts | anywhere | walls moved (severity 1): launch / route / anywhere |
|---|---|---|---|---|
| `comp_s2.pt` | 47.1 (99% finish) | 29.4 | 24.5 | 29.6 / 26.4 / 22.2 |
| `pert_s2.pt` | 41.7 (100% finish) | 31.3 | 26.3 | 30.5 / 28.0 / 24.2 |
| `return_s1.pt` | 0 (collides) | 18.8 | 13.1 | |

`comp_s2` is fastest on the exact map; `pert_s2` (trained on moved walls) is more reliable everywhere else.
Later bank versions (`assets/bank_free_v3.npz`, the default) start from any launch heading. `return_s1` predates the floor-friction fix in `quad.py`.

## Layout

```
dronehacks/
  rules.py         scoring rules + checkpoint geometry (from the organizers' diagram)
  env.py           the vectorised env: resets, scoring, observations, metrics
  quad.py          drone dynamics + Lee controller (torch); quad_warp.py is the fused GPU version
  render.py        Warp mesh queries: depth images, obstacle distance, rays
  realism.py       sensor / plant / timing randomization; stereo_warp.py is the fused GPU stereo model
  mesh.py          course mesh: OBJ import, group labels (floor/wall/tunnel/boulder/net/paint)
  bank.py path.py  reset-state bank + planner (A* on exact clearance)
  sgs.py           Success Guided Sampling over the bank
  perturb.py       randomly moved/rotated walls, gates, pad, launch mark; as-built course edits
  coursegen.py     procedurally generated courses
  scan_course.py   builds an eval course from a point-cloud scan of the built course
  policy.py        CNN+GRU actor for rsl_rl
  train.py evaluate.py stress.py video.py export_policy.py build_assets.py vio_calib.py   command-line tools
assets/            course meshes, route, reset banks, scoring, battery/plant/VIO parameters
checkpoints/       baseline policies (see above)
deploy/            early standalone copy of the policy node + a local PX4 SITL rig (deploy/README.md)
docs/APPROACH.md   approach and results
```

`AGENTS.md` has the engineering notes: conventions, design decisions and gotchas. Read it before changing the env.
