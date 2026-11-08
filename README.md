# dronehacks-sim

Approach, robustness measures, evaluation results and limitations: [docs/APPROACH.md](docs/APPROACH.md)

GPU simulator and RL training for the **DroneHacks 2026** course, the Stanford indoor drone race.
Thousands of quadrotors fly the exact course mesh in parallel. They see a depth camera and a downward
rangefinder, and they are scored by the competition rules.

- **Course:** the organizers' OBJ, ray traced exactly with NVIDIA Warp.
- **Drone:** a Lee geometric velocity controller, a rotor lag model and drag. Pure torch, no physics engine.
- **Scoring:** checkpoints in order, speed bonus, window bonus, the over-wall penalty, the pad landing,
  and the optional return leg (x1.25).
- **Policy:** CNN + GRU over the depth stack and proprioception, trained with PPO (rsl_rl). It gets no
  position or goal input; it navigates from what it sees.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[video,dev]"        # torch, warp-lang, rsl-rl-lib 5.4.1, tensordict, numpy, scipy
pytest                                  # ~20 s on CPU
```

A CUDA GPU is needed for training. Everything else also runs on CPU (`--device cpu`), just slowly.

## Use

```bash
# score a shipped checkpoint (launch run, near-route starts, anywhere starts)
python -m dronehacks.evaluate checkpoints/comp_s2.pt
python -m dronehacks.evaluate checkpoints/pert_s2.pt --perturb 0.5    # on randomly moved walls/gates

# watch it fly (FPV depth + top-down map; needs ffmpeg)
python -m dronehacks.video checkpoints/comp_s2.pt --out out/comp_s2.mp4

# train (2048 envs needs ~32 GB VRAM; use --num-envs 1024 on 24 GB). ~100M steps = a few hours.
python -m dronehacks.train --run-id my_run
python -m dronehacks.train --run-id my_pert --perturb 1.0               # layout randomisation
python -m dronehacks.train --run-id my_ret --return-phase               # + the return leg
```

### Real-world eval (`eval-realism` branch)

```bash
python -m dronehacks.stress checkpoints/comp_s2.pt                   # each real-world effect alone, then all
python -m dronehacks.evaluate checkpoints/comp_s2.pt --realism real  # everything on
python -m dronehacks.evaluate checkpoints/comp_s2.pt --realism depth,velocity --plant assets/plant_kit.json
```

Effects (`dronehacks/realism.py`, all off by default): `plant` (true airframe differs from nominal, PX4-like
velocity loop), `battery` (thrust sag), `latency` (late commands), `velocity` (VIO estimate: lag, noise, bias,
dropouts), `depth` (OAK-D stereo at 640x400: noise, min range, holes, occlusion shadows, latency), `net`
(see-through boundary net with a random room behind it), `rangefinder` (dropouts). `real` = all of them;
`real_lite` / `depth_lite` = cheap versions for training. `vio_pos` (+ `train.py --pos-obs`) adds a VIO-like course-frame
position/yaw input to the actor (calibrated by `python -m dronehacks.vio_calib` -> `assets/vio_sigma.json`). Magnitudes are best guesses; replace with
measurements (`assets/plant_nominal.json` is the template for the measured drone).

Runs go to `runs/<run-id>/` (`metrics.jsonl`, `checkpoints/`, `config.json`). Re-running a run id resumes it.

## Shipped checkpoints

Points out of 60, `python -m dronehacks.evaluate` on an L40S: mean of seeds 1000 and 2000, 512 launch episodes,
2048 starts per other condition. Each `.pt` is an rsl_rl state dict; its `.json` holds the training config.

| checkpoint | launch run | near-route starts | anywhere | walls moved (severity 1): launch / route / anywhere |
|---|---|---|---|---|
| `comp_s2.pt` | 47.1 (99% finish) | 29.4 | 24.5 | 29.6 / 26.4 / 22.2 |
| `pert_s2.pt` | 41.7 (100% finish) | 31.3 | 26.3 | 30.5 / 28.0 / 24.2 |
| `return_s1.pt` | 0 (collides) | 18.8 | 13.1 | |

`comp_s2` is fastest on the exact course; `pert_s2` (trained on moved walls) is more reliable everywhere else.
`return_s1` predates the floor-friction fix: it returns home from 74% of over-the-barrier starts and 41% of
final-approach starts, never from the pad, and its run phase now collides at launch. Retraining is under way.

## Layout

```
dronehacks/
  rules.py         scoring rules + checkpoint geometry (from the organizers' diagram)
  env.py           the vectorised env: resets, scoring, observations, metrics
  quad.py          drone dynamics + Lee controller (torch)
  render.py        Warp mesh queries: depth images, obstacle distance, rays
  mesh.py          course mesh: OBJ import, group labels (floor/wall/tunnel/boulder/net/paint)
  bank.py path.py  reset-state bank + planner (A* on exact clearance)
  perturb.py       randomly moved/rotated walls, gates, pad, launch mark
  sgs.py           Success Guided Sampling over the bank
  policy.py        CNN+GRU actor for rsl_rl
  train.py evaluate.py video.py build_assets.py   command-line tools
assets/            course mesh, route, reset banks, gate distances
checkpoints/       trained policies
```

`AGENTS.md` has the details: conventions, design decisions, known gaps vs the real rules, and results so far.
Read it before changing the env.
