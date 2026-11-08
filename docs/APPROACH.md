# Approach

## Summary

- **Simulator.** GPU-parallel (thousands of envs) PyTorch simulator. Depth images are ray traced against the
  course mesh with NVIDIA Warp kernels.
- **Plant.** Quadrotor with a Lee geometric velocity controller, first-order rotor lag and aerodynamic drag
  (`dronehacks/quad.py`). The policy commands body-frame velocity + yaw rate, which is how the real drone is flown.
- **Policy.** PPO (rsl_rl), CNN + GRU actor on stacked depth frames plus proprioception (velocity estimate,
  attitude, rangefinder). There is **no position or goal input**: the policy must recognise where it is on the
  course from depth alone.
- **Reward/score.** Episodes are scored by the competition rules (points out of 60).

## Why the method generalizes to new environments

**The reward comes from the task definition, not from tuning.** The reward is the competition score itself
(points / 60, paid when earned): checkpoints crossed in order, the pad landing, the speed bonus and the window
bonus (`dronehacks/env.py`, `dronehacks/rules.py`). A new course needs only what defines the task anyway: the
mesh, the checkpoint lines, the pad and the launch pose. There are no hand-weighted penalty or shaping terms to
re-balance per course. The one dense term (`cp_shaping`) is potential-based, so it does not change which
policy is optimal: reward per metre of approach to the next checkpoint. Its coefficient is the same for every
environment we train on: the published map, the scan-registered as-built course, perturbed layouts, and a pack
of 62 procedurally generated courses (`dronehacks/coursegen.py`).

**SGS replaces a hand-designed curriculum.** Training episodes start from a reset bank: a large set of start
states that `dronehacks/bank.py` samples automatically from the course geometry. They cover the launch pose,
points along the planned route, points near the goal, and anywhere in free space. Each start state is its own
cell with a running history of outcomes (last 100 episodes). Success Guided Sampling (SGS; Zhang et al.,
arXiv:2610.12465) draws starts in proportion to a Beta-shaped weight of each cell's success rate, peaked at
the target 0.5 (`dronehacks/sgs.py`). So:
- The curriculum is emergent. Early on, only starts near the goal succeed sometimes, so sampling concentrates
  there. As the policy improves, the 50%-success frontier moves back along the course toward the launch. The
  sparse, task-defined reward becomes learnable without staging or reward shaping.
- It adapts to whatever is hard in a given course (a tight gap, the tunnel, the landing) without anyone
  identifying it. Cells that are always solved or never solved are sampled rarely.
- A fixed share of resets (`--launch-frac`, 0.3 in our runs) still starts at the launch pose, so the full run
  from launch is always being practised.
- Nothing is course-specific. The bank is generated from geometry, and the SGS hyperparameters (target 0.5,
  concentration 1, history 100, temperature 2) are fixed across every experiment and every course.

**Robustness is built in, not bolted on.** The sensor and plant randomization below is a property of the
simulator, not of the course. It applies unchanged to any environment the course loader or generator produces.

## Robustness / domain randomization

Configured in `dronehacks/realism.py` (sensor and plant effects), `dronehacks/perturb.py` (course layout) and
`dronehacks/env.py` (curricula, start states). Per-episode unless noted.

- **Stereo depth model** (OAK-D-like: 640x400, 7.5 cm baseline, 95 px disparity search):
  - disparity noise (sigma 0.2 px) and sub-pixel quantization (1/8 px), so error grows with range;
  - occlusion shadows (invalid bands beside foreground edges);
  - spatially correlated textureless holes, fraction randomized per episode up to 45-60%;
  - left-edge invalid band, width ~ U(0, 100) px, covering the 95 px disparity search range
    (with a share of episodes at 80-100 px and a share mirrored to the right edge);
  - net returns: pixels looking at the safety net return the net with a randomized probability.
- **Timing.** Actuation and depth latency; depth frame rate 13.5 Hz (policy runs at 10 Hz).
- **Plant randomization.** Mass +-10%, inertia +-20%, thrust efficiency +-8%; battery sag; velocity-loop
  integral gain randomized.
- **State estimate.** Velocity-estimate noise (white + speed-dependent + bias random walk + dropouts),
  attitude noise.
- **Rangefinder.** Noise and dropouts plus a constant per-episode bias ~ U(-0.3, +0.3) m.
- **Net frame.** Posts/rails of the net frame randomly missing from depth (bar dropout) and offset from the net plane.
- **Layout perturbation.** 16 layouts per run: walls and gates translated and rotated, centre wall anchored,
  extra boulders added.
- **As-built course variants.** Wall positions registered to a scan of the built course; gap under the walls
  0-0.35 m; window sill and top heights randomized.
- **Speed curriculum.** Speed cap ramps 5 -> 1.2 m/s while the observation scale stays fixed, so the
  policy's inputs mean the same thing throughout training.
- **Start state.** Idle on the floor at the launch mark (the policy must take off itself).

## Evaluation protocol

Deterministic policy (mean action), 512 launches per condition, fixed layout seeds, the deployed drone's camera
band (left-edge invalid band) and the deploy-side edge fill applied. Success % / mean points (out of 60):

| condition | deployed c5_s164 | s196 (current default) | s274 (hole-robust) |
|---|---|---|---|
| as-built course, 4% stereo holes | 93.6 / 37.8 | 91.4 / 37.8 | 88.7 / 37.1 |
| 20% stereo holes | - | 86.3 / 36.7 | 88.7 / - |
| 35% stereo holes | - | 14.1 / 25.9 | 63.3 / - |
| 1 ft open gap under every wall | - | 81.6 / 33.9 | 83.4 / - |
| perturbed layouts (v5h3) | - | 79.5 / 34.2 | 77.0 / 32.4 |
| perturbed layouts (published map) | - | 81.4 / 38.3 | 76.0 / 36.3 |
| wall 0 + window at scanned position | 12.7 / 28.0 | 53.7 / 32.6 | 19.5 / 28.6 |
