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

## Limitations

- Trained on one course family (the competition course, perturbed). Transfer to unseen, procedurally generated
  courses is currently ~0%; from-scratch training on generated courses is in progress.
- Sensitive to the exact window-wall geometry (see the last table row); fine-tunes on the scanned geometry are
  in progress.
- The return-leg policy does not yet work on the as-built course.
- Sim-to-real transfer has not yet been validated in flight.
