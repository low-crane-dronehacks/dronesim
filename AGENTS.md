# AGENTS.md: context for coding agents (and humans)

Read this before changing the env, the reward or the assets. Update it when you learn something the next
person would otherwise rediscover the hard way. Keep entries short.

## The competition (what we are optimising)

DroneHacks 2026, Stanford. 20 x 20 ft netted course, 68 in (1.727 m) walls, 3 min limit. Kit: Jetson Orin Nano,
Pixhawk 6C, OAK-D S2 stereo camera, downward VL53L1X rangefinder, **no GPS**.

Scoring (`dronehacks/rules.py`, out of 60):
- checkpoints 4 / 7 / 10 / 7 / 7, cleared **in order**; one reached after flying over a wall since the last
  one earns 15%
- finished (land and disarm inside the pad's outer ring): + 20 x (fastest time / yours), + 5 if checkpoint 4
  was the window rather than Gate B
- unfinished: + 7 x (fastest time to your last checkpoint / yours)
- return bonus x1.25: after landing, take off by itself, fly back over the barrier and land within 2 ft of
  the launch mark

The organizers also run an official ROS 2 / PX4 simulator (companion container, cuVSLAM VIO, Offboard
setpoints). This repo is our own fast training sim; it is not that one. See "Deployment" below.

## How the sim works

- `env.py` is one rsl_rl VecEnv over E envs. Each step: clamp the action, run the plant for 10 x 10 ms
  substeps, check collisions, contact, scoring and gate crossings at **every substep** (a fast drone
  otherwise tunnels through a wall), compute the reward, reset the finished envs, return observations.
- `quad.py`: the action is a velocity command `[fwd, right, down, yaw_rate]` in [-1, 1], scaled by `v_max`
  (default 5, 3, 2 m/s) in the **yaw-aligned body frame**. It goes through a Lee geometric velocity
  controller, deadcat rotor allocation, 4S LiPo sag + PX4-style filtered voltage compensation, first-order
  motor lag (tau 25-35 ms, scaled by voltage) and drag. Controller gains, motor tau, battery SoC and internal
  resistance are re-drawn every episode. Internally the state is a Z-up world (x east, y north)
  with XYZW quaternions; everything outside `quad.py` is **NED** (x north, y east, z down; altitude = -z).
- `render.py`: Warp BVH over the exact course triangles. Depth is z-depth / 5 m from a 72 x 50 deg camera at
  128 x 80. Collision = distance to wall/tunnel/boulder < 0.20 m. The rangefinder casts 5 rays over 27 deg.
- Observations: `policy` = 2-frame depth stack; `proprio` = body velocity, rangefinder, roll/pitch,
  previous action (+ the latched return flag). The actor gets **no position, map or goal**. `critic` gets
  privileged state (asymmetric actor-critic).
- Resets: `assets/bank_*.npz` holds tens of thousands of validated start states (the exact launch pose,
  near the goal, near the route, anywhere in free space, moving and tilted). **SGS** (`sgs.py`) picks starts
  whose recent success is near 50%, so the curriculum follows the policy. Return banks add pad / climb /
  cross / final-approach starts.
- Reward = competition points / 60, paid when earned. **Keep it sparse and rules-exact** (team decision):
  no shaping unless it matches what the rules pay. Potential shaping toward the launch mark was tried for
  the return leg and pulled the drone into the barrier.

## Conventions

- Evaluate with `python -m dronehacks.evaluate`, never from the training curve. Training numbers are
  stochastic, SGS-weighted and can disagree with deterministic eval by 15+ points.
- GPU runs are not bit-reproducible. Treat eval differences under ~4 points as noise unless repeated with
  other seeds. Different training seeds of one config differ by far more than that (see Results).
- Anything that changes env numerics (obs layout, scaling, dynamics) breaks the shipped checkpoints'
  inputs. Bump the checkpoint/notes, and keep the obs layout unless you mean to retrain.
- Train fresh for a new setup. Warm-starting from an old checkpoint has been worse every time we tried it.
- `pytest` must pass (CPU, about 20 s). Add a test with any env or dynamics change.
- Coordinates in NED, angles in radians, SI units.

## Gotchas we hit (do not rediscover)

1. **Ground reaction.** A drone starting on the ground (launch pose, pad) is clamped at its rest height
   (0.08 m centre) until it climbs above 0.3 m. Without the clamp, any downward command at the start was
   a crash. Without **friction**, a grounded drone could skid along the floor below the 0.05 m contact
   height: the return policy learned to taxi from the pad into the barrier, and to dive from climb starts.
   Both are fixed in `quad.py`. `return_s1.pt` was trained before the friction fix.
2. **init_std 0.3, entropy 0.** With std 1.0 and a sparse reward the action noise crashes the drone, the
   policy gradient is ~0, and the std drifts up instead of down.
3. **Headings in the bank matter.** With no goal input, uniformly random start headings made the task
   unlearnable (about 10% success, flat). Banks use mostly "toward the goal +- noise" headings.
4. **Recurrent eval:** reset the GRU state on done, inside `torch.inference_mode()`, and reset it before
   each eval condition. Otherwise episodes start from the previous episode's memory.
5. `MeshScene.min_obstacle_distance` returns a persistent buffer: `.clone()` it if you keep it.
6. Observations must be fresh tensors each step (rsl_rl keeps a reference until the next step).
7. Recurrent PPO minibatches must divide `num_envs`. `policy.patch_rsl_rl()` pads one minibatch at a time;
   without it the update OOMs when episodes are short.
8. 2048 envs with 45 s episodes need ~32 GB VRAM; 1024 fits in 24 GB. Return-leg (90 s) episodes cost more.
9. **Seed variance is large.** Of three identical-config runs, one learned the exact launch pose (99%
   finish) and two failed it (one collides immediately, one never lifts off) while doing fine elsewhere.
   Always look at the launch condition separately and run more than one seed.

## Known gaps vs the real competition

- Any contact ends the episode. In reality a touch costs +5 s and the run continues.
- "Fastest team" time is a stand-in: remaining geodesic distance / 4 m/s, ratio capped at 1.
- Return-leg starts assume 50 run points were banked. The policy is not told where the launch mark is,
  only that the return flag is set; it has to remember the way.
- No depth noise, no state-estimate noise, no latency, no wind. The real stack flies on VIO (cuVSLAM), whose
  drift matters indoors.
- The course is the organizers' preliminary v3 CAD; rebuild `assets/` if they publish changes.
- Ground contact counts at 0.05 m centre height while a parked drone sits at 0.08 m, so 0.05-0.08 m
  skimming is not penalised once airborne.

## Results so far (points / 60, deterministic eval)

| policy | launch run | near-route | anywhere | notes |
|---|---|---|---|---|
| comp_s2 (100M steps) | 47.1 | 29.3 | 24.5 | lands 99% from launch, ~10 s run |
| comp_s1, comp_warm | 4.7 / 0.0 | 21.2 / 29.2 | 17.7 / 26.0 | fail the launch pose (collide / never lift off) |
| comp_s2 on perturbed layouts, severity 0.5 / 1 | 41.2 / 29.5 | 28.4 / 26.3 | 23.5 / 22.2 | walls +-15 cm, +-4 deg |
| pert_s1 (perturb-trained, 2048 envs, ramp) | 4.9 | 25.3 | 22.7 | launch collides 83%; degrades less with severity |
| return_s1 | 0 | | | 85% home from cross starts, 0% from pad starts (skid bug, now fixed) |

| pert_s2 (perturb-trained, 1024 envs, no ramp) | 41.7 | 31.3 | 26.3 | 100% finish; at severity 1: 30.5 / 28.0 / 24.2 (comp_s2: 29.6 / 26.4 / 22.2) |

## Deployment (not started)

The policy outputs body-frame velocity + yaw-rate setpoints at 10 Hz from depth + rangefinder +
velocity, which maps onto PX4 Offboard `TrajectorySetpoint` velocity on the official stack. It needs a
velocity estimate (VIO) and an OAK-D depth stream at 128 x 80, 72 x 50 deg. Expect to add depth and
velocity noise in sim before trusting it.

## Updating this file

New gotcha -> add it to the list. New result -> add a row (with seeds and eval settings). Changed
interface -> fix the "How the sim works" section.
