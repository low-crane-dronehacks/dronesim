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
  (default 5, 3, 2 m/s) in the **yaw-aligned body frame**. It goes through Isaac Lab's Lee velocity
  controller (ported verbatim), rotor allocation, first-order motor lag (tau 25-35 ms) and drag. Controller
  gains and motor tau are re-drawn every episode. Internally the state is a Z-up world (x east, y north)
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
10. **Fused GPU paths shadow the torch code.** On CUDA, `TorchQuad.step_control*` runs `quad_warp.py` (all
    10 substeps in one Warp kernel) and `Realism.corrupt_depth` runs `stereo_warp.py`. The torch code is the
    reference and the CPU path: **a change to the plant/controller or the stereo model must be made in both**,
    and `tests/test_fast_paths.py` (GPU; skipped on CPU) pins them together. `TorchQuad.fast=False` /
    `Realism.fast=False` (or `DRONEHACKS_TORCH_PHYSICS=1` / `DRONEHACKS_TORCH_STEREO=1`) force the torch path.
    The fused stereo model draws its per-pixel noise from a counter-based Warp RNG, not `Realism.gen`.
11. **Host syncs dominate small-op code here,** especially while another job shares the GPU: `x[ids] = 0.0`
    (index_put with a Python scalar is a synchronous H2D copy; use `index_fill_`), boolean-mask indexing,
    `torch.tensor(...)` per call, `int()/float()/.any()` on GPU values. `torch.cuda.set_sync_debug_mode('warn')`
    lists them. env.step had 32 per step; it has 4.

## Real-world effects (`realism.py`, eval-realism branch)

Opt-in via `CourseEnvCfg.realism` (`--realism` on evaluate, `stress.py` for one effect at a time). Off =
the ideal sim, bit-identical to before. Design rules:
- every effect uses a private generator, so turning one on never shifts the nominal random stream;
- each has a cheap form usable in training (`render_scale=1`, `real_lite`) and an accurate eval form
  (depth rendered at the sensor's 640x400 and downsampled with valid-pixel averaging);
- magnitudes are best guesses for the kit, written as config fields so measurements can replace them.
  Replace in this order once we have data: plant profile (mass, inertia, motor time constant, max
  thrust), VIO velocity error (log VIO vs motion capture or a tape-measure run), stereo error on the real
  walls, end-to-end command latency.
What each models and what it does not:
- `plant`: true mass/inertia/thrust/drag/motor tau drawn per episode while the controller assumes nominal.
  Adds a velocity-loop integral (PX4 MPC_*_VEL_I_ACC defaults) and a 45 deg tilt limit. NOT PX4's actual
  cascade (attitude P + rate PID + mixer); validate against the organizers' PX4 SITL.
- `latency`: each physics substep applies the newest command issued at least `latency` ago.
- `velocity`: lagged, biased (random walk), speed-dependent noise, dropouts that freeze then re-initialise.
- `depth`: z from quantised noisy disparity (sigma_z ~ z^2), invalid below the minimum range (0.35 m) and
  in occlusion shadows / the left border band / textureless blobs; invalid pixels are fed as `invalid_fill`
  ('far' by default, which is what a naive pipeline does and the most dangerous choice; deployment should
  decide this deliberately). Rendered at the pose of a frame 30-80 ms + up to 1/15 s old.
- `net`: the boundary net is not drawn (stereo sees through it); a random room with people/tables/chairs
  stands behind it; with `depth`, a few % of net pixels return the net.
- `battery`: thrust x U(0.85, 1) at takeoff, minus 8%/min. PX4's battery compensation would hide part of this.
- `vio_pos` (with `CourseEnvCfg.pos_obs` / `train.py --pos-obs`; new runs only, proprio 10 -> 16): the actor also
  gets a course-frame pose estimate, appended LAST: position / 3 m (NED), sin / cos yaw, VIO-tracking flag. Deploy:
  course_pose = launch_mark_pose o cuVSLAM odom. The position error integrates the SAME velocity-error process as
  `velocity` (run privately if that effect is off; own generator, so adding vio_pos leaves real_lite bit-identical)
  plus the motion seen through a yaw error that random-walks 0.3 deg/sqrt(s); z error decays (tau 0.5 s,
  rangefinder/baro). Dropout: the published pose HOLDS (flag 0), then tracking resumes from the old error plus a
  re-init jump (10 cm/axis, 2 deg). Reset: launch + liftoff starts ~1 cm / 0.5 deg; others N(0, sigma(s)) from
  `assets/vio_sigma.json` (`python -m dronehacks.vio_calib`: the route flown at 1.5-3 m/s through this process;
  rerun it after changing any vio_* field). Defaults give ~9 / 18 / 35 / 50 cm per axis at s = 2 / 5 / 10 / 14 m
  (~5% of distance; the velocity model's 0.05 m/s bias dominates: pessimistic vs typical cuVSLAM, replace with a
  measured drift; `vio_pos_drift_scale` scales it). `pos_obs` without `vio_pos` = the true pose.

### Stress-test findings (2026-11-05, comp_s2 / pert_s2, launch points; ideal 47.1 / 41.6)

- Minor: VIO-like velocity noise 42.7 / 39.1 (half of the loss is dropouts), rangefinder dropouts 44.0 / 40.6,
  latency 20-50 ms 38.5 / 37.4.
- Large: plant error 28.9 / 26.6, battery sag 19.6 / 21.4, PX4-like velocity integral alone 17.4 / 24.5,
  latency 40-120 ms 13.9 / 20.5. The policy is tuned to the training controller's exact response; it flips
  its commands constantly, so an integral term winds up in flight.
- Fatal: see-through net 4.8 / 0 (flies out: the first corridor's west wall IS the net, and training drew it
  as a solid wall), and stereo depth 0 / 0.
- Depth cause: NOT noise (noise + quantisation alone: 47.0 / 41.6). At takeoff the camera is 8 cm above the
  floor, so the floor in the lower image is inside the OAK-D's 0.35 m minimum range and invalid. Fed as 'far'
  it reads as open space -> 0.5 / 0.3; fed as 'near' -> 44.6 / 41.4. Deployment must map invalid pixels to
  near (or train with them). Occlusion bands and holes still break the policy with either fill.
- Pipeline check: the 640x400 depth path with every error off scores like ideal (47.0 / 41.6).

### Organizers' simulator capture (2026-11-06, ~/dronehacks-official/capture, read-only, nothing flown)

**RULE: the organizers' simulator is a HELD-OUT test (sim2sim generalization), never a training target.** Do not
fit realism.py, the banks, or any training setting to its numbers below: it is itself only an approximation of
the real course, and tuning to it would hide sim2real gaps instead of exposing them. Use it only to evaluate.
Fixes must be justified by the real hardware/rules (e.g. "the drone can be placed at any heading"), not by
"their sim does X".

- Geometry: their 5 cm collision voxels match our course mesh (99.9% within 6 cm, zero offset). Their start pose
  is NED (1.114, -2.515), facing NORTH (heading 0), 0.2 m north of the painted launch mark.
- Our training launch pose faced 134.6 deg (a degenerate route tangent at the vertical takeoff): comp_s2 launched
  facing north scores 0 (64/64 floor) vs 47.4 at 134.6. Every ground start must vary heading + position.
- Their stereo depth (SGBM, OAK-D S2 640x400, fx 421, baseline 7.49 cm, 16UC1 mm, ~15 Hz) on the pad, on the
  128x80 policy grid: 39% valid (ours: 71-77%). Top half 65-69%, floor rows 4-5% (textureless floor gives no
  return), a fixed ~16-column invalid band on the LEFT edge (the disparity search range, independent of depth),
  ~1/3 of wall pixels missing, half of all pixels flicker valid/invalid frame to frame. Temporal noise < 1 cm
  to 3 m, ~11 cm at 3-4 m. (Held out: NOT used to tune our depth model; it is a test of how far a policy trained
  on our model generalizes.)
- VIO at rest: < 1 mm position noise, no drift in 12 s (moving behaviour unknown). Sim RTF ~0.88.
- The bag (pad_bag, mcap, 15 s): both stereo images + their depth + IMU + cuVSLAM odometry + PX4 state; read with
  ~/venvs/rosbags (rosbags). Extract: capture/pad_extract.npz.

## Known gaps vs the real competition

- Any contact ends the episode. In reality a touch costs +5 s and the run continues.
- "Fastest team" time is a stand-in: remaining geodesic distance / 4 m/s, ratio capped at 1.
- Return-leg starts assume 50 run points were banked. The policy is not told where the launch mark is,
  only that the return flag is set; it has to remember the way.
- Training still uses the ideal sim (no depth / state-estimate noise, no latency); realism.py models these
  for evaluation. No wind, no ground effect, no prop wash near walls.
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

Numbers above for comp_s2 / pert_s2 / return_s1 were re-measured with this repo on GPU and match the original
original training-time evals within noise (the env is bit-identical except the later floor-friction fix).

## Deployment (not started)

The policy outputs body-frame velocity + yaw-rate setpoints at 10 Hz from depth + rangefinder +
velocity, which maps onto PX4 Offboard `TrajectorySetpoint` velocity on the official stack. It needs a
velocity estimate (VIO) and an OAK-D depth stream at 128 x 80, 72 x 50 deg. Expect to add depth and
velocity noise in sim before trusting it.

## Updating this file

New gotcha -> add it to the list. New result -> add a row (with seeds and eval settings). Changed
interface -> fix the "How the sim works" section.
