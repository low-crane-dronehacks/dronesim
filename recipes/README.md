# Training recipes

Exact commands behind the published policies and results. Each was regenerated from the run's saved arguments
and checked by re-parsing it with `dronehacks/train.py`'s own argument parser. Run from the repository root
(`PYTHONPATH=.`). Fine-tunes name their starting checkpoint (`--init-checkpoint`); those files are in `checkpoints/`.

The generalist recipe needs the 62-course training pack (31 MB, not in git). Rebuild it with:

```bash
python -m dronehacks.coursepack build --seeds 0-63 --out assets/packs/gen64.npz   # 2 of 64 seeds are rejected by the generator
```

The 5-course held-out evaluation pack is included: `assets/packs/heldout5.npz` (seeds 100-104).

Lineage of the run policy: `asb_s115` -> `c5_s164` -> `s196` -> (`holes60_s274`, `ftw0_s330`).

## c5_s164

Run policy on the as-built course with stereo/sensor randomization, perturbed layouts. Fine-tuned from checkpoints/asb_s115.pt (an earlier as-built-course policy from this simulator).

`recipes/c5_s164.sh`

## s196

Adds per-episode stereo holes and the drone-17 camera band; fine-tuned from c5_s164. Proposed default policy.

`recipes/s196.sh`

## holes60_s274

Hole-robust: stereo holes up to 60%, rangefinder bias, net-frame bar dropout, open gap under the walls. Fine-tuned from s196.

`recipes/holes60_s274.sh`

## ftw0_s330

Window wall (wall 0) re-laid on the scan of the built course (assets/course_w0scan.npz) with randomized window sill/top heights. Fine-tuned from s196.

`recipes/ftw0_s330.sh`

## scratch_holes60_s300

The hole-robust recipe trained FROM SCRATCH (no init checkpoint).

`recipes/scratch_holes60_s300.sh`

## gen_fast_s306

Generalist: from scratch on 62 procedurally generated courses (course pack, see below), ideal depth, 5 m/s.

`recipes/gen_fast_s306.sh`

## return_s321

Return-leg policy (pad -> launch mark), from scratch on the as-built course with stereo/latency/plant randomization.

`recipes/return_s321.sh`
