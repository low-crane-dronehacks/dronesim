#!/bin/bash
# Run policy on the as-built course with stereo/sensor randomization, perturbed layouts. Fine-tuned from checkpoints/asb_s115.pt (an earlier as-built-course policy from this simulator).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python -m dronehacks.train --run-id c5_s164 --num-envs 768 --total-steps 60000000 --seed 164 --launch-frac 0.3 --extra-boulders 5 10 --arena-half 2.85 --cp-shaping 0.02 --launch-yaw-max 45.0 --start-heading-max 90.0 --v-max-start 5.0 3.0 2.0 --v-max-anneal-steps 1 --anchor-wall wall1 --gate-extend --cp0-wall wall3 --perturb-scale 3.0 --course-mods 'shift:wall1:0.30:0.46:north:mark;rot:wall3:15:45:south;place:wall5:north:0.05:0.3:-2.25:-1.25:wall3;gap:wall3:wall5:0.80:1.0;span:wall6:wall1:-0.2:0.2:nw:-0.3:0.0' --skip-terminal --launch-liftoff-frac 0.5 --v-max 1.2 0.6 0.6 --yaw-rate-max 1.0 --perturb 1.0 --perturb-variants 16 --realism 'real{net_return_prob_max=0.15;depth_clean_frac=0.3;left_band_max_px=80;left_band_hi_frac=0.5;left_band_hi_lo=80;left_band_hi_hi=100;band_right_frac=0.5;frame_rate_hz=13.5;edge_fill_cols=16}' --realism-ramp-steps 0 --lr 0.0001 --init-checkpoint checkpoints/asb_s115.pt "$@"
