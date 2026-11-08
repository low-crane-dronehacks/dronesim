#!/bin/bash
# Return-leg policy (pad -> launch mark), from scratch on the as-built course with stereo/latency/plant randomization.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python -m dronehacks.train --run-id return_s321 --num-envs 768 --total-steps 60000000 --seed 321 --extra-boulders 5 10 --arena-half 2.85 --v-max-start 5.0 3.0 2.0 --v-max-anneal-steps 30000000 --anchor-wall wall1 --gate-extend --cp0-wall wall3 --perturb-scale 2.0 --course-mods 'move:wall1:-0.13:0.0:0;move:wall0:-0.32:-0.18:0;move:wall4:-0.08:-0.46:0;move:tunnel:0.10:-0.07:0;seg:wall2:0.75:0.98:-1.2:1.6;shift:wall1:0.30:0.46:north:mark;rot:wall3:40:50:south;place:wall5:north:0.05:0.3:-2.40:-2.10:wall3;span:wall6:wall1:-0.2:0.2:nw:-0.3:0.0' --skip-terminal --return-mode return --v-max 1.2 0.6 0.6 --yaw-rate-max 1.0 --perturb 1.0 --perturb-variants 16 --realism 'real{net_return_prob_max=0.15;depth_clean_frac=0.3;left_band_max_px=80;left_band_hi_frac=0.5;left_band_hi_lo=80;left_band_hi_hi=100;band_right_frac=0.5;frame_rate_hz=13.5;edge_fill_cols=16;hole_frac_max=0.45;rf_bias_max_m=0.3}' --realism-ramp-steps 20000000 "$@"
