#!/bin/bash
# Generalist: from scratch on 62 procedurally generated courses (course pack, see below), ideal depth, 5 m/s.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python -m dronehacks.train --run-id gen_fast_s306 --num-envs 768 --seed 306 --launch-frac 0.3 --cp-shaping 0.02 --start-heading-max 90.0 --skip-terminal --launch-liftoff-frac 0.5 --course-pack packs/gen64.npz "$@"
