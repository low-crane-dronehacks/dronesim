Read @AGENTS.md first: it has the rules we optimise, how the sim works, conventions and known gotchas.
The approach and results are in docs/APPROACH.md.

- Run `pytest` before committing; keep it passing.
- Reward is the competition score (points / 60) plus at most the potential-based `cp_shaping` term. Ask before
  adding any other shaping.
- Judge policies with `python -m dronehacks.evaluate`, never the training curve.
- The onboard ROS 2 / PX4 stack lives in https://github.com/low-crane-dronehacks/deploy; this repo is the
  training simulator.
