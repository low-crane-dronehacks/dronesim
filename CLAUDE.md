Read @AGENTS.md first: it has the rules we optimise, how the sim works, conventions and known gotchas.

- Run `pytest` before committing; keep it passing.
- Reward stays sparse and rules-exact (competition points / 60). Ask before adding shaping.
- Judge policies with `python -m dronehacks.evaluate`, never the training curve.
- The organizers' simulator (ROS 2 / PX4) is documented in `docs/official_sim/`; this repo is our fast training sim.
