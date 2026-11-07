# DroneHacks participant workspace

Read `AI_CONTEXT.md` before changing or running code. This file is for the
student's coding agent, not for the simulator's maintainers.

The project is a Linux companion application using native ROS 2 and `px4_msgs`.
Write flight code against those interfaces; keep simulator internals out of it.
Start by reading `onboard/examples/takeoff_land.py`, `onboard/README.md`, and the
executable notebook under `learn/`. Never run flight code merely to inspect it.

Ask the team to approve a proposed mission before it arms. Verify fresh PX4
mode, estimator, landed and armed reports. One teammate holds editing control;
stop your coding agent and save before handing control over. Never put secrets,
login cookies or private keys in code, notebooks, prompts or Git.

Use `CONNECT_FROM_LAPTOP.md` for access and `AI_CONTEXT.md` for architecture,
testing, simulation limits and the hardware transfer checklist. Teams with
strong fundamentals and first-principles understanding, combined with careful
AI use, will make better decisions than an agent working without supervision.

After using keyboard RC, land and release RC, then use **Reset world** before
running an automated program, including a notebook or native terminal script.
The cloud follow-up run without Reset failed; the fresh-world native run passed.
The web Run button enforces this qualification boundary; terminal Python remains
ordinary ROS and does not enforce it for you. This is not a VIO fix.
