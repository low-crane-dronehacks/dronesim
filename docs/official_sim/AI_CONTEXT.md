# DroneHacks Lab: context for your coding agent

Give this file to Claude, ChatGPT, or another coding agent. Work in the current
project; inspect its source and live interfaces before proposing changes.

## What you are building

Your code runs in an unprivileged Linux companion container, like an onboard
Jetson application. ROS 2 Jazzy and matching `px4_msgs` are installed. PX4 runs
separately as the autopilot; Micro XRCE-DDS bridges its messages into ROS 2.
On a real Pixhawk/Jetson this agent usually connects over a reviewed UART or
UDP link. In this lab the simulator supplies that transport. Do not start a
second agent, create a fake UART, install a second autopilot, or modify the host.

The selected aircraft is Sky Beta in the feature-rich DroneHacks course.
Physics uses simulation time. ROS nodes use standard `use_sim_time` and `/clock`;
hardware deployment uses the normal system clock. A slower real-time factor
means a longer wall-time run; inspect actual sensor timestamps and health.

## Files to read

- `onboard/examples/takeoff_land.py`: native ROS/PX4 takeoff, hover and landing.
  It sends actual Offboard/mode/arming commands; reading it does not run it.
- `onboard/README.md`: services, ownership, frames, camera/IMU configuration and
  hardware transfer. Read the live files; do not assume they match a template.
- `onboard/slam/cuvslam_lib_node.py`: the cuVSLAM wrapper. It consumes calibrated
  stereo and IMU through ROS; it is a starting estimator, not a finished maze
  navigation system or an automatically validated hardware deployment.
- `onboard/vio/ev_bridge.py`: validates/transforms visual estimates and sends
  PX4 external-vision odometry. Images alone do not give PX4 a position fix.
- `onboard/config/`: launch settings, calibration and estimator contracts.
- `learn/*.ipynb`: runnable lessons in a persistent Jupyter kernel. Read a cell,
  then Run it. Flight cells require explicit confirmation. Interrupting Python
  is not a Land command. Keep your own notes and saved outputs.
- `main.py`, `flight.yaml`, `route.json`, `refnav/`, `tests/`: editable reference
  mission and tests. The native example is the shortest path to inspect when
  you want to understand the underlying PX4 interface.

## First inspection and flight

Open the companion terminal (Ctrl+` in the editor), then inspect rather than
automatically flying:

```bash
pwd
python3 --version
printenv ROS_DISTRO ROS_DOMAIN_ID
ros2 topic list
python3 -m pytest tests/ -q
```

Topic names are discovery hints, not proof of content. Observe actual messages
for at least 30 seconds. PX4 status and local-position output topics use `_v1`
on this pinned image. Standard native input topics include
`/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`,
`/fmu/in/vehicle_command`, and `/fmu/in/vehicle_visual_odometry`. Read the
example and installed message definitions instead of guessing field layouts.
Camera images, CameraInfo, TF and IMU have their real ROS message types.
The cuVSLAM stereo interface is `/stereo/left/image` and `/stereo/right/image`;
inspect the launch/config files for current camera/IMU topic mappings.

The qualified G4 preset renders both mono eyes at 640x400, with unchanged
baseline/FOV and appropriately scaled calibration. Stereo/depth are configured
at 15 Hz, IMU at 480 Hz, physics at 240 Hz; delivered rates can include modeled
losses. RGB and observer views are different streams. The world observer is a
display aid, never an onboard navigation sensor or a source of position truth.

When the team explicitly approves a bounded flight, wait for READY and fresh
landed/disarmed PX4 plus valid estimator reports. Inspect the example's 1 m
height, hover duration and landing behavior, then run:

```bash
python3 onboard/examples/takeoff_land.py
```

Do not run another mission or keyboard controller concurrently. Offboard needs
a sustained setpoint stream and a usable estimator. With VIO stopped indoors,
PX4 can reject a position-controlled takeoff; a forced crash is not a more
realistic test. Manual mode stabilizes attitude without holding height/XY.
Keyboard RC is an optional simulator input device handled by PX4, not mission
code or a faithful radio model. Releasing it invokes PX4's configured RC-loss
behavior. Simulator Reset stays in the website; it is not a flight-code API.

Use Run saved file when you need the lab's managed recording and independent
grade. Native terminal/notebook execution does not automatically create a
managed Run. Inspect PX4 ULogs and explicit recorder output as appropriate.
Keep failed evidence; a process exiting successfully does not prove a safe flight.

## Working with AI and teammates

Describe the intended mission, assumptions, frames, estimator dependencies and
loss-of-data behavior before editing. Keep changes small and test the actual
behavior. Ask the agent to explain the controller and geometry rather than
accepting generated code without understanding it. Strong fundamentals plus
careful AI use are encouraged. Save files and stop background coding agents
before releasing editing control; a detached process can still write files.
Credentials and private keys stay outside the project and prompts.

## Transfer to a real aircraft

The mission should retain its ROS/PX4 algorithm. Deployment must replace the
virtual devices with an actual OAK driver, measured CameraInfo/TF and IMU-camera
calibration, matching PX4 messages, and one reviewed XRCE transport. Use an ARM
build compatible with the Jetson's JetPack/ROS/CUDA/cuVSLAM versions. This
x86-64 image's GPU binaries cannot be copied onto an Orin.

Do not copy the simulator's PX4 parameter file onto hardware. Recheck motor
order, controller tuning, mass, failsafes, RC takeover/kill switch, estimator
latency, camera mounts and lighting. The prepared scene and sensor models are
useful approximations; real netting, blur, calibration, vibration, compute
limits and contact dynamics need physical tests. No first-flight guarantee is
made. Real missions begin from an approved software/RC start mechanism and
then fly autonomously; keep a tested physical RC safety path available.

After using keyboard RC, land and release RC, then use **Reset world** before
running an automated program, including a notebook or native terminal script.
The cloud follow-up run without Reset failed; the fresh-world native run passed.
The web Run button enforces this qualification boundary; terminal Python remains
ordinary ROS and does not enforce it for you. This is not a VIO fix.
