# Local PX4 SITL rig (evaluation only: tests the policy node on real PX4 software, never a training target)

Phase 1 (target): Docker container on a lab PC with
- ROS 2 Jazzy + Gazebo Harmonic (start from `osrf/ros:jazzy-desktop-full`, which ships ros_gz/Harmonic)
- PX4-Autopilot SITL (gz_x500_depth or x500 + a forward depth camera), uXRCE-DDS agent, `px4_msgs` matching the
  PX4 version, all built from their upstream sources
- world: `deploy/rig/world/dronehacks.sdf` (our course mesh, ENU; net = invisible collision shell), spawn at the
  launch mark (`world/spawn.txt`, any yaw)
- state estimate: PX4 EKF fed with external vision from the gz ground-truth pose (+ noise) instead of GPS
- `deploy/policynav` running against it (`range_source:=altitude`, depth from the gz depth camera, 72x50 deg-ish)
Phase 2: OAK-like stereo pair (640x400, fx 421, baseline 7.5 cm) + a stereo matcher + cuVSLAM instead of truth.

Regenerate the world after a course change: `python deploy/rig/export_world.py`.
