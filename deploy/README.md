# Deploying a policy on the native ROS 2 / PX4 stack (organizers' simulator, then the aircraft)

`policynav/` runs a trained checkpoint with NumPy only (no torch) and talks native ROS 2 / PX4:
depth + PX4 velocity/attitude + a downward range in, Offboard velocity setpoints out. Safety and stop
conditions are in `policynav/mission.py`. **Nothing arms unless you pass `arm:=true`.**

1. Export a checkpoint (here, with torch): `python -m dronehacks.export_policy checkpoints/comp_s2.pt --out deploy/policynav/comp_s2.npz`
   (verifies the NumPy forward pass against torch).
2. Copy `policynav/` and `main_policy.py` into the flyx workspace (next to `main.py`).
3. Dry run (reads live sensors, runs the policy, prints what it WOULD command, publishes nothing):
   `python3 main_policy.py --ros-args -p use_sim_time:=true -p range_source:=sim`
4. Fly (team approval; Reset simulation first): add `-p arm:=true`.

`range_source`: `sim` = the lab's simulation-only voxel raycast (the simulator has no rangefinder),
`distance_sensor` = PX4's `/fmu/out/distance_sensor` (the kit's VL53L1X; aircraft), `altitude` = height above
takeoff. Depth is re-projected from the OAK depth image into the training camera (128x80, 72x50 deg);
invalid pixels read as near, never as open space.
