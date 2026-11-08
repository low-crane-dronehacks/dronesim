"""Export the course mesh as a Gazebo (gz-sim Harmonic) world for the local PX4 SITL rig.

    python deploy/rig/export_world.py --out deploy/rig/world

Writes world/course.obj (ENU, metres; Gazebo is ENU/FLU, our assets are NED) and world/dronehacks.sdf with the
course as one static collision+visual mesh, a ground plane, a light, and the PX4 spawn note. The boundary net is
NOT included as a visual (a stereo camera sees through it); it is a thin collision shell so the drone cannot
leave the arena.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from dronehacks.mesh import CourseMesh  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets')


def write_obj(mesh: CourseMesh, path: str) -> None:
    v = mesh.vertices.astype(np.float64)
    enu = np.stack([v[:, 1], v[:, 0], -v[:, 2]], 1)          # NED -> ENU (a reflection: flip winding)
    f = mesh.faces[:, [0, 2, 1]]
    # per-vertex normals (area-weighted face normals): gz-physics (dartsim/ODE) segfaults on an OBJ without them
    fn = np.cross(enu[f[:, 1]] - enu[f[:, 0]], enu[f[:, 2]] - enu[f[:, 0]])
    vn = np.zeros_like(enu)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    with open(path, 'w') as fh:
        fh.write('# DroneHacks 2026 course, ENU metres\n')
        np.savetxt(fh, enu, fmt='v %.5f %.5f %.5f')
        np.savetxt(fh, vn, fmt='vn %.5f %.5f %.5f')
        fi = f + 1
        np.savetxt(fh, np.stack([fi[:, 0], fi[:, 0], fi[:, 1], fi[:, 1], fi[:, 2], fi[:, 2]], 1), fmt='f %d//%d %d//%d %d//%d')


SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="dronehacks">
    <physics name="1ms" type="ignore"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>37.4275</latitude_deg><longitude_deg>-122.1697</longitude_deg><elevation>30</elevation></spherical_coordinates>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.3 0.2 -0.9</direction></light>
    <light type="point" name="hall"><pose>0 0 3.5 0 0 0</pose><diffuse>0.6 0.6 0.6 1</diffuse>
      <attenuation><range>20</range><linear>0.05</linear></attenuation></light>
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        <material><ambient>0.55 0.5 0.45 1</ambient><diffuse>0.55 0.5 0.45 1</diffuse></material></visual></link></model>
    <model name="course"><static>true</static><link name="l">
      <collision name="c"><geometry><mesh><uri>course.obj</uri></mesh></geometry></collision>
      <visual name="v"><geometry><mesh><uri>course.obj</uri></mesh></geometry>
        <material><ambient>0.6 0.6 0.62 1</ambient><diffuse>0.7 0.7 0.72 1</diffuse></material></visual></link></model>
    <model name="net"><static>true</static><link name="l">
      <collision name="c"><geometry><mesh><uri>net.obj</uri></mesh></geometry></collision></link></model>
  </world>
</sdf>
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'world'))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    course = CourseMesh.load(os.path.join(ASSETS, 'course.npz'))
    write_obj(course.select(('wall', 'tunnel', 'boulder', 'paint')), os.path.join(a.out, 'course.obj'))
    write_obj(course.select(('net',)), os.path.join(a.out, 'net.obj'))
    with open(os.path.join(a.out, 'dronehacks.sdf'), 'w') as f:
        f.write(SDF)
    lm = json.load(open(os.path.join(ASSETS, 'scene.json')))['landmarks']
    n, e = lm['launch_mark_ned'][:2]
    with open(os.path.join(a.out, 'spawn.txt'), 'w') as f:
        f.write(f'# PX4_GZ_MODEL_POSE (ENU x y z roll pitch yaw): launch mark; pick any yaw\n{e:.4f},{n:.4f},0.05,0,0,1.5708\n')
    print('wrote', a.out, sorted(os.listdir(a.out)))


if __name__ == '__main__':
    main()
