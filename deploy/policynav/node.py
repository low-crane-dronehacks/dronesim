#!/usr/bin/env python3
"""policynav.node: run a trained dronehacks-sim policy on the native ROS 2 / PX4 stack (sim or aircraft).

  in   depth          sensor_msgs/Image (16UC1 mm or 32FC1 m) + CameraInfo  (depth_topic, default the OAK depth)
       calibration    TF base_link <- depth frame_id (looked up once)
       PX4            /fmu/out/vehicle_local_position_v1, vehicle_attitude, vehicle_status_v1, vehicle_land_detected
       health         std_msgs/Bool (health_topic): the estimator feeding PX4 is healthy
       range          downward distance: 'distance_sensor' (/fmu/out/distance_sensor, the kit's VL53L1X),
                      'sim' (SIMULATION ONLY: a voxel raycast from dronehacks_sim, as the lab's SIM_SENSORS.md
                      describes, because the simulator has no rangefinder) or 'altitude' (height above takeoff)
  out  /fmu/in/offboard_control_mode (velocity) + trajectory_setpoint (NED velocity + yawspeed), vehicle_command
       /policynav/status (JSON)
Nothing arms unless the operator passes arm:=true. See mission.py for every stop condition.

    python3 main_policy.py --ros-args -p policy_file:=policynav/comp_s2.npz -p range_source:=sim [-p arm:=true]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from . import mission as M
from . import obs as O
from .policy_np import NumpyPolicy

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
CMD_SET_MODE, CMD_ARM, CMD_LAND = 176, 400, 21


def decode_depth(m) -> np.ndarray:
    enc = m.encoding.lower()
    if enc in ('16uc1', 'mono16'):
        a = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.step // 2)[:, :m.width].astype(np.float32) * 1e-3
    elif enc == '32fc1':
        a = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, m.step // 4)[:, :m.width].copy()
    else:
        raise ValueError(f'unsupported depth encoding {m.encoding!r}')
    a[np.isposinf(a)] = 1e3                 # +inf = nothing within range (a simulator's 'miss'): far, NOT invalid
    a[np.isnan(a) | (a <= 0)] = np.nan      # 0 / NaN = no stereo match: invalid (reads as near)
    return a


class PolicyNav(Node):
    def __init__(self):
        super().__init__('policynav')
        dp = lambda name, default: self.declare_parameter(name, default).value     # noqa: E731
        pf = dp('policy_file', 'comp_s2.npz')
        self.policy = NumpyPolicy(pf if os.path.isabs(pf) else os.path.join(os.path.dirname(os.path.abspath(__file__)), pf))
        meta = self.policy.meta
        self.meta = meta
        self.n_act = meta['num_actions']
        self.v_max, self.yaw_max = meta['v_max'], meta['yaw_rate_max']
        mp = M.MissionParams()
        for k, v in vars(M.MissionParams()).items():
            val = dp(k, v)
            if isinstance(v, bool):                       # 'false' / "false" / False -> False (bool('false') is True)
                val = val if isinstance(val, bool) else str(val).strip().strip("'\"").lower() in ('1', 'true', 'yes')
            setattr(mp, k, type(v)(val))
        self.mission = M.Mission(mp)
        self.depth_topic = dp('depth_topic', '/camera/depth/image_raw')
        self.info_topic = dp('depth_info_topic', '/camera/depth/camera_info')
        self.base_frame = dp('base_frame', 'base_link')
        self.range_source = dp('range_source', 'distance_sensor')
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand,
                                  VehicleLandDetected, VehicleLocalPosition, VehicleStatus)
        self._OCM, self._TSP, self._CMD = OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.pub_ocm = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        self.pub_status = self.create_publisher(String, dp('status_topic', '/policynav/status'), 10)
        self.v = M.Vehicle()
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._on_att, PX4_QOS)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected', self._on_land, PX4_QOS)
        self.create_subscription(Bool, dp('health_topic', '/estimator/healthy'), self._on_health, 10)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.range_m, self.sim, self.rf_offset, self.land_seen = None, None, None, False
        if self.range_source == 'distance_sensor':
            from px4_msgs.msg import DistanceSensor
            self.create_subscription(DistanceSensor, '/fmu/out/distance_sensor', self._on_range, PX4_QOS)
        elif self.range_source == 'sim':
            from dronehacks_sim import Simulation          # simulation-only sensor model (SIM_SENSORS.md)
            self.sim = Simulation()
            self.rng = np.random.default_rng(0)

        self.K = self.reproj = None
        self.stack = O.FrameStack(meta['frame_stack'])
        self.depth_img = None
        self.prev_action = np.zeros(self.n_act, np.float32)
        self.vel_sp, self.yawspeed_sp = np.zeros(3), 0.0
        self.complete = False
        self.timing = []
        self.create_timer(mp.control_dt, self._control)          # node clock: follows /clock in simulation
        self.create_timer(0.05, self._stream)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(f"policy {meta['source']} ({self.n_act} actions, v_max {self.v_max}), range "
                               f"{self.range_source}, depth {self.depth_topic}, arm={mp.arm}")

    # ── inputs ──
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_lpos(self, m):
        self.v.pos, self.v.vel = (m.x, m.y, m.z), (m.vx, m.vy, m.vz)
        self.v.pos_ok = bool(m.xy_valid and m.z_valid and m.v_xy_valid and m.v_z_valid
                             and all(math.isfinite(c) for c in (m.x, m.y, m.z, m.vx, m.vy, m.vz)))

    def _on_att(self, m):
        self.v.q = tuple(float(c) for c in m.q)

    def _on_status(self, m):
        self.v.armed, self.v.nav_state = m.arming_state == 2, int(m.nav_state)

    def _on_land(self, m):
        self.v.landed, self.land_seen = bool(m.landed), True

    def _on_health(self, m):
        self.mission.on_health(self._now(), m.data)

    def _on_range(self, m):
        if m.signal_quality != 0 and m.min_distance <= m.current_distance <= m.max_distance:
            self.range_m = float(m.current_distance)

    def _on_info(self, m):
        if m.k[0] > 0 and self.K is None:
            self.K, self.src_hw = (m.k[0], m.k[4], m.k[2], m.k[5]), (m.height, m.width)

    def _on_depth(self, m):
        if self.K is None:
            return
        if self.reproj is None:
            from rclpy.time import Time
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, m.header.frame_id, Time())
            except Exception as e:                                     # not on the TF tree yet
                self.get_logger().warning(f'no TF {self.base_frame} <- {m.header.frame_id}: {e}',
                                          throttle_duration_sec=5.0)
                return
            r, t = tf.transform.rotation, tf.transform.translation
            R = O.quat_to_mat(r.w, r.x, r.y, r.z)
            H, W = self.meta['depth_hw']
            self.reproj = O.DepthReprojector(self.K, self.src_hw, R, (t.x, t.y, t.z), out_hw=(H, W),
                                             hfov_deg=self.meta['hfov_deg'], vfov_deg=self.meta['vfov_deg'],
                                             far=self.meta['far_clip'], stride=2)
            self.get_logger().info(f'depth calibration: optical axis {self.reproj.axis_err_deg:.1f} deg off body x, '
                                   f't=({t.x:.3f},{t.y:.3f},{t.z:.3f})')
        try:
            self.depth_img = self.reproj(decode_depth(m))
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        self.mission.on_depth(self._now())

    def _range(self) -> float:
        if self.range_source == 'sim':
            try:
                pose = self.sim.pose()
                hit = self.sim.voxels.raycast(pose.position_enu_m, pose.body_to_world([0, 0, -1]), self.meta['rf_max'])
                r = self.meta['rf_max'] if hit is None else hit.distance_m
                if self.rf_offset is None and self.land_seen and self.v.landed and self.v.pos_ok:
                    # 5 cm voxels: on the ground the raycast reads ~0.0x; training read the body-centre height (0.08)
                    self.rf_offset = 0.08 - r
                    self.get_logger().info(f'sim range calibrated on the ground: raw {r:.3f} m, offset {self.rf_offset:+.3f} m')
                r = r + (self.rf_offset or 0.0)
                return float(np.clip(r + self.rng.normal(0, 0.005 + 0.01 * r), 0.04, self.meta['rf_max']))
            except Exception:
                pass
        if self.range_source == 'distance_sensor' and self.range_m is not None:
            return self.range_m
        if self.v.pos is not None and self.mission.home is not None:          # 'altitude' / fallback
            return float(np.clip(-(self.v.pos[2] - self.mission.home[2]) + 0.08, 0.04, self.meta['rf_max']))
        return 0.08

    # ── control ──
    def _control(self):
        t = self._now()
        if t <= 0:
            return
        stream, cmds = self.mission.step(t, self.v, policy_complete=self.complete)
        for e in self.mission.events:
            self.get_logger().info(e)
        self.mission.events.clear()
        if self.mission.phase == 'fly' and stream:
            if self.mission.holding(t) or self.depth_img is None:
                self.vel_sp, self.yawspeed_sp = np.zeros(3), 0.0
            else:
                t0 = time.perf_counter()
                img = self.stack.push(self.depth_img)
                pro = O.proprio(self.v.vel, self.v.q, self._range(), self.prev_action, self.v_max)
                a = np.clip(self.policy(img, pro), -1.0, 1.0)
                self.prev_action = a.astype(np.float32)
                self.complete = self.n_act == 5 and a[4] > 0
                self.vel_sp, self.yawspeed_sp = O.action_to_setpoint(a[:4], self.v.q, self.v_max, self.yaw_max)
                self.timing.append(time.perf_counter() - t0)
        elif self.mission.phase == 'wait' and self.mission.why.startswith('ready') and self.depth_img is not None:
            # DRY RUN (arm:=false): what the policy would command on the FIRST step of a flight from here (fresh
            # memory each tick; feeding its own unexecuted commands back while the drone sits still is not
            # something it ever saw). Publishes nothing.
            t0 = time.perf_counter()
            self.policy.reset()
            self.stack.reset()
            img = self.stack.push(self.depth_img)
            pro = O.proprio(self.v.vel, self.v.q, self._range(), np.zeros(self.n_act), self.v_max)
            a = np.clip(self.policy(img, pro), -1.0, 1.0)
            self.timing.append(time.perf_counter() - t0)
            self.dry_action = a
            self.dry = O.action_to_setpoint(a[:4], self.v.q, self.v_max, self.yaw_max)
            self.vel_sp, self.yawspeed_sp = np.zeros(3), 0.0
        elif self.mission.phase != 'fly':
            self.vel_sp, self.yawspeed_sp = np.zeros(3), 0.0
        self._streaming = stream
        for c in cmds:
            if c == 'offboard':
                self._command(CMD_SET_MODE, 1.0, 6.0)
            elif c == 'arm':
                self._command(CMD_ARM, 1.0)
            elif c == 'land':
                self._command(CMD_LAND)
            elif c == 'disarm':
                self._command(CMD_ARM, 0.0)
        if self.mission.phase == 'done':
            self.get_logger().info(f'mission done: {self.mission.result} ({self.mission.why})')
            raise SystemExit(0 if self.mission.result == 'complete' else 1)

    def _stream(self):
        if not getattr(self, '_streaming', False):
            return
        us = self.sys_clock.now().nanoseconds // 1000
        o = self._OCM()
        o.timestamp, o.velocity = us, True
        self.pub_ocm.publish(o)
        s = self._TSP()
        s.timestamp = us
        s.position = [math.nan] * 3
        s.velocity = [float(c) for c in self.vel_sp]
        s.acceleration = [math.nan] * 3
        s.yaw, s.yawspeed = math.nan, float(self.yawspeed_sp)
        self.pub_sp.publish(s)

    def _command(self, cmd, p1=0.0, p2=0.0):
        c = self._CMD()
        c.timestamp = self.sys_clock.now().nanoseconds // 1000
        c.command, c.param1, c.param2 = cmd, float(p1), float(p2)
        c.target_system = c.target_component = c.source_system = c.source_component = 1
        c.from_external = True
        self.pub_cmd.publish(c)

    def _publish_status(self):
        tm = self.timing[-50:]
        st = {'phase': self.mission.phase, 'why': self.mission.why, 'result': self.mission.result,
              'vel_sp': [round(float(c), 3) for c in self.vel_sp], 'yawspeed': round(self.yawspeed_sp, 3),
              'prev_action': [round(float(c), 3) for c in self.prev_action],
              'policy_ms': round(1e3 * float(np.mean(tm)), 2) if tm else None,
              'depth_valid_frac': None if self.depth_img is None else round(float((self.depth_img > 0).mean()), 3),
              'range_m': round(self._range(), 3),
              'dry_run_would_command': None if getattr(self, 'dry', None) is None else
              {'action': [round(float(c), 3) for c in self.dry_action],
               'vel_ned': [round(float(c), 3) for c in self.dry[0]], 'yawspeed': round(self.dry[1], 3)}}
        if self.mission.phase == 'wait':
            self.get_logger().info(f"{self.mission.why} | {json.dumps(st)}", throttle_duration_sec=2.0)
        self.pub_status.publish(String(data=json.dumps(st)))


def main(argv=None):
    rclpy.init(args=list(argv) if argv is not None else sys.argv)
    node = PolicyNav()
    code = 1
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = int(e.code or 0)
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
