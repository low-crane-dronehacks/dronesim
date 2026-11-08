#!/usr/bin/env python3
"""Gazebo ground-truth pose -> PX4 external vision (/fmu/in/vehicle_visual_odometry) + /estimator/healthy.

Reads the model pose from `gz topic -e /world/<w>/dynamic_pose/info --json-output` (world ENU / body FLU; the x500 has no odometry
publisher and ros_gz drops the entity names of Pose_V), converts to NED/FRD, adds small
gaussian noise (position sigma_p m, yaw/tilt sigma_a rad) and publishes at `rate` Hz. /estimator/healthy is true once PX4's
local position is valid (xy_valid, z_valid, heading_good_for_control)."""
import json
import math
import subprocess
import sys
import threading

import numpy as np
import rclpy
from px4_msgs.msg import VehicleLocalPosition, VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 history=HistoryPolicy.KEEP_LAST, depth=5)
T_NED_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
T_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def quat_to_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, R[i, i] - R[j, j] - R[k, k] + 1)) * 2
    q = np.zeros(4)
    q[1 + i] = 0.25 * s
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def rotvec_mat(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class EvBridge(Node):
    def __init__(self):
        super().__init__('ev_bridge')
        dp = lambda n, v: self.declare_parameter(n, v).value
        self.model = dp('model', 'x500_depth_0')
        self.gz_topic = dp('gz_topic', '/world/dronehacks/dynamic_pose/info')
        self.sigma_p, self.sigma_a, self.rate = dp('sigma_p', 0.02), dp('sigma_a', 0.01), dp('rate', 30.0)
        self.rng = np.random.default_rng(1)
        self.pose = None
        self.healthy = False
        self.pub = self.create_publisher(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', QOS)
        self.pub_h = self.create_publisher(Bool, '/estimator/healthy', 10)
        threading.Thread(target=self._read_gz, daemon=True).start()
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._on_lpos, QOS)
        self.create_timer(1.0 / self.rate, self._tick)
        self.create_timer(0.2, lambda: self.pub_h.publish(Bool(data=self.healthy)))

    def _read_gz(self):
        p = subprocess.Popen(['gz', 'topic', '-e', '-t', self.gz_topic, '--json-output'], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            try:
                for e in json.loads(line).get('pose', []):
                    if e.get('name') == self.model:
                        t, o = e.get('position', {}), e.get('orientation', {})
                        self.pose = (np.array([t.get('x', 0.0), t.get('y', 0.0), t.get('z', 0.0)]),
                                     quat_to_mat(o.get('w', 0.0), o.get('x', 0.0), o.get('y', 0.0), o.get('z', 0.0)))
            except ValueError:
                pass

    def _on_lpos(self, m):
        self.healthy = bool(m.xy_valid and m.z_valid and m.heading_good_for_control)

    def _tick(self):
        if self.pose is None:
            return
        p, R = self.pose
        p_ned = T_NED_ENU @ p + self.rng.normal(0, self.sigma_p, 3)
        R_ned_frd = rotvec_mat(self.rng.normal(0, self.sigma_a, 3)) @ T_NED_ENU @ R @ T_FLU_FRD
        q = mat_to_quat(R_ned_frd)
        msg = VehicleOdometry()
        msg.timestamp = msg.timestamp_sample = int(self.get_clock().now().nanoseconds / 1000)   # system clock, like PX4's examples
        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.position = [float(x) for x in p_ned]
        msg.q = [float(x) for x in q]
        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_UNKNOWN if hasattr(VehicleOdometry, 'VELOCITY_FRAME_UNKNOWN') else 0
        nan = float('nan')
        msg.velocity = [nan, nan, nan]
        msg.angular_velocity = [nan, nan, nan]
        msg.position_variance = [self.sigma_p ** 2] * 3
        msg.orientation_variance = [self.sigma_a ** 2] * 3
        msg.velocity_variance = [nan, nan, nan]
        self.pub.publish(msg)


def main():
    rclpy.init(args=sys.argv)
    rclpy.spin(EvBridge())


if __name__ == '__main__':
    main()
