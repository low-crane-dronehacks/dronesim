#!/usr/bin/env python3
"""/camera/depth/image_raw_gz (gz frame_id) -> /camera/depth/image_raw with frame_id camera_optical_frame (the static TF's child)."""
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Relay(Node):
    def __init__(self):
        super().__init__('depth_relay')
        self.pub = self.create_publisher(Image, '/camera/depth/image_raw', qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw_gz', self._cb, qos_profile_sensor_data)

    def _cb(self, m):
        m.header.frame_id = 'camera_optical_frame'
        self.pub.publish(m)


if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    rclpy.spin(Relay())
