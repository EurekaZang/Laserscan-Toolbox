#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

# 目标激光点的数量
TARGET_POINTS = 350

class LaserScanDownsampler(Node):
    def __init__(self):
        super().__init__('laser_scan_downsampler')
        self.subscription = self.create_subscription(
            LaserScan,
            '/orbbec_scan',
            self.listener_callback,
            30)
        self.publisher_ = self.create_publisher(
            LaserScan,
            '/scan_downsampled',
            30) # QoS profile depth
        self.get_logger().info(f"LaserScan Downsampler Node Started. Subscribing to /scan, Publishing to /scan_downsampled with approx {TARGET_POINTS} points.")

    def listener_callback(self, msg: LaserScan):
        self.get_logger().info(f"Received scan with {len(msg.ranges)} points.")

        original_num_points = len(msg.ranges)

        if original_num_points == 0:
            self.get_logger().warn("Received scan with 0 points. Publishing as is.")
            self.publisher_.publish(msg)
            return

        new_scan_msg = LaserScan()

        new_scan_msg.header = msg.header
        new_scan_msg.angle_min = msg.angle_min
        new_scan_msg.angle_max = msg.angle_max
        new_scan_msg.time_increment = msg.time_increment
        new_scan_msg.scan_time = msg.scan_time
        new_scan_msg.range_min = msg.range_min
        new_scan_msg.range_max = msg.range_max

        if original_num_points > 1:
            indices_to_pick = np.linspace(0, original_num_points - 1, num=TARGET_POINTS, endpoint=True, dtype=int)
        elif original_num_points == 1:
            indices_to_pick = np.zeros(TARGET_POINTS, dtype=int)
            indices_to_pick = np.array([], dtype=int)


        original_ranges_np = np.array(msg.ranges)
        new_scan_msg.ranges = original_ranges_np[indices_to_pick].tolist()

        if msg.intensities:
            original_intensities_np = np.array(msg.intensities)
            if len(original_intensities_np) == original_num_points:
                new_scan_msg.intensities = original_intensities_np[indices_to_pick].tolist()
            else:
                self.get_logger().warn(f"Intensities array length ({len(original_intensities_np)}) does not match ranges array length ({original_num_points}). Publishing empty intensities.")
                new_scan_msg.intensities = []
        else:
            new_scan_msg.intensities = []

        num_new_points = len(new_scan_msg.ranges)
        if num_new_points > 1:
            new_scan_msg.angle_increment = (new_scan_msg.angle_max - new_scan_msg.angle_min) / (num_new_points - 1)
        elif num_new_points == 1:
            new_scan_msg.angle_increment = 0.0
        else:
            new_scan_msg.angle_increment = 0.0


        self.publisher_.publish(new_scan_msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaserScanDownsampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('LaserScan Downsampler Node shutting down cleanly.')
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
