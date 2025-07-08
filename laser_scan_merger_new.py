#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
import message_filters
import numpy as np
import math

# 导入TF2相关的库
import tf2_ros
from tf_transformations import euler_from_quaternion

class LaserScanMerger(Node):
    def __init__(self):
        super().__init__('laser_scan_merger_optimized')
        
        # 初始化TF2的buffer和listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 定义传感器数据的QoS配置
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5 # 稍微给一点buffer
        )
        
        self.scan_360_sub = message_filters.Subscriber(
            self, LaserScan, '/scan', qos_profile=sensor_qos_profile
        )
        self.scan_75_sub = message_filters.Subscriber(
            self, LaserScan, '/scan_downsampled', qos_profile=sensor_qos_profile
        )
        
        # 保持slop不变，0.05s是合理的默认值
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [self.scan_360_sub, self.scan_75_sub], 
            queue_size=10,
            slop=0.1
        )
        self.ats.registerCallback(self.synchronized_callback)
        
        self.merged_scan_pub = self.create_publisher(
            LaserScan, '/merged_scan', qos_profile=sensor_qos_profile
        )
        
        self.get_logger().info('Optimized LaserScan Merger Node Started.')

    def synchronized_callback(self, scan_360_msg, scan_75_msg):
        # 测量回调函数的起始时间，用于性能分析
        start_time = self.get_clock().now()

        # 使用主传感器的时间戳
        merged_scan = LaserScan()
        merged_scan.header.stamp = scan_360_msg.header.stamp
        merged_scan.header.frame_id = scan_360_msg.header.frame_id
        
        merged_scan.angle_min = scan_360_msg.angle_min
        merged_scan.angle_max = scan_360_msg.angle_max
        merged_scan.angle_increment = scan_360_msg.angle_increment
        merged_scan.time_increment = scan_360_msg.time_increment
        merged_scan.scan_time = scan_360_msg.scan_time
        merged_scan.range_min = min(scan_360_msg.range_min, scan_75_msg.range_min)
        merged_scan.range_max = max(scan_360_msg.range_max, scan_75_msg.range_max)
        
        merged_ranges_np = np.array(scan_360_msg.ranges)
        if scan_360_msg.intensities:
            merged_intensities_np = np.array(scan_360_msg.intensities)
        else:
            merged_intensities_np = np.zeros_like(merged_ranges_np)

        merged_ranges_np, merged_intensities_np, replaced_count = self.replace_scan_data_vectorized(
            merged_ranges_np, 
            merged_intensities_np,
            scan_75_msg, 
            scan_360_msg
        )
        
        merged_scan.ranges = merged_ranges_np.tolist()
        merged_scan.intensities = merged_intensities_np.tolist()
        
        self.merged_scan_pub.publish(merged_scan)

        end_time = self.get_clock().now()
        processing_time = (end_time - start_time).nanoseconds / 1e6 # 毫秒
        self.get_logger().info(
            f'Published merged scan. Replaced {replaced_count} points. '
            f'Processing time: {processing_time:.2f} ms.'
        )

    def replace_scan_data_vectorized(self, merged_ranges_np, merged_intensities_np, scan_75, scan_360):
        source_frame = scan_75.header.frame_id
        target_frame = scan_360.header.frame_id
        
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, scan_75.header.stamp, timeout=Duration(seconds=0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF transform not available from {source_frame} to {target_frame}: {e}', throttle_duration_sec=5.0)
            return merged_ranges_np, merged_intensities_np, 0

        translation = trans.transform.translation
        rotation_q = trans.transform.rotation
        
        _, _, yaw = euler_from_quaternion([rotation_q.x, rotation_q.y, rotation_q.z, rotation_q.w])
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        ranges_75 = np.array(scan_75.ranges)
        valid_indices = np.isfinite(ranges_75) & (ranges_75 >= scan_75.range_min) & (ranges_75 <= scan_75.range_max)
        
        if not np.any(valid_indices):
            return merged_ranges_np, merged_intensities_np, 0
            
        valid_ranges = ranges_75[valid_indices]
        num_points_75 = len(scan_75.ranges)
        angles_75 = scan_75.angle_min + np.arange(num_points_75) * scan_75.angle_increment
        valid_angles = angles_75[valid_indices]
        
        x_source = valid_ranges * np.cos(valid_angles)
        y_source = valid_ranges * np.sin(valid_angles)

        x_target = x_source * cos_yaw - y_source * sin_yaw + translation.x
        y_target = x_source * sin_yaw + y_source * cos_yaw + translation.y

        new_ranges = np.sqrt(x_target**2 + y_target**2)
        new_angles = np.arctan2(y_target, x_target)

        range_mask = (new_ranges >= scan_360.range_min) & (new_ranges <= scan_360.range_max)
        if not np.any(range_mask):
            return merged_ranges_np, merged_intensities_np, 0

        final_new_ranges = new_ranges[range_mask]
        final_new_angles = new_angles[range_mask]
        
        target_indices = np.round((final_new_angles - scan_360.angle_min) / scan_360.angle_increment).astype(int)

        valid_target_mask = (target_indices >= 0) & (target_indices < len(merged_ranges_np))
        final_indices = target_indices[valid_target_mask]
        final_ranges_to_update = final_new_ranges[valid_target_mask]

        merged_ranges_np[final_indices] = final_ranges_to_update
        replaced_count = len(final_indices)
        
        if scan_75.intensities:
            intensities_75 = np.array(scan_75.intensities)
            valid_intensities = intensities_75[valid_indices]
            final_intensities_to_update = valid_intensities[range_mask][valid_target_mask]
            merged_intensities_np[final_indices] = final_intensities_to_update

        return merged_ranges_np, merged_intensities_np, replaced_count


def main(args=None):
    rclpy.init(args=args)
    laser_scan_merger = LaserScanMerger()
    try:
        rclpy.spin(laser_scan_merger)
    except KeyboardInterrupt:
        pass
    finally:
        laser_scan_merger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

