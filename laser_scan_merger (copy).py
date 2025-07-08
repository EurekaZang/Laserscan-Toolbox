#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
import message_filters
import numpy as np
import math

# 导入TF2相关的库
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

class LaserScanMerger(Node):
    def __init__(self):
        super().__init__('laser_scan_merger')
        
        # 初始化TF2的buffer和listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.scan_360_sub = message_filters.Subscriber(self, LaserScan, '/scan')
        self.scan_75_sub = message_filters.Subscriber(self, LaserScan, '/scan_downsampled')
        
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [self.scan_360_sub, self.scan_75_sub], 
            queue_size=10,
            slop=0.05
        )
        self.ats.registerCallback(self.synchronized_callback)
        
        self.merged_scan_pub = self.create_publisher(LaserScan, '/merged_scan', 30)
        
        self.get_logger().info('LaserScan Merger Node Started. Using ApproximateTimeSynchronizer and TF2.')

    def synchronized_callback(self, scan_360_msg, scan_75_msg):
        time_diff = abs(
            rclpy.time.Time.from_msg(scan_360_msg.header.stamp).nanoseconds - 
            rclpy.time.Time.from_msg(scan_75_msg.header.stamp).nanoseconds
        ) / 1e9
        self.get_logger().info(f'Received synchronized scans (time diff: {time_diff:.4f}s). Merging...')
        
        merged_scan = LaserScan()
        merged_scan.header.stamp = self.get_clock().now().to_msg()
        merged_scan.header.frame_id = scan_360_msg.header.frame_id # 目标坐标系是360度雷达的
        
        merged_scan.angle_min = scan_360_msg.angle_min
        merged_scan.angle_max = scan_360_msg.angle_max
        merged_scan.angle_increment = scan_360_msg.angle_increment
        merged_scan.time_increment = scan_360_msg.time_increment
        merged_scan.scan_time = scan_360_msg.scan_time
        merged_scan.range_min = min(scan_360_msg.range_min, scan_75_msg.range_min)
        merged_scan.range_max = max(scan_360_msg.range_max, scan_75_msg.range_max)
        
        merged_ranges = list(scan_360_msg.ranges)
        merged_intensities = [0.0] * len(merged_ranges)
        if scan_360_msg.intensities:
            merged_intensities = list(scan_360_msg.intensities)

        # [关键修改] 调用新的、使用TF的替换函数
        replaced_count = self.replace_scan_data_with_tf(
            merged_ranges, 
            merged_intensities,
            scan_75_msg, 
            scan_360_msg
        )
        
        merged_scan.ranges = merged_ranges
        merged_scan.intensities = merged_intensities
        
        self.merged_scan_pub.publish(merged_scan)
        self.get_logger().info(f'Published merged scan. Replaced {replaced_count} points.')

    def replace_scan_data_with_tf(self, merged_ranges, merged_intensities, scan_75, scan_360):
        num_points_75 = len(scan_75.ranges)
        if num_points_75 == 0:
            return 0
        
        source_frame = scan_75.header.frame_id
        target_frame = scan_360.header.frame_id
        
        replaced_count = 0
        
        try:
            self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time(), timeout=Duration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF transform from {source_frame} to {target_frame} not available: {e}')
            return 0 # 如果TF不可用，直接返回，不再继续

        for i in range(num_points_75):
            range_val = scan_75.ranges[i]
            
            if math.isnan(range_val) or math.isinf(range_val):
                continue
                
            angle = scan_75.angle_min + i * scan_75.angle_increment
            x = range_val * math.cos(angle)
            y = range_val * math.sin(angle)
            
            point_in_source_frame = PointStamped()
            # 注意：header仍然使用原始的，因为它包含了正确的frame_id
            point_in_source_frame.header = scan_75.header
            
            # [主要修正] 在进行变换前，将时间戳清零，以请求最新的可用变换
            # 这样做可以解决ExtrapolationException
            point_in_source_frame.header.stamp = rclpy.time.Time().to_msg()
            
            point_in_source_frame.point.x = x
            point_in_source_frame.point.y = y
            point_in_source_frame.point.z = 0.0

            try:
                # 现在这个变换将使用最新的可用TF，而不是scan_75的旧时间戳
                transformed_point = self.tf_buffer.transform(
                    point_in_source_frame,
                    target_frame
                )
                
                x_prime = transformed_point.point.x
                y_prime = transformed_point.point.y
                
                new_angle = math.atan2(y_prime, x_prime)
                new_range = math.sqrt(x_prime**2 + y_prime**2)
                
                if not (scan_360.range_min <= new_range <= scan_360.range_max):
                    continue

                target_index = int(round((new_angle - scan_360.angle_min) / scan_360.angle_increment))
                
                if 0 <= target_index < len(merged_ranges):
                    merged_ranges[target_index] = new_range
                    if scan_75.intensities and i < len(scan_75.intensities):
                        merged_intensities[target_index] = scan_75.intensities[i]
                    replaced_count += 1

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                # 理论上不应该再进入这里，但作为保护措施保留
                self.get_logger().warn(f'Could not transform point from {source_frame} to {target_frame}: {e}', throttle_duration_sec=5)

        return replaced_count


# main函数保持不变
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

