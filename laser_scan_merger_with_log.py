#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
import message_filters
import numpy as np
import math
import time

import tf2_ros
from tf_transformations import euler_from_quaternion

class LaserScanMerger(Node):
    def __init__(self):
        super().__init__('laser_scan_merger_detailed')

        # --- TF Buffer and Listener ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- QoS Profile ---
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        # --- 独立监控订阅者 (用于调试) ---
        self.scan_360_monitor_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_360_monitor_callback, sensor_qos_profile
        )
        self.scan_75_monitor_sub = self.create_subscription(
            LaserScan, '/scan_downsampled', self.scan_75_monitor_callback, sensor_qos_profile
        )

        self.scan_360_sub = message_filters.Subscriber(
            self, LaserScan, '/scan', qos_profile=sensor_qos_profile
        )
        self.scan_75_sub = message_filters.Subscriber(
            self, LaserScan, '/scan_downsampled', qos_profile=sensor_qos_profile
        )

        self.ats = message_filters.ApproximateTimeSynchronizer(
            [self.scan_360_sub, self.scan_75_sub],
            queue_size=15,
            slop=0.1
        )
        self.ats.registerCallback(self.synchronized_callback)
        
        self.merged_scan_pub = self.create_publisher(
            LaserScan, '/merged_scan', qos_profile=sensor_qos_profile
        )
        
        self.callback_count = 0
        self.get_logger().info("=" * 60)
        self.get_logger().info(" Detailed LaserScan Merger Node has been started.")
        self.get_logger().info(" -> Monitoring '/scan' and '/scan_downsampled' for synchronization.")
        self.get_logger().info(f" -> Synchronization Slop: {self.ats.slop} seconds.")
        self.get_logger().info(" -> Publishing merged scan to '/merged_scan'.")
        self.get_logger().info("=" * 60)

    def scan_360_monitor_callback(self, msg):
        self.get_logger().info("Received a message from '/scan'", throttle_duration_sec=5)

    def scan_75_monitor_callback(self, msg):
        self.get_logger().info("Received a message from '/scan_downsampled'", throttle_duration_sec=5)

    def validate_scan_data(self, scan_msg, scan_name):
        """检查传入的LaserScan数据是否有效"""
        if not scan_msg:
            self.get_logger().error(f"  [Validation] {scan_name} message is None!")
            return False
        if not scan_msg.ranges:
            self.get_logger().warn(f"  [Validation] {scan_name} has an empty 'ranges' array. Skipping.")
            return False
        if scan_msg.angle_max <= scan_msg.angle_min:
            self.get_logger().warn(f"  [Validation] {scan_name} has invalid angle limits. Skipping.")
            return False
        return True

    def synchronized_callback(self, scan_360_msg, scan_75_msg):
        # 回调主函数
        start_time = time.perf_counter()
        self.callback_count += 1
        
        self.get_logger().info(f"\n{'='*20} [Executing Callback #{self.callback_count}] {'='*20}")
        
        self.get_logger().info("Step 1: Validating incoming scan data...")
        if not self.validate_scan_data(scan_360_msg, "'/scan' (360)"):
            self.get_logger().info("=" * 60)
            return
        if not self.validate_scan_data(scan_75_msg, "'/scan_downsampled' (75)"):
            self.get_logger().info("=" * 60)
            return
        self.get_logger().info("  -> Validation PASSED for both scans.")
        
        self.get_logger().info("Step 2: Initializing merged scan from the base scan ('/scan').")
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
        
        merged_ranges_np = np.array(scan_360_msg.ranges, dtype=np.float32)
        if scan_360_msg.intensities:
            merged_intensities_np = np.array(scan_360_msg.intensities, dtype=np.float32)
        else:
            merged_intensities_np = np.zeros_like(merged_ranges_np)
        self.get_logger().info(f"  -> Base scan initialized with {len(merged_ranges_np)} points.")

        self.get_logger().info("Step 3: Replacing data using '/scan_downsampled'.")
        merged_ranges_np, merged_intensities_np, replaced_count = self.replace_scan_data_vectorized(
            merged_ranges_np,
            merged_intensities_np,
            scan_75_msg,
            scan_360_msg
        )
        
        self.get_logger().info("Step 4: Publishing the final merged scan.")
        merged_scan.ranges = merged_ranges_np.tolist()
        merged_scan.intensities = merged_intensities_np.tolist()
        self.merged_scan_pub.publish(merged_scan)

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        self.get_logger().info("--- Final Summary ---")
        self.get_logger().info(f"  Points Replaced: {replaced_count}")
        self.get_logger().info(f"  Total Processing Time: {processing_time_ms:.2f} ms")
        self.get_logger().info(f"{'='*25} [Callback Finished] {'='*25}\n")


    def replace_scan_data_vectorized(self, merged_ranges_np, merged_intensities_np, scan_75, scan_360):
        source_frame = scan_75.header.frame_id
        target_frame = scan_360.header.frame_id
        
        self.get_logger().info(f"  [3a] Looking up transform from '{source_frame}' to '{target_frame}'...")
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=Duration(seconds=0.05)
            )
            translation = trans.transform.translation
            rotation_q = trans.transform.rotation
            _, _, yaw = euler_from_quaternion([rotation_q.x, rotation_q.y, rotation_q.z, rotation_q.w])
            self.get_logger().info(f"    -> Transform found! Translation: [x:{translation.x:.2f}, y:{translation.y:.2f}], Yaw: {math.degrees(yaw):.2f}°")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"    -> TF transform not available: {e}", throttle_duration_sec=5.0)
            return merged_ranges_np, merged_intensities_np, 0

        self.get_logger().info("  [3b] Filtering and preparing source scan points...")
        ranges_75 = np.array(scan_75.ranges, dtype=np.float32)
        valid_indices = np.isfinite(ranges_75) & (ranges_75 >= scan_75.range_min) & (ranges_75 <= scan_75.range_max)
        
        num_valid_points = np.sum(valid_indices)
        if num_valid_points == 0:
            self.get_logger().info("    -> No valid points found in source scan after initial filtering. Aborting replacement.")
            return merged_ranges_np, merged_intensities_np, 0
        self.get_logger().info(f"    -> Found {num_valid_points} valid points in source scan.")
            
        self.get_logger().info("  [3c] Transforming points to the target frame...")
        valid_ranges = ranges_75[valid_indices]
        angles_75 = scan_75.angle_min + np.arange(len(scan_75.ranges)) * scan_75.angle_increment
        valid_angles = angles_75[valid_indices]
        
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        x_source = valid_ranges * np.cos(valid_angles)
        y_source = valid_ranges * np.sin(valid_angles)
        x_target = x_source * cos_yaw - y_source * sin_yaw + translation.x
        y_target = x_source * sin_yaw + y_source * cos_yaw + translation.y

        self.get_logger().info("  [3d] Calculating new ranges/angles and applying filters...")
        new_ranges = np.sqrt(x_target**2 + y_target**2)
        new_angles = np.arctan2(y_target, x_target)

        range_mask = (new_ranges >= scan_360.range_min) & (new_ranges <= scan_360.range_max)
        num_in_range = np.sum(range_mask)
        if num_in_range == 0:
            self.get_logger().info("    -> All transformed points are outside the target scan's range limits. Aborting.")
            return merged_ranges_np, merged_intensities_np, 0
        self.get_logger().info(f"    -> {num_in_range} transformed points are within target's range limits.")
        
        final_new_ranges = new_ranges[range_mask]
        final_new_angles = new_angles[range_mask]
        
        self.get_logger().info("  [3e] Calculating target indices and updating arrays...")
        target_indices = np.round((final_new_angles - scan_360.angle_min) / scan_360.angle_increment).astype(int)
        
        valid_target_mask = (target_indices >= 0) & (target_indices < len(merged_ranges_np))
        num_to_replace = np.sum(valid_target_mask)
        if num_to_replace == 0:
            self.get_logger().info("    -> No points map to valid indices in the target scan. Aborting.")
            return merged_ranges_np, merged_intensities_np, 0

        final_indices = target_indices[valid_target_mask]
        final_ranges_to_update = final_new_ranges[valid_target_mask]
        
        merged_ranges_np[final_indices] = final_ranges_to_update
        self.get_logger().info(f"    -> Successfully updated {num_to_replace} range points.")
        
        if scan_75.intensities and len(scan_75.intensities) == len(ranges_75):
            intensities_75 = np.array(scan_75.intensities, dtype=np.float32)
            valid_intensities = intensities_75[valid_indices]
            final_intensities_to_update = valid_intensities[range_mask][valid_target_mask]
            merged_intensities_np[final_indices] = final_intensities_to_update
            self.get_logger().info(f"    -> Successfully updated {num_to_replace} intensity points.")

        return merged_ranges_np, merged_intensities_np, num_to_replace


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Detailed LaserScan Merger Node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

