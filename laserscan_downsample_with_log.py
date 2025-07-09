#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import time # 导入 time 模块以计算耗时
import math # 导入 math 模块以进行弧度/角度转换

# 目标激光点的数量
TARGET_POINTS = 350

class LaserScanDownsampler(Node):
    """
    一个ROS2节点，用于订阅LaserScan消息，对其进行降采样，然后发布新的消息。
    它提供了详细的日志，以监控处理过程。
    """
    def __init__(self):
        super().__init__('laser_scan_downsampler')
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/orbbec_scan',
            self.listener_callback,
            30) # QoS profile depth
            
        self.publisher_ = self.create_publisher(
            LaserScan,
            '/scan_downsampled',
            30) # QoS profile depth
            
        self.get_logger().info("="*60)
        self.get_logger().info(" LaserScan Downsampler Node has been started.")
        self.get_logger().info(f" Subscribing to topic: '/orbbec_scan'")
        self.get_logger().info(f" Publishing to topic: '/scan_downsampled'")
        self.get_logger().info(f" Target number of points: {TARGET_POINTS}")
        self.get_logger().info("="*60)

    def listener_callback(self, msg: LaserScan):
        """
        接收到LaserScan消息时的回调函数。
        """
        # 记录处理开始时间
        start_time = time.perf_counter()

        original_num_points = len(msg.ranges)
        
        if original_num_points <= TARGET_POINTS:
            if original_num_points == 0:
                self.get_logger().warn("Received a scan with 0 points. Publishing as is.")
            else:
                self.get_logger().info(f"Received a scan with {original_num_points} points, which is <= target {TARGET_POINTS}. Passthrough.")
            
            self.publisher_.publish(msg)
            
            # 记录处理结束时间并计算耗时
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000

            # --- 打印格式化的日志摘要 (Passthrough) ---
            self.get_logger().info("--- Processing Summary (Passthrough) " + "-"*27)
            self.get_logger().info(f"  Points:          {original_num_points}")
            self.get_logger().info(f"  Angle Min:       {msg.angle_min:.4f} rad ({math.degrees(msg.angle_min):.2f}°)")
            self.get_logger().info(f"  Angle Max:       {msg.angle_max:.4f} rad ({math.degrees(msg.angle_max):.2f}°)")
            self.get_logger().info(f"  Angle Increment: {msg.angle_increment:.6f} rad ({math.degrees(msg.angle_increment):.4f}°)")
            self.get_logger().info(f"  Processing Time: {processing_time_ms:.2f} ms")
            self.get_logger().info("-" * 60 + "\n")
            return

        # --- 如果点数 > 目标点数，则执行降采样 ---
        self.get_logger().info(f"Received a scan with {original_num_points} points. Downsampling to ~{TARGET_POINTS} points.")
        
        new_scan_msg = LaserScan()
        new_scan_msg.header = msg.header
        new_scan_msg.angle_min = msg.angle_min
        new_scan_msg.angle_max = msg.angle_max
        new_scan_msg.time_increment = msg.time_increment
        new_scan_msg.scan_time = msg.scan_time
        new_scan_msg.range_min = msg.range_min
        new_scan_msg.range_max = msg.range_max

        indices_to_pick = np.linspace(0, original_num_points - 1, num=TARGET_POINTS, endpoint=True, dtype=int)

        # 从原始数据中拾取点
        original_ranges_np = np.array(msg.ranges)
        new_scan_msg.ranges = original_ranges_np[indices_to_pick].tolist()

        # 如果存在强度信息，也进行同样的处理
        if msg.intensities and len(msg.intensities) == original_num_points:
            original_intensities_np = np.array(msg.intensities)
            new_scan_msg.intensities = original_intensities_np[indices_to_pick].tolist()
        else:
            if msg.intensities:
                self.get_logger().warn(f"Intensities array length ({len(msg.intensities)}) does not match ranges array length ({original_num_points}). Publishing empty intensities.")
            new_scan_msg.intensities = []

        # 重新计算角度增量
        num_new_points = len(new_scan_msg.ranges)
        if num_new_points > 1:
            new_scan_msg.angle_increment = (new_scan_msg.angle_max - new_scan_msg.angle_min) / (num_new_points - 1)
        else:
            new_scan_msg.angle_increment = 0.0

        # 发布降采样后的消息
        self.publisher_.publish(new_scan_msg)

        # 记录处理结束时间并计算耗时
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        # --- 打印格式化的日志摘要 (Downsampled) ---
        self.get_logger().info("--- Processing Summary (Downsampled) " + "-"*27)
        # 输入详情
        self.get_logger().info("  Input Scan Details:")
        self.get_logger().info(f"    Points:          {original_num_points}")
        self.get_logger().info(f"    Angle Min:       {msg.angle_min:.4f} rad ({math.degrees(msg.angle_min):.2f}°)")
        self.get_logger().info(f"    Angle Max:       {msg.angle_max:.4f} rad ({math.degrees(msg.angle_max):.2f}°)")
        self.get_logger().info(f"    Angle Increment: {msg.angle_increment:.6f} rad ({math.degrees(msg.angle_increment):.4f}°)")
        # 输出详情
        self.get_logger().info("  Output Scan Details:")
        self.get_logger().info(f"    Points:          {num_new_points}")
        self.get_logger().info(f"    Angle Min:       {new_scan_msg.angle_min:.4f} rad ({math.degrees(new_scan_msg.angle_min):.2f}°)")
        self.get_logger().info(f"    Angle Max:       {new_scan_msg.angle_max:.4f} rad ({math.degrees(new_scan_msg.angle_max):.2f}°)")
        self.get_logger().info(f"    Angle Increment: {new_scan_msg.angle_increment:.6f} rad ({math.degrees(new_scan_msg.angle_increment):.4f}°)")
        # 总体耗时
        self.get_logger().info("-" * 50)
        self.get_logger().info(f"  Processing Time: {processing_time_ms:.2f} ms")
        self.get_logger().info("=" * 60 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanDownsampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('LaserScan Downsampler Node shutting down cleanly.')
    finally:
        # 销毁节点并关闭rclpy
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
