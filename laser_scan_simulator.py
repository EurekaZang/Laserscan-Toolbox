#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import random

class LaserScanSimulator(Node):
    def __init__(self):
        super().__init__('laser_scan_simulator')
        
        # 发布两个LaserScan话题
        self.scan_360_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.scan_75_pub = self.create_publisher(LaserScan, '/scan_by_orbbec', 10)
        
        # 定时器，以30Hz频率发布数据
        self.timer = self.create_timer(0.033, self.publish_scans)
        
        self.get_logger().info('LaserScan Simulator Started')
        
        # 计数器用于生成变化的数据
        self.counter = 0
    
    def publish_scans(self):
        """发布360度和75度的LaserScan数据"""
        current_time = self.get_clock().now().to_msg()
        
        # 发布360度scan
        scan_360 = self.create_360_scan(current_time)
        self.scan_360_pub.publish(scan_360)
        
        # 发布75度scan
        scan_75 = self.create_75_scan(current_time)
        self.scan_75_pub.publish(scan_75)
        
        self.counter += 1
        
        if self.counter % 50 == 0:  # 每5秒打印一次日志
            self.get_logger().info(f'Published scan data #{self.counter}')
    
    def create_360_scan(self, timestamp):
        """创建360度LaserScan数据"""
        scan = LaserScan()
        
        # Header
        scan.header.stamp = timestamp
        scan.header.frame_id = "laser"
        
        # Scan parameters (基于你提供的/scan数据)
        scan.angle_min = -3.1415927410125732
        scan.angle_max = 3.1415927410125732
        scan.angle_increment = 0.0037691574543714523
        scan.time_increment = 5.003871046938002e-05
        scan.scan_time = 0.08341452479362488
        scan.range_min = 0.0
        scan.range_max = 10.0
        
        # 计算点数
        num_points = int((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
        
        # 生成ranges数据
        ranges = []
        intensities = []
        
        for i in range(num_points):
            angle = scan.angle_min + i * scan.angle_increment
            
            # 根据角度生成不同的距离模式
            if -0.7 <= angle <= 0.7:  # 前方区域，模拟墙壁
                base_distance = 2.0 + 0.5 * math.sin(angle * 3)
                noise = random.uniform(-0.1, 0.1)
                distance = max(scan.range_min, base_distance + noise)
            elif -1.5 <= angle <= -0.7 or 0.7 <= angle <= 1.5:  # 侧方区域
                distance = random.uniform(1.0, 5.0)
            else:  # 后方区域，大部分为inf
                if random.random() < 0.3:  # 30%概率有有效数据
                    distance = random.uniform(2.0, 10.0)
                else:
                    distance = float('inf')
            
            ranges.append(distance)
            intensities.append(0.0)  # 强度都设为0
        
        scan.ranges = ranges
        scan.intensities = intensities
        
        return scan
    
    def create_75_scan(self, timestamp):
        """创建75度LaserScan数据（基于你提供的实际数据）"""
        scan = LaserScan()
        
        # Header
        scan.header.stamp = timestamp
        scan.header.frame_id = "camera_depth_frame"
        
        # Scan parameters (基于你提供的/scan_by_orbbec数据)
        scan.angle_min = -0.6334390640258789
        scan.angle_max = 0.6413999795913696
        scan.time_increment = 0.0
        scan.scan_time = 0.0010000000474974513
        scan.range_min = 0.20000000298023224
        scan.range_max = 5.0
        scan.angle_increment = 0.0019950533751398325
        
        # 计算点数
        num_points = int((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
        
        # 生成ranges数据（模拟深度相机检测到的物体）
        ranges = []
        
        for i in range(num_points):
            angle = scan.angle_min + i * scan.angle_increment
            
            # 前36个点设为NaN（模拟检测不到）
            if i < 36:
                ranges.append(float('nan'))
            # 中间一段检测到远处物体（约2米）
            elif i < 48:
                base_distance = 2.0 + 0.1 * math.sin(angle * 10)
                noise = random.uniform(-0.05, 0.05)
                ranges.append(base_distance + noise)
            # 一段NaN
            elif i < 60:
                ranges.append(float('nan'))
            # 检测到近处物体（约0.65米）
            else:
                base_distance = 0.65 + 0.01 * math.sin(angle * 20 + self.counter * 0.1)
                noise = random.uniform(-0.005, 0.005)
                distance = max(scan.range_min, base_distance + noise)
                ranges.append(distance)
        
        scan.ranges = ranges
        scan.intensities = []  # 空强度数组
        
        return scan

def main(args=None):
    rclpy.init(args=args)
    
    simulator = LaserScanSimulator()
    
    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        pass
    finally:
        simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
