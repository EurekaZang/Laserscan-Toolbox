#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
from scipy.optimize import minimize
from collections import deque
import message_filters
from rclpy.time import Time

class DepthLidarCalibrator(Node):
    def __init__(self):
        super().__init__('depth_lidar_calibrator')
        
        # 参数
        self.declare_parameter('calibration_mode', 'linear')
        self.declare_parameter('sample_size', 100)
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('max_range', 15.0)
        self.declare_parameter('time_sync_tolerance', 0.5)  # 时间同步容差
        self.declare_parameter('debug_mode', True)
        # NEW: Add parameter for angle matching tolerance (e.g., in radians)
        self.declare_parameter('angle_tolerance', 0.005)  # ~0.3 degrees, adjust as needed
        
        self.calibration_mode = self.get_parameter('calibration_mode').get_parameter_value().string_value
        self.sample_size = self.get_parameter('sample_size').get_parameter_value().integer_value
        self.min_range = self.get_parameter('min_range').get_parameter_value().double_value
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value
        self.time_tolerance = self.get_parameter('time_sync_tolerance').get_parameter_value().double_value
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        self.angle_tolerance = self.get_parameter('angle_tolerance').get_parameter_value().double_value
        
        # 数据存储
        self.scan_pairs = deque(maxlen=self.sample_size)
        self.lock = threading.Lock()
        
        # 统计信息
        self.lidar_count = 0
        self.depth_count = 0
        self.paired_count = 0
        self.valid_pairs_count = 0
        
        # 当前参数
        self.current_params = [1.0, 0.0, 15.0]  # [scale, shift, multiplier]
        
        # 方法1: 使用message_filters进行时间同步
        self.use_sync = True
        
        if self.use_sync:
            # 使用message_filters同步话题
            self.lidar_sub = message_filters.Subscriber(self, LaserScan, '/scan')
            self.depth_sub = message_filters.Subscriber(self, LaserScan, '/scan_unscaled')
            
            # 时间同步器
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.lidar_sub, self.depth_sub], 
                queue_size=10, 
                slop=self.time_tolerance
            )
            self.ts.registerCallback(self.synchronized_callback)
        else:
            # 方法2: 使用缓存队列进行手动同步
            self.lidar_buffer = deque(maxlen=50)
            self.depth_buffer = deque(maxlen=50)
            
            self.lidar_sub = self.create_subscription(
                LaserScan, '/scan', self.lidar_callback, 50)
            self.depth_scan_sub = self.create_subscription(
                LaserScan, '/scan_unscaled', self.depth_scan_callback, 50)
        
        # 发布校准参数
        self.param_pub = self.create_publisher(Float64MultiArray, '/calibration_params', 10)
        
        # 定时器
        self.calibration_timer = self.create_timer(5.0, self.calibrate_parameters)
        self.stats_timer = self.create_timer(2.0, self.print_statistics)
        
        self.get_logger().info(f'Depth-LiDAR calibrator started in {self.calibration_mode} mode.')
        self.get_logger().info(f'Using {"message_filters sync" if self.use_sync else "manual sync"}')
    
    def synchronized_callback(self, lidar_msg, depth_msg):
        self.get_logger().info('Received synchronized pair!')  # NEW: Confirm reception
        """使用message_filters同步的回调"""
        self.lidar_count += 1
        self.depth_count += 1
        self.paired_count += 1
        
        if self.debug_mode:
            time_diff = abs(self.msg_to_sec(lidar_msg.header.stamp) - 
                           self.msg_to_sec(depth_msg.header.stamp))
            self.get_logger().debug(f'Synchronized pair received, time diff: {time_diff:.4f}s')
        
        self.store_scan_pair(lidar_msg, depth_msg)
    
    def lidar_callback(self, msg):
        self.get_logger().info('Received LiDAR message!')  # NEW: Confirm reception
        """LiDAR数据回调（手动同步模式）"""
        self.lidar_count += 1
        with self.lock:
            self.lidar_buffer.append(msg)
        self.try_manual_sync()
    
    def depth_scan_callback(self, msg):
        self.get_logger().info('Received Depth message!')  # NEW: Confirm reception
        """深度扫描数据回调（手动同步模式）"""
        self.depth_count += 1
        with self.lock:
            self.depth_buffer.append(msg)
        self.try_manual_sync()
    
    def try_manual_sync(self):
        """尝试手动同步"""
        with self.lock:
            if not self.lidar_buffer or not self.depth_buffer:
                return
            
            # 寻找最佳匹配
            best_pair = None
            best_time_diff = float('inf')
            
            for lidar_msg in self.lidar_buffer:
                for depth_msg in self.depth_buffer:
                    time_diff = abs(self.msg_to_sec(lidar_msg.header.stamp) - 
                                   self.msg_to_sec(depth_msg.header.stamp))
                    
                    if time_diff < best_time_diff and time_diff < self.time_tolerance:
                        best_time_diff = time_diff
                        best_pair = (lidar_msg, depth_msg)
            
            if best_pair:
                self.paired_count += 1
                if self.debug_mode:
                    self.get_logger().debug(f'Manual sync found pair, time diff: {best_time_diff:.4f}s')
                self.store_scan_pair(best_pair[0], best_pair[1])
    
    def msg_to_sec(self, stamp):
        """将ROS时间戳转换为秒"""
        return stamp.sec + stamp.nanosec * 1e-9
    
    def store_scan_pair(self, lidar_scan, depth_scan):
        """存储配对的扫描数据 - Updated to match based on angles instead of assuming equal lengths"""
        
        # Compute angles for LiDAR
        lidar_angles = np.arange(
            lidar_scan.angle_min,
            lidar_scan.angle_max + lidar_scan.angle_increment / 2,  # Add half increment to include endpoint
            lidar_scan.angle_increment
        )
        if len(lidar_angles) > len(lidar_scan.ranges):
            lidar_angles = lidar_angles[:len(lidar_scan.ranges)]  # Trim if needed (floating point issues)
        
        # Compute angles for depth
        depth_angles = np.arange(
            depth_scan.angle_min,
            depth_scan.angle_max + depth_scan.angle_increment / 2,
            depth_scan.angle_increment
        )
        if len(depth_angles) > len(depth_scan.ranges):
            depth_angles = depth_angles[:len(depth_scan.ranges)]
        
        # Extract ranges as numpy arrays for easier handling
        lidar_ranges = np.array(lidar_scan.ranges)
        depth_ranges = np.array(depth_scan.ranges)
        
        # Find matching pairs based on angle proximity
        valid_pairs = []
        valid_lidar = 0
        valid_depth = 0
        
        # For each depth angle, find closest LiDAR angle within tolerance
        for d_idx, d_angle in enumerate(depth_angles):
            d_range = depth_ranges[d_idx]
            
            # Count valid depth points independently
            if not np.isnan(d_range) and not np.isinf(d_range) and d_range > 0:
                valid_depth += 1
            
            # Find matching LiDAR angle
            angle_diffs = np.abs(lidar_angles - d_angle)
            closest_idx = np.argmin(angle_diffs)
            if angle_diffs[closest_idx] <= self.angle_tolerance:
                l_range = lidar_ranges[closest_idx]
                
                # Count valid LiDAR points (only for matched angles)
                if self.min_range <= l_range <= self.max_range:
                    valid_lidar += 1
                
                # If both valid, add pair
                if (self.min_range <= l_range <= self.max_range and
                    not np.isnan(d_range) and not np.isinf(d_range) and
                    d_range > 0):
                    valid_pairs.append((d_range, l_range))
        
        total_possible = min(len(lidar_angles), len(depth_angles))  # Rough estimate for logging
        
        if self.debug_mode:
            self.get_logger().debug(f'Scan analysis: LiDAR points={len(lidar_angles)}, '
                                  f'Depth points={len(depth_angles)}, '
                                  f'Valid LiDAR (matched)={valid_lidar}, Valid Depth={valid_depth}, '
                                  f'Valid Pairs={len(valid_pairs)}')
        
        if valid_pairs:
            with self.lock:
                self.scan_pairs.append(valid_pairs)
                self.valid_pairs_count += len(valid_pairs)
                
            self.get_logger().info(f'Stored {len(valid_pairs)} valid range pairs. '
                                 f'Total scan samples: {len(self.scan_pairs)}')
        else:
            self.get_logger().warning('No valid pairs found in this scan')
    
    def print_statistics(self):
        """打印统计信息"""
        if self.debug_mode:
            self.get_logger().info(f'Statistics: LiDAR={self.lidar_count}, '
                                 f'Depth={self.depth_count}, Paired={self.paired_count}, '
                                 f'Valid pairs={self.valid_pairs_count}, '
                                 f'Scan samples={len(self.scan_pairs)}')
    
    def calibrate_parameters(self):
        with self.lock:
            scan_samples = len(self.scan_pairs)
            
            if scan_samples < 5:
                self.get_logger().info(f'Insufficient scan samples for calibration. '
                                     f'Current: {scan_samples}, Need: 5+')
                return
            
            all_pairs = []
            for pairs in self.scan_pairs:
                all_pairs.extend(pairs)
            
            if len(all_pairs) < 50:
                self.get_logger().info(f'Insufficient data points for calibration. '
                                     f'Current: {len(all_pairs)}, Need: 50+')
                return
            
            depth_values = np.array([pair[0] for pair in all_pairs])
            lidar_values = np.array([pair[1] for pair in all_pairs])
        
        self.get_logger().info(f'Starting calibration with {len(all_pairs)} data points from {scan_samples} scans...')
        
        self.get_logger().info(f'Depth range: {depth_values.min():.3f} - {depth_values.max():.3f}')
        self.get_logger().info(f'LiDAR range: {lidar_values.min():.3f} - {lidar_values.max():.3f}')
        
        correlation = np.corrcoef(depth_values, lidar_values)[0,1]
        self.get_logger().info(f'Depth-LiDAR correlation: {correlation:.4f}')
        mean_depth = np.mean(depth_values)
        mean_lidar = np.mean(lidar_values)
        self.get_logger().info(f'Mean depth: {mean_depth:.3f}, Mean LiDAR: {mean_lidar:.3f}')
        
        try:
            if self.calibration_mode == 'linear':
                optimized_params = self.optimize_linear_params(depth_values, lidar_values)
            else:
                optimized_params = self.optimize_nonlinear_params(depth_values, lidar_values)
            
            # 计算校准前后的误差
            old_error = self.calculate_error(depth_values, lidar_values, self.current_params)
            new_error = self.calculate_error(depth_values, lidar_values, optimized_params)
            
            self.get_logger().info(f'Calibration completed!')
            self.get_logger().info(f'Old parameters: {self.current_params}')
            self.get_logger().info(f'New parameters: {optimized_params}')
            self.get_logger().info(f'Error: {old_error:.4f} -> {new_error:.4f} '
                                 f'({((old_error-new_error)/old_error*100):.1f}% improvement)')
            
            # 更新参数
            self.current_params = optimized_params
            
            # 发布新参数
            param_msg = Float64MultiArray()
            param_msg.data = optimized_params
            self.param_pub.publish(param_msg)
            
            # 重置统计
            self.valid_pairs_count = 0
            
        except Exception as e:
            self.get_logger().error(f'Calibration failed: {e}')
    
    def optimize_linear_params(self, depth_values, lidar_values):
        """优化线性参数 - Updated to use analytical linear regression for stability"""
        # Since the model is effectively linear (lidar = a * depth + b), use polyfit for optimal fit
        # Then map to [scale, shift, multiplier] by setting multiplier=1.0, scale=a, shift=b
        if len(depth_values) < 2:
            self.get_logger().warn('Too few points for linear fit')
            return self.current_params
        
        # Perform linear regression: lidar = scale * depth + shift (implicit multiplier=1)
        scale, shift = np.polyfit(depth_values, lidar_values, 1)  # Returns [slope, intercept]
        
        optimized_params = [float(scale), float(shift), 1.0]  # Fix multiplier to 1.0
        
        # Check if fit is valid
        if np.isnan(scale) or np.isnan(shift):
            self.get_logger().warn('Linear fit resulted in NaN values')
            return self.current_params
        
        self.get_logger().info(f'Linear regression results: scale={scale:.4f}, shift={shift:.4f}, multiplier=1.0')
        
        return optimized_params
    
    def optimize_nonlinear_params(self, depth_values, lidar_values):
        """优化非线性参数"""
        def objective(params):
            a, b, c = params
            try:
                predicted = a * np.power(depth_values, b) + c
                return np.sum((predicted - lidar_values) ** 2)
            except:
                return np.inf
        
        initial_guess = [1.0, 1.0, 0.0]
        bounds = [(0.001, 100.0), (0.1, 3.0), (-10.0, 10.0)]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return result.x.tolist()
        else:
            self.get_logger().warn(f'Nonlinear optimization failed: {result.message}')
            return [1.0, 1.0, 0.0]
    
    def calculate_error(self, depth_values, lidar_values, params):
        """计算预测误差"""
        if len(params) == 3:
            scale, shift, multiplier = params
            predicted = (depth_values * scale + shift) * multiplier
        else:
            a, b, c = params
            predicted = a * np.power(depth_values, b) + c
        
        return np.sqrt(np.mean((predicted - lidar_values) ** 2))

def main(args=None):
    rclpy.init(args=args)
    node = DepthLidarCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()