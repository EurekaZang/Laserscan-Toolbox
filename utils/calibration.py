#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import message_filters
import numpy as np
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from cv_bridge import CvBridge
from scipy.optimize import minimize
import threading

class DepthScaleCalibrator(Node):
    def __init__(self):
        super().__init__('depth_scale_calibrator')
        
        # --- Parameters ---
        # The topics we listen to
        self.declare_parameter('raw_depth_topic', '/fake_camera/depth/raw_unscaled')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('real_scan_topic', '/scan')
        # Initial guess for optimization. Based on your original values.
        # A = scale * 19, B = shift * 19
        self.declare_parameter('initial_guess_A', 0.016655 * 19)
        self.declare_parameter('initial_guess_B', 0.000132 * 19)

        raw_depth_topic = self.get_parameter('raw_depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        real_scan_topic = self.get_parameter('real_scan_topic').get_parameter_value().string_value
        self.initial_A = self.get_parameter('initial_guess_A').get_parameter_value().double_value
        self.initial_B = self.get_parameter('initial_guess_B').get_parameter_value().double_value
        
        self.bridge = CvBridge()
        self.camera_info = None
        self.optimization_result = None
        self.lock = threading.Lock()

        # --- Subscribers ---
        # We need CameraInfo only once
        self.info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.info_callback, 1)

        # Synchronize the raw depth image and the real laser scan
        self.depth_sub = message_filters.Subscriber(self, Image, raw_depth_topic)
        self.scan_sub = message_filters.Subscriber(self, LaserScan, real_scan_topic)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.scan_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Depth Calibrator node started.")
        self.get_logger().info("Waiting for CameraInfo...")

    def info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info(f"Received CameraInfo for a {msg.width}x{msg.height} image.")
            # We don't need it anymore
            self.destroy_subscription(self.info_sub)

    def sync_callback(self, depth_msg, scan_msg):
        if not self.camera_info:
            self.get_logger().warn("No CameraInfo received yet, skipping callback.", throttle_duration_sec=5)
            return

        if self.optimization_result is not None:
            self.get_logger().info("Optimization already completed.", throttle_duration_sec=30)
            return

        # Use a lock to prevent running multiple optimizations at once
        if self.lock.acquire(blocking=False):
            try:
                self.get_logger().info("Received synchronized data, starting optimization...")
                
                # The objective function to minimize. It captures the necessary data.
                def objective_function(params):
                    return self.calculate_error(params, depth_msg, scan_msg)

                # Run the optimization
                result = minimize(
                    objective_function,
                    [self.initial_A, self.initial_B], # Initial guess
                    method='Nelder-Mead', # A robust method that doesn't need gradients
                    options={'xatol': 1e-4, 'disp': False}
                )
                
                self.optimization_result = result
                self.get_logger().info("--- CALIBRATION COMPLETE ---")
                self.get_logger().info(f"Optimization successful: {result.success}")
                self.get_logger().info(f"Final Error: {result.fun:.4f}")
                self.get_logger().info(f"Optimal parameter A (scale): {result.x[0]:.6f}")
                self.get_logger().info(f"Optimal parameter B (shift): {result.x[1]:.6f}")
                self.get_logger().info("-----------------------------")
                self.get_logger().info("You can now stop this node (Ctrl+C).")
                
            finally:
                self.lock.release()

    def calculate_error(self, params, depth_msg, scan_msg):
        A, B = params
        
        # 1. Convert raw depth image to a simulated scan
        raw_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        
        # `depthimage_to_laserscan` typically uses the horizontal centerline
        center_y = self.camera_info.height // 2
        depth_center_row = raw_depth_image[center_y, :]
        
        # Apply the candidate parameters to get metric depth
        metric_depth_row = depth_center_row * A + B

        # 2. Get ranges from the real LiDAR scan
        real_scan_ranges = np.array(scan_msg.ranges)
        
        # 3. Align the two scans by their angles
        # Create angular array for the camera-based scan
        cam_angles = np.linspace(
            -self.camera_info.width / 2 / self.camera_info.k[0] * 1, # fx
             self.camera_info.width / 2 / self.camera_info.k[0] * 1,
            self.camera_info.width
        )
        # Create angular array for the real LiDAR scan
        lidar_angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(real_scan_ranges))

        # Find overlapping angular region
        overlap_min_angle = max(np.min(cam_angles), np.min(lidar_angles))
        overlap_max_angle = min(np.max(cam_angles), np.max(lidar_angles))

        # Filter out non-overlapping data
        valid_cam_indices = (cam_angles >= overlap_min_angle) & (cam_angles <= overlap_max_angle)
        valid_lidar_indices = (lidar_angles >= overlap_min_angle) & (lidar_angles <= overlap_max_angle)

        cam_angles_overlap = cam_angles[valid_cam_indices]
        metric_depth_overlap = metric_depth_row[valid_cam_indices]
        
        lidar_angles_overlap = lidar_angles[valid_lidar_indices]
        real_scan_overlap = real_scan_ranges[valid_lidar_indices]

        # Ignore invalid LiDAR readings (inf, nan)
        valid_lidar_readings = np.isfinite(real_scan_overlap)
        lidar_angles_overlap = lidar_angles_overlap[valid_lidar_readings]
        real_scan_overlap = real_scan_overlap[valid_lidar_readings]

        if len(lidar_angles_overlap) < 10: # Not enough data to compare
            return 1e9 # Return a large error

        # 4. Interpolate LiDAR data to match camera's angular points for direct comparison
        interpolated_lidar_ranges = np.interp(cam_angles_overlap, lidar_angles_overlap, real_scan_overlap)

        # 5. Calculate error (Mean Absolute Error is robust to outliers)
        error = np.mean(np.abs(metric_depth_overlap - interpolated_lidar_ranges))
        
        # We want to encourage finding positive depth values
        if np.any(metric_depth_overlap <= 0):
            error += 100 # Add a large penalty for non-positive depth

        return error

def main(args=None):
    rclpy.init(args=args)
    node = DepthScaleCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
