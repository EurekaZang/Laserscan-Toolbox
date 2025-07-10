#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class PreprocessorNode(Node):
    def __init__(self):
        super().__init__('preprocessor_node')
        
        # Declare parameters that will be tuned by the calibrator
        # We start with non-influential defaults (A=1, B=0)
        self.declare_parameter('calib_A', 1.0)
        self.declare_parameter('calib_B', 0.0)
        
        # Create a timer to periodically check for updated parameters
        self.timer = self.create_timer(1.0, self.update_parameters)
        
        self.bridge = CvBridge()
        self.calib_A = self.get_parameter('calib_A').get_parameter_value().double_value
        self.calib_B = self.get_parameter('calib_B').get_parameter_value().double_value

        self.subscription = self.create_subscription(
            Image,
            '/fake_camera/depth/raw_unscaled',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/fake_camera/depth/raw_processed', 10)
        
        self.get_logger().info('PreprocessorNode started. Waiting for raw depth images...')

    def update_parameters(self):
        """Periodically check for and apply updated ROS parameters."""
        self.calib_A = self.get_parameter('calib_A').get_parameter_value().double_value
        self.calib_B = self.get_parameter('calib_B').get_parameter_value().double_value
        self.get_logger().info(f'Updated params: A={self.calib_A:.4f}, B={self.calib_B:.4f}', throttle_duration_sec=5)

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            raw_depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            processed_depth = self.calib_A / (raw_depth_cv + self.calib_B + 1e-8)
            
            # Ensure no negative depths are produced
            np.clip(processed_depth, 0, None, out=processed_depth)

            # Convert OpenCV image back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_depth.astype(np.float32), encoding='32FC1')
            processed_msg.header = msg.header # Preserve header info
            
            self.publisher.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = PreprocessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
