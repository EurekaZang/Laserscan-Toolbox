#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import numpy as np
import threading

class DepthCalibrationProcessor(Node):
    def __init__(self):
        super().__init__('depth_calibration_processor')
        
        # 默认参数
        self.scale = 1.0
        self.shift = 0.0
        self.multiplier = 15.0
        
        # 非线性参数 (可选)
        self.use_nonlinear = False
        self.nonlinear_a = 1.0
        self.nonlinear_b = 1.0
        self.nonlinear_c = 0.0
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # 订阅原始深度图和校准参数
        self.depth_sub = self.create_subscription(
            Image, '/fake_camera/depth/image', self.depth_callback, 30)
            # Image, '/fake_camera/depth/raw_unscaled', self.depth_callback, 30)
        self.param_sub = self.create_subscription(
            Float64MultiArray, '/calibration_params', self.param_callback, 30)
        
        # 发布处理后的深度图
        self.depth_pub = self.create_publisher(Image, '/raw_unscaled_processed', 30)
        
        self.get_logger().info('Depth calibration processor started.')
    
    def param_callback(self, msg):
        """更新校准参数"""
        with self.lock:
            if msg.data[0] is None or msg.data[1] is None or msg.data[2] is None:
                msg.data = [1.0, 0.0, 15.0]  # 默认值
            if len(msg.data) >= 3:
                self.scale = msg.data[0]
                self.shift = msg.data[1]
                self.multiplier = msg.data[2]
                
                if len(msg.data) >= 6:  # 非线性参数
                    self.use_nonlinear = True
                    self.nonlinear_a = msg.data[3]
                    self.nonlinear_b = msg.data[4]
                    self.nonlinear_c = msg.data[5]
                
                self.get_logger().info(f'Updated calibration params: scale={self.scale:.6f}, '
                                     f'shift={self.shift:.6f}, multiplier={self.multiplier:.6f}')
    
    def depth_callback(self, msg):
        """处理深度图像"""
        try:
            # 转换为numpy数组
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            with self.lock:
                if self.use_nonlinear:
                    # 非线性转换: depth = a * raw^b + c
                    depth_processed = self.nonlinear_a * np.power(depth_raw, self.nonlinear_b) + self.nonlinear_c
                else:
                    # 线性转换: depth = (raw * scale + shift) * multiplier
                    depth_processed = (depth_raw * self.scale + self.shift) * self.multiplier
            
            # 发布处理后的深度图
            processed_msg = self.bridge.cv2_to_imgmsg(depth_processed.astype(np.float32), encoding='32FC1')
            processed_msg.header = msg.header
            self.depth_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DepthCalibrationProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
