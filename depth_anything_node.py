#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters  # 用于同步话题

import cv2
import numpy as np
import time
import os
import threading

# --- TensorRT Imports ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化CUDA，确保只在主线程执行一次

def preprocess_image(image, target_height, target_width):
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    chw_image = np.transpose(normalized, (2, 0, 1))
    batch_image = np.expand_dims(chw_image, axis=0)
    return batch_image.astype(np.float32)

class TrtInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        print(f"Loading engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shape = self.engine.get_tensor_shape(tensor_name)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shape = self.engine.get_tensor_shape(tensor_name)

        self.stream = cuda.Stream()
        print("TensorRT Engine initialized.")
        print(f"Input Shape: {self.input_shape}")
        print(f"Output Shape: {self.output_shape}")

    def infer(self, image):
        input_data = preprocess_image(image, self.input_shape[2], self.input_shape[3])
        np.copyto(self.host_inputs[0], input_data.ravel())

        start = time.time()
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        self.stream.synchronize()
        end = time.time()
        
        print(f"Inference time: {(end - start) * 1000:.2f} ms")
        return self.host_outputs[0]

class DepthAnythingTensorRTNode(Node):
    def __init__(self):
        super().__init__('depth_anything_tensorrt_node')

        self.declare_parameter('engine_path', 'models/depth_anything_v2_small.engine')
        self.declare_parameter('param_scale', 1.0)
        self.declare_parameter('param_shift', 0.0)
        self.declare_parameter('calib_A', 1.0) # <-- PUT YOUR OPTIMIZED A HERE
        self.declare_parameter('calib_B', 0.0) # <-- PUT YOUR OPTIMIZED B HERE
        self.calib_A = self.get_parameter('calib_A').get_parameter_value().double_value
        self.calib_B = self.get_parameter('calib_B').get_parameter_value().double_value
        self.declare_parameter('input_image_topic', '/camera/color/image_raw')
        self.declare_parameter('input_info_topic', '/camera/color/camera_info')
        
        engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.param_scale = self.get_parameter('param_scale').get_parameter_value().double_value
        self.param_shift = self.get_parameter('param_shift').get_parameter_value().double_value
        input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        input_info_topic = self.get_parameter('input_info_topic').get_parameter_value().string_value

        self.get_logger().info(f"Using TensorRT engine: {engine_path}")

        self.trt_model = TrtInference(engine_path)
        self.bridge = CvBridge()

        self.depth_pub = self.create_publisher(Image, '/depth_anything/image', 30)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/depth_anything/camera_info', 30)

        self.lock = threading.Lock()

        self.image_sub = message_filters.Subscriber(self, Image, input_image_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, input_info_topic)

        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], 10)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info('Depth Anything TensorRT node has started.')

    def image_callback(self, image_msg, info_msg):
        if not self.lock.acquire(blocking=False):
            self.get_logger().warn('Dropping a frame, inference is not fast enough for the input rate.')
            return

        try:
            t0 = self.get_clock().now()
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            original_shape = cv_image.shape
            t1 = self.get_clock().now()
            raw_output = self.trt_model.infer(cv_image)
            t2 = self.get_clock().now()

            depth_map = raw_output.reshape(self.trt_model.output_shape[-2:])
            depth_resized = cv2.resize(depth_map, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

            metric_depth = ((depth_resized * self.param_scale + self.param_shift) / 2.3).astype(np.float32)
            # metric_depth = 1.0 / np.clip(metric_depth, 1e-6, None)  # Avoid division by zero
            t3 = self.get_clock().now()

            # depth_msg = self.bridge.cv2_to_imgmsg(depth_resized, encoding='32FC1')
            depth_msg = self.bridge.cv2_to_imgmsg(metric_depth, encoding='32FC1')
            depth_msg.header = image_msg.header
            self.depth_pub.publish(depth_msg)

            info_msg.header = image_msg.header
            self.depth_info_pub.publish(info_msg)
            t4 = self.get_clock().now()

            time_ros_to_cv = (t1 - t0).nanoseconds / 1e6
            time_inference_full = (t2 - t1).nanoseconds / 1e6
            time_post_processing = (t3 - t2).nanoseconds / 1e6
            time_publishing_and_viz = (t4 - t3).nanoseconds / 1e6
            time_total = (t4 - t0).nanoseconds / 1e6

            log_message = (
                f"\n--- Timing Breakdown (ms) ---\n"
                f"  1. ROS->CV Convert:    {time_ros_to_cv:6.2f}\n"
                f"  2. Full Inference:     {time_inference_full:6.2f} (incl. preprocess)\n"
                f"  3. Post-Processing:    {time_post_processing:6.2f} (resize & scale)\n"
                f"  4. Viz & Publishing:   {time_publishing_and_viz:6.2f} (colorize, convert, pub)\n"
                f"--------------------------------\n"
                f"  Total Cycle Time:      {time_total:6.2f}"
            )
            self.get_logger().info(log_message)

        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        finally:
            self.lock.release()

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingTensorRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
