#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
import os
import threading

# --- TensorRT Imports ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
print("TensorRT Version: ", trt.__version__)

class TensorRTDepthNode(Node):
    def __init__(self):
        super().__init__('tensorrt_depth_node')

        # --- ROS2 参数声明 ---
        self.declare_parameter('engine_path', 'depth_anything_v2_small.engine')
        self.declare_parameter('shift', 0.000438)
        self.declare_parameter('scale', 0.021146)
        self.declare_parameter('input_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', '/fake_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_info_topic', '/fake_camera/depth/camera_info')

        # --- 获取参数 ---
        self.engine_path = self.get_parameter('engine_path').value
        self.shift = self.get_parameter('shift').value
        self.scale = self.get_parameter('scale').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        depth_info_topic = self.get_parameter('depth_info_topic').value

        # --- 初始化变量 ---
        self.bridge = CvBridge()
        self.engine = None
        self.stream = None
        self.input_buffer = None
        self.output_buffer = None
        self.input_host = None
        self.output_host = None
        self.input_shape = None
        self.output_shape = None
        self.camera_info = None
        self.tensorrt_ready = False
        self.lock = threading.Lock()

        # --- 日志信息 ---
        self.get_logger().info("=" * 50)
        self.get_logger().info("TensorRT Depth Estimation Node Starting")
        self.get_logger().info(f"Engine Path: {self.engine_path}")
        self.get_logger().info(f"Input Topic: {input_topic}")
        self.get_logger().info(f"Output Topic: {output_topic}")
        self.get_logger().info("=" * 50)

        # --- 初始化 TensorRT ---
        if not self.init_tensorrt():
            self.get_logger().error("Failed to initialize TensorRT. Exiting.")
            return

        # --- ROS2 Publishers and Subscribers ---
        self.depth_pub = self.create_publisher(Image, output_topic, 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, depth_info_topic, 10)
        self.image_sub = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info("TensorRT Depth Node initialized successfully!")

    def init_tensorrt(self):
        """初始化 TensorRT 引擎和上下文"""
        try:
            # 检查引擎文件是否存在
            if not os.path.exists(self.engine_path):
                self.get_logger().error(f"Engine file not found: {self.engine_path}")
                return False

            # 创建 TensorRT Logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # 创建运行时并反序列化引擎
            self.get_logger().info("Loading TensorRT engine...")
            with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                self.get_logger().error("Failed to load TensorRT engine")
                return False
            else:
                self.get_logger().info("TensorRT engine loaded successfully")

            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            if self.context is None:
                self.get_logger().error("Failed to create execution context")
                return False

            self.get_logger().info("Execution context created successfully")

            # 创建 CUDA 流
            self.stream = cuda.Stream()

            # 获取输入输出信息
            self.get_logger().info("Setting up input/output buffers...")

            # 获取输入输出的绑定信息
            input_binding = self.engine.get_tensor_name(0)  # 第一个是输入
            output_binding = self.engine.get_tensor_name(1)  # 第二个是输出

            # 获取输入形状并设置动态形状
            self.input_shape = self.engine.get_tensor_shape(input_binding)
            self.get_logger().info(f"Input shape from engine: {self.input_shape}")

            # 对于动态形状，设置具体的输入尺寸
            if self.input_shape[0] == -1:  # 如果批次大小是动态的
                actual_input_shape = (1, self.input_shape[1], self.input_shape[2], self.input_shape[3])
                self.context.set_input_shape(input_binding, actual_input_shape)
                self.input_shape = actual_input_shape
                self.get_logger().info(f"Set dynamic input shape to: {self.input_shape}")

            # 获取输出形状
            self.output_shape = self.context.get_tensor_shape(output_binding)
            self.get_logger().info(f"Output shape: {self.output_shape}")

            # 分配内存
            input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
            output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize

            # 分配设备内存
            self.input_buffer = cuda.mem_alloc(input_size)
            self.output_buffer = cuda.mem_alloc(output_size)

            # 分配主机内存
            self.input_host = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
            self.output_host = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

            # 设置张量地址
            self.context.set_tensor_address(input_binding, int(self.input_buffer))
            self.context.set_tensor_address(output_binding, int(self.output_buffer))

            self.get_logger().info("Memory allocation completed")
            self.get_logger().info(f"Input buffer size: {input_size} bytes")
            self.get_logger().info(f"Output buffer size: {output_size} bytes")

            self.tensorrt_ready = True
            return True

        except Exception as e:
            self.get_logger().error(f"TensorRT initialization failed: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            return False

    def camera_info_callback(self, msg):
        """接收相机信息"""
        with self.lock:
            self.camera_info = msg

    def image_callback(self, msg):
        """处理输入图像"""
        if not self.tensorrt_ready:
            self.get_logger().warn("TensorRT not ready, skipping frame")
            return

        try:
            start_time = time.time()
            
            # 转换 ROS 图像到 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 预处理图像
            processed_image = self.preprocess_image(cv_image)

            # 运行推理
            depth_map = self.run_inference(processed_image)
            
            if depth_map is not None:
                # 后处理并发布深度图
                self.postprocess_and_publish(depth_map, msg.header, cv_image.shape)

                # 计算处理时间
                processing_time = time.time() - start_time
                self.get_logger().info(f"Processing time: {processing_time:.3f}s")
            
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def preprocess_image(self, image):
        """预处理输入图像"""
        # 获取目标尺寸（从输入形状中获取）
        target_height = self.input_shape[2]
        target_width = self.input_shape[3]
        
        # 调整图像大小
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 转换为 RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # 标准化 (ImageNet 标准)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # 转换为 CHW 格式并添加批次维度
        chw_image = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        batch_image = np.expand_dims(chw_image, axis=0)  # 添加批次维度
        
        return batch_image.astype(np.float32)

    def run_inference(self, input_data):
        """运行 TensorRT 推理"""
        try:
            with self.lock:
                # 将输入数据复制到分页锁定的主机内存
                np.copyto(self.input_host, input_data.ravel())
                
                # 将数据从主机复制到设备
                cuda.memcpy_htod_async(self.input_buffer, self.input_host, self.stream)
                
                # 执行推理
                self.context.execute_async_v3(stream_handle=self.stream.handle)
                
                # 将结果从设备复制到主机
                cuda.memcpy_dtoh_async(self.output_host, self.output_buffer, self.stream)
                
                # 同步流
                self.stream.synchronize()
                
                # 重塑输出数据
                output_data = self.output_host.reshape(self.output_shape)
                
                return output_data
                
        except Exception as e:
            self.get_logger().error(f"Inference error: {str(e)}")
            return None

    def postprocess_and_publish(self, depth_output, header, original_shape):
        """后处理深度图并发布"""
        try:
            # 移除批次维度并获取深度图
            if len(depth_output.shape) == 4:
                depth_map = depth_output[0, 0, :, :]  # 假设输出是 (1, 1, H, W)
            else:
                depth_map = depth_output[0, :, :]  # 假设输出是 (1, H, W)
            
            # 调整深度图尺寸到原始图像尺寸
            original_height, original_width = original_shape[:2]
            depth_resized = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            
            # 应用尺度和偏移
            depth_resized = depth_resized * self.scale + self.shift
            
            # 转换为毫米 (ROS 深度图通常使用毫米)
            depth_mm = (depth_resized * 1000).astype(np.uint16)
            
            # 创建深度图消息
            depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="mono16")
            depth_msg.header = header
            depth_msg.header.frame_id = header.frame_id
            
            # 发布深度图
            self.depth_pub.publish(depth_msg)
            
            # 发布深度相机信息
            if self.camera_info is not None:
                depth_info = self.camera_info
                depth_info.header = header
                self.depth_info_pub.publish(depth_info)
            
            self.get_logger().info(f"Published depth map - Shape: {depth_mm.shape}, "
                                 f"Min: {np.min(depth_mm)}, Max: {np.max(depth_mm)}")
            
        except Exception as e:
            self.get_logger().error(f"Postprocessing error: {str(e)}")

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'input_buffer') and self.input_buffer:
                self.input_buffer.free()
            if hasattr(self, 'output_buffer') and self.output_buffer:
                self.output_buffer.free()
            if hasattr(self, 'stream') and self.stream:
                self.stream.synchronize()
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TensorRTDepthNode()
        
        if node.tensorrt_ready:
            print("TensorRT Depth Node is ready!")
            rclpy.spin(node)
        else:
            print("Failed to initialize TensorRT Depth Node")
            
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()