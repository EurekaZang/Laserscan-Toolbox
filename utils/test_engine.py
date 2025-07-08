# test_engine.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 重要：初始化CUDA
import numpy as np
import cv2
import time

# --- 配置 ---
ENGINE_PATH = 'depth_anything_v2_small_rebuilt.engine' # <--- 使用新生成的引擎
TEST_IMAGE_PATH = 'test_image.jpg'                 # <--- 你的测试图片
OUTPUT_IMAGE_PATH = 'depth_output.png'             # <--- 输出的可视化深度图

# 你的ROS节点中使用的参数，用于后处理
# 这些值需要根据你的模型进行微调！
# 初始值可以从你的ROS节点代码中获取
PARAM_SHIFT = 0.000438
PARAM_SCALE = 0.021146

# 从 `test_onnx.py` 复制过来的预处理函数，保持一致性
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
        # 预处理
        input_data = preprocess_image(image, self.input_shape[2], self.input_shape[3])
        np.copyto(self.host_inputs[0], input_data.ravel())

        # 推理
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

def postprocess_and_visualize(raw_output, output_shape, original_shape):
    # 重塑输出
    depth_map = raw_output.reshape(output_shape[1:]) # 移除批次维度
    if len(depth_map.shape) == 3:
        depth_map = depth_map[0] # 移除通道维度
    
    # 调整到原始尺寸
    depth_resized = cv2.resize(depth_map, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 应用偏移和缩放
    depth_scaled = depth_resized * PARAM_SCALE + PARAM_SHIFT

    # 可视化
    print(f"Processed depth map stats: min={np.min(depth_scaled):.4f}, max={np.max(depth_scaled):.4f}, mean={np.mean(depth_scaled):.4f}")
    
    # 归一化到 0-255 用于显示
    depth_normalized = cv2.normalize(depth_scaled, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    
    return colored_depth

def main():
    trt_model = TrtInference(ENGINE_PATH)
    
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image at {TEST_IMAGE_PATH}")
        return
        
    raw_output = trt_model.infer(image)
    
    colored_depth_map = postprocess_and_visualize(raw_output, trt_model.output_shape, image.shape)
    
    cv2.imwrite(OUTPUT_IMAGE_PATH, colored_depth_map)
    print(f"Success! Visualized depth map saved to {OUTPUT_IMAGE_PATH}")
    
    # 如果你想预览
    # cv2.imshow("Original", image)
    # cv2.imshow("Depth", colored_depth_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
