# test_onnx.py
import onnxruntime as ort
import numpy as np
import cv2
import time

# --- 配置 ---
ONNX_PATH = 'depth_anything_v2_small.onnx' # <--- 修改为你的 ONNX 文件路径
TEST_IMAGE_PATH = 'test_image.jpg'     # <--- 修改为你的测试图片路径

def preprocess_image(image, target_height, target_width):
    """
    预处理输入图像，必须与模型训练时的方式完全一致。
    这是 DepthAnything 模型的标准预处理流程。
    """
    # 调整图像大小
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # 转换为 RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 归一化到 [0, 1]
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # 标准化 (ImageNet 标准)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # 转换为 CHW 格式并添加批次维度
    chw_image = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
    batch_image = np.expand_dims(chw_image, axis=0)  # 添加批次维度
    
    return batch_image.astype(np.float32)

def main():
    print("--- ONNX Model Sanity Check ---")

    # 检查 ONNX Runtime 是否在使用 GPU
    print(f"ONNX Runtime Providers: {ort.get_available_providers()}")
    assert 'CUDAExecutionProvider' in ort.get_available_providers(), "CUDA provider not found for ONNX Runtime"

    # 创建推理会话
    print(f"Loading ONNX model from: {ONNX_PATH}")
    session = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 获取输入信息
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape # e.g., ['batch', 3, 518, 518]
    print(f"Model Input Name: {input_name}")
    print(f"Model Input Shape (from ONNX): {input_shape}")

    # 对于动态输入，我们需要确定一个具体的尺寸
    # DepthAnything 通常使用 518x518
    target_height = input_shape[2] if isinstance(input_shape[2], int) else 518
    target_width = input_shape[3] if isinstance(input_shape[3], int) else 518
    actual_input_shape = [1, 3, target_height, target_width]
    print(f"Using actual input shape for inference: {actual_input_shape}")

    # 获取输出信息
    output_info = session.get_outputs()[0]
    output_name = output_info.name
    print(f"Model Output Name: {output_name}")

    # 加载并预处理测试图像
    print(f"Loading test image from: {TEST_IMAGE_PATH}")
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image at {TEST_IMAGE_PATH}")
        return

    preprocessed_image = preprocess_image(image, target_height, target_width)
    print("Running warm-up inference...")
    _ = session.run([output_name], {input_name: preprocessed_image})
    print("Warm-up complete.")

    # 2. 性能测试
    # 多次运行并取平均值，以获得更稳定的性能数据
    print("Running performance test...")
    num_runs = 20
    start_time = time.time()
    for _ in range(num_runs):
        outputs = session.run([output_name], {input_name: preprocessed_image})
    end_time = time.time()

    # 提取最后一次运行的结果用于分析
    depth_map_raw = outputs[0]

    # 计算平均推理时间
    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / num_runs) * 1000

    print(f"Performance test completed over {num_runs} runs.")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms") # 使用毫秒(ms)为单位更直观
    print(f"Raw output shape: {depth_map_raw.shape}")
    print(f"Raw output stats: min={np.min(depth_map_raw):.4f}, max={np.max(depth_map_raw):.4f}, mean={np.mean(depth_map_raw):.4f}")

    print("\n--- ONNX model check PASSED with GPU! ---")
    print("If this script runs without errors, your ONNX file is valid and running on CUDA.")

if __name__ == '__main__':
    main()