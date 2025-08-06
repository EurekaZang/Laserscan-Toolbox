# build_engine.py
import tensorrt as trt
import os

# --- 配置 ---
ONNX_PATH = 'depth_anything_v2_vits.onnx' # <--- 你的 ONNX 文件路径
ENGINE_PATH = 'depth-anything-v2-small-relative_depth.engine' # <--- 输出的引擎文件路径
USE_FP16 = True # 使用半精度浮点数可以显著提速，通常对深度估计模型影响不大

# 假设的输入尺寸，必须与第一步中验证的尺寸一致！
INPUT_H = 518
INPUT_W = 518

def build_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    print("--- Building TensorRT Engine ---")
    print(f"From ONNX: {ONNX_PATH}")
    print(f"To Engine: {ENGINE_PATH}")
    print("TensorRT Version: ", trt.__version__)
    
    # 1. 创建 Builder, Network, 和 Parser
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 2. 配置 Builder
    config = builder.create_builder_config()
    # 根据你的GPU内存调整
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB

    # 3. 解析 ONNX 模型
    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX file not found at {ONNX_PATH}")
        return

    print("Parsing ONNX model...")
    with open(ONNX_PATH, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("ONNX model parsed successfully.")

    # 4. 设置输入形状 (对于有动态输入的模型)
    input_tensor = network.get_input(0)
    print(f"Network input name: {input_tensor.name}")
    print(f"Network input shape (from parser): {input_tensor.shape}")

    # 如果模型是动态形状 (-1), 我们需要提供一个优化配置
    if -1 in input_tensor.shape:
        print("Input shape is dynamic. Creating an optimization profile.")
        profile = builder.create_optimization_profile()
        # 定义最小、最优、最大的输入尺寸
        # 对于深度估计，批次大小通常是1
        min_shape = (1, 3, INPUT_H, INPUT_W)
        opt_shape = (1, 3, INPUT_H, INPUT_W)
        max_shape = (1, 3, INPUT_H, INPUT_W)
        profile.set_shape(input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)
    else:
        print("Input shape is static.")


    # 5. 构建引擎
    print("Building serialized engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return
        
    # 6. 保存引擎到文件
    print(f"Engine built successfully. Saving to {ENGINE_PATH}")
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized_engine)
        
    print("\n--- Engine build complete! ---")

if __name__ == '__main__':
    build_engine()
