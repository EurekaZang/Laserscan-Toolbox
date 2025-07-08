import tensorrt as trt
import os

def build_engine(onnx_path, engine_path, use_fp16=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
        
        if use_fp16 and builder.platform_has_fast_fp16:
            print("FP16 mode enabled.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("FP16 mode not enabled or not supported.")

        print(f"Loading ONNX file from: {onnx_path}")
        if not os.path.exists(onnx_path):
            print(f"ERROR: ONNX file not found at {onnx_path}")
            return
            
        with open(onnx_path, 'rb') as model:
            print("Parsing ONNX model...")
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
        
        print("ONNX model parsed successfully.")

        # ===================================================================
        #  vvv --- 新增的核心代码：创建和配置优化配置文件 --- vvv
        # ===================================================================
        
        print("Defining optimization profile for dynamic inputs...")
        # 1. 创建一个优化配置文件对象
        profile = builder.create_optimization_profile()
        
        # 2. 获取ONNX模型的输入信息
        #    我们假设模型只有一个输入，即索引为0的输入
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape # 例如: (-1, 3, 518, 518)
        
        # 3. 定义输入尺寸的范围 (min, opt, max)
        #    由于我们的ROS节点一次处理一张图，所以batch size的min, opt, max都设为1
        min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        max_shape = (1, input_shape[1], input_shape[2], input_shape[3])
        
        # 4. 将这个范围设置到配置文件中
        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        
        # 5. 将配置文件添加到构建器的配置中
        config.add_optimization_profile(profile)
        
        print(f"Optimization profile defined for input '{input_name}':")
        print(f"  MIN shape: {min_shape}")
        print(f"  OPT shape: {opt_shape}")
        print(f"  MAX shape: {max_shape}")
        
        # ===================================================================
        #  ^^^ --- 新增代码结束 --- ^^^
        # ===================================================================

        print("Building serialized TensorRT engine... (This may take a few minutes)")
        
        # 现在，使用更新后的config来构建引擎
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("ERROR: Failed to build the engine.")
            return

        print("Engine built successfully.")

        print(f"Saving engine to file: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print("Engine saved.")


if __name__ == '__main__':
    # 确保路径正确，如果脚本在utils目录，ONNX文件在上一级
    # 根据你的文件结构进行调整
    ONNX_FILE_PATH = '../depth_anything_v2_small.onnx' 
    ENGINE_FILE_PATH = '../depth_anything_v2_small.engine'
    
    build_engine(ONNX_FILE_PATH, ENGINE_FILE_PATH, use_fp16=True)
