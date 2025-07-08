import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import os

def export_model_to_onnx():
    # --- Configuration ---
    # IMPORTANT: Use the exact same model name as in your ROS node
    model_name = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'
    output_onnx_path = 'depth_anything_v2_small.onnx'
    
    # --- Check Model Input Size ---
    # The V2-Small model expects a fixed input size. We get this from the processor config.
    # It's usually 518x518 for this specific model.
    temp_processor = AutoImageProcessor.from_pretrained(model_name)
    input_size = temp_processor.size['height'] # height and width are the same
    print(f"Model: {model_name}")
    print(f"Expected input size (H, W): ({input_size}, {input_size})")

    # --- Model Loading ---
    print("Loading PyTorch model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    
    # --- Create a Dummy Input ---
    # This dummy input must match the model's expected input shape and type.
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # --- Export to ONNX ---
    print(f"Exporting model to {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=['input'],            # Name for the input layer
        output_names=['output'],          # Name for the output layer
        opset_version=14,                 # A stable opset version
        dynamic_axes={
            'input': {0: 'batch_size'},   # Allow variable batch size
            'output': {0: 'batch_size'}
        }
    )
    
    print("Export complete!")
    print(f"ONNX model saved to: {os.path.abspath(output_onnx_path)}")

if __name__ == '__main__':
    export_model_to_onnx()
