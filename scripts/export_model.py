"""
Export trained model to ONNX and TensorRT formats.

Usage:
    ONNX only
    python scripts/export_model.py outputs/final/final_model.pth

    ONNX + TensorRT FP32
    python scripts/export_model.py outputs/final/final_model.pth --tensorrt

    ONNX + TensorRT FP16 + Benchmark
    python scripts/export_model.py outputs/final/final_model.pth --tensorrt --fp16 --benchmark

    ResNet18
    python scripts/export_model.py outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18 --tensorrt --fp16 --benchmark

Run from project root directory: C:\AWrk\cats_dogs_project> 
"""

import argparse
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

warnings.filterwarnings('ignore')


def load_model(model_path, model_type='mobilenet'):
    """Load trained model."""
    if model_type == 'mobilenet':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, 2)
    elif model_type == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def export_onnx(model, output_path, image_size=224):
    """Export model to ONNX format."""
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # No dynamic_axes - fixed batch size for TensorRT compatibility
    )
    
    print(f"ONNX model saved: {output_path}")
    return output_path


def verify_onnx(onnx_path):
    """Verify ONNX model."""
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model verified")


def test_onnx(onnx_path, image_size=224):
    """Test ONNX inference."""
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        onnx_path, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy})
    
    print(f"ONNX inference test passed")
    print(f"  Output shape: {output[0].shape}")


def export_tensorrt(onnx_path, output_path, fp16=False):
    """Convert ONNX model to TensorRT engine."""
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed. Skipping.")
        return None
    
    print(f"Building TensorRT engine ({'FP16' if fp16 else 'FP32'})...")
    print("This may take a few minutes...")
    
    # Work in ONNX directory
    original_dir = os.getcwd()
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
    onnx_file = os.path.basename(onnx_path)
    os.chdir(onnx_dir)
    
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        
        parser = trt.OnnxParser(network, logger)
        with open(onnx_file, 'rb') as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX:")
                for i in range(parser.num_errors):
                    print(f"  {parser.get_error(i)}")
                return None
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
        
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            print("Failed to build engine")
            return None
        
        os.chdir(original_dir)
        with open(output_path, 'wb') as f:
            f.write(engine)
        
        print(f"TensorRT engine saved: {output_path}")
        return output_path
        
    finally:
        os.chdir(original_dir)


def test_tensorrt(engine_path, image_size=224):
    """Test TensorRT inference."""
    try:
        import tensorrt as trt
    except ImportError:
        return
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    output_shape = context.get_tensor_shape(output_name)
    
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    d_input = torch.from_numpy(dummy).cuda()
    d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device='cuda')
    
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())
    
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
    print(f"TensorRT inference test passed")
    print(f"  Output shape: {tuple(output_shape)}")


def benchmark(onnx_path, trt_path, image_size=224, iterations=100):
    """Compare inference speeds."""
    import time
    import onnxruntime as ort
    
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    
    print(f"\nBenchmarking ({iterations} iterations)...")
    
    # ONNX
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    for _ in range(10):  # warmup
        session.run(None, {input_name: dummy})
    
    start = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: dummy})
    onnx_time = (time.time() - start) / iterations * 1000
    print(f"  ONNX Runtime: {onnx_time:.2f} ms")
    
    # TensorRT
    if trt_path and Path(trt_path).exists():
        try:
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            
            with open(trt_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            context = engine.create_execution_context()
            input_name = engine.get_tensor_name(0)
            output_name = engine.get_tensor_name(1)
            output_shape = context.get_tensor_shape(output_name)
            
            d_input = torch.from_numpy(dummy).cuda()
            d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device='cuda')
            
            context.set_tensor_address(input_name, d_input.data_ptr())
            context.set_tensor_address(output_name, d_output.data_ptr())
            stream = torch.cuda.current_stream().cuda_stream
            
            for _ in range(10):  # warmup
                context.execute_async_v3(stream)
                torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(iterations):
                context.execute_async_v3(stream)
                torch.cuda.synchronize()
            trt_time = (time.time() - start) / iterations * 1000
            
            print(f"  TensorRT:     {trt_time:.2f} ms")
            print(f"  Speedup:      {onnx_time/trt_time:.2f}x")
        except Exception as e:
            print(f"  TensorRT benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export model')
    parser.add_argument('model_path', help='Path to .pth file')
    parser.add_argument('--model-type', default='mobilenet', choices=['mobilenet', 'resnet18'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--tensorrt', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    model_name = model_path.stem
    
    print("=" * 50)
    print("Model Export")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Type: {args.model_type}")
    if args.tensorrt:
        print(f"TensorRT: {'FP16' if args.fp16 else 'FP32'}")
    print()
    
    # Load
    print("Loading model...")
    model = load_model(model_path, args.model_type)
    
    # ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    print("\nExporting ONNX...")
    export_onnx(model, str(onnx_path), args.image_size)
    verify_onnx(str(onnx_path))
    test_onnx(str(onnx_path), args.image_size)
    
    # TensorRT
    trt_path = None
    if args.tensorrt:
        precision = 'fp16' if args.fp16 else 'fp32'
        trt_path = output_dir / f"{model_name}_{precision}.engine"
        print(f"\nExporting TensorRT...")
        result = export_tensorrt(str(onnx_path), str(trt_path), args.fp16)
        if result:
            test_tensorrt(str(trt_path), args.image_size)
    
    # Benchmark
    if args.benchmark:
        benchmark(str(onnx_path), str(trt_path) if trt_path else None, args.image_size)
    
    # Sizes
    print("\n" + "=" * 50)
    print("File Sizes")
    print("=" * 50)
    print(f"  PyTorch:  {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  ONNX:     {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    if trt_path and trt_path.exists():
        print(f"  TensorRT: {trt_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
