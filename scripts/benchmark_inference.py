"""
Benchmark inference speeds across all model formats.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --iterations 500

Run from project root directory.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =============================================================================
# Model Loading
# =============================================================================

def load_pytorch_model(model_path, model_type):
    if model_type == 'mobilenet':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, 2)
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
    
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model = model.cuda().eval()
    return model


def load_onnx_model(onnx_path):
    import onnxruntime as ort
    return ort.InferenceSession(
        str(onnx_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )


def load_tensorrt_model(engine_path):
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.ERROR)  # Suppress warnings
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    return engine, context


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess(image_path, image_size=224):
    """Preprocess image using OpenCV."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0).astype(np.float32)


# =============================================================================
# Inference
# =============================================================================

def infer_pytorch(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    torch.cuda.synchronize()
    return output


def infer_onnx(session, input_array):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_array})[0]


def infer_tensorrt(context, engine, input_array, d_input, d_output):
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    d_input.copy_(torch.from_numpy(input_array))
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
    return d_output


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_preprocess(image_paths, iterations):
    for _ in range(10):
        preprocess(image_paths[0])
    
    start = time.perf_counter()
    for i in range(iterations):
        preprocess(image_paths[i % len(image_paths)])
    
    return (time.perf_counter() - start) / iterations * 1000


def benchmark_inference(model_type, paths, input_np, iterations):
    results = {}
    input_torch = torch.from_numpy(input_np).cuda()
    
    # PyTorch
    if paths.get('pytorch') and paths['pytorch'].exists():
        model = load_pytorch_model(paths['pytorch'], model_type)
        for _ in range(10):
            infer_pytorch(model, input_torch)
        
        start = time.perf_counter()
        for _ in range(iterations):
            infer_pytorch(model, input_torch)
        results['pytorch'] = (time.perf_counter() - start) / iterations * 1000
        del model
        torch.cuda.empty_cache()
    
    # ONNX
    if paths.get('onnx') and paths['onnx'].exists():
        session = load_onnx_model(paths['onnx'])
        for _ in range(10):
            infer_onnx(session, input_np)
        
        start = time.perf_counter()
        for _ in range(iterations):
            infer_onnx(session, input_np)
        results['onnx'] = (time.perf_counter() - start) / iterations * 1000
        del session
    
    # TensorRT
    if paths.get('tensorrt') and paths['tensorrt'].exists():
        engine, context = load_tensorrt_model(paths['tensorrt'])
        output_shape = context.get_tensor_shape(engine.get_tensor_name(1))
        d_input = torch.empty((1, 3, 224, 224), dtype=torch.float32, device='cuda')
        d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device='cuda')
        
        for _ in range(10):
            infer_tensorrt(context, engine, input_np, d_input, d_output)
        
        start = time.perf_counter()
        for _ in range(iterations):
            infer_tensorrt(context, engine, input_np, d_input, d_output)
        results['tensorrt'] = (time.perf_counter() - start) / iterations * 1000
        del engine, context
        torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 70)
    print("INFERENCE BENCHMARK")
    print("=" * 70)
    print(f"Iterations: {args.iterations}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Test images
    data_dir = Path('data/CleanPetImages')
    image_paths = list(data_dir.glob('Cat/*.jpg'))[:100] + list(data_dir.glob('Dog/*.jpg'))[:100]
    print(f"Test images: {len(image_paths)}")
    
    # Preprocessing
    print(f"\n{'=' * 70}")
    print("PREPROCESSING (OpenCV)")
    print("=" * 70)
    preproc_time = benchmark_preprocess(image_paths, args.iterations)
    print(f"Time per image: {preproc_time:.2f} ms")
    
    # Model paths
    mobilenet_paths = {
        'pytorch': Path('outputs/final/final_model.pth'),
        'onnx': Path('outputs/final/final_model.onnx'),
        'tensorrt': Path('outputs/final/final_model_fp16.engine'),
    }
    resnet_paths = {
        'pytorch': Path('outputs/final_resnet18/final_resnet18_model.pth'),
        'onnx': Path('outputs/final_resnet18/final_resnet18_model.onnx'),
        'tensorrt': Path('outputs/final_resnet18/final_resnet18_model_fp16.engine'),
    }
    
    input_np = preprocess(image_paths[0])
    
    # Inference benchmarks
    print(f"\n{'=' * 70}")
    print("INFERENCE (model only)")
    print("=" * 70)
    
    mobilenet_results = benchmark_inference('mobilenet', mobilenet_paths, input_np, args.iterations)
    resnet_results = benchmark_inference('resnet18', resnet_paths, input_np, args.iterations)
    
    print(f"\n{'Model':<15} {'Format':<12} {'Time':<12} {'FPS':<10}")
    print("-" * 50)
    for name, results in [("MobileNetV2", mobilenet_results), ("ResNet18", resnet_results)]:
        for fmt, t in results.items():
            print(f"{name:<15} {fmt:<12} {t:>6.2f} ms    {1000/t:>6.0f}")
    
    # End-to-end
    print(f"\n{'=' * 70}")
    print("END-TO-END (preprocessing + inference)")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'Format':<12} {'Preproc':<10} {'Infer':<10} {'Total':<10} {'FPS':<8}")
    print("-" * 70)
    for name, results in [("MobileNetV2", mobilenet_results), ("ResNet18", resnet_results)]:
        for fmt, inf in results.items():
            total = preproc_time + inf
            print(f"{name:<15} {fmt:<12} {preproc_time:>6.2f} ms  {inf:>6.2f} ms  {total:>6.2f} ms  {1000/total:>6.0f}")
    
    # Speedup
    print(f"\n{'=' * 70}")
    print("SPEEDUP")
    print("=" * 70)
    for name, results in [("MobileNetV2", mobilenet_results), ("ResNet18", resnet_results)]:
        if 'pytorch' in results and 'tensorrt' in results:
            print(f"{name}: TensorRT is {results['pytorch']/results['tensorrt']:.1f}x faster than PyTorch")
        if 'onnx' in results and 'tensorrt' in results:
            print(f"{name}: TensorRT is {results['onnx']/results['tensorrt']:.1f}x faster than ONNX")


if __name__ == '__main__':
    main()