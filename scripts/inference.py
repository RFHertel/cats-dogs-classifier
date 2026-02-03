"""
Run inference on a single image.

Usage:
    python scripts/inference.py <image_path>
    python scripts/inference.py <image_path> --model outputs/final/final_model.pth
    python scripts/inference.py <image_path> --model outputs/final/final_model_fp16.engine
    Or any image you have on your computer:
    python scripts/inference.py C:\path\to\your\photo.jpg --model outputs/final/final_model_fp16.engine

Examples:
    python scripts/inference.py data/CleanPetImages/Cat/1.jpg
    python scripts/inference.py data/CleanPetImages/Cat/1.jpg --model outputs/final/final_model_fp16.engine
    Run from project root directory: C:\AWrk\cats_dogs_project> 
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASSES = ['Cat', 'Dog']


def preprocess(image_path, size=224):
    """Load and preprocess image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, 0).astype(np.float32)


def load_pytorch_model(model_path, model_type='mobilenet'):
    """Load PyTorch model."""
    if model_type == 'mobilenet':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, 2)
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def infer_pytorch(model, input_np):
    """Run PyTorch inference."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    input_tensor = torch.from_numpy(input_np).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    
    return probs.cpu().numpy()[0]


def infer_tensorrt(engine_path, input_np):
    """Run TensorRT inference."""
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    output_shape = context.get_tensor_shape(output_name)
    
    d_input = torch.from_numpy(input_np).cuda()
    d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device='cuda')
    
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
    output = d_output.cpu().numpy()[0]
    
    # Softmax
    exp_out = np.exp(output - np.max(output))
    probs = exp_out / exp_out.sum()
    
    return probs


def main():
    parser = argparse.ArgumentParser(description='Classify cat or dog')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--model', default='outputs/final/final_model.pth',
                        help='Path to model (.pth or .engine)')
    parser.add_argument('--model-type', default='mobilenet',
                        choices=['mobilenet', 'resnet18'],
                        help='Model architecture (for .pth files)')
    args = parser.parse_args()
    
    image_path = Path(args.image)
    model_path = Path(args.model)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    # Preprocess
    input_np = preprocess(image_path)
    
    # Inference
    if model_path.suffix == '.engine':
        probs = infer_tensorrt(model_path, input_np)
        model_type = "TensorRT"
    else:
        model = load_pytorch_model(model_path, args.model_type)
        probs = infer_pytorch(model, input_np)
        model_type = "PyTorch"
    
    # Results
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100
    
    print(f"\nImage: {image_path}")
    print(f"Model: {model_path.name} ({model_type})")
    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"  Cat: {probs[0]*100:.1f}%")
    print(f"  Dog: {probs[1]*100:.1f}%")


if __name__ == '__main__':
    main()