"""
Export trained model to ONNX format.

Usage:
    python scripts/export_model.py outputs/final/final_model.pth
    python scripts/export_model.py outputs/final/final_model.pth --fp16
    python scripts/export_model.py outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18

Run from project root directory.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


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
        opset_version=18,  # Changed from 11 to 18
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    
    print(f"ONNX model saved: {output_path}")


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


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('model_path', help='Path to .pth model file')
    parser.add_argument('--model-type', default='mobilenet', choices=['mobilenet', 'resnet18'])
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    model_name = model_path.stem
    onnx_path = output_dir / f"{model_name}.onnx"
    
    print("="*50)
    print("Model Export to ONNX")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Type: {args.model_type}")
    print()
    
    # Load
    print("Loading PyTorch model...")
    model = load_model(model_path, args.model_type)
    
    # Export
    print("Exporting to ONNX...")
    export_onnx(model, str(onnx_path), args.image_size)
    
    # Verify
    print("Verifying...")
    verify_onnx(str(onnx_path))
    
    # Test
    print("Testing inference...")
    test_onnx(str(onnx_path), args.image_size)
    
    # File sizes
    print(f"\nFile sizes:")
    print(f"  PyTorch: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  ONNX: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == '__main__':
    main()