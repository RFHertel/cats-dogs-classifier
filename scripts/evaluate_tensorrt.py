"""
Evaluate a TensorRT engine on validation or test set.

Usage:
    python scripts/evaluate_tensorrt.py <engine_path> [options]

Examples:
    python scripts/evaluate_tensorrt.py outputs/final/final_model_fp16.engine
    python scripts/evaluate_tensorrt.py outputs/final/final_model_fp16.engine --dataset test
    python scripts/evaluate_tensorrt.py outputs/final_resnet18/final_resnet18_model_fp16.engine --dataset test

Run from project root directory: C:\AWrk\cats_dogs_project> 
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import argparse
import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CONFIG = {
    'data_dir': Path('data/CleanPetImages'),
    'split_file': Path('outputs/splits/train_val_test_split.json'),
    'image_size': 224,
}


def load_engine(engine_path):
    """Load TensorRT engine."""
    import tensorrt as trt
    
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    return engine, context


def preprocess(image_path, size=224):
    """Preprocess single image."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0).astype(np.float32)


def infer_tensorrt(context, engine, input_np, d_input, d_output):
    """Run TensorRT inference."""
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    d_input.copy_(torch.from_numpy(input_np))
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    
    return d_output.cpu().numpy()


def softmax(x):
    """Compute softmax."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def load_eval_data(config, dataset='val'):
    """Load file paths and labels."""
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    if dataset == 'val':
        files = split_data['val_files']
        labels = split_data['val_labels']
    elif dataset == 'test':
        files = split_data['test_files']
        labels = split_data['test_labels']
    else:
        raise ValueError(f"dataset must be 'val' or 'test'")
    
    return files, labels


def run_evaluation(engine, context, files, labels, data_dir, image_size):
    """Run inference on all images."""
    # Preallocate GPU buffers
    output_shape = context.get_tensor_shape(engine.get_tensor_name(1))
    d_input = torch.empty((1, 3, image_size, image_size), dtype=torch.float32, device='cuda')
    d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device='cuda')
    
    all_preds = []
    all_probs = []
    
    for i, filepath in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(files)} images...")
        
        img_path = Path(data_dir) / filepath
        input_np = preprocess(img_path, image_size)
        
        output = infer_tensorrt(context, engine, input_np, d_input, d_output)
        probs = softmax(output)[0]
        pred = np.argmax(probs)
        
        all_preds.append(pred)
        all_probs.append(probs)
    
    return np.array(all_preds), np.array(all_probs), np.array(labels)


def save_predictions_csv(files, predictions, probabilities, output_path):
    """Save predictions to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Cat', 'Dog']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'predicted_label', 'probability'])
        
        for filepath, pred, prob in zip(files, predictions, probabilities):
            label = class_names[pred]
            confidence = prob[pred]
            writer.writerow([filepath, label, f"{confidence:.4f}"])
    
    print(f"Predictions saved to: {output_path}")


def save_example_images(files, predictions, probabilities, labels, data_dir, output_dir, num_examples=8):
    """Save example images with predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Cat', 'Dog']
    
    correct_idx = [i for i in range(len(predictions)) if predictions[i] == labels[i]]
    incorrect_idx = [i for i in range(len(predictions)) if predictions[i] != labels[i]]
    
    n_correct = min(num_examples // 2, len(correct_idx))
    n_incorrect = min(num_examples - n_correct, len(incorrect_idx))
    
    np.random.seed(42)
    sample_correct = np.random.choice(correct_idx, n_correct, replace=False) if n_correct > 0 else []
    sample_incorrect = np.random.choice(incorrect_idx, n_incorrect, replace=False) if n_incorrect > 0 else []
    
    sample_indices = list(sample_correct) + list(sample_incorrect)
    
    n_cols = 4
    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(sample_indices):
            ax.axis('off')
            continue
        
        idx = sample_indices[i]
        filepath = files[idx]
        pred = predictions[idx]
        true_label = labels[idx]
        prob = probabilities[idx][pred]
        
        img_path = Path(data_dir) / filepath
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        
        pred_name = class_names[pred]
        true_name = class_names[true_label]
        is_correct = pred == true_label
        
        if is_correct:
            title = f"Pred: {pred_name} ({prob:.1%})"
            color = 'green'
        else:
            title = f"Pred: {pred_name} ({prob:.1%})\nActual: {true_name}"
            color = 'red'
        
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = output_dir / 'example_predictions_tensorrt.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Example images saved to: {fig_path}")


def print_metrics(y_true, y_pred, dataset_name):
    """Print evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Results on {dataset_name.upper()} set")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Cat   Dog")
    print(f"Actual Cat   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Actual Dog   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    total = cm.sum()
    errors = cm[0][1] + cm[1][0]
    print(f"\nTotal: {total} | Correct: {total - errors} | Errors: {errors}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm.tolist()}


def main():
    parser = argparse.ArgumentParser(description='Evaluate TensorRT engine')
    parser.add_argument('engine_path', help='Path to .engine file')
    parser.add_argument('--dataset', default='val', choices=['val', 'test'])
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    
    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        print(f"Error: Engine not found: {engine_path}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else engine_path.parent
    
    print("="*50)
    print("TensorRT Model Evaluation")
    print("="*50)
    print(f"Engine: {engine_path}")
    print(f"Dataset: {args.dataset}")
    
    # Load engine
    print("\nLoading TensorRT engine...")
    engine, context = load_engine(engine_path)
    
    # Load data
    print(f"Loading {args.dataset} data...")
    files, labels = load_eval_data(CONFIG, args.dataset)
    print(f"Images: {len(files)}")
    
    # Run evaluation
    print("\nRunning inference...")
    predictions, probabilities, ground_truth = run_evaluation(
        engine, context, files, labels, 
        CONFIG['data_dir'], CONFIG['image_size']
    )
    
    # Print metrics
    metrics = print_metrics(ground_truth, predictions, args.dataset)
    
    # Save predictions CSV
    csv_path = output_dir / f'{args.dataset}_predictions_tensorrt.csv'
    save_predictions_csv(files, predictions, probabilities, csv_path)
    
    # Save example images
    save_example_images(
        files, predictions, probabilities, ground_truth,
        CONFIG['data_dir'], output_dir, num_examples=8
    )
    
    # Save metrics JSON
    results = {
        'engine_path': str(engine_path),
        'dataset': args.dataset,
        'num_images': len(files),
        'metrics': metrics,
    }
    
    results_path = output_dir / f'{args.dataset}_evaluation_tensorrt.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {results_path}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == '__main__':
    main()