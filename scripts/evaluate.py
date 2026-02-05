"""
Evaluate a trained model on validation or test set.

Usage:
    python scripts/evaluate.py <model_path> [options]

Examples:
    Validation MobilenetV2: python scripts/evaluate.py outputs/final/final_model.pth # This use the default config for mobilenet for the val set
    Test MobilenetV2: python scripts/evaluate.py outputs/final/final_model.pth --dataset test
    Validation ResNet18: python scripts/evaluate.py outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18
    Test ResNet18: python scripts/evaluate.py outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18 --dataset test

Run from project root directory: C:\AWrk\cats_dogs_project> 
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import argparse
import json
import csv
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from train import CatsDogsDataset, IMAGENET_MEAN, IMAGENET_STD

import sys

def silent_imread(path):
    """Load image while suppressing libjpeg warnings."""
    fd = sys.stderr.fileno()
    old_stderr = os.dup(fd)
    os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
    img = cv2.imread(str(path))
    os.dup2(old_stderr, fd)
    os.close(old_stderr)
    return img

# Config
CONFIG = {
    'data_dir': Path('data/CleanPetImages'),
    'split_file': Path('outputs/splits/train_val_test_split.json'),
    'batch_size': 32,
    'num_workers': 2,
    'image_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_model(model_path, model_type='mobilenet'):
    """Load model from .pth file."""
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
    return model


def get_transforms(image_size=224):
    """Standard eval transforms - no augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_eval_data(config, dataset='val'):
    """Load val or test data."""
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    if dataset == 'val':
        files = split_data['val_files']
        labels = split_data['val_labels']
    elif dataset == 'test':
        files = split_data['test_files']
        labels = split_data['test_labels']
    else:
        raise ValueError(f"dataset must be 'val' or 'test', got {dataset}")
    
    ds = CatsDogsDataset(files, labels, config['data_dir'], get_transforms(config['image_size']))
    
    loader = DataLoader(
        ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    
    return loader, files, labels


def run_evaluation(model, loader, device):
    """Run model on all batches, collect predictions."""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def save_predictions_csv(files, predictions, probabilities, output_path):
    """
    Save predictions to CSV file.
    Format: filename, predicted_label, probability
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Cat', 'Dog']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'predicted_label', 'probability'])
        
        for filepath, pred, prob in zip(files, predictions, probabilities):
            label = class_names[pred]
            confidence = prob[pred]  # probability of predicted class
            writer.writerow([filepath, label, f"{confidence:.4f}"])
    
    print(f"Predictions saved to: {output_path}")


def save_example_images(files, predictions, probabilities, labels, data_dir, output_dir, num_examples=8):
    """
    Save example images with predictions overlaid.
    Shows both correct and incorrect predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['Cat', 'Dog']
    
    # Find correct and incorrect predictions
    correct_idx = [i for i in range(len(predictions)) if predictions[i] == labels[i]]
    incorrect_idx = [i for i in range(len(predictions)) if predictions[i] != labels[i]]
    
    # Pick examples: half correct, half incorrect (if available)
    n_correct = min(num_examples // 2, len(correct_idx))
    n_incorrect = min(num_examples - n_correct, len(incorrect_idx))
    
    # Random sample
    np.random.seed(42)
    sample_correct = np.random.choice(correct_idx, n_correct, replace=False) if n_correct > 0 else []
    sample_incorrect = np.random.choice(incorrect_idx, n_incorrect, replace=False) if n_incorrect > 0 else []
    
    sample_indices = list(sample_correct) + list(sample_incorrect)
    
    # Create figure
    n_cols = 4
    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(sample_indices):
            ax.axis('off')
            continue
        
        idx = sample_indices[i]
        filepath = files[idx]
        pred = predictions[idx]
        true_label = labels[idx]
        prob = probabilities[idx][pred]
        
        # Load original image
        img_path = Path(data_dir) / filepath
        img = silent_imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        
        # Title with prediction info
        pred_name = class_names[pred]
        true_name = class_names[true_label]
        is_correct = pred == true_label
        
        if is_correct:
            title = f"✓ Pred: {pred_name} ({prob:.1%})"
            color = 'green'
        else:
            title = f"✗ Pred: {pred_name} ({prob:.1%})\nActual: {true_name}"
            color = 'red'
        
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'example_predictions.png'
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
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('model_path', help='Path to .pth model file')
    parser.add_argument('--dataset', default='val', choices=['val', 'test'],
                        help='Which dataset to evaluate on')
    parser.add_argument('--model-type', default='mobilenet', choices=['mobilenet', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save results (default: same as model)')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    # Output directory - default to model's directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent
    
    print("="*50)
    print("Model Evaluation")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {CONFIG['device']}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(model_path, args.model_type)
    model = model.to(CONFIG['device'])
    
    # Load data
    print(f"Loading {args.dataset} data...")
    loader, files, labels = load_eval_data(CONFIG, args.dataset)
    print(f"Images: {len(files)}")
    
    # Run evaluation
    print("Running inference...")
    predictions, probabilities, ground_truth = run_evaluation(model, loader, CONFIG['device'])
    
    # Print metrics
    metrics = print_metrics(ground_truth, predictions, args.dataset)
    
    # Save predictions to CSV (assignment requirement)
    csv_path = output_dir / f'{args.dataset}_predictions.csv'
    save_predictions_csv(files, predictions, probabilities, csv_path)
    
    # Save example images (assignment requirement)
    save_example_images(
        files, predictions, probabilities, ground_truth,
        CONFIG['data_dir'], output_dir, num_examples=8
    )
    
    # Save metrics JSON
    results = {
        'model_path': str(model_path),
        'model_type': args.model_type,
        'dataset': args.dataset,
        'num_images': len(files),
        'metrics': metrics,
    }
    
    results_path = output_dir / f'{args.dataset}_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {results_path}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == '__main__':
    main()

