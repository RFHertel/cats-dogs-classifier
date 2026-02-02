"""
Run hyperparameter experiments.

Usage:
    python scripts/run_experiments.py

Estimated time: ~2 hours
Earlier with 2 workers 1 experiment finsied with elapsed 3.4 min  and a second finished with elapsed 6.2min

Results saved to outputs/experiments/
Run from project root directory: C:\AWrk\cats_dogs_project> 
"""


# Imports

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from multiprocessing import freeze_support

import random
import numpy as np

from train import (
    CatsDogsDataset, 
    Trainer, 
    create_model,
    IMAGENET_MEAN, 
    IMAGENET_STD
)

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Base Configuration

BASE_CONFIG = {
    'data_dir': Path('data/CleanPetImages'),
    'split_file': Path('outputs/splits/train_val_test_split.json'),
    'model_dir': Path('outputs/models'),
    'use_subset': True,
    'batch_size': 32,
    'num_workers': 2,
    'image_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Hyperparameter Grid
PARAM_GRID = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'scheduler': [None, 'cosine'],
    'augmentation': ['none', 'light', 'moderate', 'heavy'],
}

# Training settings
MAX_EPOCHS = 20
PATIENCE = 5  # Early stopping

# Augmentation Presets
def get_augmentation(name, image_size=224):
    """Get augmentation by preset name."""
    
    if name == 'none':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    elif name == 'light':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    elif name == 'moderate':
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    elif name == 'heavy':
        return A.Compose([
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),  # CORRECT
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation: {name}")


def get_val_transforms(image_size=224):
    """Validation transforms."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

# Generate Experiments
def generate_experiments(param_grid):
    """Generate all combinations from parameter grid."""
    keys = param_grid.keys()
    values = param_grid.values()
    
    experiments = []
    for combo in product(*values):
        exp = dict(zip(keys, combo))
        
        # Create experiment name
        lr_str = f"lr{exp['learning_rate']}"
        sched_str = exp['scheduler'] if exp['scheduler'] else 'nosched'
        aug_str = exp['augmentation']
        exp['name'] = f"{lr_str}_{sched_str}_{aug_str}"
        
        experiments.append(exp)
    
    return experiments

def test_dataloader_speed(config):
    """Test that parallel data loading is working."""
    print("Testing parallel data loading...")
    
    # Create a minimal dataset
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    val_transform = get_val_transforms(config['image_size'])
    val_dataset = CatsDogsDataset(
        split_data['val_files'], split_data['val_labels'],
        config['data_dir'], val_transform
    )
    
    # Test different num_workers
    for num_workers in [0, 2]:
        loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        
        batch_times = []
        for i, _ in enumerate(loader):
            start = time.time()
            if i > 0:  # Skip timing first batch in the list
                batch_times.append(time.time() - last_end)
            last_end = time.time()
        
        first_10 = sum(batch_times[:10]) / 10 * 1000 if len(batch_times) >= 10 else 0
        last_10 = sum(batch_times[-10:]) / 10 * 1000 if len(batch_times) >= 10 else 0
        avg_all = sum(batch_times) / len(batch_times) * 1000 if batch_times else 0
        
        print(f"  num_workers={num_workers}: "
              f"first 10={first_10:.0f}ms/batch, "
              f"last 10={last_10:.0f}ms/batch, "
              f"avg={avg_all:.0f}ms/batch")
        
        del loader
    
    del val_dataset
    print()

# Run Single Experiment
def run_experiment(exp_config, base_config, exp_num, total_exp):
    """Run a single experiment and return results."""
    
    name = exp_config['name']
    print(f"\n{'='*60}")
    print(f"[{exp_num}/{total_exp}] Experiment: {name}")
    print(f"{'='*60}")
    print(f"  Learning rate: {exp_config['learning_rate']}")
    print(f"  Scheduler: {exp_config['scheduler']}")
    print(f"  Augmentation: {exp_config['augmentation']}")
    print()
    
    # Merge configs
    config = {**base_config}
    config['learning_rate'] = exp_config['learning_rate']
    config['scheduler'] = exp_config['scheduler']
    config['epochs'] = MAX_EPOCHS
    
    # Load split data
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    train_files = split_data['train_files']
    train_labels = split_data['train_labels']
    
    if config['use_subset']:
        indices = split_data['subset_indices']
        train_files = [train_files[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
    
    # Create datasets
    train_transform = get_augmentation(exp_config['augmentation'], config['image_size'])
    val_transform = get_val_transforms(config['image_size'])
    
    train_dataset = CatsDogsDataset(
        train_files, train_labels, config['data_dir'], train_transform
    )
    val_dataset = CatsDogsDataset(
        split_data['val_files'], split_data['val_labels'],
        config['data_dir'], val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True, persistent_workers=True
    )
    
    # Create fresh model for each experiment
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(config['device'])
    
    # Train with early stopping
    trainer = Trainer(model, train_loader, val_loader, config)
    start_time = time.time()
    history = trainer.fit(config['epochs'], patience=PATIENCE)
    train_time = time.time() - start_time
    
    # Save model for this experiment
    trainer.save_model(f"{name}_best.pth")
    
    # Get extended metrics on best model
    trainer.model.load_state_dict(torch.load(config['model_dir'] / f"{name}_best.pth"))
    _, val_acc, precision, recall, f1 = trainer.validate(return_extended_metrics=True)
    
    return {
        'name': name,
        'config': {
            'learning_rate': exp_config['learning_rate'],
            'scheduler': exp_config['scheduler'],
            'augmentation': exp_config['augmentation'],
        },
        'best_val_acc': trainer.best_val_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'epochs_trained': len(history['train_loss']),
        'train_time_seconds': train_time,
        'history': history,
    }

# Save Results
def save_results(results, output_dir):
    """Save full results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results with complete history per experiment
    full_results = []
    for r in results:
        full_results.append({
            'name': r['name'],
            'config': r['config'],
            'best_val_acc': r['best_val_acc'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'epochs_trained': r['epochs_trained'],
            'train_time_seconds': round(r['train_time_seconds'], 1),
            'history': {
                'train_loss': r['history']['train_loss'],
                'train_acc': r['history']['train_acc'],
                'val_loss': r['history']['val_loss'],
                'val_acc': r['history']['val_acc'],
                'lr': r['history']['lr'],
            }
        })
    
    full_file = output_dir / f'full_results_{timestamp}.json'
    with open(full_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # Save summary (quick reference)
    summary = []
    for r in results:
        summary.append({
            'name': r['name'],
            'learning_rate': r['config']['learning_rate'],
            'scheduler': r['config']['scheduler'],
            'augmentation': r['config']['augmentation'],
            'best_val_acc': r['best_val_acc'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'epochs_trained': r['epochs_trained'],
            'train_time_minutes': round(r['train_time_seconds'] / 60, 1),
        })
    
    summary_file = output_dir / f'summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return full_file, summary_file


def save_checkpoint(results, output_dir):
    """Save intermediate results after each experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = output_dir / 'checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump([{
            'name': r['name'],
            'config': r['config'],
            'best_val_acc': r['best_val_acc'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'epochs_trained': r['epochs_trained'],
            'train_time_seconds': round(r['train_time_seconds'], 1),
            'history': r['history'],
        } for r in results], f, indent=2)

# Main
def main():
    print("=" * 60)
    set_seed(SEED)
    print("=" * 60)
    print("Hyperparameter Experiments")
    print("=" * 60)
    
    # Verify GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Aborting.")
        print("This script requires GPU for reasonable training times.")
        return
    
    device = BASE_CONFIG['device']
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Test dataloader before starting experiments
    test_dataloader_speed(BASE_CONFIG)
    
    print(f"Max epochs per experiment: {MAX_EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    print()
    
    # Run all experiments
    results = []
    total_start = time.time()
    experiments = generate_experiments(PARAM_GRID)
    
    for i, exp in enumerate(experiments):
        result = run_experiment(exp, BASE_CONFIG, i + 1, len(experiments))
        results.append(result)
        save_checkpoint(results, 'outputs/experiments')
        
        # Print running summary
        elapsed = (time.time() - total_start) / 60
        remaining = elapsed / (i + 1) * (len(experiments) - i - 1)
        print(f"\nProgress: {i+1}/{len(experiments)} | "
              f"Elapsed: {elapsed:.1f}min | "
              f"Remaining: ~{remaining:.1f}min")
    
    # Save results
    full_file, summary_file = save_results(results, 'outputs/experiments')
    
    # Print final summary
    total_time = (time.time() - total_start) / 60
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} minutes\n")
    
    # Sort by best val accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"{'Experiment':<35} {'Val Acc':<10} {'F1':<10} {'Epochs':<8}")
    print("-" * 65)
    for r in results[:10]:  # Top 10
        print(f"{r['name']:<35} {r['best_val_acc']:.4f}     {r['f1']:.4f}     {r['epochs_trained']}")
    
    print(f"\nFull results: {full_file}")
    print(f"Summary: {summary_file}")
    print(f"\nBest: {results[0]['name']} ({results[0]['best_val_acc']:.4f})")


if __name__ == '__main__':
    freeze_support()
    main()

