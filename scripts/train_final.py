"""
Final model training on full dataset.

Uses best hyperparameters from experiments:
- Learning rate: 0.0001
- Scheduler: cosine
- Augmentation: light (horizontal flip only)

Usage:
    python scripts/train_final.py

Run from project root directory: C:\AWrk\cats_dogs_project> 
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from train import (
    CatsDogsDataset,
    Trainer,
    create_model,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# Configuration - Best from experiments
config = {
    # Paths
    'data_dir': Path('data/CleanPetImages'),
    'split_file': Path('outputs/splits/train_val_test_split.json'),
    'model_dir': Path('outputs/final'),
    'output_dir': Path('outputs/final'),
    
    # Best hyperparameters from experiments
    'learning_rate': 0.0001,
    'scheduler': 'cosine',
    'augmentation': 'light',
    
    # Training settings
    'use_subset': False,  # FULL DATASET
    'batch_size': 32,
    'num_workers': 2,
    'epochs': 30,
    'patience': 7,
    
    # Image settings
    'image_size': 224,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# Augmentation (matching experiment presets)
def get_train_transforms(image_size=224):
    """Light augmentation - best from experiments."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        ToTensorV2(),
    ])


def get_val_transforms(image_size=224):
    """Validation transforms - no augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# Main
def main():
    print("=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Device info
    print(f"Device: {config['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # Print config
    print("Configuration:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Scheduler: {config['scheduler']}")
    print(f"  Augmentation: {config['augmentation']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Max epochs: {config['epochs']}")
    print(f"  Early stopping patience: {config['patience']}")
    print(f"  Full dataset: {not config['use_subset']}")
    print()
    
    # Load split data
    print("Loading data...")
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    train_files = split_data['train_files']
    train_labels = split_data['train_labels']
    
    # Create datasets
    train_transform = get_train_transforms(config['image_size'])
    val_transform = get_val_transforms(config['image_size'])
    
    train_dataset = CatsDogsDataset(
        train_files, train_labels, config['data_dir'], train_transform
    )
    val_dataset = CatsDogsDataset(
        split_data['val_files'], split_data['val_labels'],
        config['data_dir'], val_transform
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(config['device'])
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: MobileNetV2 ({param_count:,} parameters)")
    print()
    
    # Train
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    start_time = time.time()
    history = trainer.fit(config['epochs'], patience=config['patience'])
    train_time = time.time() - start_time
    
    # Get final metrics with extended evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    best_model_path = config['model_dir'] / 'best_model.pth'
    model.load_state_dict(torch.load(best_model_path))
    trainer.model = model
    
    val_loss, val_acc, precision, recall, f1 = trainer.validate(return_extended_metrics=True)
    
    print(f"\nBest model metrics (on validation set):")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Loss:      {val_loss:.4f}")
    
    # Save results
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    results = {
        'config': {
            'learning_rate': config['learning_rate'],
            'scheduler': config['scheduler'],
            'augmentation': config['augmentation'],
            'batch_size': config['batch_size'],
            'epochs_max': config['epochs'],
            'patience': config['patience'],
            'image_size': config['image_size'],
        },
        'dataset': {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
        },
        'results': {
            'best_val_acc': trainer.best_val_acc,
            'final_val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'epochs_trained': len(history['train_loss']),
            'train_time_minutes': round(train_time / 60, 2),
        },
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }
    
    results_file = config['output_dir'] / 'final_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Copy best model to final output
    import shutil
    final_model_path = config['output_dir'] / 'final_model.pth'
    shutil.copy(best_model_path, final_model_path)
    
    print(f"\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {train_time / 60:.1f} minutes")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"\nOutputs saved to {config['output_dir']}/")
    print(f"  - final_model.pth")
    print(f"  - final_training_results.json")


if __name__ == '__main__':
    main()

