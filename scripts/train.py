"""
Train cats vs dogs classifier.

Usage:
    python scripts/train.py

Uses parallel data loading (num_workers=2) for speed.
Run from project root directory.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# Configuration
# =============================================================================

config = {
    # Paths (relative to project root)
    'data_dir': Path('data/CleanPetImages'),
    'split_file': Path('outputs/splits/train_val_test_split.json'),
    'model_dir': Path('outputs/models'),
    
    # Training settings
    'use_subset': True,      # True for fast iteration, False for full training
    'batch_size': 32,
    'num_workers': 2,        # Parallel loading
    'epochs': 10,
    'learning_rate': 0.001,
    
    # Image settings
    'image_size': 224,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# =============================================================================
# Dataset
# =============================================================================

class CatsDogsDataset(Dataset):
    """Dataset for cats vs dogs classification."""
    
    def __init__(self, file_list, labels, data_dir, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.data_dir = Path(data_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        rel_path = self.file_list[index]
        label = self.labels[index]
        
        img_path = str(self.data_dir / rel_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


# =============================================================================
# Transforms
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=224):
    """Training transforms with augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=224):
    """Validation transforms. No augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =============================================================================
# Data Loading
# =============================================================================

def load_data(config):
    """Load split data and create dataloaders."""
    
    # Load split
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    # Get training files
    train_files = split_data['train_files']
    train_labels = split_data['train_labels']
    
    # Use subset if configured
    if config['use_subset']:
        subset_indices = split_data['subset_indices']
        train_files = [train_files[i] for i in subset_indices]
        train_labels = [train_labels[i] for i in subset_indices]
    
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
    
    # Create dataloaders with parallel loading
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
    
    return train_loader, val_loader

# =============================================================================
# Model
# =============================================================================

def create_model(num_classes=2, pretrained=True):
    """Create MobileNetV2 with pretrained weights."""
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)
    
    # Replace classifier head for our task
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Handles model training and validation."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config['device']
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate']
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / total, correct / total
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / total, correct / total
    
    def fit(self, epochs):
        """Train for multiple epochs."""
        print(f"Training for {epochs} epochs...")
        print("-" * 60)
        
        for epoch in range(epochs):
            start = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            elapsed = time.time() - start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {elapsed:.1f}s")
        
        print("-" * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("Cats vs Dogs Training")
    print("=" * 60)
    print()
    
    print(f"Device: {config['device']}")
    print(f"Subset mode: {config['use_subset']}")
    print(f"Batch size: {config['batch_size']}")
    print()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = load_data(config)
    print(f"Train: {len(train_loader.dataset)} images")
    print(f"Val: {len(val_loader.dataset)} images")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(config['device'])
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: MobileNetV2 ({param_count:,} parameters)")
    print()
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit(config['epochs'])