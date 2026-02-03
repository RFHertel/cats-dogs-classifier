"""
Active Learning - Entropy + Diversity Sampling

Uses entropy-based uncertainty with k-means clustering for diversity.

Usage:
    python scripts/train_final_al_entropy_diversity.py

Run from project root directory: C:\AWrk\cats_dogs_project> 
"""

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import json
import time
import random
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

from train import (
    CatsDogsDataset,
    create_model,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# =============================================================================
# Configuration
# =============================================================================

STRATEGY = 'entropy_diversity'

config = {
    # Paths
    'data_dir': Path('data/CleanPetImages'),
    'split_file': Path('outputs/splits/train_val_test_split.json'),
    'output_dir': Path('outputs/final_active_learning_entropy'),  # Same folder as entropy
    
    # Active learning settings
    'initial_pool_size': 2000,
    'samples_per_round': 1000,
    'max_rounds': 18,
    
    # Diversity settings
    'diversity_pool_multiplier': 2,
    
    # Training settings (per round)
    'learning_rate': 0.0001,
    'scheduler': 'cosine',
    'batch_size': 32,
    'num_workers': 0,  # Set to 0 for Windows stability, 2 for speed on Linux
    'persistent_workers': False,  # Set to True if num_workers > 0 for speed
    'epochs_per_round': 10,
    'patience': 3,
    
    # Image settings
    'image_size': 224,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Reproducibility
    'seed': 42,
}


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Augmentation
# =============================================================================

def get_train_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# =============================================================================
# Feature Extraction and Scoring
# =============================================================================

def extract_features(model, dataloader, device):
    """
    Extract features from the penultimate layer of MobileNetV2.
    """
    model.eval()
    features = []
    
    activation = {}
    def hook_fn(module, input, output):
        activation['features'] = output.detach()
    
    handle = model.classifier[0].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            features.append(activation['features'].cpu().numpy())
    
    handle.remove()
    
    return np.vstack(features)


def compute_entropy(model, dataloader, device):
    """
    Entropy sampling: H(p) = -sum(p * log(p))
    Higher entropy = more uncertain
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            eps = 1e-10
            probs = torch.clamp(probs, eps, 1 - eps)
            
            entropy = -torch.sum(probs * torch.log(probs), dim=1)
            entropies.extend(entropy.cpu().numpy())
    
    return np.array(entropies)


def select_with_diversity(scores, features, n_select, pool_multiplier=2):
    """
    Select diverse samples from the highest-scoring ones.
    """
    candidate_size = min(n_select * pool_multiplier, len(scores))
    candidate_indices = np.argsort(scores)[::-1][:candidate_size]
    
    candidate_features = features[candidate_indices]
    candidate_scores = scores[candidate_indices]
    
    n_clusters = min(n_select, len(candidate_indices))
    
    if n_clusters < n_select:
        return candidate_indices.tolist()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(candidate_features)
    
    selected = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_local_indices = np.where(cluster_mask)[0]
        
        cluster_scores = candidate_scores[cluster_local_indices]
        best_in_cluster = cluster_local_indices[np.argmax(cluster_scores)]
        
        selected.append(candidate_indices[best_in_cluster])
    
    return selected


# =============================================================================
# Training Function
# =============================================================================

def train_model(train_loader, val_loader, config):
    """Train a fresh model and return best validation accuracy."""
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(config['device'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs_per_round']
        )
    else:
        scheduler = None
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config['epochs_per_round']):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config['device'])
                labels = labels.to(config['device'])
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / total
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if scheduler:
            scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= config['patience']:
            break
    
    model.load_state_dict(best_model_state)
    
    return best_val_acc, model, history


def evaluate_model(model, dataloader, device):
    """Get full metrics on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# =============================================================================
# Active Learning Loop
# =============================================================================

def run_active_learning(train_dataset, val_loader, config):
    """Run active learning with entropy + diversity sampling."""
    print(f"\n{'='*60}")
    print(f"Strategy: ENTROPY + DIVERSITY")
    print(f"{'='*60}")
    
    n_total = len(train_dataset)
    
    all_indices = list(range(n_total))
    random.shuffle(all_indices)
    
    labeled_indices = all_indices[:config['initial_pool_size']]
    unlabeled_indices = all_indices[config['initial_pool_size']:]
    
    results = []
    final_model = None
    
    for round_num in range(config['max_rounds']):
        print(f"\n--- Round {round_num + 1} ---")
        print(f"Labeled: {len(labeled_indices)} | Unlabeled: {len(unlabeled_indices)}")
        
        labeled_subset = Subset(train_dataset, labeled_indices)
        train_loader = DataLoader(
            labeled_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=config['persistent_workers'] and config['num_workers'] > 0,
        )
        
        print("  Training...")
        start_time = time.time()
        val_acc, model, history = train_model(train_loader, val_loader, config)
        train_time = time.time() - start_time
        
        final_model = model
        
        metrics = evaluate_model(model, val_loader, config['device'])
        
        print(f"  Val Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
              f"Time: {train_time:.1f}s")
        
        results.append({
            'round': round_num + 1,
            'labeled_size': len(labeled_indices),
            'val_acc': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'train_time': train_time,
            'epochs_trained': len(history['train_loss']),
        })
        
        # Checkpoint after each round
        checkpoint = {
            'strategy': STRATEGY,
            'results': results,
            'current_round': round_num + 1,
            'timestamp': datetime.now().isoformat(),
        }
        checkpoint_file = config['output_dir'] / f'checkpoint_{STRATEGY}.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  Checkpoint saved: round {round_num + 1}")
        
        if len(unlabeled_indices) == 0:
            print("  Unlabeled pool exhausted. Stopping.")
            break
        
        # Compute entropy and features
        print("  Computing entropy and features...")
        unlabeled_subset = Subset(train_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(
            unlabeled_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=config['persistent_workers'] and config['num_workers'] > 0,
        )
        
        entropies = compute_entropy(model, unlabeled_loader, config['device'])
        features = extract_features(model, unlabeled_loader, config['device'])
        
        # Select with diversity
        print("  Selecting diverse samples...")
        n_to_add = min(config['samples_per_round'], len(unlabeled_indices))
        selected_local = select_with_diversity(
            entropies, features, n_to_add,
            pool_multiplier=config['diversity_pool_multiplier']
        )
        selected_indices = [unlabeled_indices[i] for i in selected_local]
        
        # Clear memory
        del model, unlabeled_loader, entropies, features
        torch.cuda.empty_cache()
        
        labeled_indices.extend(selected_indices)
        unlabeled_indices = [i for i in unlabeled_indices if i not in set(selected_indices)]
    
    return results, final_model


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print(f"ACTIVE LEARNING - {STRATEGY.upper()}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    set_seed(config['seed'])
    
    print(f"Device: {config['device']}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    with open(config['split_file'], 'r') as f:
        split_data = json.load(f)
    
    train_files = split_data['train_files']
    train_labels = split_data['train_labels']
    
    train_transform = get_train_transforms(config['image_size'])
    val_transform = get_val_transforms(config['image_size'])
    
    train_dataset = CatsDogsDataset(
        train_files, train_labels, config['data_dir'], train_transform
    )
    
    val_dataset = CatsDogsDataset(
        split_data['val_files'], split_data['val_labels'],
        config['data_dir'], val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=config['persistent_workers'] and config['num_workers'] > 0,
    )
    
    print(f"Total training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    
    total_start = time.time()
    
    results, model = run_active_learning(train_dataset, val_loader, config)
    
    total_time = time.time() - total_start
    
    # Save model
    model_path = config['output_dir'] / f'final_model_{STRATEGY}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved: {model_path}")
    
    # Save results
    results_data = {
        'strategy': STRATEGY,
        'config': {
            'initial_pool_size': config['initial_pool_size'],
            'samples_per_round': config['samples_per_round'],
            'max_rounds': config['max_rounds'],
            'epochs_per_round': config['epochs_per_round'],
            'patience': config['patience'],
            'learning_rate': config['learning_rate'],
            'diversity_pool_multiplier': config['diversity_pool_multiplier'],
            'seed': config['seed'],
        },
        'results': results,
        'total_time_minutes': round(total_time / 60, 2),
        'timestamp': datetime.now().isoformat(),
    }
    
    results_file = config['output_dir'] / f'results_{STRATEGY}.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Labeled':<12} {'Val Acc':<12} {'F1':<12}")
    print("-" * 36)
    for r in results:
        print(f"{r['labeled_size']:<12} {r['val_acc']:<12.4f} {r['f1']:<12.4f}")
    
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Results: {results_file}")


if __name__ == '__main__':
    main()