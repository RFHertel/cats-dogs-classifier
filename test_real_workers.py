# test_persistent_workers.py
import time
import json
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("Main script starting...")


class CatsDogsDataset(Dataset):
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


def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def benchmark_loader(loader, num_epochs=3, batches_per_epoch=10):
    """Run multiple epochs and return timing stats."""
    first_batch_times = []
    batch_times = []
    
    for epoch in range(num_epochs):
        epoch_batch_times = []
        start = time.time()
        
        for i, (images, labels) in enumerate(loader):
            if i == 0:
                first_batch_times.append(time.time() - start)
                start = time.time()
            else:
                epoch_batch_times.append(time.time() - start)
                start = time.time()
            
            if i >= batches_per_epoch:
                break
        
        batch_times.extend(epoch_batch_times)
    
    return {
        'first_batch_avg': sum(first_batch_times) / len(first_batch_times),
        'first_batch_all': first_batch_times,
        'batch_avg_ms': (sum(batch_times) / len(batch_times)) * 1000,
        'batch_std_ms': (sum((t - sum(batch_times)/len(batch_times))**2 for t in batch_times) / len(batch_times)) ** 0.5 * 1000,
    }


if __name__ == '__main__':
    # Load split
    with open('outputs/splits/train_val_test_split.json', 'r') as f:
        split_data = json.load(f)
    
    # Use subset
    subset_indices = split_data['subset_indices'][:500]
    train_files = [split_data['train_files'][i] for i in subset_indices]
    train_labels = [split_data['train_labels'][i] for i in subset_indices]
    
    print(f"Testing with {len(train_files)} REAL images")
    print(f"Dataset: data/CleanPetImages\n")
    
    data_dir = Path('data/CleanPetImages')
    transform = get_transforms()
    dataset = CatsDogsDataset(train_files, train_labels, data_dir, transform)
    
    NUM_EPOCHS = 3
    BATCHES_PER_EPOCH = 10
    
    print(f"Config: {NUM_EPOCHS} epochs, {BATCHES_PER_EPOCH} batches per epoch")
    print("=" * 70)
    
    # TEST 1: num_workers=0 baseline
    print("\nTEST 1: num_workers=0 (baseline - no multiprocessing)")
    print("-" * 70)
    
    loader_0 = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    stats_0 = benchmark_loader(loader_0, NUM_EPOCHS, BATCHES_PER_EPOCH)
    print(f"  First batch avg:    {stats_0['first_batch_avg']:.3f}s  (per epoch: {stats_0['first_batch_all']})")
    print(f"  Batch avg:          {stats_0['batch_avg_ms']:.1f}ms ± {stats_0['batch_std_ms']:.1f}ms")
    
    # TEST 2: num_workers=2, persistent_workers=True
    print("\nTEST 2: num_workers=2, persistent_workers=True")
    print("-" * 70)
    
    loader_2 = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True,
    )
    
    print("  Spawning 2 workers (one-time cost)...")
    stats_2 = benchmark_loader(loader_2, NUM_EPOCHS, BATCHES_PER_EPOCH)
    print(f"  First batch avg:    {stats_2['first_batch_avg']:.3f}s  (per epoch: {[f'{t:.2f}' for t in stats_2['first_batch_all']]})")
    print(f"  Batch avg:          {stats_2['batch_avg_ms']:.1f}ms ± {stats_2['batch_std_ms']:.1f}ms")
    
    del loader_2  # Clean up workers
    
    # TEST 3: num_workers=4, persistent_workers=True
    print("\nTEST 3: num_workers=4, persistent_workers=True")
    print("-" * 70)
    
    loader_4 = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
    )
    
    print("  Spawning 4 workers (one-time cost)...")
    stats_4 = benchmark_loader(loader_4, NUM_EPOCHS, BATCHES_PER_EPOCH)
    print(f"  First batch avg:    {stats_4['first_batch_avg']:.3f}s  (per epoch: {[f'{t:.2f}' for t in stats_4['first_batch_all']]})")
    print(f"  Batch avg:          {stats_4['batch_avg_ms']:.1f}ms ± {stats_4['batch_std_ms']:.1f}ms")
    
    del loader_4  # Clean up workers
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'Batch Avg':>12} {'Speedup vs 0':>15}")
    print("-" * 70)
    print(f"{'num_workers=0 (baseline)':<35} {stats_0['batch_avg_ms']:>9.1f}ms {1.0:>14.2f}x")
    print(f"{'num_workers=2, persistent':<35} {stats_2['batch_avg_ms']:>9.1f}ms {stats_0['batch_avg_ms']/stats_2['batch_avg_ms']:>14.2f}x")
    print(f"{'num_workers=4, persistent':<35} {stats_4['batch_avg_ms']:>9.1f}ms {stats_0['batch_avg_ms']/stats_4['batch_avg_ms']:>14.2f}x")
    print("-" * 70)
    print(f"{'4 workers vs 2 workers:':<35} {stats_2['batch_avg_ms']/stats_4['batch_avg_ms']:>14.2f}x faster")
    print("=" * 70)
