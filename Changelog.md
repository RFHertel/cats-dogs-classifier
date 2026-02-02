# Changelog

## [Unreleased]

### 2026-02-01 - Data Cleaning Complete  
- Completed CLIP-based filtering with V3+ algorithm
- Created CleanPetImages dataset: 24,924 images (12,456 cats, 12,468 dogs)
- Excluded 76 problematic images (corrupted, duplicates, non-photos)

### 2026-02-01 - Data Preprocessing

**Configuration and file loading**
- Created 03_data_preprocessing.ipynb
- Added config dict to centralize all settings (paths, batch size, split ratios)
- Loaded file paths from CleanPetImages: 24,924 images (12,456 cats, 12,468 dogs)

**Train/val/test split**
- Split dataset 80/10/10 using stratified sampling to maintain class balance
- Created iteration subset: 4,000 images from training set for fast experiments
- Saved splits to JSON for reproducibility across notebooks
- Added scikit-learn and tqdm to requirements.txt

**Dataset class**
- Created CatsDogsDataset class following PyTorch Dataset pattern
- Used OpenCV for image loading (faster than PIL)
- Designed for Albumentations transform compatibility

**Image transforms (Albumentations)**
- Chose Albumentations over torchvision for speed (numpy-based vs PIL-based)
- Training augmentation: HorizontalFlip, small rotation (±10°), slight color jitter
- Augmentation kept simple intentionally - aggressive augmentation adds complexity with diminishing returns for binary classification
- Validation/test: resize and normalize only (no augmentation for consistent evaluation)
- ImageNet normalization required for pretrained model compatibility
- Added visual verification showing augmentation effects

**Verification**
- Added batch visualization to verify transforms working correctly
- Added preprocessing summary with dataset statistics
- Documented that train.py will use parallel workers for faster loading

**Data loading and config**
- Created scripts/train.py for training with parallel data loading
- num_workers=2 with persistent_workers for 2.5x faster loading
- Configurable subset mode for fast iteration

**Model setup**
- MobileNetV2 with pretrained ImageNet weights
- Replaced classifier head for binary classification
- 2.2M parameters

**Trainer class**
- Encapsulates training and validation logic
- train_epoch() and validate() methods
- fit() method runs full training

**Saving and history**
- Tracks loss and accuracy per epoch
- Saves best model by validation accuracy
- Saves final model after training
- Models saved to outputs/models/

**train.py - Core training module**
- CatsDogsDataset class with OpenCV loading for Albumentations compatibility
- Trainer class encapsulating training loop, validation, and model saving
- MobileNetV2 with pretrained ImageNet weights, custom classifier head
- Early stopping with configurable patience
- Optional learning rate schedulers (step, cosine annealing)
- Extended metrics: precision, recall, F1 score via sklearn
- History tracking: per-epoch loss, accuracy, learning rate
- Parallel data loading: num_workers=2, persistent_workers=True
- Achieved 2.5x speedup (29ms vs 73ms per batch) over sequential loading

**run_experiments.py - Hyperparameter search**
- Imports core components from train.py (DRY principle)
- Grid search: 4 learning rates × 2 schedulers × 4 augmentation levels = 32 experiments
- Augmentation presets: none, light, moderate, heavy
- 32 experiments completed in 145 minutes
- Best result: lr=0.0001, cosine scheduler, light augmentation (98.96% val acc)
- Checkpoint saving after each experiment for crash recovery
- Full training history saved per experiment
- GPU verification fails fast if CUDA unavailable
- Reproducibility via set_seed() for deterministic results

**train_final.py - Final model training**
- Uses best hyperparameters from experiments (lr=0.0001, cosine scheduler, light augmentation)
- Trains on full dataset (20k images vs 4k subset)
- 30 epochs max with patience=7 early stopping
- Achieved 99.20% validation accuracy in 21 minutes (15 epochs)
- Final metrics: Precision 0.9944, Recall 0.9896, F1 0.9920
- Output: final_model.pth + final_training_results.json

**train_final_resnet18.py - Architecture comparison**
- ResNet18 training using best hyperparameters from MobileNetV2 experiments
- Tests whether larger model (11M params) outperforms smaller (2.2M params)
- Results: 99.00% val accuracy, F1 0.9899 in 12 minutes (early stopping at epoch 10)
- Slightly below MobileNetV2 (99.20%) despite 5x more parameters
- Note: 0.2% difference on ~2,500 images ≈ 5 images—within statistical noise
- Conclusion: Hyperparameters tuned for one architecture don't fully transfer
- ResNet18 likely needs adjusted LR, scheduler warmup, or stronger regularization
- MobileNetV2 preferred for this task (faster inference, smaller, equal/better accuracy)
- ResNet18 kept for potential ensemble use

**evaluate.py - Model evaluation**
- Evaluates any saved .pth model on validation or test set
- Supports MobileNetV2 and ResNet18 architectures
- Outputs predictions CSV (filename, label, probability) per assignment spec
- Saves example images showing correct and incorrect predictions
- Reports accuracy, precision, recall, F1, confusion matrix
- Uses parallel data loading for efficient batch processing
- Output directory defaults to model's parent directory