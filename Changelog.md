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