# Cats vs Dogs Classifier

Binary image classifier using transfer learning with PyTorch. Trained on the Microsoft Cats vs Dogs dataset (~25,000 images) with MobileNetV2 and ResNet18 architectures. Includes model export to ONNX and TensorRT for deployment, with a native C++ inference application.

Built on Windows with an RTX 3050 Laptop GPU (4GB VRAM).


## Setup

```
conda create -n catdog python=3.10 -y
conda activate catdog

# PyTorch with CUDA (install first, separate from requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Everything else
pip install -r requirements.txt
```

Verify GPU:
```
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # should show your GPU
```

For TensorRT and C++ setup, see INSTALL.md.


## Dataset

Download from: https://www.kaggle.com/competitions/dogs-vs-cats/data

Extract to data/PetImages/. The raw dataset has ~25,000 images across Cat/ and Dog/ folders.

Data cleaning (see notebook 02) uses CLIP zero-shot classification to automatically filter out artwork, cartoons, text overlays, and corrupted files. This reduces the dataset to 24,924 clean images saved to data/CleanPetImages/. The split is 80/10/10 train/val/test, stratified.


## Project Structure

```
cats_dogs_project/

    data/
        PetImages/                          Raw dataset
        CleanPetImages/                     Cleaned dataset
            Cat/
            Dog/

    notebooks/
        01_explore_data.ipynb               Dataset exploration and statistics
        02_data_preprocessing.ipynb         CLIP filtering and train/val/test splits
        03_train_experiments.ipynb          Hyperparameter experiment analysis
        04_results_analysis.ipynb           Final results and model comparison

    scripts/
        train.py                            Core training module (dataset, trainer, model)
        run_experiments.py                  Hyperparameter grid search (32 experiments)
        train_final.py                      Final MobileNetV2 training
        train_final_resnet18.py             Final ResNet18 training
        evaluate.py                         Evaluate PyTorch models on val/test set
        evaluate_tensorrt.py                Evaluate TensorRT engines on val/test set
        export_model.py                     Export to ONNX and TensorRT
        inference.py                        Single image classification
        benchmark_inference.py              Inference speed comparison across formats

    cpp_inference/
        CMakeLists.txt
        main.cpp                            Native C++ TensorRT inference


    outputs/
        splits/                             Train/val/test split files
        experiments/                        Hyperparameter search results
        final/                              MobileNetV2 model and exports
        final_resnet18/                     ResNet18 model and exports

    requirements.txt
    INSTALL.md
    README.md
```


## Training Pipeline

### 1. Data Exploration

```
jupyter notebook notebooks/01_explore_data.ipynb
```

Explores the raw dataset -- class balance, image sizes, broken files, that sort of thing.

### 2. Data Preprocessing

```
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

Uses CLIP (clip-vit-large-patch14-336) to classify images as real photos vs artwork/cartoons/text. Combined with Laplacian blur detection to handle edge cases. Removed 76 problematic images from the dataset.

### 3. Hyperparameter Search

```
python scripts/run_experiments.py
```

Grid search over 32 combinations:
- Learning rates: 0.0001, 0.0005, 0.001, 0.005
- Schedulers: cosine, none
- Augmentation: none, light, moderate, heavy

Runs on a 4,000 image subset for speed (~2.5 hours total). Results analyzed in notebook 03.

### 4. Final Training

```
# MobileNetV2 with best hyperparameters
python scripts/train_final.py

# ResNet18 for architecture comparison
python scripts/train_final_resnet18.py
```

Best hyperparameters from grid search: lr=0.0001, cosine scheduler, light augmentation (horizontal flip only). Trained on full dataset (~20,000 images).


## Results

### Model Accuracy

```
Model           Accuracy    Params    Training Time
MobileNetV2     99.20%      2.2M      21 min
ResNet18        99.00%      11M       12 min
```

MobileNetV2 is the better model here -- higher accuracy with 5x fewer parameters.

### What I Learned From the Hyperparameter Search

- Lower learning rates (0.0001) consistently outperform higher ones for transfer learning
- Heavy augmentation actually hurts -- the dataset is already diverse enough
- Cosine scheduling helps recover from suboptimal learning rates without ever making things worse
- Light augmentation (just horizontal flip) is the sweet spot


## Evaluation

```
# PyTorch models
python scripts/evaluate.py outputs/final/final_model.pth
python scripts/evaluate.py outputs/final/final_model.pth --dataset test
python scripts/evaluate.py outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18 --dataset test

# TensorRT engines
python scripts/evaluate_tensorrt.py outputs/final/final_model_fp16.engine
python scripts/evaluate_tensorrt.py outputs/final/final_model_fp16.engine --dataset test
python scripts/evaluate_tensorrt.py outputs/final_resnet18/final_resnet18_model_fp16.engine --dataset test
```

Outputs accuracy, precision, recall, F1, confusion matrix. Saves predictions to CSV and example images with predictions overlaid.


## Model Export

Export pipeline: PyTorch (.pth) -> ONNX (.onnx) -> TensorRT (.engine)

TensorRT requires manual installation -- see INSTALL.md.

```
# ONNX only
python scripts/export_model.py outputs/final/final_model.pth

# ONNX + TensorRT FP16 + benchmark
python scripts/export_model.py outputs/final/final_model.pth --tensorrt --fp16 --benchmark

# ResNet18
python scripts/export_model.py outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18 --tensorrt --fp16 --benchmark
```

### trtexec Validation

The exported engines were validated with NVIDIA's native C++ trtexec tool to confirm they work outside of Python:

```
Model               Throughput      Mean Latency    GPU Compute
MobileNetV2 FP16    1,662 qps       0.544 ms        0.487 ms
ResNet18 FP16       2,074 qps       0.435 ms        0.380 ms
```


## Inference

### Single Image

```
# Default (MobileNetV2 PyTorch)
python scripts/inference.py data/CleanPetImages/Cat/1.jpg

# TensorRT engine (faster)
python scripts/inference.py data/CleanPetImages/Cat/1.jpg --model outputs/final/final_model_fp16.engine

# ResNet18
python scripts/inference.py data/CleanPetImages/Dog/1.jpg --model outputs/final_resnet18/final_resnet18_model.pth --model-type resnet18

# Your own photo
python scripts/inference.py C:\path\to\photo.jpg --model outputs/final/final_model_fp16.engine
```

Example output:
```
Image: data/CleanPetImages/Cat/1.jpg
Model: final_model_fp16.engine (TensorRT)

Prediction: Cat
Confidence: 100.0%
  Cat: 100.0%
  Dog: 0.0%
```

### Benchmark All Formats

```
python scripts/benchmark_inference.py
python scripts/benchmark_inference.py --iterations 500
```

### Inference Speed (RTX 3050 Laptop GPU)

```
Model           Format              Inference       End-to-End FPS
MobileNetV2     PyTorch             4.64 ms         167
MobileNetV2     ONNX                1.99 ms         299
MobileNetV2     TensorRT FP16       0.79 ms         468
ResNet18        PyTorch             2.33 ms         272
ResNet18        ONNX                2.16 ms         284
ResNet18        TensorRT FP16       0.57 ms         521
```

TensorRT FP16 gives 2.5-5.9x speedup over the alternatives. Preprocessing (OpenCV resize + normalize) adds ~1.3ms per image.


## C++ Inference

Native C++ TensorRT inference in cpp_inference/. Uses OpenCV for image preprocessing (bilinear resize and normalization), matching the Python training pipeline exactly.

### Requirements
- Visual Studio Build Tools 2022+
- CMake 3.18+
- CUDA Toolkit 12.x
- TensorRT 10.x
- OpenCV 4.x (prebuilt)

### Build

From x64 Native Tools Command Prompt:
```
cd cpp_inference
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Run

```
cpp_inference\build\Release\catdog_inference.exe <engine_file> <image_file>
```

```
# MobileNetV2
cpp_inference\build\Release\catdog_inference.exe outputs\final\final_model_fp16.engine data\CleanPetImages\Cat\1.jpg
cpp_inference\build\Release\catdog_inference.exe outputs\final\final_model_fp16.engine data\CleanPetImages\Dog\1.jpg

# ResNet18
cpp_inference\build\Release\catdog_inference.exe outputs\final_resnet18\final_resnet18_model_fp16.engine data\CleanPetImages\Cat\1.jpg
cpp_inference\build\Release\catdog_inference.exe outputs\final_resnet18\final_resnet18_model_fp16.engine data\CleanPetImages\Dog\1.jpg
```

### MobileNetV2 Example

```
================================
TensorRT C++ Inference
================================
Engine: outputs\final\final_model_fp16.engine
Image: data\CleanPetImages\Cat\1.jpg

================================
Result
================================
Prediction: Cat
Confidence: 100%
  Cat: 100%
  Dog: 5.16308e-05%

================================
Performance
================================
Inference: 0.53 ms
Throughput: 1888 FPS
```

### ResNet18 Example

```
================================
TensorRT C++ Inference
================================
Engine: outputs\final_resnet18\final_resnet18_model_fp16.engine
Image: data\CleanPetImages\Dog\1.jpg

================================
Result
================================
Prediction: Dog
Confidence: 99.9889%
  Cat: 0.0110573%
  Dog: 99.9889%

================================
Performance
================================
Inference: 0.42 ms
Throughput: 2391 FPS
```

Run from x64 Native Tools Command Prompt, or make sure TensorRT bin folder is in your system PATH.


## Available Models

```
Model                       File                                                    Size        Speed
MobileNetV2 PyTorch         outputs/final/final_model.pth                           8.7 MB      4.6 ms
MobileNetV2 ONNX            outputs/final/final_model.onnx                          0.2 MB      2.0 ms
MobileNetV2 TensorRT FP16   outputs/final/final_model_fp16.engine                   4.6 MB      0.8 ms
ResNet18 PyTorch            outputs/final_resnet18/final_resnet18_model.pth          42.7 MB     2.3 ms
ResNet18 ONNX               outputs/final_resnet18/final_resnet18_model.onnx         0.1 MB      2.2 ms
ResNet18 TensorRT FP16      outputs/final_resnet18/final_resnet18_model_fp16.engine  21.7 MB     0.6 ms
```

TensorRT engine files are GPU-specific. They were built on an RTX 3050 Laptop GPU and will need to be regenerated on different hardware using the export script.
