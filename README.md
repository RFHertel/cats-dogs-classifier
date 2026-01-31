# Cats vs Dogs Classifier

## Setup

conda create -n catdog python=3.10 -y
conda activate catdog

# PyTorch with CUDA (must be first, separate from requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Everything else
pip install -r requirements.txt

## Dataset

Download from: https://www.microsoft.com/en-us/download/details.aspx?id=54765

Extracted images: data/PetImages/

Verify GPU is working:
```
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # should show your GPU
```

## Project Structure
```
cats_dogs_project/
├── data/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
├── notebooks/
│   └── 01_explore_data.ipynb
├── outputs/
├── requirements.txt
└── README.md
```