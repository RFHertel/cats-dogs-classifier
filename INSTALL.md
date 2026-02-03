# Installation Guide

This covers the full setup for the project. The Python section is required. TensorRT and C++ are optional but needed if you want to run the optimized inference pipeline.


## Python Environment

```
conda create -n catdog python=3.10 -y
conda activate catdog
```

Install PyTorch with CUDA first, before anything else:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then install the rest:
```
pip install -r requirements.txt
```

Check that CUDA is working:
```
conda activate catdog
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

You should see your GPU name and CUDA: True. If CUDA is False, you probably installed the CPU version of PyTorch -- uninstall it and reinstall with the cu126 URL above.


## TensorRT (optional)

TensorRT can't be installed from pip. You need to download it manually from NVIDIA. This is only needed for the export and TensorRT inference scripts.

### Download

1. Go to https://developer.nvidia.com/tensorrt/download
2. Create or log into an NVIDIA Developer account (free)
3. Download TensorRT 10.x for Windows, matching your CUDA version
   - For CUDA 12.x grab the zip that says cuda-12.x in the filename

### Install

Extract the zip to a permanent location. I used C:\TensorRT\ so the folder ends up at C:\TensorRT\TensorRT-10.15.1.29\.

Install the Python wheels. Make sure you're in the catdog conda environment, not base:
```
conda activate catdog
cd C:\TensorRT\TensorRT-10.15.1.29\python
pip install tensorrt-10.15.1.29-cp310-none-win_amd64.whl
pip install tensorrt_dispatch-10.15.1.29-cp310-none-win_amd64.whl
pip install tensorrt_lean-10.15.1.29-cp310-none-win_amd64.whl
```

The version numbers in the wheel filenames will vary depending on what you downloaded. Use whatever matches.

### Add to PATH

TensorRT needs its DLLs on your system PATH. The DLLs are in the bin folder, not lib.

1. Press Windows key, search "environment variables"
2. Click "Edit the system environment variables"
3. Click "Environment Variables..."
4. Under System variables, find Path, click Edit
5. Click New
6. Add: C:\TensorRT\TensorRT-10.15.1.29\bin
7. OK, OK, OK

Important: it's the bin folder. The lib folder has .lib files for C++ linking, not the DLLs you need at runtime.

### Verify

Close your terminal completely and open a new one, then:
```
conda activate catdog
python -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"
```

Should print TensorRT: 10.15.1.29 (or whatever version you installed).

If you get "Could not find: nvinfer_10.dll", your PATH isn't set correctly. Double check that the bin folder is in your system PATH and that you restarted your terminal after changing it.


## C++ Build Tools (optional)

Only needed if you want to build the C++ inference application in cpp_inference/.

### Visual Studio Build Tools

1. Go to https://visualstudio.microsoft.com/downloads/
2. Scroll down to "Tools for Visual Studio"
3. Download "Build Tools for Visual Studio 2022" (or newer)
4. Run the installer
5. Select "Desktop development with C++"
6. Install

This is a few GB download. After it finishes you should be able to open "x64 Native Tools Command Prompt" from the Start Menu.

Verify:
```
where cl
```

Should show the path to cl.exe.

### CMake

1. Go to https://cmake.org/download/
2. Download the Windows x64 Installer (.msi)
3. Run it
4. Check "Add CMake to system PATH for all users" during install

Verify (new terminal):
```
cmake --version
```

### CUDA Toolkit

You need the full CUDA Toolkit, not just the runtime that comes with PyTorch.

1. Go to https://developer.nvidia.com/cuda-toolkit-archive
2. Download CUDA Toolkit 12.6 (or whatever matches your PyTorch CUDA version)
3. Select Windows, x86_64, exe (local)
4. Run the installer with default options

This is about 3 GB. After installing, verify (new terminal):
```
nvcc --version
```

### Building the C++ Application

Once all three are installed, open the x64 Native Tools Command Prompt and run:
```
cd C:\AWrk\cats_dogs_project\cpp_inference
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

If cmake can't find CUDA or TensorRT, check the paths in cpp_inference/CMakeLists.txt and update them to match where you installed things.

Test it:
```
.\Release\catdog_inference.exe ..\..\outputs\final\final_model_fp16.engine ..\..\data\CleanPetImages\Cat\1.jpg
```

Note: the exe needs TensorRT DLLs at runtime. If it runs from the x64 command prompt but silently does nothing from PowerShell or VS Code, your TensorRT bin folder isn't in PATH for that terminal session. Restart VS Code or add it to your PATH.


## Full Verification

Once everything is installed, this should all work:

```
conda activate catdog

python -c "
import torch
import tensorrt
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('TensorRT:', tensorrt.__version__)
"
```

And from x64 Native Tools Command Prompt:
```
where cl
cmake --version
nvcc --version
```
