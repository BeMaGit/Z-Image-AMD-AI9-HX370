# Z-Image on AMD HX 370 (Radeon 890M)

This guide details how to run Z-Image on the AMD Ryzen AI 9 HX 370 processor with integrated Radeon 890M graphics (ROCm).

## Overview

Running the Z-Image Turbo model on the Radeon 890M (8GB shared VRAM) requires significant memory optimizations. The default implementation exceeds the available memory. We have modified the codebase to support:

*   **Layer-wise Offloading**: The transformer model is kept on the CPU, and individual layers are moved to the GPU only when needed for computation.
*   **CPU Text Encoding**: The text encoder is run entirely on the CPU.
*   **Sequential Offloading**: Other components (VAE) are offloaded when not in use.

**Note**: These optimizations trade speed for memory. Generation will be significantly slower than on a dedicated high-VRAM GPU, but it will function without crashing.

## Prerequisites

1.  **Linux OS**: Tested on Linux.
2.  **ROCm Support**: Ensure you have the necessary ROCm drivers installed.
3.  **Python 3.10+**

## Installation

1.  **Clone the Repository** (if you haven't already):
    ```bash
    git clone https://github.com/Tongyi-MAI/Z-Image
    cd Z-Image
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install PyTorch with ROCm Support**:
    You must install the ROCm version of PyTorch.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    ```
    *Check [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command matching your ROCm version.*

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install huggingface_hub
    ```

## Usage

We have provided a custom script `z_gen.py` that handles the memory optimizations automatically.

### Running the Generator

```bash
./venv/bin/python z_gen.py "Your prompt here" --output output.png
```

**Arguments:**
*   `prompt`: The text description of the image you want to generate.
*   `--output`: Filename for the generated image (default: `generated.png`).
*   `--steps`: Number of inference steps (default: 8).
*   `--seed`: Random seed (default: 42).
*   `--height`: Image height (default: 1024).
*   `--width`: Image width (default: 1024).
*   `--cpu`: Force execution on CPU (useful if GPU memory is insufficient even with optimizations).

### Example

```bash
# Run on GPU (default)
./venv/bin/python z_gen.py "A futuristic city with flying cars" --output city.png

# Run on CPU
./venv/bin/python z_gen.py "A futuristic city with flying cars" --cpu --output city_cpu.png
```

## Technical Details & Troubleshooting

### Performance Optimizations
*   **Multi-threading**: The script automatically sets the number of threads to the number of available CPU cores to maximize performance during text encoding and data processing.
*   **Memory Management**: Excessive memory clearing has been removed to improve speed.

### Environment Variables
The script automatically sets `HSA_OVERRIDE_GFX_VERSION=11.0.0` to ensure compatibility with the RDNA3 architecture of the Radeon 890M.

### "HIP out of memory" Errors
If you still encounter OOM errors:
1.  Ensure no other GPU-intensive applications are running.
2.  Try reducing the image resolution (e.g., `--height 512 --width 512`).
3.  The script uses `expandable_segments:True` for memory allocation if supported, but on some ROCm versions, this might not be enough.

### Slowness
Generation is expected to take several minutes due to the constant data transfer between system RAM and VRAM (layer-wise offloading). This is the only way to fit the model into the shared memory pool.
