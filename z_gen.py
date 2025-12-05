#!/usr/bin/env python3
import argparse
import os
import sys

# Set environment variable for AMD Radeon 890M (gfx1150) support
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import multiprocessing
torch.set_num_threads(multiprocessing.cpu_count())
from huggingface_hub import snapshot_download
from utils import load_from_local_dir, set_attention_backend
from zimage import generate

def ensure_model_downloaded(model_path="ckpts/Z-Image-Turbo"):
    """Checks if model exists, downloads if not."""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading from Hugging Face...")
        try:
            snapshot_download(repo_id="Tongyi-MAI/Z-Image-Turbo", local_dir=model_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)
    else:
        print(f"Model found at {model_path}.")

def main():
    parser = argparse.ArgumentParser(description="Generate images using Z-Image Turbo")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated.png", help="Output filename (default: generated.png)")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    
    args = parser.parse_args()

    model_path = "ckpts/Z-Image-Turbo"
    ensure_model_downloaded(model_path)

    # Device selection
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'xla') and torch.xla.is_available(): # Check for TPU properly if needed, simplified here
         # Simplified TPU check as per original inference.py logic roughly
         try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
         except:
             device = "cpu"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")

    dtype = torch.float16 if device == "cuda" or device == "tpu" else torch.float32 # bfloat16 might not be supported on all CPUs/MPS
    
    # Load models
    print("Loading model...")
    try:
        # Load to CPU first to save memory
        components = load_from_local_dir(model_path, device="cpu", dtype=dtype, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # set_attention_backend("_native_flash") 

    print(f"Generating image for prompt: '{args.prompt}'")
    
    # Disable offloading if running on CPU (not needed and might cause issues)
    use_offload = device != "cpu"
    
    images = generate(
        prompt=args.prompt,
        **components,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator(device).manual_seed(args.seed),
        device=device,
        sequential_offload=use_offload,
        force_text_encoder_cpu=use_offload,
    )

    images[0].save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()
