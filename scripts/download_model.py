#!/usr/bin/env python3
"""
Script to download and cache the DDPM MNIST model from HuggingFace.
Model: https://huggingface.co/1aurent/ddpm-mnist
"""

import os
from pathlib import Path
from diffusers import DDPMPipeline


def download_ddpm_mnist(cache_dir="./ddpm-mnist"):
    """
    Download the DDPM MNIST model from HuggingFace and cache it locally.
    
    Args:
        cache_dir: Directory to cache the model (default: ./ddpm-mnist)
    
    Returns:
        DDPMPipeline: The loaded model pipeline
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading DDPM MNIST model to {cache_path}...")
    
    # Download and cache the model
    pipeline = DDPMPipeline.from_pretrained(
        "1aurent/ddpm-mnist",
        cache_dir=str(cache_path)
    )
    
    print("Model downloaded successfully!")
    print(f"Model cached at: {cache_path}")
    
    return pipeline


def main():
    """Main function to download the model."""
    pipeline = download_ddpm_mnist()
    
    # Print model info
    print(f"\nModel information:")
    print(f"  - Scheduler: {type(pipeline.scheduler).__name__}")
    print(f"  - UNet config: {pipeline.unet.config}")
    
    # Generate a sample image to verify the model works
    print("\nGenerating a test image to verify model...")
    image = pipeline(num_inference_steps=25).images[0]
    
    # Save test image
    output_path = Path("test_sample.png")
    image.save(output_path)
    print(f"Test image saved to: {output_path}")
    

if __name__ == "__main__":
    main()
