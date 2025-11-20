"""
Modal app for Stable Diffusion 3.5 latent-space SLERP video generation.

This script generates a video by performing spherical linear interpolation (SLERP)
between two latent noise vectors in SD3.5's latent space, creating smooth transitions
between generated images.
"""

import os
from pathlib import Path
from typing import List

import modal

# Modal app setup (v3 style)
app = modal.App("sd3-slerp-video")

# Create a Modal volume for storing generated videos
volume = modal.Volume.from_name("sd3-slerp-videos", create_if_missing=True)

# Create a Modal volume for caching model weights (avoids re-downloading ~10GB on each run)
model_cache_volume = modal.Volume.from_name("sd3-model-cache", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "diffusers>=0.30.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "safetensors>=0.4.0",
        "imageio[ffmpeg]>=2.31.0",
        "Pillow>=10.0.0",
        "sentencepiece>=0.1.99",  # Required for SD3.5 tokenizer
    )
)


def rand_latent(pipe, height: int, width: int, seed: int):
    """
    Generate a deterministic random latent tensor for SD3.5.
    
    Args:
        pipe: StableDiffusion3Pipeline instance
        height: Image height in pixels
        width: Image width in pixels
        seed: Random seed for reproducibility
    
    Returns:
        Random latent tensor with proper shape for SD3.5
    """
    import torch
    
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # SD3.5 uses 16 latent channels and 8x downsampling
    latent_height = height // 8
    latent_width = width // 8
    num_channels = 16
    
    latent = torch.randn(
        (1, num_channels, latent_height, latent_width),
        generator=generator,
        device=pipe.device,
        dtype=pipe.dtype,
    )
    
    return latent


def slerp(t: float, z0, z1):
    """
    Spherical linear interpolation (SLERP) between two tensors.
    
    Performs great-circle interpolation in the flattened vector space,
    then reshapes back to the original tensor shape.
    
    Args:
        t: Interpolation parameter in [0, 1]
        z0: Starting latent tensor
        z1: Ending latent tensor
    
    Returns:
        Interpolated latent tensor
    """
    import torch
    
    # Store original shape
    original_shape = z0.shape
    
    # Flatten to vectors
    v0 = z0.flatten()
    v1 = z1.flatten()
    
    # Normalize
    v0_norm = v0 / (torch.norm(v0) + 1e-10)
    v1_norm = v1 / (torch.norm(v1) + 1e-10)
    
    # Compute angle between vectors
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    
    # Handle near-parallel vectors (fallback to linear interpolation)
    if theta < 1e-5:
        result = (1 - t) * v0 + t * v1
    else:
        # Great-circle interpolation
        sin_theta = torch.sin(theta)
        result = (torch.sin((1 - t) * theta) / sin_theta) * v0 + (torch.sin(t * theta) / sin_theta) * v1
    
    # Reshape back to original shape
    return result.reshape(original_shape)


def ellipse_slerp(t: float, z0, z1, origin=None):
    """
    Spherical linear interpolation along an ellipse in latent space.
    
    Creates an ellipse centered at the origin (or computed center) where z0 and z1
    are two sample points on the ellipse. Traverses the full ellipse for smooth looping.
    
    Args:
        t: Interpolation parameter in [0, 1], where 0 and 1 represent the same point
        z0: First sample point on the ellipse (at angle 0)
        z1: Second sample point on the ellipse (defines ellipse shape)
        origin: Center of the ellipse. If None, uses the midpoint between z0 and z1
    
    Returns:
        Interpolated latent tensor on the ellipse
    """
    import torch
    import math
    
    # Store original shape
    original_shape = z0.shape
    
    # Flatten to vectors
    v0 = z0.flatten()
    v1 = z1.flatten()
    
    # Compute or use provided origin
    if origin is None:
        # Use midpoint as center for a symmetric ellipse
        v_origin = (v0 + v1) / 2.0
    else:
        v_origin = origin.flatten()
    
    # Translate to origin-centered coordinates
    u0 = v0 - v_origin
    u1 = v1 - v_origin
    
    # Compute angle for full ellipse traversal (0 to 2*pi)
    angle = 2 * math.pi * t
    
    # Normalize the basis vectors to define the ellipse axes
    # u0 defines the first axis (at angle 0)
    # u1 defines a point on the ellipse, we compute the perpendicular component
    
    # Project u1 onto u0 to get the component parallel to u0
    u0_norm_sq = torch.dot(u0, u0) + 1e-10
    projection = (torch.dot(u1, u0) / u0_norm_sq) * u0
    
    # Get the perpendicular component (defines the second axis)
    u1_perp = u1 - projection
    
    # Compute point on ellipse using parametric form:
    # E(t) = origin + cos(angle)*axis1 + sin(angle)*axis2
    result = v_origin + torch.cos(torch.tensor(angle)) * u0 + torch.sin(torch.tensor(angle)) * u1_perp
    
    # Reshape back to original shape
    return result.reshape(original_shape)


def write_video(frames: List, out_path: str, fps: int = 8, save_debug_frames: bool = False, output_tar_path: str = None):
    """
    Write a list of PIL images to an MP4 video file using imageio.
    
    Args:
        frames: List of PIL Image objects
        out_path: Output path for the video file
        fps: Frames per second for the video
        save_debug_frames: If True, save first and last frames as PNGs to the same directory
        output_tar_path: If provided, save all frames as PNGs in a tar.gz archive at this path
    """
    import numpy as np
    import imageio
    import tarfile
    import tempfile
    
    if not frames:
        raise ValueError("No frames to write")
    
    # Save debug frames if requested
    if save_debug_frames:
        out_dir = os.path.dirname(out_path)
        first_frame_path = os.path.join(out_dir, "debug_frame_first.png")
        last_frame_path = os.path.join(out_dir, "debug_frame_last.png")
        frames[0].save(first_frame_path)
        frames[-1].save(last_frame_path)
        print(f"Debug frames saved: {first_frame_path}, {last_frame_path}")
    
    # Save all frames as tar.gz if requested
    if output_tar_path:
        print(f"Creating tar.gz archive with all frames at {output_tar_path}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(frames):
                frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
                frame.save(frame_path)
            
            with tarfile.open(output_tar_path, "w:gz") as tar:
                for i in range(len(frames)):
                    frame_filename = f"frame_{i:04d}.png"
                    frame_path = os.path.join(tmpdir, frame_filename)
                    tar.add(frame_path, arcname=frame_filename)
            
            print(f"Tar.gz archive saved: {output_tar_path} ({len(frames)} frames)")
    
    # Convert PIL images to numpy arrays
    frames_np = [np.array(frame.convert("RGB")) for frame in frames]
    
    # Write video using imageio with H.264 codec
    # CRF 23 is the default for x264 (good quality, reasonable size)
    imageio.mimsave(
        out_path,
        frames_np,
        fps=fps,
        codec='libx264',
        pixelformat='yuv420p',
        output_params=['-crf', '23'],
    )
    
    print(f"Video saved to {out_path}")


@app.function(
    image=image,
    gpu=modal.gpu.L4(count=1),  # L4 GPU for efficient SD3.5 inference
    secrets=[modal.Secret.from_name("hf-token")],  # Hugging Face token from Modal secret
    volumes={
        "/videos": volume,
        "/model_cache": model_cache_volume,  # Cache SD3.5 weights to avoid re-downloading
    },
    timeout=1800,  # 30 minutes timeout
)
def generate_slerp_video(
    prompt: str = "highly detailed illustration of a floating city at sunset",
    height: int = 1024,
    width: int = 1024,
    seed0: int = 1234,
    seed1: int = 5678,
    num_frames: int = 16,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    fps: int = 8,
    output_name: str = "sd3_slerp.mp4",
    save_debug_frames: bool = False,
    output_tar_name: str = None,
    use_ellipse: bool = True,
) -> str:
    """
    Generate a SLERP video using Stable Diffusion 3.5 Medium.
    
    This function:
    1. Loads the SD3.5 pipeline using HF_TOKEN from Modal secret
    2. Generates two deterministic latent endpoints (z0, z1) from different seeds
    3. Interpolates between them using SLERP or ellipse interpolation
    4. Runs SD3.5 pipeline for each interpolated latent
    5. Saves the resulting images as an MP4 video
    
    Args:
        prompt: Text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        seed0: Random seed for first latent sample point
        seed1: Random seed for second latent sample point
        num_frames: Number of frames to interpolate
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        fps: Frames per second for output video
        output_name: Name of output video file
        save_debug_frames: If True, save first and last frames as PNGs
        output_tar_name: If provided, save all frames as PNGs in a tar.gz archive
        use_ellipse: If True, traverse full ellipse for smooth looping (default: True)
    
    Returns:
        Absolute path to the generated video in the container
    """
    import torch
    from diffusers import StableDiffusion3Pipeline
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This function requires a GPU. "
            "Make sure the Modal function is configured with gpu parameter."
        )
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print("Loading Stable Diffusion 3.5 Medium model...")
    # Note: Can also use "stabilityai/stable-diffusion-3.5-large" for higher quality (requires more VRAM)
    # or "stabilityai/stable-diffusion-3.5-large-turbo" for faster inference
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    
    # Load the model using HF_TOKEN from Modal secret (env var HF_TOKEN)
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment. Ensure 'hf-token' secret is configured.")
    
    # Use cached model weights from Modal volume to avoid re-downloading ~10GB on each run
    cache_dir = "/model_cache"
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        token=hf_token,
        cache_dir=cache_dir,
    )
    pipe = pipe.to("cuda")
    
    # Commit the cache volume to persist downloaded weights for future runs
    model_cache_volume.commit()
    
    print(f"Generating {num_frames} frames with {'ellipse' if use_ellipse else 'linear SLERP'} interpolation...")
    
    # Generate two deterministic latent sample points from different seeds
    z0 = rand_latent(pipe, height, width, seed0)
    z1 = rand_latent(pipe, height, width, seed1)
    
    # Sample interpolation values in [0, 1)
    # endpoint=False ensures first and last frames are not identical for seamless looping
    alphas = torch.linspace(0, 1, num_frames, endpoint=False)
    
    frames = []
    
    for i, alpha in enumerate(alphas):
        print(f"Generating frame {i+1}/{num_frames} (alpha={alpha:.3f})...")
        
        # Perform interpolation in latent space
        if use_ellipse:
            lat = ellipse_slerp(alpha.item(), z0, z1)
        else:
            lat = slerp(alpha.item(), z0, z1)
        
        # Run the pipeline with the interpolated latent
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=lat,
        )
        
        # Extract the PIL image
        image = output.images[0]
        frames.append(image)
    
    # Write video to the mounted volume
    output_path = f"/videos/{output_name}"
    
    # Prepare tar.gz path if output_tar_name is provided
    output_tar_path = None
    if output_tar_name:
        output_tar_path = f"/videos/{output_tar_name}"
    
    write_video(frames, output_path, fps=fps, save_debug_frames=save_debug_frames, output_tar_path=output_tar_path)
    
    # Commit the volume to persist the video
    volume.commit()
    print(f"Video saved and volume committed: {output_path}")
    if output_tar_name:
        print(f"Tar.gz archive saved and volume committed: {output_tar_path}")
    
    return output_path


@app.local_entrypoint()
def main(
    prompt: str = "highly detailed illustration of a floating city at sunset",
    output_name: str = "sd3_slerp.mp4",
    height: int = 1024,
    width: int = 1024,
    seed0: int = 1234,
    seed1: int = 5678,
    num_frames: int = 16,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    fps: int = 8,
    save_debug_frames: bool = False,
    output_tar_name: str = None,
    use_ellipse: bool = True,
):
    """
    Local entrypoint for generating SD3.5 SLERP videos.
    
    Usage:
        modal run app_sd3_slerp.py --prompt "your prompt here" --output-name "my_video.mp4"
    
    With tar.gz archive of all frames:
        modal run app_sd3_slerp.py --output-tar-name "frames.tar.gz"
    
    With ellipse interpolation for smooth looping (default):
        modal run app_sd3_slerp.py --use-ellipse
    
    With linear SLERP (original behavior):
        modal run app_sd3_slerp.py --no-use-ellipse
    
    After completion, download the video with:
        modal volume get sd3-slerp-videos <output_name> <local_path>
    
    For debugging, use --save-debug-frames to save first and last frames as PNGs.
    """
    print(f"Starting SD3.5 SLERP video generation...")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_name}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {num_frames} at {fps} FPS")
    print(f"Seeds: {seed0} -> {seed1}")
    print(f"Interpolation: {'Ellipse (full loop)' if use_ellipse else 'Linear SLERP'}")
    if output_tar_name:
        print(f"Tar.gz archive: {output_tar_name}")
    
    # Call the remote function
    video_path = generate_slerp_video.remote(
        prompt=prompt,
        height=height,
        width=width,
        seed0=seed0,
        seed1=seed1,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        fps=fps,
        output_name=output_name,
        save_debug_frames=save_debug_frames,
        output_tar_name=output_tar_name,
        use_ellipse=use_ellipse,
    )
    
    print(f"\n✅ Video generation complete!")
    print(f"Video saved to Modal volume at: {video_path}")
    print(f"\nTo download the video, run:")
    print(f"  modal volume get sd3-slerp-videos {output_name} ./{output_name}")
    
    if output_tar_name:
        print(f"\nTo download the tar.gz archive, run:")
        print(f"  modal volume get sd3-slerp-videos {output_tar_name} ./{output_tar_name}")
