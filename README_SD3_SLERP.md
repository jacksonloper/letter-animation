# SD3.5 SLERP Video Generator

This Modal app generates videos using Stable Diffusion 3.5 with spherical linear interpolation (SLERP) in latent space.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
3. **Hugging Face Token**: Generate a token with read permissions at https://huggingface.co/settings/tokens
4. **Modal Secret**: Create a Modal secret named `hf-token` containing your Hugging Face token:
   ```bash
   modal secret create hf-token HF_TOKEN=hf_your_token_here
   ```

## Usage

### Basic Usage

Generate a video with default settings:

```bash
modal run app_sd3_slerp.py
```

### Custom Prompt

Generate with a custom prompt:

```bash
modal run app_sd3_slerp.py --prompt "a serene mountain landscape at dawn"
```

### Advanced Configuration

```bash
modal run app_sd3_slerp.py \
  --prompt "cyberpunk city neon lights" \
  --output-name "cyberpunk_slerp.mp4" \
  --num-frames 24 \
  --width 1024 \
  --height 1024 \
  --seed0 1111 \
  --seed1 9999 \
  --fps 12
```

### All Parameters

- `--prompt`: Text prompt for generation (default: "highly detailed illustration of a floating city at sunset")
- `--output-name`: Output video filename (default: "sd3_slerp.mp4")
- `--height`: Image height in pixels (default: 1024)
- `--width`: Image width in pixels (default: 1024)
- `--seed0`: Starting random seed (default: 1234)
- `--seed1`: Ending random seed (default: 5678)
- `--num-frames`: Number of frames to generate (default: 16)
- `--num-inference-steps`: Diffusion steps per frame (default: 28)
- `--guidance-scale`: CFG scale (default: 3.5)
- `--fps`: Video framerate (default: 8)

## Downloading Results

After generation completes, download the video from the Modal volume:

```bash
modal volume get sd3-slerp-videos sd3_slerp.mp4 ./sd3_slerp.mp4
```

Or list all videos in the volume:

```bash
modal volume ls sd3-slerp-videos
```

## How It Works

1. **Model Caching**: Downloads and caches SD3.5 weights (~10GB) in a Modal volume on first run, reusing them on subsequent runs
2. **Latent Generation**: Creates two deterministic latent noise vectors from different random seeds
3. **SLERP Interpolation**: Performs spherical linear interpolation between the latents in the normalized vector space
4. **Image Generation**: For each interpolated latent, runs the SD3.5 pipeline to generate an image
5. **Video Creation**: Combines all frames into an MP4 video using OpenCV and ffmpeg

## Technical Details

- **Model**: Stable Diffusion 3.5 Medium (can be switched to `-large` or `-large-turbo` variants for different quality/speed tradeoffs)
- **GPU**: Runs on Modal L4 GPU
- **Model Caching**: Weights cached in `sd3-model-cache` Modal volume (persists across runs)
- **Latent Space**: 16 channels, 8x downsampling (128x128 latents for 1024x1024 images)
- **Video Codec**: H.264 (MP4 container)
- **Storage**: Persistent Modal volume (`sd3-slerp-videos`)

## Performance Notes

- Generation time depends on `num_frames` and `num_inference_steps`
- Typical generation: ~30 seconds per frame on L4
- 16 frames with 28 steps: approximately 8-10 minutes total
- **First run**: Includes model download time (~10GB, 5-10 minutes depending on network)
- **Subsequent runs**: Model loaded from cache (much faster startup)
- Model weights are automatically cached in `sd3-model-cache` volume

## Troubleshooting

### "HF_TOKEN not found in environment"
Ensure the Modal secret is properly configured:
```bash
modal secret create hf-token HF_TOKEN=hf_your_token_here
```

### Model Access Denied
Make sure your Hugging Face token has access to the SD3.5 model. Visit the model page and accept the license if required.

### Out of Memory
Reduce resolution or use the turbo variant for faster inference with less memory.
