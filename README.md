# FreeMorph on Replicate

This repository contains everything needed to run FreeMorph on Replicate, including:
- Cog configuration for containerized deployment
- Prediction interface for running inference
- GitHub Actions for testing builds and pushing to Replicate

## About FreeMorph

FreeMorph is a character animation model that creates smooth morphing animations between letters, shapes, and other visual elements.

## Prerequisites

- [Docker](https://www.docker.com/)
- [Cog](https://github.com/replicate/cog) - Install with:
  ```bash
  sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
  sudo chmod +x /usr/local/bin/cog
  ```

## Local Development

### Building the Docker image

```bash
cog build
```

### Running predictions locally

```bash
# Create a test image
python3 -c "from PIL import Image; img = Image.new('RGB', (256, 256), color='blue'); img.save('source.png')"

# Run prediction
cog predict -i source_image=@source.png -i num_frames=30 -i fps=15
```

### Testing with custom images

```bash
cog predict \
  -i source_image=@/path/to/source.png \
  -i target_image=@/path/to/target.png \
  -i num_frames=60 \
  -i fps=30
```

## Deployment to Replicate

### Setup

1. Create a model on [Replicate](https://replicate.com/)
2. Get your API token from your [Replicate account settings](https://replicate.com/account)
3. Add the following secrets to your GitHub repository:
   - `REPLICATE_API_TOKEN`: Your Replicate API token
   - `REPLICATE_MODEL_NAME`: Your model name (e.g., `username/model-name`)

### Automatic Deployment

The repository includes two GitHub Actions workflows:

1. **Test Build** (`.github/workflows/test-build.yml`)
   - Runs on every push and pull request
   - Tests that the Cog build completes successfully
   - Ensures the container can be built

2. **Push to Replicate** (`.github/workflows/push-to-replicate.yml`)
   - Runs when you create a version tag (e.g., `v1.0.0`)
   - Can also be triggered manually from the Actions tab
   - Builds and pushes the model to Replicate

### Manual Deployment

```bash
# Login to Replicate
cog login

# Push to Replicate
cog push r8.im/username/model-name
```

### Creating a Release

To trigger automatic deployment to Replicate:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── test-build.yml         # Tests Cog build
│       └── push-to-replicate.yml  # Pushes to Replicate
├── .gitignore                     # Ignore patterns
├── cog.yaml                       # Cog configuration
├── predict.py                     # Prediction interface
└── README.md                      # This file
```

## Configuration

### cog.yaml

The `cog.yaml` file defines:
- GPU requirements
- System packages
- Python version and packages
- Entry point for predictions

### predict.py

The `predict.py` file contains the `Predictor` class with:
- `setup()`: Load model weights and initialize
- `predict()`: Run inference on input images

## API Usage

Once deployed to Replicate, use the model via:

### Python

```python
import replicate

output = replicate.run(
    "username/model-name:version",
    input={
        "source_image": "https://example.com/source.png",
        "target_image": "https://example.com/target.png",
        "num_frames": 30,
        "fps": 15
    }
)
```

### cURL

```bash
curl -s -X POST \
  -H "Authorization: Token YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "VERSION_ID",
    "input": {
      "source_image": "https://example.com/source.png",
      "num_frames": 30,
      "fps": 15
    }
  }' \
  https://api.replicate.com/v1/predictions
```

## Contributing

1. Make your changes
2. Test locally with `cog build` and `cog predict`
3. Submit a pull request
4. The test workflow will automatically verify your changes

## License

See LICENSE file for details.