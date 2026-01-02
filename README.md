# ImageRouter

Python module for video and image generation via the ImageRouter.io unified API.

Access multiple AI video/image generation models (Google Veo, Kling, Sora, Seedance, FLUX, GPT-Image, etc.) through a single API with cost estimation before execution.

## Installation

```bash
pip install -e .
```

## Configuration

Set your API key as an environment variable:

```bash
export IMAGEROUTER_API_KEY=ir_xxx
```

Optional settings:

```bash
export IMAGEROUTER_TIMEOUT=300      # Request timeout in seconds (default: 300)
export IMAGEROUTER_MAX_RETRIES=3    # Retry attempts (default: 3)
```

## CLI Usage

### Estimate Costs

Estimate generation costs without making actual API calls:

```bash
# Estimate video generation cost
imagerouter estimate --type video --model google/veo-3.1-fast --seconds 4

# Estimate with multiple outputs
imagerouter estimate --type video --model kwaivgi/kling-2.1-standard --seconds 10 --count 3

# Estimate image generation
imagerouter estimate --type image --model openai/gpt-image-1 --count 5

# Output as JSON
imagerouter estimate --type video --model google/veo-3.1-fast --seconds 4 --json
```

Example output:

```
Model: google/veo-3.1-fast
Type: video
Duration: 4s
Count: 1
Price per unit: $0.9000

Estimated total cost:
  Minimum: $0.6000
  Average: $0.9000
  Maximum: $1.2000
```

### Generate Content

Generate videos or images (requires `--execute` flag):

```bash
# Text-to-video
imagerouter generate --execute --type video \
  --model google/veo-3.1-fast \
  --prompt "A cat playing piano in a jazz club" \
  --seconds 4 \
  --output cat_piano.mp4

# Image-to-video
imagerouter generate --execute --type video \
  --model kwaivgi/kling-2.1-standard \
  --image input.jpg \
  --prompt "The subject walks forward slowly" \
  --seconds 5

# Text-to-image
imagerouter generate --execute --type image \
  --model openai/gpt-image-1 \
  --prompt "A futuristic cityscape at night" \
  --quality high \
  --size 1024x1024 \
  --output city.png

# Image-to-image editing
imagerouter generate --execute --type image \
  --model openai/gpt-image-1 \
  --image landscape.jpg \
  --prompt "Add a dramatic sunset sky" \
  --output landscape_sunset.png

# With mask for targeted editing
imagerouter generate --execute --type image \
  --model openai/gpt-image-1 \
  --image photo.jpg \
  --mask mask.png \
  --prompt "Replace the background with a beach scene"
```

### List Models

View available models and pricing:

```bash
# List all models
imagerouter models

# Filter by type
imagerouter models --type video
imagerouter models --type image

# Output as JSON
imagerouter models --type video --json
```

Example output:

```
Available video models:

  google/veo-3.1-fast
    Provider: Gemini
    Price: $0.60 - $1.20
    Durations: [4, 6, 8]s

  kwaivgi/kling-2.1-standard
    Provider: Runware
    Price: $0.18 - $0.37
    Durations: [5, 10]s
    Supports edit: Yes
```

### Check Credits

View account balance:

```bash
# Human-readable format
imagerouter credits

# JSON format
imagerouter credits --json
```

Example output:

```
Account Balance:
  Remaining credits: $50.00
  Total usage: $25.50
  Total deposits: $75.50
```

## Python API Usage

```python
from imagerouter import (
    ImageRouterClient,
    CostEstimator,
    VideoGenerator,
    ImageGenerator,
)

# Initialize client
client = ImageRouterClient()  # Uses IMAGEROUTER_API_KEY env var

# Estimate costs before generation
estimator = CostEstimator(client)
estimate = estimator.estimate_video(model="google/veo-3.1-fast", seconds=4)
print(f"Estimated cost: ${estimate.total_average:.2f}")

# Generate video
video_gen = VideoGenerator(client)
result = video_gen.text_to_video(
    prompt="A serene mountain landscape at sunset",
    model="google/veo-3.1-fast",
    seconds=4,
    output_path="mountain.mp4",
)
print(f"Video URL: {result['data'][0]['url']}")

# Generate image
image_gen = ImageGenerator(client)
result = image_gen.text_to_image(
    prompt="A cyberpunk street scene",
    model="openai/gpt-image-1",
    quality="high",
    output_path="cyberpunk.png",
)
print(f"Image URL: {result['data'][0]['url']}")
```

## Error Handling

The module raises specific exceptions for different error scenarios:

```python
from imagerouter import (
    ImageRouterClient,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ModelNotFoundError,
    ValidationError,
    GenerationError,
    NetworkError,
)

try:
    client = ImageRouterClient()
    # ... perform operations
except AuthenticationError as e:
    # Invalid or missing API key (HTTP 401)
    print(f"Auth failed: {e.message}")

except InsufficientCreditsError as e:
    # Account balance too low (HTTP 402)
    print(f"Need more credits: {e.message}")

except RateLimitError as e:
    # Too many requests (HTTP 429)
    print(f"Rate limited: {e.message}")

except ModelNotFoundError as e:
    # Model doesn't exist (HTTP 404)
    print(f"Bad model: {e.message}")

except ValidationError as e:
    # Invalid parameters (HTTP 400)
    print(f"Invalid request: {e.message}")

except GenerationError as e:
    # Generation failed on provider side (HTTP 5xx)
    print(f"Generation failed: {e.message}")

except NetworkError as e:
    # Connection/timeout issues
    print(f"Network error: {e.message}")
```

### CLI Exit Codes

| Exit Code | Description |
|-----------|-------------|
| 0 | Success |
| 1 | General error (API error, validation error, etc.) |
| 130 | Interrupted (Ctrl+C) |

## Free Tier Models

Use these models for development/testing without cost:

- `ir/test-video` - Video test endpoint
- `openai/gpt-image-1.5:free` - Free GPT image generation
- `black-forest-labs/FLUX-1-schnell:free` - Free FLUX image generation

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=imagerouter --cov-report=term-missing

# Type checking
mypy src/imagerouter/

# Linting
ruff check src/
ruff format src/
```

## License

MIT
