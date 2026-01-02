"""ImageRouter - Python module for video and image generation via ImageRouter.io.

This module provides programmatic access to multiple AI video/image generation
models through a unified API.

Example:
    >>> from imagerouter import ImageRouterClient, CostEstimator, VideoGenerator
    >>> client = ImageRouterClient()
    >>> estimator = CostEstimator(client)
    >>> estimate = estimator.estimate_video(model="google/veo-3.1-fast", seconds=4)
    >>> print(f"Estimated cost: ${estimate.total_average:.2f}")
"""

from .client import ImageRouterClient
from .estimator import CostEstimate, CostEstimator
from .exceptions import (
    AuthenticationError,
    GenerationError,
    ImageRouterError,
    InsufficientCreditsError,
    ModelNotFoundError,
    NetworkError,
    RateLimitError,
    ValidationError,
)
from .generators import ImageGenerator, VideoGenerator
from .models import ModelInfo, ModelRegistry, PricingInfo

__version__ = "0.1.0"

__all__ = [
    # Client
    "ImageRouterClient",
    # Generators
    "VideoGenerator",
    "ImageGenerator",
    # Estimation
    "CostEstimator",
    "CostEstimate",
    # Models
    "ModelRegistry",
    "ModelInfo",
    "PricingInfo",
    # Exceptions
    "ImageRouterError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "ModelNotFoundError",
    "ValidationError",
    "GenerationError",
    "NetworkError",
]
