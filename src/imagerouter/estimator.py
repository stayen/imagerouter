"""Cost estimation for video and image generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .exceptions import ValidationError
from .models import ModelRegistry

if TYPE_CHECKING:
    from .client import ImageRouterClient


@dataclass
class CostEstimate:
    """Cost estimate for a generation request.

    Attributes:
        model: Model ID used for the estimate.
        generation_type: Type of generation ('video' or 'image').
        duration_seconds: Video duration (for video estimates).
        count: Number of outputs to generate.
        price_per_unit: Price per single generation (average).
        total_min: Minimum total cost.
        total_max: Maximum total cost.
        total_average: Average/expected total cost.
        currency: Currency code (always 'USD').
    """

    model: str
    generation_type: str
    duration_seconds: int | None
    count: int
    price_per_unit: float
    total_min: float
    total_max: float
    total_average: float
    currency: str = "USD"

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "model": self.model,
            "type": self.generation_type,
            "count": self.count,
            "price_per_unit": self.price_per_unit,
            "total_min": self.total_min,
            "total_max": self.total_max,
            "total_average": self.total_average,
            "currency": self.currency,
        }
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        return result

    def format_summary(self) -> str:
        """Format estimate as human-readable summary."""
        lines = [
            f"Model: {self.model}",
            f"Type: {self.generation_type}",
        ]
        if self.duration_seconds is not None:
            lines.append(f"Duration: {self.duration_seconds}s")
        lines.extend([
            f"Count: {self.count}",
            f"Price per unit: ${self.price_per_unit:.4f}",
            "",
            "Estimated total cost:",
            f"  Minimum: ${self.total_min:.4f}",
            f"  Average: ${self.total_average:.4f}",
            f"  Maximum: ${self.total_max:.4f}",
        ])
        return "\n".join(lines)


class CostEstimator:
    """Estimate generation costs before execution.

    Uses model pricing data to calculate expected costs without
    making actual generation requests.

    Args:
        client: ImageRouterClient instance for fetching model data.

    Example:
        >>> estimator = CostEstimator(client)
        >>> estimate = estimator.estimate_video(
        ...     model="google/veo-3.1-fast",
        ...     seconds=4,
        ...     count=1
        ... )
        >>> print(f"Expected cost: ${estimate.total_average:.2f}")
    """

    def __init__(self, client: "ImageRouterClient") -> None:
        self._registry = ModelRegistry(client)

    def estimate_video(
        self,
        model: str,
        seconds: int | None = None,
        count: int = 1,
    ) -> CostEstimate:
        """Estimate video generation cost.

        Args:
            model: Model ID (e.g., 'google/veo-3.1-fast').
            seconds: Video duration in seconds. If None, uses model default.
            count: Number of videos to generate.

        Returns:
            CostEstimate with pricing breakdown.

        Raises:
            ModelNotFoundError: If the model doesn't exist.
            ValidationError: If the model doesn't support video or invalid duration.

        Example:
            >>> estimate = estimator.estimate_video(
            ...     model="kwaivgi/kling-2.1-standard",
            ...     seconds=10,
            ...     count=2
            ... )
        """
        if count < 1:
            raise ValidationError("Count must be at least 1")

        model_info = self._registry.get_model(model)

        if not model_info.is_video_model():
            raise ValidationError(f"Model '{model}' does not support video generation")

        # Use provided duration or model default
        duration = seconds
        if duration is None:
            duration = model_info.get_default_duration()
            if duration is None:
                raise ValidationError(
                    f"Model '{model}' requires explicit duration (seconds parameter)"
                )

        # Validate duration against supported values
        if model_info.supported_durations and duration not in model_info.supported_durations:
            raise ValidationError(
                f"Duration {duration}s not supported. "
                f"Valid options: {model_info.supported_durations}"
            )

        # Get base pricing
        min_price, avg_price, max_price = model_info.pricing.get_estimate()

        # Calculate total
        total_min = min_price * count
        total_avg = avg_price * count
        total_max = max_price * count

        return CostEstimate(
            model=model,
            generation_type="video",
            duration_seconds=duration,
            count=count,
            price_per_unit=avg_price,
            total_min=total_min,
            total_max=total_max,
            total_average=total_avg,
        )

    def estimate_image(
        self,
        model: str,
        quality: str = "auto",
        size: str = "auto",
        count: int = 1,
    ) -> CostEstimate:
        """Estimate image generation cost.

        Args:
            model: Model ID (e.g., 'openai/gpt-image-1').
            quality: Quality level ('auto', 'low', 'medium', 'high').
            size: Output resolution or 'auto'.
            count: Number of images to generate.

        Returns:
            CostEstimate with pricing breakdown.

        Raises:
            ModelNotFoundError: If the model doesn't exist.
            ValidationError: If the model doesn't support image generation.

        Example:
            >>> estimate = estimator.estimate_image(
            ...     model="openai/gpt-image-1",
            ...     quality="high",
            ...     count=5
            ... )
        """
        if count < 1:
            raise ValidationError("Count must be at least 1")

        model_info = self._registry.get_model(model)

        if not model_info.is_image_model():
            raise ValidationError(f"Model '{model}' does not support image generation")

        # Get base pricing
        min_price, avg_price, max_price = model_info.pricing.get_estimate()

        # Calculate total
        total_min = min_price * count
        total_avg = avg_price * count
        total_max = max_price * count

        return CostEstimate(
            model=model,
            generation_type="image",
            duration_seconds=None,
            count=count,
            price_per_unit=avg_price,
            total_min=total_min,
            total_max=total_max,
            total_average=total_avg,
        )

    def refresh_models(self) -> None:
        """Refresh cached model data from API."""
        self._registry.refresh()
