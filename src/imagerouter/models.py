"""Model registry and pricing data management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .exceptions import ModelNotFoundError

if TYPE_CHECKING:
    from .client import ImageRouterClient


@dataclass
class PricingInfo:
    """Pricing information for a model.

    Attributes:
        pricing_type: Type of pricing ('fixed', 'calculated', 'post_generation').
        value: Fixed price value (for 'fixed' type).
        min_price: Minimum price in range.
        max_price: Maximum price in range.
        average_price: Average price in range.
    """

    pricing_type: str
    value: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    average_price: float | None = None

    @classmethod
    def from_api_data(cls, pricing_data: dict[str, Any]) -> "PricingInfo":
        """Create PricingInfo from API response data.

        Args:
            pricing_data: Pricing dict from API response.

        Returns:
            PricingInfo instance.
        """
        pricing_type = pricing_data.get("type", "unknown")

        if pricing_type == "fixed":
            value = float(pricing_data.get("value", 0))
            return cls(
                pricing_type=pricing_type,
                value=value,
                min_price=value,
                max_price=value,
                average_price=value,
            )

        # Handle range-based pricing (calculated, post_generation)
        range_data = pricing_data.get("range", {})
        return cls(
            pricing_type=pricing_type,
            min_price=float(range_data.get("min", 0)),
            max_price=float(range_data.get("max", 0)),
            average_price=float(range_data.get("average", 0)),
        )

    def get_estimate(self) -> tuple[float, float, float]:
        """Get price estimate as (min, average, max).

        Returns:
            Tuple of (min_price, average_price, max_price).
        """
        if self.pricing_type == "fixed" and self.value is not None:
            return (self.value, self.value, self.value)
        return (
            self.min_price or 0,
            self.average_price or 0,
            self.max_price or 0,
        )


@dataclass
class ModelInfo:
    """Information about an available model.

    Attributes:
        id: Model identifier (e.g., 'google/veo-3.1-fast').
        name: Human-readable model name.
        provider: Provider name.
        output_types: List of output types ('image', 'video').
        pricing: Pricing information.
        supported_durations: For video models, list of supported durations in seconds.
        supported_sizes: List of supported output sizes.
        supports_edit: Whether the model supports image editing/input.
        raw_data: Original API response data.
    """

    id: str
    name: str
    provider: str
    output_types: list[str]
    pricing: PricingInfo
    supported_durations: list[int] | None = None
    supported_sizes: list[str] | None = None
    supports_edit: bool = False
    raw_data: dict[str, Any] | None = None

    @classmethod
    def from_api_data(cls, data: dict[str, Any]) -> "ModelInfo":
        """Create ModelInfo from API response data.

        Args:
            data: Model dict from API response.

        Returns:
            ModelInfo instance.
        """
        pricing_data = data.get("pricing", {})
        supported_params = data.get("supported_params", {})

        return cls(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            provider=data.get("provider", "unknown"),
            output_types=data.get("output", []),
            pricing=PricingInfo.from_api_data(pricing_data),
            supported_durations=data.get("seconds"),
            supported_sizes=data.get("sizes"),
            supports_edit=supported_params.get("edit", False),
            raw_data=data,
        )

    def is_video_model(self) -> bool:
        """Check if this is a video generation model."""
        return "video" in self.output_types

    def is_image_model(self) -> bool:
        """Check if this is an image generation model."""
        return "image" in self.output_types

    def get_default_duration(self) -> int | None:
        """Get default video duration for the model.

        Returns:
            Default duration in seconds, or None if not a video model.
        """
        if self.supported_durations:
            return self.supported_durations[0]
        return None


class ModelRegistry:
    """Registry for available models with caching.

    Manages fetching and caching model information from the API.

    Args:
        client: ImageRouterClient instance for API requests.

    Example:
        >>> registry = ModelRegistry(client)
        >>> video_models = registry.get_video_models()
        >>> model = registry.get_model("google/veo-3.1-fast")
    """

    def __init__(self, client: "ImageRouterClient") -> None:
        self._client = client
        self._models: dict[str, ModelInfo] | None = None

    def _ensure_loaded(self) -> None:
        """Ensure models are loaded from API."""
        if self._models is None:
            self.refresh()

    def refresh(self) -> None:
        """Refresh model data from API."""
        raw_models = self._client.list_models()
        self._models = {}
        for model_id, data in raw_models.items():
            self._models[model_id] = ModelInfo.from_api_data(data)

    def get_model(self, model_id: str) -> ModelInfo:
        """Get model information by ID.

        Args:
            model_id: The model identifier.

        Returns:
            ModelInfo for the requested model.

        Raises:
            ModelNotFoundError: If the model is not found.
        """
        self._ensure_loaded()
        assert self._models is not None

        if model_id not in self._models:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return self._models[model_id]

    def get_all_models(self) -> dict[str, ModelInfo]:
        """Get all available models.

        Returns:
            Dict mapping model_id to ModelInfo.
        """
        self._ensure_loaded()
        assert self._models is not None
        return self._models.copy()

    def get_video_models(self) -> dict[str, ModelInfo]:
        """Get all video generation models.

        Returns:
            Dict mapping model_id to ModelInfo for video models.
        """
        self._ensure_loaded()
        assert self._models is not None
        return {k: v for k, v in self._models.items() if v.is_video_model()}

    def get_image_models(self) -> dict[str, ModelInfo]:
        """Get all image generation models.

        Returns:
            Dict mapping model_id to ModelInfo for image models.
        """
        self._ensure_loaded()
        assert self._models is not None
        return {k: v for k, v in self._models.items() if v.is_image_model()}

    def get_models_by_type(self, output_type: str) -> dict[str, ModelInfo]:
        """Get models by output type.

        Args:
            output_type: Either 'image' or 'video'.

        Returns:
            Dict mapping model_id to ModelInfo.
        """
        if output_type == "video":
            return self.get_video_models()
        elif output_type == "image":
            return self.get_image_models()
        else:
            return self.get_all_models()
