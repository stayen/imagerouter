"""Tests for cost estimator module."""

import pytest
from unittest.mock import MagicMock

from imagerouter.estimator import CostEstimate, CostEstimator
from imagerouter.exceptions import ModelNotFoundError, ValidationError
from imagerouter.models import ModelInfo, PricingInfo

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from fixtures.mock_responses import MOCK_MODELS_RESPONSE


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_to_dict(self):
        """Test converting estimate to dictionary."""
        estimate = CostEstimate(
            model="google/veo-3.1-fast",
            generation_type="video",
            duration_seconds=4,
            count=2,
            price_per_unit=0.90,
            total_min=1.20,
            total_max=2.40,
            total_average=1.80,
        )

        result = estimate.to_dict()

        assert result["model"] == "google/veo-3.1-fast"
        assert result["type"] == "video"
        assert result["duration_seconds"] == 4
        assert result["count"] == 2
        assert result["price_per_unit"] == 0.90
        assert result["total_min"] == 1.20
        assert result["total_max"] == 2.40
        assert result["total_average"] == 1.80
        assert result["currency"] == "USD"

    def test_to_dict_without_duration(self):
        """Test dictionary for image estimate (no duration)."""
        estimate = CostEstimate(
            model="openai/gpt-image-1",
            generation_type="image",
            duration_seconds=None,
            count=1,
            price_per_unit=0.15,
            total_min=0.01,
            total_max=0.30,
            total_average=0.15,
        )

        result = estimate.to_dict()
        assert "duration_seconds" not in result

    def test_format_summary(self):
        """Test formatting estimate as summary."""
        estimate = CostEstimate(
            model="google/veo-3.1-fast",
            generation_type="video",
            duration_seconds=4,
            count=1,
            price_per_unit=0.90,
            total_min=0.60,
            total_max=1.20,
            total_average=0.90,
        )

        summary = estimate.format_summary()

        assert "google/veo-3.1-fast" in summary
        assert "video" in summary
        assert "4s" in summary
        assert "$0.60" in summary
        assert "$1.20" in summary


class TestCostEstimator:
    """Tests for CostEstimator class."""

    def _create_mock_client(self):
        """Create a mock client with model data."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            model["id"]: model for model in MOCK_MODELS_RESPONSE["data"]
        }
        return mock_client

    def test_estimate_video(self):
        """Test video cost estimation."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimate = estimator.estimate_video(
            model="google/veo-3.1-fast",
            seconds=4,
            count=1,
        )

        assert estimate.model == "google/veo-3.1-fast"
        assert estimate.generation_type == "video"
        assert estimate.duration_seconds == 4
        assert estimate.count == 1
        assert estimate.total_min == 0.60
        assert estimate.total_max == 1.20

    def test_estimate_video_multiple(self):
        """Test video cost estimation for multiple outputs."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimate = estimator.estimate_video(
            model="kwaivgi/kling-2.1-standard",
            seconds=5,
            count=3,
        )

        assert estimate.count == 3
        assert estimate.total_min == 0.18 * 3
        assert estimate.total_max == 0.37 * 3

    def test_estimate_video_default_duration(self):
        """Test video estimation with default duration."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimate = estimator.estimate_video(
            model="google/veo-3.1-fast",
            seconds=None,  # Should use first supported duration (4)
        )

        assert estimate.duration_seconds == 4

    def test_estimate_video_invalid_duration(self):
        """Test video estimation with invalid duration."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        with pytest.raises(ValidationError) as exc:
            estimator.estimate_video(
                model="google/veo-3.1-fast",
                seconds=99,  # Not in [4, 6, 8]
            )
        assert "not supported" in str(exc.value)

    def test_estimate_video_wrong_model_type(self):
        """Test video estimation with image model."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        with pytest.raises(ValidationError) as exc:
            estimator.estimate_video(model="openai/gpt-image-1")
        assert "does not support video" in str(exc.value)

    def test_estimate_video_invalid_count(self):
        """Test video estimation with invalid count."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        with pytest.raises(ValidationError) as exc:
            estimator.estimate_video(
                model="google/veo-3.1-fast",
                seconds=4,
                count=0,
            )
        assert "at least 1" in str(exc.value)

    def test_estimate_image(self):
        """Test image cost estimation."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimate = estimator.estimate_image(
            model="openai/gpt-image-1",
            count=1,
        )

        assert estimate.model == "openai/gpt-image-1"
        assert estimate.generation_type == "image"
        assert estimate.duration_seconds is None
        assert estimate.total_min == 0.01
        assert estimate.total_max == 0.30

    def test_estimate_image_multiple(self):
        """Test image estimation for multiple outputs."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimate = estimator.estimate_image(
            model="openai/gpt-image-1",
            count=5,
        )

        assert estimate.count == 5
        assert estimate.total_min == 0.01 * 5
        assert estimate.total_max == 0.30 * 5

    def test_estimate_image_wrong_model_type(self):
        """Test image estimation with video model."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        with pytest.raises(ValidationError) as exc:
            estimator.estimate_image(model="google/veo-3.1-fast")
        assert "does not support image" in str(exc.value)

    def test_estimate_free_model(self):
        """Test estimation for free model."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimate = estimator.estimate_image(
            model="openai/gpt-image-1.5:free",
            count=10,
        )

        assert estimate.total_min == 0.0
        assert estimate.total_max == 0.0
        assert estimate.total_average == 0.0

    def test_model_not_found(self):
        """Test estimation with non-existent model."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {}
        estimator = CostEstimator(mock_client)

        with pytest.raises(ModelNotFoundError):
            estimator.estimate_video(model="nonexistent/model")

    def test_refresh_models(self):
        """Test refreshing model data."""
        client = self._create_mock_client()
        estimator = CostEstimator(client)

        estimator.estimate_video(model="google/veo-3.1-fast", seconds=4)
        estimator.refresh_models()
        estimator.estimate_video(model="google/veo-3.1-fast", seconds=4)

        assert client.list_models.call_count == 2
