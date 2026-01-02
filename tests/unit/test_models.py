"""Tests for models module."""

import pytest
from unittest.mock import MagicMock

from imagerouter.models import ModelInfo, ModelRegistry, PricingInfo
from imagerouter.exceptions import ModelNotFoundError

# Import mock data
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from fixtures.mock_responses import MOCK_MODELS_RESPONSE


class TestPricingInfo:
    """Tests for PricingInfo dataclass."""

    def test_from_fixed_pricing(self):
        """Test parsing fixed pricing data."""
        data = {"type": "fixed", "value": 0.23}
        pricing = PricingInfo.from_api_data(data)

        assert pricing.pricing_type == "fixed"
        assert pricing.value == 0.23
        assert pricing.min_price == 0.23
        assert pricing.max_price == 0.23
        assert pricing.average_price == 0.23

    def test_from_range_pricing(self):
        """Test parsing range-based pricing data."""
        data = {
            "type": "calculated",
            "range": {"min": 0.60, "average": 0.90, "max": 1.20},
        }
        pricing = PricingInfo.from_api_data(data)

        assert pricing.pricing_type == "calculated"
        assert pricing.value is None
        assert pricing.min_price == 0.60
        assert pricing.max_price == 1.20
        assert pricing.average_price == 0.90

    def test_get_estimate_fixed(self):
        """Test get_estimate for fixed pricing."""
        pricing = PricingInfo(pricing_type="fixed", value=0.50)
        min_p, avg_p, max_p = pricing.get_estimate()

        assert min_p == 0.50
        assert avg_p == 0.50
        assert max_p == 0.50

    def test_get_estimate_range(self):
        """Test get_estimate for range pricing."""
        pricing = PricingInfo(
            pricing_type="calculated",
            min_price=0.10,
            average_price=0.20,
            max_price=0.30,
        )
        min_p, avg_p, max_p = pricing.get_estimate()

        assert min_p == 0.10
        assert avg_p == 0.20
        assert max_p == 0.30


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_from_api_data_video_model(self):
        """Test parsing video model data."""
        data = MOCK_MODELS_RESPONSE["data"][0]  # google/veo-3.1-fast
        model = ModelInfo.from_api_data(data)

        assert model.id == "google/veo-3.1-fast"
        assert model.name == "Veo 3.1 Fast"
        assert model.provider == "Gemini"
        assert model.output_types == ["video"]
        assert model.supported_durations == [4, 6, 8]
        assert model.is_video_model()
        assert not model.is_image_model()

    def test_from_api_data_image_model(self):
        """Test parsing image model data."""
        data = MOCK_MODELS_RESPONSE["data"][2]  # openai/gpt-image-1
        model = ModelInfo.from_api_data(data)

        assert model.id == "openai/gpt-image-1"
        assert model.is_image_model()
        assert not model.is_video_model()
        assert model.supports_edit

    def test_get_default_duration(self):
        """Test getting default duration."""
        model = ModelInfo(
            id="test",
            name="Test",
            provider="Test",
            output_types=["video"],
            pricing=PricingInfo(pricing_type="fixed", value=0),
            supported_durations=[5, 10],
        )
        assert model.get_default_duration() == 5

    def test_get_default_duration_none(self):
        """Test getting default duration when not available."""
        model = ModelInfo(
            id="test",
            name="Test",
            provider="Test",
            output_types=["image"],
            pricing=PricingInfo(pricing_type="fixed", value=0),
        )
        assert model.get_default_duration() is None


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_get_model(self):
        """Test getting a model by ID."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            model["id"]: model for model in MOCK_MODELS_RESPONSE["data"]
        }

        registry = ModelRegistry(mock_client)
        model = registry.get_model("google/veo-3.1-fast")

        assert model.id == "google/veo-3.1-fast"
        mock_client.list_models.assert_called_once()

    def test_get_model_not_found(self):
        """Test getting a non-existent model."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {}

        registry = ModelRegistry(mock_client)

        with pytest.raises(ModelNotFoundError):
            registry.get_model("nonexistent/model")

    def test_get_video_models(self):
        """Test filtering video models."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            model["id"]: model for model in MOCK_MODELS_RESPONSE["data"]
        }

        registry = ModelRegistry(mock_client)
        video_models = registry.get_video_models()

        assert "google/veo-3.1-fast" in video_models
        assert "ir/test-video" in video_models
        assert "openai/gpt-image-1" not in video_models

    def test_get_image_models(self):
        """Test filtering image models."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            model["id"]: model for model in MOCK_MODELS_RESPONSE["data"]
        }

        registry = ModelRegistry(mock_client)
        image_models = registry.get_image_models()

        assert "openai/gpt-image-1" in image_models
        assert "openai/gpt-image-1.5:free" in image_models
        assert "google/veo-3.1-fast" not in image_models

    def test_caching(self):
        """Test that models are cached."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            model["id"]: model for model in MOCK_MODELS_RESPONSE["data"]
        }

        registry = ModelRegistry(mock_client)

        # First call
        registry.get_all_models()
        # Second call should use cache
        registry.get_all_models()

        # Should only call API once
        mock_client.list_models.assert_called_once()

    def test_refresh(self):
        """Test refreshing model data."""
        mock_client = MagicMock()
        mock_client.list_models.return_value = {
            model["id"]: model for model in MOCK_MODELS_RESPONSE["data"]
        }

        registry = ModelRegistry(mock_client)
        registry.get_all_models()
        registry.refresh()
        registry.get_all_models()

        # Should call API twice (initial + refresh)
        assert mock_client.list_models.call_count == 2
