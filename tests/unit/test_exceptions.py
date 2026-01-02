"""Tests for custom exceptions."""

import pytest

from imagerouter.exceptions import (
    AuthenticationError,
    GenerationError,
    ImageRouterError,
    InsufficientCreditsError,
    ModelNotFoundError,
    NetworkError,
    RateLimitError,
    ValidationError,
)


class TestImageRouterError:
    """Tests for the base ImageRouterError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ImageRouterError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.status_code is None
        assert error.response_data is None

    def test_error_with_status_code(self):
        """Test error with status code."""
        error = ImageRouterError("Bad request", status_code=400)
        assert error.message == "Bad request"
        assert error.status_code == 400

    def test_error_with_response_data(self):
        """Test error with response data."""
        data = {"error": {"code": "invalid_param", "message": "Bad request"}}
        error = ImageRouterError("Bad request", status_code=400, response_data=data)
        assert error.response_data == data


class TestSpecificExceptions:
    """Tests for specific exception types."""

    def test_authentication_error(self):
        """Test AuthenticationError inherits from ImageRouterError."""
        error = AuthenticationError("Invalid API key", status_code=401)
        assert isinstance(error, ImageRouterError)
        assert error.status_code == 401

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limit exceeded", status_code=429)
        assert isinstance(error, ImageRouterError)
        assert error.status_code == 429

    def test_insufficient_credits_error(self):
        """Test InsufficientCreditsError."""
        error = InsufficientCreditsError("Not enough credits")
        assert isinstance(error, ImageRouterError)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("Model 'foo/bar' not found", status_code=404)
        assert isinstance(error, ImageRouterError)

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid duration")
        assert isinstance(error, ImageRouterError)

    def test_generation_error(self):
        """Test GenerationError."""
        error = GenerationError("Generation failed", status_code=500)
        assert isinstance(error, ImageRouterError)

    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Connection timeout")
        assert isinstance(error, ImageRouterError)
