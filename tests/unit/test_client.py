"""Tests for the ImageRouter client."""

import pytest
from unittest.mock import MagicMock, patch

from imagerouter.client import ImageRouterClient, BASE_URL
from imagerouter.exceptions import (
    AuthenticationError,
    GenerationError,
    ModelNotFoundError,
    RateLimitError,
    ValidationError,
)

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from fixtures.mock_responses import (
    MOCK_CREDITS_RESPONSE,
    MOCK_MODELS_RESPONSE,
)


class TestImageRouterClientInit:
    """Tests for client initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = ImageRouterClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_init_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict("os.environ", {"IMAGEROUTER_API_KEY": "env_key"}):
            client = ImageRouterClient()
            assert client.api_key == "env_key"

    def test_init_no_key(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError) as exc:
                ImageRouterClient()
            assert "API key is required" in str(exc.value)

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = ImageRouterClient(api_key="test", timeout=600)
        assert client.timeout == 600

    def test_init_custom_retries(self):
        """Test initialization with custom retries."""
        client = ImageRouterClient(api_key="test", max_retries=5)
        assert client.max_retries == 5

    def test_init_from_env_timeout(self):
        """Test timeout from environment variable."""
        with patch.dict("os.environ", {
            "IMAGEROUTER_API_KEY": "test",
            "IMAGEROUTER_TIMEOUT": "120"
        }):
            client = ImageRouterClient()
            assert client.timeout == 120


class TestImageRouterClientRequests:
    """Tests for client HTTP requests."""

    @patch("imagerouter.client.requests.request")
    def test_list_models(self, mock_request):
        """Test listing models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MODELS_RESPONSE
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")
        models = client.list_models()

        assert "google/veo-3.1-fast" in models
        assert "openai/gpt-image-1" in models
        mock_request.assert_called_once()

    @patch("imagerouter.client.requests.request")
    def test_list_models_filter_video(self, mock_request):
        """Test listing models with video filter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_MODELS_RESPONSE
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")
        models = client.list_models(output_type="video")

        assert "google/veo-3.1-fast" in models
        assert "openai/gpt-image-1" not in models

    @patch("imagerouter.client.requests.request")
    def test_get_credits(self, mock_request):
        """Test getting credits."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_CREDITS_RESPONSE
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")
        credits = client.get_credits()

        assert credits["remaining_credits"] == 50.00
        assert credits["credit_usage"] == 25.50

    @patch("imagerouter.client.requests.request")
    def test_test_auth(self, mock_request):
        """Test auth validation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")
        result = client.test_auth()

        assert result is True

    @patch("imagerouter.client.requests.request")
    def test_headers(self, mock_request):
        """Test request headers."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="my_api_key")
        client.list_models()

        call_kwargs = mock_request.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer my_api_key"


class TestImageRouterClientErrors:
    """Tests for client error handling."""

    @patch("imagerouter.client.requests.request")
    def test_authentication_error(self, mock_request):
        """Test 401 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="bad_key")

        with pytest.raises(AuthenticationError) as exc:
            client.list_models()
        assert exc.value.status_code == 401

    @patch("imagerouter.client.requests.request")
    def test_rate_limit_error(self, mock_request):
        """Test 429 error handling with retry."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}
        mock_response_429.json.return_value = {"error": {"message": "Rate limit"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"data": []}

        mock_request.side_effect = [mock_response_429, mock_response_200]

        client = ImageRouterClient(api_key="test", max_retries=2)
        result = client.list_models()

        assert mock_request.call_count == 2

    @patch("imagerouter.client.requests.request")
    def test_validation_error(self, mock_request):
        """Test 400 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Invalid parameter"}}
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")

        with pytest.raises(ValidationError) as exc:
            client.post_json("/v1/test", {"bad": "param"})
        assert exc.value.status_code == 400

    @patch("imagerouter.client.requests.request")
    def test_model_not_found_error(self, mock_request):
        """Test 404 model error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"message": "Model not found"}}
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")

        with pytest.raises(ModelNotFoundError):
            client.post_json("/v1/test", {"model": "bad/model"})

    @patch("imagerouter.client.requests.request")
    def test_generation_error(self, mock_request):
        """Test 500 error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Internal error"}}
        mock_request.return_value = mock_response

        client = ImageRouterClient(api_key="test")

        with pytest.raises(GenerationError) as exc:
            client.list_models()
        assert exc.value.status_code == 500
