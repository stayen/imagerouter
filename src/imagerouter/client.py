"""ImageRouter API client for authentication and base requests."""

from __future__ import annotations

import os
import time
from typing import Any, BinaryIO

import requests
from dotenv import load_dotenv

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

# Load environment variables from .env file
load_dotenv()

BASE_URL = "https://api.imagerouter.io"
DEFAULT_TIMEOUT = 300  # 5 minutes for video generation
DEFAULT_MAX_RETRIES = 3


class ImageRouterClient:
    """Main API client for ImageRouter.io.

    Handles authentication, request execution, and error handling for all
    ImageRouter API endpoints.

    Args:
        api_key: API key for authentication. Falls back to IMAGEROUTER_API_KEY env var.
        timeout: Request timeout in seconds. Defaults to IMAGEROUTER_TIMEOUT env var or 300.
        max_retries: Maximum retry attempts for transient errors.
            Defaults to IMAGEROUTER_MAX_RETRIES env var or 3.

    Raises:
        AuthenticationError: If no API key is provided or found in environment.

    Example:
        >>> client = ImageRouterClient()
        >>> models = client.list_models(output_type="video")
        >>> credits = client.get_credits()
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("IMAGEROUTER_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "API key is required. Provide via api_key parameter or "
                "IMAGEROUTER_API_KEY environment variable."
            )

        self.timeout = timeout or int(os.environ.get("IMAGEROUTER_TIMEOUT", DEFAULT_TIMEOUT))
        self.max_retries = max_retries or int(
            os.environ.get("IMAGEROUTER_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )
        self.base_url = BASE_URL

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "imagerouter-python/0.1.0",
        }

    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the API.

        Args:
            response: The HTTP response object.

        Raises:
            AuthenticationError: For 401 responses.
            RateLimitError: For 429 responses.
            ValidationError: For 400 responses.
            ModelNotFoundError: For 404 responses with model-related errors.
            InsufficientCreditsError: For credit-related errors.
            GenerationError: For 5xx responses.
            ImageRouterError: For other error responses.
        """
        status_code = response.status_code
        try:
            data = response.json()
        except ValueError:
            data = {"error": {"message": response.text}}

        error_message = data.get("error", {}).get("message", response.text)

        if status_code == 401:
            raise AuthenticationError(error_message, status_code, data)

        if status_code == 429:
            raise RateLimitError(error_message, status_code, data)

        if status_code == 400:
            raise ValidationError(error_message, status_code, data)

        if status_code == 404:
            if "model" in error_message.lower():
                raise ModelNotFoundError(error_message, status_code, data)
            raise ImageRouterError(error_message, status_code, data)

        if status_code == 402 or "credit" in error_message.lower():
            raise InsufficientCreditsError(error_message, status_code, data)

        if 500 <= status_code < 600:
            raise GenerationError(error_message, status_code, data)

        raise ImageRouterError(error_message, status_code, data)

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, tuple[str, BinaryIO, str]] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint (e.g., '/v1/models').
            json_data: JSON body for the request.
            data: Form data for multipart requests.
            files: Files for multipart upload.
            timeout: Override default timeout for this request.

        Returns:
            Parsed JSON response data.

        Raises:
            NetworkError: For connection or timeout errors after retries.
            Various ImageRouterError subclasses for API errors.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        request_timeout = timeout or self.timeout

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    data=data,
                    files=files,
                    timeout=request_timeout,
                )

                if response.status_code >= 400:
                    # Don't retry client errors (4xx) except rate limits
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                        time.sleep(retry_after)
                        continue
                    self._handle_error_response(response)

                return response.json()

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Request failed: {e}") from e

        raise NetworkError(
            f"Request failed after {self.max_retries} attempts: {last_exception}"
        )

    def list_models(self, output_type: str | None = None) -> dict[str, Any]:
        """Fetch available models with pricing information.

        Args:
            output_type: Filter by output type ('image' or 'video').
                If None, returns all models.

        Returns:
            Dict mapping model_id to model info including pricing data.
            Structure: {model_id: {name, provider, output, pricing, ...}}

        Example:
            >>> models = client.list_models(output_type="video")
            >>> for model_id, info in models.items():
            ...     print(f"{model_id}: {info['pricing']}")
        """
        response = self._request("GET", "/v1/models")
        models = response.get("data", [])

        # Convert list to dict keyed by model ID
        result = {}
        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue

            # Filter by output type if specified
            if output_type:
                model_outputs = model.get("output", [])
                if output_type not in model_outputs:
                    continue

            result[model_id] = model

        return result

    def get_credits(self) -> dict[str, Any]:
        """Get account credit balance.

        Returns:
            Dict with credit information:
            {
                "remaining_credits": float,
                "credit_usage": float,
                "total_deposits": float
            }

        Example:
            >>> credits = client.get_credits()
            >>> print(f"Remaining: ${credits['remaining_credits']:.2f}")
        """
        response = self._request("GET", "/v1/credits")
        return response

    def test_auth(self) -> bool:
        """Validate API key.

        Returns:
            True if the API key is valid.

        Raises:
            AuthenticationError: If the API key is invalid.

        Example:
            >>> if client.test_auth():
            ...     print("API key is valid")
        """
        self._request("POST", "/v1/auth/test")
        return True

    def post_json(
        self, endpoint: str, data: dict[str, Any], timeout: int | None = None
    ) -> dict[str, Any]:
        """Make a POST request with JSON body.

        Args:
            endpoint: API endpoint.
            data: JSON data to send.
            timeout: Optional timeout override.

        Returns:
            Parsed JSON response.
        """
        return self._request("POST", endpoint, json_data=data, timeout=timeout)

    def post_multipart(
        self,
        endpoint: str,
        data: dict[str, Any],
        files: dict[str, tuple[str, BinaryIO, str]],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Make a POST request with multipart form data.

        Args:
            endpoint: API endpoint.
            data: Form field data.
            files: Files to upload as {field_name: (filename, file_object, mime_type)}.
            timeout: Optional timeout override.

        Returns:
            Parsed JSON response.
        """
        return self._request("POST", endpoint, data=data, files=files, timeout=timeout)
