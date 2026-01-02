"""Custom exception classes for ImageRouter module."""


class ImageRouterError(Exception):
    """Base exception for ImageRouter errors.

    Args:
        message: Error description.
        status_code: Optional HTTP status code from API response.
        response_data: Optional response data from API.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data


class AuthenticationError(ImageRouterError):
    """Invalid or missing API key.

    Raised when the API returns 401 Unauthorized or when no API key is provided.
    """


class RateLimitError(ImageRouterError):
    """Rate limit exceeded.

    Raised when the API returns 429 Too Many Requests.
    """


class InsufficientCreditsError(ImageRouterError):
    """Account balance too low.

    Raised when the account doesn't have enough credits for the requested operation.
    """


class ModelNotFoundError(ImageRouterError):
    """Requested model not available.

    Raised when attempting to use a model that doesn't exist or is not accessible.
    """


class ValidationError(ImageRouterError):
    """Invalid request parameters.

    Raised when request parameters fail validation before sending to the API,
    or when the API returns 400 Bad Request.
    """


class GenerationError(ImageRouterError):
    """Generation failed on provider side.

    Raised when the generation request was accepted but failed during processing.
    """


class NetworkError(ImageRouterError):
    """Network-related error.

    Raised when there are connection issues, timeouts, or other network problems.
    """
