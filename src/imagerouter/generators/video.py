"""Video generation operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import ValidationError
from ..utils import (
    close_file_handles,
    download_file,
    prepare_multiple_images,
    save_base64_content,
    validate_prompt,
)

if TYPE_CHECKING:
    from ..client import ImageRouterClient


class VideoGenerator:
    """Video generation operations.

    Handles text-to-video and image-to-video generation requests.

    Args:
        client: ImageRouterClient instance for API requests.

    Example:
        >>> generator = VideoGenerator(client)
        >>> result = generator.text_to_video(
        ...     prompt="A cat playing piano",
        ...     model="google/veo-3.1-fast",
        ...     seconds=4
        ... )
        >>> print(result["data"][0]["url"])
    """

    ENDPOINT = "/v1/openai/videos/generations"

    def __init__(self, client: "ImageRouterClient") -> None:
        self._client = client

    def text_to_video(
        self,
        prompt: str,
        model: str,
        seconds: int | str = "auto",
        size: str = "auto",
        response_format: str = "url",
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate video from text prompt.

        Args:
            prompt: Text description (max 10000 chars).
            model: Model ID (e.g., 'google/veo-3.1-fast').
            seconds: Duration in seconds or 'auto' for model default.
            size: Resolution or 'auto' for model default.
            response_format: Output format ('url', 'b64_json', 'b64_ephemeral').
            output_path: Optional local path to save the video.

        Returns:
            API response dict with video URL or base64 data:
            {
                "created": int,
                "data": [{"url": str, "revised_prompt": str}]
            }

        Raises:
            ValidationError: If parameters are invalid.
            Various API errors from the client.

        Example:
            >>> result = generator.text_to_video(
            ...     prompt="A serene mountain landscape at sunset",
            ...     model="google/veo-3.1-fast",
            ...     seconds=4,
            ...     output_path="mountain.mp4"
            ... )
        """
        clean_prompt = validate_prompt(prompt)

        payload: dict[str, Any] = {
            "prompt": clean_prompt,
            "model": model,
            "response_format": response_format,
        }

        if seconds != "auto":
            payload["seconds"] = seconds

        if size != "auto":
            payload["size"] = size

        result = self._client.post_json(self.ENDPOINT, payload)

        # Handle output file saving
        if output_path:
            self._save_output(result, output_path)

        return result

    def image_to_video(
        self,
        image_path: str | list[str],
        prompt: str,
        model: str,
        seconds: int | str = "auto",
        size: str = "auto",
        response_format: str = "url",
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate video from image(s).

        Args:
            image_path: Path to image file or list of paths.
            prompt: Text description for animation.
            model: Model ID.
            seconds: Duration in seconds or 'auto'.
            size: Resolution or 'auto'.
            response_format: Output format.
            output_path: Optional local path to save the video.

        Returns:
            API response dict with video URL or base64 data.

        Raises:
            ValidationError: If parameters or files are invalid.

        Example:
            >>> result = generator.image_to_video(
            ...     image_path="flower.jpg",
            ...     prompt="The flower blooms and petals fall",
            ...     model="kwaivgi/kling-2.1-standard",
            ...     seconds=5
            ... )
        """
        clean_prompt = validate_prompt(prompt)

        # Prepare image files
        paths = [image_path] if isinstance(image_path, str) else image_path
        file_tuples = prepare_multiple_images(paths)

        try:
            # Build form data
            form_data: dict[str, Any] = {
                "prompt": clean_prompt,
                "model": model,
                "response_format": response_format,
            }

            if seconds != "auto":
                form_data["seconds"] = str(seconds)

            if size != "auto":
                form_data["size"] = size

            # Build files dict for multipart upload
            files: dict[str, tuple[str, Any, str]] = {}
            for i, (filename, file_obj, mime_type) in enumerate(file_tuples):
                files[f"image[{i}]"] = (filename, file_obj, mime_type)

            result = self._client.post_multipart(self.ENDPOINT, form_data, files)

            # Handle output file saving
            if output_path:
                self._save_output(result, output_path)

            return result

        finally:
            close_file_handles(file_tuples)

    def _save_output(self, result: dict[str, Any], output_path: str) -> Path:
        """Save generation output to local file.

        Args:
            result: API response dict.
            output_path: Local path to save to.

        Returns:
            Path to the saved file.
        """
        data = result.get("data", [])
        if not data:
            raise ValidationError("No output data in response")

        first_item = data[0]

        # Check for URL response
        if "url" in first_item:
            return download_file(first_item["url"], output_path)

        # Check for base64 response
        if "b64_json" in first_item:
            return save_base64_content(first_item["b64_json"], output_path)

        raise ValidationError("Response contains neither URL nor base64 data")
