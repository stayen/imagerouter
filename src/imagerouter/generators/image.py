"""Image generation operations."""

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


class ImageGenerator:
    """Image generation operations.

    Handles text-to-image and image-to-image generation requests.

    Args:
        client: ImageRouterClient instance for API requests.

    Example:
        >>> generator = ImageGenerator(client)
        >>> result = generator.text_to_image(
        ...     prompt="A futuristic cityscape",
        ...     model="openai/gpt-image-1",
        ...     quality="high"
        ... )
        >>> print(result["data"][0]["url"])
    """

    GENERATION_ENDPOINT = "/v1/openai/images/generations"
    EDIT_ENDPOINT = "/v1/openai/images/edits"

    def __init__(self, client: "ImageRouterClient") -> None:
        self._client = client

    def text_to_image(
        self,
        prompt: str,
        model: str,
        quality: str = "auto",
        size: str = "auto",
        output_format: str | None = None,
        response_format: str = "url",
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate image from text prompt.

        Args:
            prompt: Text description.
            model: Model ID (e.g., 'openai/gpt-image-1').
            quality: Quality level ('auto', 'low', 'medium', 'high').
            size: Resolution or 'auto' (model-specific).
            output_format: Optional format specification.
            response_format: Output format ('url', 'b64_json', 'b64_ephemeral').
            output_path: Optional local path to save the image.

        Returns:
            API response dict with image URL or base64 data:
            {
                "created": int,
                "data": [{"url": str, "revised_prompt": str}]
            }

        Raises:
            ValidationError: If parameters are invalid.

        Example:
            >>> result = generator.text_to_image(
            ...     prompt="A cyberpunk street scene at night",
            ...     model="openai/gpt-image-1",
            ...     quality="high",
            ...     size="1024x1024",
            ...     output_path="cyberpunk.png"
            ... )
        """
        clean_prompt = validate_prompt(prompt)

        payload: dict[str, Any] = {
            "prompt": clean_prompt,
            "model": model,
            "response_format": response_format,
        }

        if quality != "auto":
            payload["quality"] = quality

        if size != "auto":
            payload["size"] = size

        if output_format:
            payload["output_format"] = output_format

        result = self._client.post_json(self.GENERATION_ENDPOINT, payload)

        # Handle output file saving
        if output_path:
            self._save_output(result, output_path)

        return result

    def image_to_image(
        self,
        image_path: str | list[str],
        prompt: str,
        model: str,
        mask_path: str | list[str] | None = None,
        quality: str = "auto",
        size: str = "auto",
        response_format: str = "url",
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Edit/transform existing image(s).

        Args:
            image_path: Source image path or list of paths (up to 16).
            prompt: Edit instructions.
            model: Model ID (must support edit).
            mask_path: Optional mask image path(s) for targeted editing.
            quality: Quality level.
            size: Output size.
            response_format: Output format.
            output_path: Optional local path to save the result.

        Returns:
            API response dict with image URL or base64 data.

        Raises:
            ValidationError: If parameters or files are invalid.

        Example:
            >>> result = generator.image_to_image(
            ...     image_path="landscape.jpg",
            ...     prompt="Add a sunset sky",
            ...     model="openai/gpt-image-1",
            ...     output_path="landscape_sunset.png"
            ... )
        """
        clean_prompt = validate_prompt(prompt)

        # Prepare image files
        image_paths = [image_path] if isinstance(image_path, str) else image_path
        image_tuples = prepare_multiple_images(image_paths)

        # Prepare mask files if provided
        mask_tuples = []
        if mask_path:
            mask_paths = [mask_path] if isinstance(mask_path, str) else mask_path
            mask_tuples = prepare_multiple_images(mask_paths)

        try:
            # Build form data
            form_data: dict[str, Any] = {
                "prompt": clean_prompt,
                "model": model,
                "response_format": response_format,
            }

            if quality != "auto":
                form_data["quality"] = quality

            if size != "auto":
                form_data["size"] = size

            # Build files dict for multipart upload
            files: dict[str, tuple[str, Any, str]] = {}

            for i, (filename, file_obj, mime_type) in enumerate(image_tuples):
                files[f"image[{i}]"] = (filename, file_obj, mime_type)

            for i, (filename, file_obj, mime_type) in enumerate(mask_tuples):
                files[f"mask[{i}]"] = (filename, file_obj, mime_type)

            result = self._client.post_multipart(self.EDIT_ENDPOINT, form_data, files)

            # Handle output file saving
            if output_path:
                self._save_output(result, output_path)

            return result

        finally:
            close_file_handles(image_tuples)
            close_file_handles(mask_tuples)

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
