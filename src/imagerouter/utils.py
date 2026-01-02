"""Utility functions for file handling and validation."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import BinaryIO

import requests

from .exceptions import NetworkError, ValidationError

# Supported image formats for upload
SUPPORTED_IMAGE_FORMATS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Supported video formats for output
SUPPORTED_VIDEO_FORMATS = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def validate_image_path(path: str) -> Path:
    """Validate that a file path points to a supported image.

    Args:
        path: Path to the image file.

    Returns:
        Resolved Path object.

    Raises:
        ValidationError: If the file doesn't exist or has unsupported format.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise ValidationError(f"Image file not found: {path}")

    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise ValidationError(
            f"Unsupported image format '{suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS.keys())}"
        )

    return file_path.resolve()


def get_mime_type(path: str | Path) -> str:
    """Get MIME type for a file.

    Args:
        path: Path to the file.

    Returns:
        MIME type string.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_IMAGE_FORMATS:
        return SUPPORTED_IMAGE_FORMATS[suffix]
    if suffix in SUPPORTED_VIDEO_FORMATS:
        return SUPPORTED_VIDEO_FORMATS[suffix]

    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def prepare_image_for_upload(
    path: str,
) -> tuple[str, BinaryIO, str]:
    """Prepare an image file for multipart upload.

    Args:
        path: Path to the image file.

    Returns:
        Tuple of (filename, file_object, mime_type) for use with requests.

    Raises:
        ValidationError: If the file is invalid.
    """
    file_path = validate_image_path(path)
    mime_type = get_mime_type(file_path)
    file_obj = open(file_path, "rb")
    return (file_path.name, file_obj, mime_type)


def prepare_multiple_images(
    paths: list[str],
) -> list[tuple[str, BinaryIO, str]]:
    """Prepare multiple image files for upload.

    Args:
        paths: List of paths to image files.

    Returns:
        List of (filename, file_object, mime_type) tuples.

    Raises:
        ValidationError: If any file is invalid or too many files.
    """
    if len(paths) > 16:
        raise ValidationError("Maximum 16 images allowed per request")

    return [prepare_image_for_upload(p) for p in paths]


def close_file_handles(files: list[tuple[str, BinaryIO, str]]) -> None:
    """Close file handles after upload.

    Args:
        files: List of file tuples from prepare_image_for_upload.
    """
    for _, file_obj, _ in files:
        try:
            file_obj.close()
        except Exception:
            pass


def download_file(url: str, output_path: str, timeout: int = 60) -> Path:
    """Download a file from URL to local path.

    Args:
        url: URL to download from.
        output_path: Local path to save the file.
        timeout: Request timeout in seconds.

    Returns:
        Path object for the saved file.

    Raises:
        NetworkError: If download fails.
        ValidationError: If output directory doesn't exist.
    """
    out_path = Path(output_path)

    # Ensure parent directory exists
    if out_path.parent and not out_path.parent.exists():
        raise ValidationError(f"Output directory does not exist: {out_path.parent}")

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return out_path.resolve()

    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Failed to download file: {e}") from e


def save_base64_content(data: str, output_path: str) -> Path:
    """Save base64-encoded content to a file.

    Args:
        data: Base64-encoded string.
        output_path: Local path to save the file.

    Returns:
        Path object for the saved file.

    Raises:
        ValidationError: If output directory doesn't exist or data is invalid.
    """
    out_path = Path(output_path)

    # Ensure parent directory exists
    if out_path.parent and not out_path.parent.exists():
        raise ValidationError(f"Output directory does not exist: {out_path.parent}")

    try:
        decoded = base64.b64decode(data)
        with open(out_path, "wb") as f:
            f.write(decoded)
        return out_path.resolve()

    except Exception as e:
        raise ValidationError(f"Failed to decode base64 content: {e}") from e


def infer_output_extension(generation_type: str, output_path: str | None = None) -> str:
    """Infer the appropriate file extension for output.

    Args:
        generation_type: Either 'video' or 'image'.
        output_path: Optional output path that may already have extension.

    Returns:
        File extension including the dot (e.g., '.mp4', '.png').
    """
    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext:
            return ext

    if generation_type == "video":
        return ".mp4"
    return ".png"


def ensure_output_path(
    output_path: str | None,
    generation_type: str,
    model: str,
) -> Path:
    """Ensure output path is valid and has appropriate extension.

    Args:
        output_path: User-provided output path or None for auto-generated.
        generation_type: Either 'video' or 'image'.
        model: Model ID for generating filename.

    Returns:
        Resolved output Path.
    """
    if output_path:
        path = Path(output_path)
        # Add extension if missing
        if not path.suffix:
            ext = infer_output_extension(generation_type)
            path = path.with_suffix(ext)
        return path

    # Auto-generate filename
    import time

    timestamp = int(time.time())
    model_safe = model.replace("/", "_").replace(":", "_")
    ext = infer_output_extension(generation_type)
    return Path(f"{model_safe}_{timestamp}{ext}")


def validate_prompt(prompt: str, max_length: int = 10000) -> str:
    """Validate and clean a text prompt.

    Args:
        prompt: The text prompt.
        max_length: Maximum allowed length.

    Returns:
        Cleaned prompt string.

    Raises:
        ValidationError: If prompt is empty or too long.
    """
    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty")

    cleaned = prompt.strip()

    if len(cleaned) > max_length:
        raise ValidationError(
            f"Prompt too long ({len(cleaned)} chars). Maximum is {max_length} characters."
        )

    return cleaned
